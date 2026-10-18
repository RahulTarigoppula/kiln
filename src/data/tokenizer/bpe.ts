/**
 * Byte-level BPE tokenizer in the style of GPT-2.
 *
 * - Text is pre-tokenised with the GPT-2 regex, each pre-token is UTF-8 encoded and every byte is mapped
 *   to a printable unicode "byte character" (GPT-2's `bytes_to_unicode`), so merges operate on strings and
 *   any input round-trips exactly.
 * - Training counts adjacent symbol pairs over the unique pre-tokens (weighted by frequency) and merges the
 *   most frequent pair repeatedly; pair counts are updated incrementally (only words containing the merged
 *   pair are revisited), so 1 MB of text trains a 1k vocabulary in a few seconds.
 * - Ids 0..255 are the byte characters (GPT-2 order), followed by one id per merge, then special tokens.
 */

/** Serialised form of {@link ByteLevelBPE}. */
export interface BpeJSON {
  type: "bpe";
  version: 1;
  /** token string (byte-character alphabet) → id, including special tokens. */
  vocab: Record<string, number>;
  /** Merge rules in rank order, each as "left right". */
  merges: string[];
  specialTokens: string[];
}

/** Options for {@link ByteLevelBPE.train}. */
export interface BpeTrainOptions {
  /** Stop merging when the best pair occurs fewer times than this (default 2). */
  minFrequency?: number;
  /** Special tokens appended after the merges (default `["<|endoftext|>"]`). */
  specialTokens?: string[];
  /** Progress callback (mergesDone, mergesTotal). */
  onProgress?: (done: number, total: number) => void;
}

/** Options for {@link ByteLevelBPE.encode}. */
export interface BpeEncodeOptions {
  /** Recognise special-token strings in the input and emit their ids (default true). */
  allowSpecial?: boolean;
}

/** GPT-2 pre-tokenisation pattern. */
export const GPT2_PRETOKENIZE_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

/** Default end-of-text special token. */
export const ENDOFTEXT = "<|endoftext|>";

/** Byte order used by GPT-2 for ids 0..255, and the byte → character map. */
function buildByteMaps(): { byteOrder: number[]; byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const byteToChar = new Array<string>(256);
  const charToByte = new Map<string, number>();
  bs.forEach((b, i) => {
    const ch = String.fromCodePoint(cs[i] as number);
    byteToChar[b] = ch;
    charToByte.set(ch, b);
  });
  return { byteOrder: bs, byteToChar, charToByte };
}

const BYTE_MAPS = buildByteMaps();
const SEP = "\u0000"; // never produced by the byte alphabet (byte 0 maps to U+0100)
const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder("utf-8");

/** Map a pre-token to its byte-character symbols. */
function toByteChars(word: string): string[] {
  const bytes = textEncoder.encode(word);
  const out = new Array<string>(bytes.length);
  for (let i = 0; i < bytes.length; i++) out[i] = BYTE_MAPS.byteToChar[bytes[i] as number] as string;
  return out;
}

function escapeRegex(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** GPT-2-compatible byte-level BPE. */
export class ByteLevelBPE {
  private readonly encoder = new Map<string, number>();
  private readonly decoder = new Map<number, string>();
  private readonly ranks = new Map<string, number>();
  private readonly mergeList: string[];
  private readonly specials: string[];
  private specialRegex: RegExp | null = null;
  private readonly cache = new Map<string, number[]>();

  /**
   * @param vocab token → id (must include the 256 byte characters)
   * @param merges "left right" rules in rank order
   * @param specialTokens special-token strings present in `vocab`
   */
  constructor(vocab: Record<string, number>, merges: readonly string[], specialTokens: readonly string[] = []) {
    for (const [tok, id] of Object.entries(vocab)) {
      this.encoder.set(tok, id);
      this.decoder.set(id, tok);
    }
    this.mergeList = [...merges];
    this.mergeList.forEach((m, i) => {
      const sp = m.indexOf(" ");
      if (sp < 0) throw new Error(`ByteLevelBPE: malformed merge "${m}"`);
      this.ranks.set(m.slice(0, sp) + SEP + m.slice(sp + 1), i);
    });
    this.specials = [...specialTokens];
    for (const s of this.specials) if (!this.encoder.has(s)) throw new Error(`ByteLevelBPE: special token ${s} missing from vocab`);
    if (this.specials.length) {
      const sorted = [...this.specials].sort((a, b) => b.length - a.length);
      this.specialRegex = new RegExp(sorted.map(escapeRegex).join("|"), "g");
    }
  }

  /** Total number of ids (bytes + merges + specials). */
  get vocabSize(): number {
    return this.encoder.size;
  }

  /** Merge rules in rank order ("left right"). */
  get merges(): readonly string[] {
    return this.mergeList;
  }

  /** Special-token strings. */
  get specialTokens(): readonly string[] {
    return this.specials;
  }

  /** Id of a token string (byte-character alphabet or special token), or -1. */
  tokenToId(token: string): number {
    return this.encoder.get(token) ?? -1;
  }

  /** Human-readable text of an id (bytes decoded leniently). */
  idToText(id: number): string {
    const tok = this.decoder.get(id);
    if (tok === undefined) return "";
    if (this.specials.includes(tok)) return tok;
    return textDecoder.decode(this.tokenBytes(tok));
  }

  // ------------------------------------------------------------------ train

  /** Train merges from `text` until `vocabSize` ids exist (bytes + merges + specials). */
  static train(text: string, vocabSize: number, opts: BpeTrainOptions = {}): ByteLevelBPE {
    const minFrequency = opts.minFrequency ?? 2;
    const specials = opts.specialTokens ?? [ENDOFTEXT];
    const targetMerges = vocabSize - 256 - specials.length;
    if (targetMerges < 0) throw new Error(`ByteLevelBPE.train: vocabSize ${vocabSize} is smaller than 256 + ${specials.length} specials`);

    // 1. unique pre-tokens with frequencies
    const freq = new Map<string, number>();
    for (const m of text.matchAll(GPT2_PRETOKENIZE_PATTERN)) freq.set(m[0], (freq.get(m[0]) ?? 0) + 1);
    const words: string[][] = [];
    const wfreq: number[] = [];
    for (const [w, f] of freq) {
      words.push(toByteChars(w));
      wfreq.push(f);
    }

    // 2. pair statistics
    const counts = new Map<string, number>();
    const where = new Map<string, Set<number>>();
    const addWord = (wi: number, sign: number): void => {
      const syms = words[wi] as string[];
      const f = (wfreq[wi] as number) * sign;
      for (let i = 0; i + 1 < syms.length; i++) {
        const key = (syms[i] as string) + SEP + (syms[i + 1] as string);
        const c = (counts.get(key) ?? 0) + f;
        if (c <= 0) counts.delete(key);
        else counts.set(key, c);
        if (sign > 0) {
          let set = where.get(key);
          if (!set) {
            set = new Set();
            where.set(key, set);
          }
          set.add(wi);
        }
      }
    };
    for (let wi = 0; wi < words.length; wi++) addWord(wi, 1);

    // 3. merge loop
    const merges: string[] = [];
    while (merges.length < targetMerges) {
      let bestKey: string | null = null;
      let bestCount = 0;
      for (const [key, c] of counts) {
        if (c > bestCount || (c === bestCount && bestKey !== null && key < bestKey)) {
          bestCount = c;
          bestKey = key;
        }
      }
      if (bestKey === null || bestCount < minFrequency) break;
      const sep = bestKey.indexOf(SEP);
      const left = bestKey.slice(0, sep);
      const right = bestKey.slice(sep + 1);
      const merged = left + right;
      merges.push(`${left} ${right}`);

      const affected = where.get(bestKey);
      if (affected) {
        for (const wi of affected) {
          const syms = words[wi] as string[];
          let has = false;
          for (let i = 0; i + 1 < syms.length; i++) {
            if (syms[i] === left && syms[i + 1] === right) {
              has = true;
              break;
            }
          }
          if (!has) continue;
          addWord(wi, -1);
          const out: string[] = [];
          for (let i = 0; i < syms.length; i++) {
            if (i + 1 < syms.length && syms[i] === left && syms[i + 1] === right) {
              out.push(merged);
              i++;
            } else {
              out.push(syms[i] as string);
            }
          }
          words[wi] = out;
          addWord(wi, 1);
        }
        where.delete(bestKey);
      }
      counts.delete(bestKey);
      opts.onProgress?.(merges.length, targetMerges);
    }

    // 4. vocabulary
    const vocab: Record<string, number> = {};
    BYTE_MAPS.byteOrder.forEach((b, i) => {
      vocab[BYTE_MAPS.byteToChar[b] as string] = i;
    });
    merges.forEach((m, i) => {
      vocab[m.replace(" ", "")] = 256 + i;
    });
    specials.forEach((s, i) => {
      vocab[s] = 256 + merges.length + i;
    });
    return new ByteLevelBPE(vocab, merges, specials);
  }

  // ----------------------------------------------------------------- encode

  /** Apply merges to one pre-token's symbols. */
  private bpe(symsIn: string[]): string[] {
    let syms = symsIn;
    while (syms.length > 1) {
      let bestRank = Infinity;
      let bestIdx = -1;
      for (let i = 0; i + 1 < syms.length; i++) {
        const r = this.ranks.get((syms[i] as string) + SEP + (syms[i + 1] as string));
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestIdx = i;
        }
      }
      if (bestIdx < 0) break;
      const left = syms[bestIdx] as string;
      const right = syms[bestIdx + 1] as string;
      const merged = left + right;
      const out: string[] = [];
      for (let i = 0; i < syms.length; i++) {
        if (i + 1 < syms.length && syms[i] === left && syms[i + 1] === right) {
          out.push(merged);
          i++;
        } else {
          out.push(syms[i] as string);
        }
      }
      syms = out;
    }
    return syms;
  }

  private encodeWord(word: string, out: number[]): void {
    let ids = this.cache.get(word);
    if (!ids) {
      ids = this.bpe(toByteChars(word)).map((tok) => {
        const id = this.encoder.get(tok);
        if (id === undefined) throw new Error(`ByteLevelBPE: token ${JSON.stringify(tok)} missing from vocab`);
        return id;
      });
      if (this.cache.size < 100_000) this.cache.set(word, ids);
    }
    for (const id of ids) out.push(id);
  }

  private encodeOrdinary(text: string, out: number[]): void {
    for (const m of text.matchAll(GPT2_PRETOKENIZE_PATTERN)) this.encodeWord(m[0], out);
  }

  /** Encode text to token ids. */
  encode(text: string, opts: BpeEncodeOptions = {}): Int32Array {
    const out: number[] = [];
    const allowSpecial = opts.allowSpecial ?? true;
    if (allowSpecial && this.specialRegex) {
      let last = 0;
      this.specialRegex.lastIndex = 0;
      for (const m of text.matchAll(this.specialRegex)) {
        const idx = m.index ?? 0;
        if (idx > last) this.encodeOrdinary(text.slice(last, idx), out);
        out.push(this.encoder.get(m[0]) as number);
        last = idx + m[0].length;
      }
      if (last < text.length) this.encodeOrdinary(text.slice(last), out);
    } else {
      this.encodeOrdinary(text, out);
    }
    return Int32Array.from(out);
  }

  // ----------------------------------------------------------------- decode

  private tokenBytes(tok: string): Uint8Array {
    const bytes: number[] = [];
    for (const ch of tok) {
      const b = BYTE_MAPS.charToByte.get(ch);
      if (b !== undefined) bytes.push(b);
    }
    return Uint8Array.from(bytes);
  }

  /** Decode ids to text. Special tokens are emitted verbatim; unknown ids are skipped. */
  decode(ids: ArrayLike<number>): string {
    let out = "";
    let pending: number[] = [];
    const flush = (): void => {
      if (pending.length) {
        out += textDecoder.decode(Uint8Array.from(pending));
        pending = [];
      }
    };
    for (let i = 0; i < ids.length; i++) {
      const tok = this.decoder.get(ids[i] as number);
      if (tok === undefined) continue;
      if (this.specials.includes(tok)) {
        flush();
        out += tok;
        continue;
      }
      for (const ch of tok) {
        const b = BYTE_MAPS.charToByte.get(ch);
        if (b !== undefined) pending.push(b);
      }
    }
    flush();
    return out;
  }

  // -------------------------------------------------------------- serialise

  toJSON(): BpeJSON {
    const vocab: Record<string, number> = {};
    for (const [tok, id] of this.encoder) vocab[tok] = id;
    return { type: "bpe", version: 1, vocab, merges: [...this.mergeList], specialTokens: [...this.specials] };
  }

  static fromJSON(json: BpeJSON): ByteLevelBPE {
    if (json.type !== "bpe") throw new Error("ByteLevelBPE.fromJSON: not a bpe tokenizer");
    return new ByteLevelBPE(json.vocab, json.merges, json.specialTokens ?? []);
  }

  /** GPT-2 style files: `vocab.json` contents and `merges.txt` contents (with the `#version` header). */
  toGpt2Files(): { vocabJson: string; mergesTxt: string } {
    const vocab: Record<string, number> = {};
    for (const [tok, id] of this.encoder) vocab[tok] = id;
    return { vocabJson: JSON.stringify(vocab), mergesTxt: ["#version: 0.2", ...this.mergeList].join("\n") + "\n" };
  }

  /** Load from GPT-2 style `vocab.json` / `merges.txt` contents. Special tokens must exist in the vocab. */
  static fromGpt2Files(vocabJson: string, mergesTxt: string, specialTokens: string[] = [ENDOFTEXT]): ByteLevelBPE {
    const vocab = JSON.parse(vocabJson) as Record<string, number>;
    const merges = mergesTxt
      .split(/\r?\n/)
      .filter((l) => l.length > 0 && !l.startsWith("#version"));
    const specials = specialTokens.filter((s) => s in vocab);
    return new ByteLevelBPE(vocab, merges, specials);
  }
}
