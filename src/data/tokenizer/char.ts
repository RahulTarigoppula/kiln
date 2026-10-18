/**
 * Character-level tokenizer: vocabulary = sorted unique code points of a corpus.
 */

/** Serialised form of {@link CharTokenizer}. */
export interface CharTokenizerJSON {
  type: "char";
  /** Vocabulary as an array of single-code-point strings, index = id. */
  chars: string[];
}

/** Options for {@link CharTokenizer.encode}. */
export interface CharEncodeOptions {
  /** What to do with characters outside the vocabulary: skip them (default) or throw. */
  unknown?: "skip" | "throw";
}

/** Maps every code point of a corpus to an integer id. */
export class CharTokenizer {
  private readonly chars: readonly string[];
  private readonly ids = new Map<string, number>();

  /** Build from an explicit list of single-code-point strings (index = id). */
  constructor(chars: readonly string[]) {
    this.chars = [...chars];
    this.chars.forEach((c, i) => {
      if (this.ids.has(c)) throw new Error(`CharTokenizer: duplicate character ${JSON.stringify(c)}`);
      this.ids.set(c, i);
    });
  }

  /** Build from a corpus: vocabulary is its sorted (by code point) set of characters. */
  static fromText(text: string): CharTokenizer {
    const set = new Set<string>();
    for (const ch of text) set.add(ch);
    const chars = [...set].sort((a, b) => (a.codePointAt(0) ?? 0) - (b.codePointAt(0) ?? 0));
    return new CharTokenizer(chars);
  }

  /** Number of distinct characters. */
  get vocabSize(): number {
    return this.chars.length;
  }

  /** True when `ch` (one code point) is in the vocabulary. */
  hasChar(ch: string): boolean {
    return this.ids.has(ch);
  }

  /** Token id for a character, or -1. */
  idOf(ch: string): number {
    return this.ids.get(ch) ?? -1;
  }

  /** Character for an id (empty string if out of range). */
  charOf(id: number): string {
    return this.chars[id] ?? "";
  }

  /** Encode a string to ids. Unknown characters are skipped unless `unknown: "throw"`. */
  encode(text: string, opts: CharEncodeOptions = {}): Int32Array {
    const out: number[] = [];
    for (const ch of text) {
      const id = this.ids.get(ch);
      if (id === undefined) {
        if (opts.unknown === "throw") throw new Error(`CharTokenizer: unknown character ${JSON.stringify(ch)}`);
        continue;
      }
      out.push(id);
    }
    return Int32Array.from(out);
  }

  /** Decode ids back to a string (ids out of range are skipped). */
  decode(ids: ArrayLike<number>): string {
    let s = "";
    for (let i = 0; i < ids.length; i++) s += this.chars[ids[i] as number] ?? "";
    return s;
  }

  toJSON(): CharTokenizerJSON {
    return { type: "char", chars: [...this.chars] };
  }

  static fromJSON(json: CharTokenizerJSON): CharTokenizer {
    if (json.type !== "char" || !Array.isArray(json.chars)) throw new Error("CharTokenizer.fromJSON: not a char tokenizer");
    return new CharTokenizer(json.chars);
  }
}
