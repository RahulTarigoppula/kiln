import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { ByteLevelBPE, CharTokenizer, ENDOFTEXT, tokenizerFromJSON } from "../src/data/index";
import { Random } from "../src/tensor/random";

const corpus = readFileSync(new URL("../data/tinyshakespeare.txt", import.meta.url), "utf8");
const sample = corpus.slice(0, 200 * 1024);

/** Random unicode strings mixing ASCII, emoji, CJK, accents, whitespace and newlines. */
function randomUnicodeStrings(count: number, seed: number): string[] {
  const rng = new Random(seed);
  const pools = [
    "the quick brown fox jumps over the lazy dog ",
    "Hello, World! 123 4.56 (test) [ok] {x} <y> 'quoted' \"dq\" ",
    "😀🎉🚀🧠🔥👍🏽👩‍👩‍👧‍👦🇯🇵",
    "日本語のテキスト 中文文本 한국어 텍스트",
    "Ünïcödé àccénts ñ ß æ ø å çé",
    "\n\n\t  \r\n   ",
    "Привет мир Ελληνικά עברית العربية",
    "𝔘𝔫𝔦𝔠𝔬𝔡𝔢 ∑∫∂√∞ ≠ ≤ ≥",
  ];
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    const parts = 1 + rng.int(6);
    let s = "";
    for (let p = 0; p < parts; p++) {
      const pool = [...(pools[rng.int(pools.length)] as string)];
      const start = rng.int(pool.length);
      const len = 1 + rng.int(pool.length - start);
      s += pool.slice(start, start + len).join("");
    }
    out.push(s);
  }
  return out;
}

describe("ByteLevelBPE", () => {
  const vocabSize = 512;
  const t0 = performance.now();
  const bpe = ByteLevelBPE.train(sample, vocabSize);
  const trainMs = performance.now() - t0;

  it("trains to the requested vocab size quickly", () => {
    expect(bpe.vocabSize).toBe(vocabSize);
    expect(bpe.merges.length).toBe(vocabSize - 256 - 1);
    expect(bpe.specialTokens).toEqual([ENDOFTEXT]);
    expect(trainMs).toBeLessThan(20_000);
  });

  it("compresses the training text", () => {
    const text = sample.slice(0, 20_000);
    const ids = bpe.encode(text);
    const bytes = new TextEncoder().encode(text).length;
    expect(ids.length).toBeLessThan(bytes * 0.6);
    expect(bpe.decode(ids)).toBe(text);
  });

  it("round-trips 100 random unicode strings exactly", () => {
    for (const s of randomUnicodeStrings(100, 7)) {
      const ids = bpe.encode(s);
      expect(ids).toBeInstanceOf(Int32Array);
      expect(bpe.decode(ids)).toBe(s);
    }
  });

  it("round-trips the whole 200 KB sample", () => {
    expect(bpe.decode(bpe.encode(sample))).toBe(sample);
  });

  it("handles special tokens", () => {
    const eot = bpe.tokenToId(ENDOFTEXT);
    expect(eot).toBe(vocabSize - 1);
    const ids = bpe.encode(`First${ENDOFTEXT}Second`);
    expect(Array.from(ids)).toContain(eot);
    expect(bpe.decode(ids)).toBe(`First${ENDOFTEXT}Second`);
    const plain = bpe.encode(`First${ENDOFTEXT}Second`, { allowSpecial: false });
    expect(Array.from(plain)).not.toContain(eot);
  });

  it("serialises to JSON and back", () => {
    const json = bpe.toJSON();
    const text = JSON.stringify(json);
    const again = ByteLevelBPE.fromJSON(JSON.parse(text));
    expect(again.vocabSize).toBe(bpe.vocabSize);
    const s = "ROMEO: But soft, what light through yonder window breaks? 🌙";
    expect(Array.from(again.encode(s))).toEqual(Array.from(bpe.encode(s)));
    const generic = tokenizerFromJSON(json);
    expect(generic.decode(generic.encode(s))).toBe(s);
  });

  it("exports and imports GPT-2 style vocab.json / merges.txt", () => {
    const { vocabJson, mergesTxt } = bpe.toGpt2Files();
    expect(mergesTxt.startsWith("#version: 0.2\n")).toBe(true);
    const vocab = JSON.parse(vocabJson) as Record<string, number>;
    expect(vocab["!"]).toBe(0); // GPT-2 byte ordering
    expect(Object.keys(vocab).length).toBe(vocabSize);
    const again = ByteLevelBPE.fromGpt2Files(vocabJson, mergesTxt);
    const s = "To be, or not to be: that is the question.";
    expect(Array.from(again.encode(s))).toEqual(Array.from(bpe.encode(s)));
  });

  it("is deterministic", () => {
    const b2 = ByteLevelBPE.train(sample.slice(0, 50_000), 300);
    const b3 = ByteLevelBPE.train(sample.slice(0, 50_000), 300);
    expect(b2.merges).toEqual(b3.merges);
  });

  it("stops early when pairs fall below minFrequency", () => {
    const tiny = ByteLevelBPE.train("ab ab ab cd", 1000, { minFrequency: 2 });
    expect(tiny.vocabSize).toBeLessThan(1000);
    expect(tiny.decode(tiny.encode("ab ab ab cd xyz"))).toBe("ab ab ab cd xyz");
  });
});

describe("CharTokenizer", () => {
  const tok = CharTokenizer.fromText(sample);

  it("has a sorted unique vocabulary", () => {
    const chars = tok.toJSON().chars;
    expect(new Set(chars).size).toBe(chars.length);
    for (let i = 1; i < chars.length; i++) {
      expect((chars[i] as string).codePointAt(0)!).toBeGreaterThan((chars[i - 1] as string).codePointAt(0)!);
    }
    expect(tok.vocabSize).toBe(chars.length);
    expect(tok.vocabSize).toBeGreaterThan(50);
  });

  it("round-trips text and skips unknown characters", () => {
    const s = sample.slice(1000, 2000);
    const ids = tok.encode(s);
    expect(ids.length).toBe([...s].length);
    expect(tok.decode(ids)).toBe(s);
    expect(tok.decode(tok.encode("abc😀"))).toBe("abc");
    expect(() => tok.encode("😀", { unknown: "throw" })).toThrow();
  });

  it("serialises", () => {
    const again = CharTokenizer.fromJSON(JSON.parse(JSON.stringify(tok.toJSON())));
    expect(Array.from(again.encode("Hello"))).toEqual(Array.from(tok.encode("Hello")));
    expect(() => new CharTokenizer(["a", "a"])).toThrow();
  });
});
