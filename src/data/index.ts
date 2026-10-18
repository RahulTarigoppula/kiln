/** Data utilities: tokenizers and the token dataset. */

export { CharTokenizer, type CharTokenizerJSON, type CharEncodeOptions } from "./tokenizer/char";
export {
  ByteLevelBPE,
  ENDOFTEXT,
  GPT2_PRETOKENIZE_PATTERN,
  type BpeJSON,
  type BpeTrainOptions,
  type BpeEncodeOptions,
} from "./tokenizer/bpe";
export { TokenDataset, type BatchRng } from "./dataset";

import { CharTokenizer, type CharTokenizerJSON } from "./tokenizer/char";
import { ByteLevelBPE, type BpeJSON } from "./tokenizer/bpe";

/** Any serialised tokenizer. */
export type TokenizerJSON = CharTokenizerJSON | BpeJSON;

/** Common tokenizer surface used by the trainer/UI. */
export interface Tokenizer {
  readonly vocabSize: number;
  encode(text: string): Int32Array;
  decode(ids: ArrayLike<number>): string;
  toJSON(): TokenizerJSON;
}

/** Revive a tokenizer from its JSON form. */
export function tokenizerFromJSON(json: TokenizerJSON): Tokenizer {
  if (json.type === "char") return CharTokenizer.fromJSON(json);
  if (json.type === "bpe") return ByteLevelBPE.fromJSON(json);
  throw new Error("tokenizerFromJSON: unknown tokenizer type");
}
