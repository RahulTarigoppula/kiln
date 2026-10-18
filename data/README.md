# Bundled data

## `tinyshakespeare.txt`

The complete works of William Shakespeare as a single ~1.1 MB plain-text file (public domain).

- Source: Andrej Karpathy's char-rnn repository —
  <https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>
- Used by `tests/tokenizer.test.ts` (BPE training sample), `scripts/train-tiny.ts` (the shipped char-level
  checkpoint) and the Studio page's default dataset.

Re-download with:

```sh
curl -sSL -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
