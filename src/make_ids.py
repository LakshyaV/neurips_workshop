"""Build a token-id stream for a new tokenizer from the v1 GPT-2 stream (same underlying text).

v1's evaluation data is results/wikitext2_test_gpt2_ids.npy (WikiText-2 test, GPT-2 BPE,
concatenated stream reshaped into 1024-token rows at load time). GPT-2 BPE is lossless, so
decoding that stream recovers the exact text; re-encoding it with the target tokenizer gives
the v2 stream over identical text — no second trip through dataset preprocessing that could
drift. No special tokens are inserted: as in v1, sequences are contiguous 1024-token cuts and
the first token of each row is context only (nll is computed on positions 1:).

Usage: python -m src.make_ids --model meta-llama/Llama-3.1-8B --out results/wikitext2_test_llama31_ids.npy
"""
import argparse, os
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src", default=os.path.join(RES, "wikitext2_test_gpt2_ids.npy"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    gpt2 = AutoTokenizer.from_pretrained("gpt2")
    tok = AutoTokenizer.from_pretrained(args.model)
    src = np.load(args.src)
    text = gpt2.decode(src.tolist())
    rt = gpt2(text, add_special_tokens=False)["input_ids"]
    assert rt == src.tolist(), "gpt2 decode/encode round trip failed; text recovery is not exact"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    np.save(args.out, np.asarray(ids, dtype=np.int64))
    print(f"{args.src}: {len(src)} gpt2 tokens -> {len(ids)} {args.model} tokens "
          f"({len(ids)/len(src):.3f}x); wrote {args.out}")


if __name__ == "__main__":
    main()
