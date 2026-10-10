#!/usr/bin/env python3
"""Score tokenizers -- any supported format, or a named reference -- on held-out text, one
table per domain (e.g. english / code / math).

WHAT CAN BE BENCHMARKED
-----------------------
Any mix of these, as positional arguments:

    a file                      detected by its CONTENT, not its name:
        naive_bbpe .model           encoded and decoded by our own binary
        .tiktoken                   via tiktoken; needs its .tiktoken.json beside it (the regex)
        tokenizer.json              via HuggingFace tokenizers (any HF tokenizer file)
    tiktoken:<name>             one of tiktoken's built-ins: gpt2, cl100k_base, o200k_base, ...
    hf:<repo>                   a HuggingFace model's tokenizer, e.g. hf:HuggingFaceTB/SmolLM2-135M

Each text file is encoded WHOLE, in one call, by every tokenizer -- only the test files,
never the training corpus. Text that looks like a special token (a literal <|endoftext|>)
is encoded as ORDINARY text by every tokenizer, so all are scored on the same input.

THE COLUMNS (one table per domain)
----------------------------------
    vocab        ids the tokenizer has, special tokens included
    bytes/tok    file bytes / tokens -- higher is more text per token; needs no notion of "word"
    tok/word     FERTILITY, the primary score -- tokens / words, words = text split on
                 whitespace. Every tokenizer is divided by the same word count, so it is a
                 fair comparison between them; on code and math a "word" is rough, so compare
                 tokenizers with each other there, not with published numbers.
    renyi        Renyi efficiency (alpha 2.5) = H_alpha / log(vocab): how evenly the
                 vocabulary is used on this text. A DIAGNOSTIC, not a score to maximise.
                 H_a beside it is the raw entropy. Same code as the step-11 bench.py.
    round trip   decoding the ids gives back the file byte for byte
    ids          every id is below the vocab size

Exits non-zero if any round trip or id check fails; the tables are printed either way.

COMPARE AT EQUAL VOCAB SIZE. A bigger vocabulary always gives fewer tokens per word, so
compare our slice at a reference's size (v49152 vs SmolLM2, v50257 vs gpt2).

USAGE
-----
    python3 scripts/eval/bench_tokenizers.py \\
        runs/prod/sliced/v49152/tokenizer.tiktoken  hf:HuggingFaceTB/SmolLM2-135M \\
        runs/prod/sliced/v50257/tokenizer.tiktoken  tiktoken:gpt2 \\
        --text english=test_english.txt code=test_code.txt math=test_math.txt \\
        --csv runs/prod/bench.csv
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from bench import renyi  # noqa: E402  -- the step-11 implementation, not a copy
from export_tokenizer import DEFAULT_BINARY, load_tiktoken, read_model  # noqa: E402
from slice_tokenizer import NaiveModel, Tiktoken  # noqa: E402  -- format detection


class Refused(Exception):
    pass


class Loaded:
    """One tokenizer, reduced to what the benchmark needs: a label, its vocab size, and
    encode / decode functions that treat special-token text as ordinary text."""

    def __init__(self, label, vocab, encode, decode):
        self.label, self.vocab, self.encode, self.decode = label, vocab, encode, decode


def load_hf_file(label, path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(path))
    tok.encode_special_tokens = True          # special-token text is ordinary text
    return Loaded(label, tok.get_vocab_size(with_added_tokens=True),
                  lambda t: tok.encode(t, add_special_tokens=False).ids,
                  lambda ids: tok.decode(ids, skip_special_tokens=False))


def load_tiktoken_encoding(label, enc):
    return Loaded(label, enc.n_vocab, enc.encode_ordinary, enc.decode)


def load_ours(label, path, binary):
    if not Path(binary).exists():
        raise Refused(f"{label}: {binary} not found -- run `cargo build --release` first")
    m = read_model(path)
    vocab = len(m["tokens"]) + len(m["specials"])

    def encode(text):
        run = subprocess.run([str(binary), "encode", "--model", str(path), "--allow-special",
                              "none"], input=text.encode("utf-8"), capture_output=True)
        if run.returncode != 0:
            raise RuntimeError(f"naive_bbpe encode failed: {run.stderr.decode()}")
        return [int(x) for x in run.stdout.split()]

    def decode(ids):
        run = subprocess.run([str(binary), "decode", "--model", str(path)],
                             input=" ".join(map(str, ids)).encode(), capture_output=True)
        if run.returncode != 0:
            raise RuntimeError(f"naive_bbpe decode failed: {run.stderr.decode()}")
        return run.stdout.decode("utf-8")

    return Loaded(label, vocab, encode, decode)


def load(spec, binary):
    """Turn one command-line argument into a Loaded tokenizer, or refuse with the reason."""
    if spec.startswith("tiktoken:"):
        import tiktoken
        name = spec.split(":", 1)[1]
        try:
            return load_tiktoken_encoding(spec, tiktoken.get_encoding(name))
        except ValueError:
            raise Refused(f"{spec}: tiktoken has no encoding named {name!r} "
                          f"(it has: {', '.join(tiktoken.list_encoding_names())})")
    if spec.startswith("hf:"):
        # huggingface_hub, not tokenizers' own from_pretrained: measured at prod.6, the
        # latter hung on this machine while hf_hub_download fetched the same file in 0.6 s.
        from huggingface_hub import hf_hub_download
        repo = spec.split(":", 1)[1]
        try:
            return load_hf_file(spec, hf_hub_download(repo, "tokenizer.json"))
        except Exception as e:
            raise Refused(f"{spec}: could not download its tokenizer.json ({type(e).__name__}: {e})")

    path = Path(spec)
    if not path.is_file():
        raise Refused(f"{spec}: no such file (and not tiktoken:<name> or hf:<repo>)")
    if NaiveModel.detect(path):
        return load_ours(spec, path, binary)
    if Tiktoken.detect(path):
        sidecar = Path(f"{path}.json")
        if not str(path).endswith(".tiktoken") or not sidecar.exists():
            raise Refused(f"{spec}: a .tiktoken rank file needs {sidecar.name} beside it -- "
                          "it holds the regex tiktoken splits text with")
        return load_tiktoken_encoding(spec, load_tiktoken(str(path)[:-len(".tiktoken")]))
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("model"), dict):
            return load_hf_file(spec, path)
    except (ValueError, UnicodeDecodeError):
        pass
    raise Refused(f"{spec}: not a naive_bbpe .model, a .tiktoken rank file, or a "
                  "HuggingFace tokenizer.json")


def score(tok, text, n_bytes, n_words):
    ids = tok.encode(text)
    h_alpha, efficiency = renyi(ids, tok.vocab)
    return {
        "tokenizer": tok.label, "vocab_size": tok.vocab, "bytes": n_bytes, "words": n_words,
        "tokens": len(ids),
        "bytes_per_token": n_bytes / len(ids), "tokens_per_word": len(ids) / n_words,
        "renyi_h_alpha": h_alpha, "renyi_efficiency": efficiency,
        "round_trip": tok.decode(ids) == text,
        "ids_in_range": max(ids) < tok.vocab and min(ids) >= 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tokenizers", nargs="+", metavar="TOKENIZER",
                    help="a .model / .tiktoken / tokenizer.json file, tiktoken:<name>, or hf:<repo>")
    ap.add_argument("--text", nargs="+", required=True, metavar="DOMAIN=FILE",
                    help="held-out text per domain, e.g. english=test_english.txt")
    ap.add_argument("--csv", help="also write every row here")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY), help="naive_bbpe, for .model files")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    # Everything that can be refused is refused before any encoding starts.
    try:
        domains = []
        for item in a.text:
            if "=" not in item:
                raise Refused(f"--text {item}: expected DOMAIN=FILE, e.g. english=test_english.txt")
            name, path = item.split("=", 1)
            if not Path(path).is_file():
                raise Refused(f"--text {item}: no such file")
            domains.append((name, Path(path)))
        toks = [load(spec, a.binary) for spec in a.tokenizers]
    except Refused as e:
        raise SystemExit(f"refused: {e}")

    rows, failures = [], []
    width = max(len(t.label) for t in toks)
    for name, path in domains:
        text = path.read_bytes().decode("utf-8")
        n_bytes, n_words = len(text.encode("utf-8")), len(text.split())
        print(f"\n{name.upper()}  ({path}: {n_bytes:,} bytes, {n_words:,} words)")
        print(f"  {'tokenizer':<{width}}  {'vocab':>7}  {'bytes/tok':>9}  {'tok/word':>8}  "
              f"{'renyi':>6}  {'H_a':>6}  {'round trip':>10}  {'ids':>4}")
        for tok in toks:
            r = score(tok, text, n_bytes, n_words)
            rows.append({"domain": name, **r})
            ok = lambda b: "ok" if b else "FAIL"
            print(f"  {r['tokenizer']:<{width}}  {r['vocab_size']:>7,}  {r['bytes_per_token']:>9.3f}  "
                  f"{r['tokens_per_word']:>8.4f}  {r['renyi_efficiency']:>6.4f}  "
                  f"{r['renyi_h_alpha']:>6.3f}  {ok(r['round_trip']):>10}  {ok(r['ids_in_range']):>4}")
            for check in ("round_trip", "ids_in_range"):
                if not r[check]:
                    failures.append(f"{tok.label} on {name}: {check.replace('_', ' ')} FAILED")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")
    if failures:
        print("\n" + "\n".join(failures))
        sys.exit(1)


if __name__ == "__main__":
    main()
