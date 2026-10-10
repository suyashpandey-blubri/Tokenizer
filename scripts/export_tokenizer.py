#!/usr/bin/env python3
"""Export a trained naive_bbpe `.model` to HuggingFace `tokenizer.json` and/or tiktoken.

WHY THIS EXISTS
---------------
Our `.model` is our own text format. Inference stacks read their own formats instead:
HuggingFace `tokenizers` reads `tokenizer.json`, tiktoken reads a `.tiktoken` rank file plus
a regex and special tokens supplied in code. Both libraries implement the same thing we do
-- byte-level BPE behind a regex pre-tokenizer -- so the tokenizer can be expressed in either
EXACTLY, with the same ids: bytes 0-255, merges from 256 in learned order, specials on top.
(SentencePiece cannot: its pieces must be whole characters, and it has no regex stage.)

"Exactly" is a claim, so it is checked rather than assumed: `--check FILE` encodes FILE with
our binary and with every exported file through the real library, and fails on any
difference in any id.

WHAT IS WRITTEN (into --out-dir)
--------------------------------
    tokenizer.json              HuggingFace:  tokenizers.Tokenizer.from_file(...)
    tokenizer.tiktoken          tiktoken:     one "<base64 token bytes> <id>" line per token
    tokenizer.tiktoken.json     tiktoken:     the regex and the special tokens

tiktoken's own `load_tiktoken_bpe` refuses local files unless the optional `blobfile`
package is installed, so `load_tiktoken()` below reads the rank file directly instead.

Two things a `.model` can hold that an export cannot, both refused rather than mis-exported:
a normalizer for tiktoken (tiktoken has no normalization stage; HF does), and two tokens
with identical bytes (both formats key tokens by their bytes; never seen in practice).

USAGE
-----
    python3 scripts/export_tokenizer.py models/mytok.model --format both --out-dir out/ \\
        --check some_text.txt
"""

import argparse
import base64
import json
import subprocess
import sys
import unicodedata
from pathlib import Path

MODEL_HEADER = "naive_bbpe v1"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_BINARY = REPO / "target" / "release" / "naive_bbpe"


def read_model(path):
    """Parse our `.model` and rebuild every token's bytes by replaying the merges."""
    lines = Path(path).read_bytes().decode("utf-8").split("\n")
    if lines[0] != MODEL_HEADER:
        raise SystemExit(f"{path} is not a {MODEL_HEADER} model")
    pattern, normalizer, n_special = lines[1], lines[2], int(lines[3])
    specials = []
    for line in lines[4:4 + n_special]:
        token, sid = line.rsplit(" ", 1)          # rsplit: a special token may contain spaces
        specials.append((token, int(sid)))
    merges = [tuple(map(int, l.split())) for l in lines[4 + n_special:] if l.strip()]
    tokens = [bytes([i]) for i in range(256)]
    for p0, p1 in merges:                         # new id is implicit: 256 + line index
        tokens.append(tokens[p0] + tokens[p1])

    if len(set(tokens)) != len(tokens):
        seen, dups = set(), []
        for i, t in enumerate(tokens):
            (dups.append((i, t)) if t in seen else seen.add(t))
        raise SystemExit(f"{path}: {len(dups)} tokens repeat another token's bytes, e.g. {dups[:3]}; "
                         "tokenizer.json and .tiktoken key tokens by bytes and cannot represent that")
    return dict(pattern=pattern, normalizer=normalizer, specials=specials, merges=merges,
                tokens=tokens)


def bytes_to_unicode():
    """GPT-2's byte -> printable character table, which HF's ByteLevel stage uses."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def write_hf(m, path):
    b2u = bytes_to_unicode()
    s = lambda b: "".join(b2u[x] for x in b)
    tokens = m["tokens"]
    doc = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": sid, "content": tok, "single_word": False, "lstrip": False, "rstrip": False,
             "normalized": False, "special": True}
            for tok, sid in m["specials"]
        ],
        "normalizer": {"none": None, "nfc": {"type": "NFC"}, "nfd": {"type": "NFD"}}[m["normalizer"]],
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            # Our regex first, then ByteLevel only to map bytes to its printable alphabet --
            # use_regex=False so it does not split a second time with GPT-2's own pattern.
            {"type": "Split", "pattern": {"Regex": m["pattern"]}, "behavior": "Isolated",
             "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
             "use_regex": False},
        ]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                    "use_regex": True},
        "model": {
            "type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
            "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False,
            # ignore_merges=False: never shortcut a whole word straight to its vocab id; apply
            # merges in rank order, as naive_bbpe does.
            "ignore_merges": False,
            "vocab": {s(t): i for i, t in enumerate(tokens)},
            "merges": [[s(tokens[a]), s(tokens[b])] for a, b in m["merges"]],
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False)


def write_tiktoken(m, prefix):
    if m["normalizer"] != "none":
        raise SystemExit(f"this model normalizes text ({m['normalizer']}) and tiktoken has no "
                         "normalization stage; export --format hf instead")
    with open(f"{prefix}.tiktoken", "w", encoding="ascii") as f:
        for i, t in enumerate(m["tokens"]):
            f.write(f"{base64.b64encode(t).decode('ascii')} {i}\n")
    with open(f"{prefix}.tiktoken.json", "w", encoding="utf-8") as f:
        json.dump({"pattern": m["pattern"], "special_tokens": dict(m["specials"])}, f,
                  ensure_ascii=False, indent=1)


def load_tiktoken(prefix, name="naive_bbpe"):
    """Build a tiktoken.Encoding from the two exported files (no `blobfile` needed)."""
    import tiktoken
    ranks = {}
    with open(f"{prefix}.tiktoken", "rb") as f:
        for line in f:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    meta = json.loads(Path(f"{prefix}.tiktoken.json").read_text(encoding="utf-8"))
    return tiktoken.Encoding(name, pat_str=meta["pattern"], mergeable_ranks=ranks,
                             special_tokens=meta["special_tokens"])


def check(model_path, out_dir, formats, text_paths, binary):
    """Encode each file with ours and every export; return True only if all ids agree.

    Specials in the text are treated as ordinary text on every side (ours: --allow-special
    none; tiktoken: encode_ordinary; HF: encode_special_tokens=True), so the comparison is of
    the BPE itself.
    """
    encoders = {}
    if "tiktoken" in formats:
        enc = load_tiktoken(str(Path(out_dir) / "tokenizer"))
        encoders["tiktoken"] = (enc.encode_ordinary, enc.decode)
    if "hf" in formats:
        from tokenizers import Tokenizer
        hf = Tokenizer.from_file(str(Path(out_dir) / "tokenizer.json"))
        hf.encode_special_tokens = True
        encoders["hf"] = (lambda t: hf.encode(t, add_special_tokens=False).ids,
                          lambda ids: hf.decode(ids, skip_special_tokens=False))

    # A normalizer rewrites text before encoding (e + U+0301 -> é), so decoding returns the
    # NORMALIZED text, exactly as naive_bbpe's own decode does. Compare against that.
    form = {"nfc": "NFC", "nfd": "NFD"}.get(read_model(model_path)["normalizer"])
    all_ok = True
    for path in text_paths:
        text = Path(path).read_bytes().decode("utf-8")
        expected = unicodedata.normalize(form, text) if form else text
        run = subprocess.run([str(binary), "encode", "--model", str(model_path),
                              "--allow-special", "none"], input=text.encode("utf-8"),
                             capture_output=True)
        if run.returncode != 0:
            raise SystemExit(f"naive_bbpe encode failed on {path}: {run.stderr.decode()}")
        ours = [int(x) for x in run.stdout.split()]
        for name, (encode, decode) in encoders.items():
            theirs = encode(text)
            if theirs == ours:
                back = decode(theirs) == expected
                rt = ("decodes back exactly" + (f" (as {form})" if form else "")) if back else "DECODE DIFFERS"
                print(f"  {name:8} == naive_bbpe on {Path(path).name}: {len(ours):,} ids identical; {rt}")
                all_ok &= back
            else:
                all_ok = False
                i = next((k for k, (a, b) in enumerate(zip(ours, theirs)) if a != b),
                         min(len(ours), len(theirs)))
                print(f"  {name:8} != naive_bbpe on {Path(path).name}: lengths {len(ours):,} vs "
                      f"{len(theirs):,}, first difference at position {i}: "
                      f"ours {ours[i:i + 5]} vs {name} {theirs[i:i + 5]}")
    return all_ok


def export(model_path, out_dir, formats):
    m = read_model(model_path)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    written = []
    if "hf" in formats:
        write_hf(m, Path(out_dir) / "tokenizer.json")
        written.append("tokenizer.json")
    if "tiktoken" in formats:
        write_tiktoken(m, Path(out_dir) / "tokenizer")
        written += ["tokenizer.tiktoken", "tokenizer.tiktoken.json"]
    print(f"exported {len(m['tokens']):,} tokens + {len(m['specials'])} special "
          f"({', '.join(f'{t}={i}' for t, i in m['specials']) or 'none'}) to "
          f"{out_dir}: {', '.join(written)}")
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="a naive_bbpe .model file")
    ap.add_argument("--format", required=True, choices=["hf", "tiktoken", "both"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--check", nargs="*", default=[], metavar="TEXT_FILE",
                    help="verify the export against naive_bbpe on these UTF-8 text files")
    ap.add_argument("--binary", default=str(DEFAULT_BINARY), help="the naive_bbpe binary")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # keep progress lines in order when piped
    formats = ["hf", "tiktoken"] if a.format == "both" else [a.format]
    export(a.model, a.out_dir, formats)
    if a.check:
        ok = check(a.model, a.out_dir, formats, a.check, a.binary)
        print("CHECK PASSED" if ok else "CHECK FAILED")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
