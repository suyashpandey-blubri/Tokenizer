#!/usr/bin/env python3
"""Cut a trained tokenizer down to smaller vocabulary sizes, in whatever format it is in.

WHY A SLICE IS A REAL SMALLER TOKENIZER
---------------------------------------
BPE training learns merges one at a time, and merge k is chosen using only merges 1..k-1.
So the first k merges of a 65k run are exactly the merges a run stopped at 256 + k would
have learned -- nothing about merge k depends on what comes after it. Cutting a trained
tokenizer to size N therefore means keeping ids 0..N-1 and dropping the rest; no retraining.

Every supported format stores tokens in id order (bytes 0-255, then merges in learned
order), so each can be cut directly, in its own format, without converting:

    naive_bbpe .model        keep the first N-256 merge lines
    .tiktoken (+ .json)      keep the lines whose id is below N; the .json is copied as is
    HF tokenizer.json        keep vocab entries with id below N, and the first N-256 merges

Each output is written to <out-dir>/v<N>/ under the input's own file name, then LOADED BACK
through its own library (our replay, tiktoken, HF tokenizers) and required to hold exactly
N tokens -- so every file this writes is one its library accepts, at the size asked for.

This command does not encode any text. Whether the tokenizers work on real text is what the
benchmark command answers; that slicing itself is correct was proven when this script was
built (a slice is byte-identical to a model trained directly at that size).

USAGE
-----
    python3 scripts/slice_tokenizer.py runs/x/ours.model runs/x/tokenizer.tiktoken \\
        --vocab-sizes 32768 49152 --out-dir runs/x/sliced
"""

import argparse
import base64
import json
import shutil
import sys
from pathlib import Path

# The `.model` reader and the tiktoken loader live in export_tokenizer.py; reuse them rather
# than keep a second copy that could drift.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_tokenizer import MODEL_HEADER, load_tiktoken, read_model  # noqa: E402

BYTE_ALPHABET = 256


class Refused(Exception):
    """An input this script will not slice, with the reason."""


# ---------------------------------------------------------------------------
# One class per format. Each knows how to read itself, how big it is, how to write a cut
# copy, and how to load that copy back to confirm it.
# ---------------------------------------------------------------------------

class NaiveModel:
    """Our own `.model`: 4 header lines, the specials, then one merge per line."""
    kind = ".model"

    @staticmethod
    def detect(path):
        with open(path, "rb") as f:
            return f.readline().rstrip(b"\n") == MODEL_HEADER.encode()

    def __init__(self, path):
        self.path = Path(path)
        lines = self.path.read_bytes().decode("utf-8").split("\n")
        n_special = int(lines[3])
        if n_special:
            raise Refused(f"{path} has {n_special} special token(s); slicing would strand their "
                          "ids far above the cut vocabulary, and special tokens come later")
        self.header = lines[:4]
        self.merges = [l for l in lines[4:] if l.strip()]

    def size(self):
        return BYTE_ALPHABET + len(self.merges)

    def write(self, n, out_dir):
        out = out_dir / self.path.name
        # Same layout `save` writes: every line, including the last, ends in "\n".
        out.write_text("".join(l + "\n" for l in self.header + self.merges[:n - BYTE_ALPHABET]),
                       encoding="utf-8")
        return [out]

    @staticmethod
    def loaded_size(out):
        # Replays every merge, and refuses two tokens with the same bytes.
        return len(read_model(out)["tokens"])


class Tiktoken:
    """A `.tiktoken` rank file: one `<base64 token bytes> <id>` line per token, ids 0..n-1.
    `<file>.json` beside it, if present, holds the pattern and specials (export writes it)."""
    kind = ".tiktoken"

    @staticmethod
    def detect(path):
        with open(path, "rb") as f:
            first = f.readline().split()
        if len(first) != 2 or not first[1].isdigit():
            return False
        try:
            base64.b64decode(first[0], validate=True)
            return True
        except ValueError:
            return False

    def __init__(self, path):
        self.path = Path(path)
        self.lines = self.path.read_bytes().splitlines(keepends=True)
        for i, line in enumerate(self.lines):
            if int(line.split()[1]) != i:
                raise Refused(f"{path}: line {i + 1} has id {line.split()[1].decode()}, not {i}; "
                              "only files whose ids run 0, 1, 2, ... in order can be cut by id")
        self.sidecar = Path(f"{path}.json")
        if self.sidecar.exists():
            specials = json.loads(self.sidecar.read_text(encoding="utf-8")).get("special_tokens")
            if specials:
                raise Refused(f"{self.sidecar} lists special tokens {specials}; special tokens "
                              "come later")

    def size(self):
        return len(self.lines)

    def write(self, n, out_dir):
        out = out_dir / self.path.name
        out.write_bytes(b"".join(self.lines[:n]))
        written = [out]
        if self.sidecar.exists():
            shutil.copyfile(self.sidecar, out_dir / self.sidecar.name)
            written.append(out_dir / self.sidecar.name)
        return written

    @staticmethod
    def loaded_size(out):
        out = str(out)
        if out.endswith(".tiktoken") and Path(f"{out}.json").exists():
            # The way export_tokenizer.py and the benchmark load it: rank file + sidecar.
            return load_tiktoken(out[:-len(".tiktoken")]).n_vocab
        # No sidecar (e.g. a downloaded rank file): build the encoding directly. The pattern
        # matters only for encoding, which this script never does.
        import tiktoken
        ranks = {base64.b64decode(t): int(r)
                 for t, r in (l.split() for l in Path(out).read_bytes().splitlines())}
        return tiktoken.Encoding("slice", pat_str=r"\S+|\s+", mergeable_ranks=ranks,
                                 special_tokens={}).n_vocab


class HfJson:
    """A HuggingFace `tokenizer.json` holding byte-level BPE, as export_tokenizer.py writes it."""
    kind = "tokenizer.json"

    @staticmethod
    def detect(path):
        with open(path, "rb") as f:
            if f.read(1) != b"{":
                return False
        try:
            doc = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:
            return False
        return isinstance(doc, dict) and isinstance(doc.get("model"), dict) and "vocab" in doc["model"]

    def __init__(self, path):
        self.path = Path(path)
        self.doc = json.loads(self.path.read_text(encoding="utf-8"))
        model = self.doc["model"]
        if model.get("type") != "BPE" or "ByteLevel" not in json.dumps(self.doc.get("pre_tokenizer")):
            raise Refused(f"{path} is not byte-level BPE; only that layout can be cut by id")
        if self.doc.get("added_tokens"):
            raise Refused(f"{path} has added/special tokens; special tokens come later")
        ids = sorted(model["vocab"].values())
        if ids != list(range(len(ids))) or len(model["merges"]) != len(ids) - BYTE_ALPHABET:
            raise Refused(f"{path}: expected ids 0..n-1 with one merge per id above 255 "
                          f"(found {len(ids)} ids, {len(model['merges'])} merges)")

    def size(self):
        return len(self.doc["model"]["vocab"])

    def write(self, n, out_dir):
        doc = json.loads(json.dumps(self.doc))          # a copy; the input stays whole
        doc["model"]["vocab"] = {t: i for t, i in doc["model"]["vocab"].items() if i < n}
        doc["model"]["merges"] = doc["model"]["merges"][:n - BYTE_ALPHABET]
        out = out_dir / self.path.name
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)          # as export_tokenizer.py writes it
        return [out]

    @staticmethod
    def loaded_size(out):
        from tokenizers import Tokenizer
        return Tokenizer.from_file(str(out)).get_vocab_size()


FORMATS = [NaiveModel, Tiktoken, HfJson]


def open_artifact(path):
    """Identify `path` by its CONTENT, not its name, and read it."""
    for fmt in FORMATS:
        if fmt.detect(path):
            return fmt(path)
    raise Refused(f"{path}: not a naive_bbpe .model, a .tiktoken rank file, or a byte-level "
                  "BPE tokenizer.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("inputs", nargs="+", metavar="FILE",
                    help="tokenizer files to cut: .model, .tiktoken, tokenizer.json, in any mix")
    ap.add_argument("--vocab-sizes", nargs="+", type=int, required=True, metavar="N",
                    help="total vocabulary sizes to cut to (including the 256 byte tokens)")
    ap.add_argument("--out-dir", required=True, help="each size is written to <out-dir>/v<N>/")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    # Read and validate EVERYTHING before writing anything, so a refusal never leaves half
    # the outputs behind.
    try:
        artifacts = [open_artifact(p) for p in a.inputs]
        names = [x.path.name for x in artifacts]
        if len(set(names)) != len(names):
            raise Refused(f"two inputs share a file name ({names}); outputs would overwrite "
                          "each other inside v<N>/")
        for n in a.vocab_sizes:
            if n < BYTE_ALPHABET:
                raise Refused(f"--vocab-sizes {n}: below {BYTE_ALPHABET}, the byte tokens alone")
            for x in artifacts:
                if n > x.size():
                    raise Refused(f"--vocab-sizes {n}: larger than {x.path} ({x.size():,} tokens)")
    except Refused as e:
        raise SystemExit(f"refused: {e}")

    for x in artifacts:
        print(f"{x.path}  ({x.kind}, {x.size():,} tokens)")
    for n in a.vocab_sizes:
        out_dir = Path(a.out_dir) / f"v{n}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for x in artifacts:
            written = x.write(n, out_dir)
            got = type(x).loaded_size(written[0])
            if got != n:
                raise SystemExit(f"BUG: {written[0]} loads back with {got:,} tokens, not {n:,}")
            print(f"  v{n}: {', '.join(str(w) for w in written)}  (loads back: {got:,} tokens)")


if __name__ == "__main__":
    main()
