#!/usr/bin/env python3
"""Encode one corpus with all four tokenizers and print one table.

WHY THIS EXISTS
---------------
`train_comparators.py` produces four models trained on the same corpus at the same
vocabulary size. This scores them. The question the whole of step 11 exists to answer is
narrow: trained on the same English corpus at the same vocab size, do we land next to
rustbpe, HuggingFace and sentencepiece? If yes, the implementation is sound and every
later measurement rests on something known-good.

WHY THE METRIC CODE NEVER TOUCHES A TOKENIZER LIBRARY
------------------------------------------------------
Each tokenizer is reduced to two things: a list of token ids, and the set of ids that
stand for a single high byte. Everything after that -- fertility, bytes per token,
high-byte rate, Renyi -- is computed by the same code for all four. A metric that
special-cased one implementation would be measuring the special case.

WHY ROUND TRIP RUNS BEFORE ANY METRIC
--------------------------------------
A tokenizer that quietly drops information gets better compression for free: less text
to encode means fewer tokens, which reads as a win. sentencepiece does this by default.
So `decode(encode(text)) == text` is checked first and a row that fails it is reported
as VOID rather than scored -- its numbers would be flattering and meaningless.

WHY IDS CANNOT BE COMPARED ACROSS TOKENIZERS, AND WHAT IS USED INSTEAD
-----------------------------------------------------------------------
"Is this id in 128..255" is the right question for us and for rustbpe, where id equals
byte value by construction. It is the wrong question everywhere else:

    HF    id 0 is b'!'. ByteLevel orders its alphabet printable-first, per GPT-2,
          so the byte values are a permutation of the ids.
    SP    id 0 is <unk> and ids 1..256 are the byte-fallback pieces <0x00>..<0xFF>,
          so everything is shifted by one.

So the high-byte set is derived per tokenizer from the token *bytes* -- via the .ranks
files for rustbpe and HF, and by parsing <0xNN> pieces for sentencepiece. Same question,
asked in a way that survives the id permutation.

WHY THE CORPUS IS ENCODED IN ONE SHOT
--------------------------------------
Not line by line. Encoding per line without the newlines silently deletes one character
per line, which shows up as a free compression win: measured with cl100k on 1000 lines
of book.txt, whole-document gave 14,567 tokens and per-line-without-newline gave 13,819,
a 5.13% improvement bought entirely by throwing data away. One shot for everyone means
there is no artifact to correct for.

USAGE
-----
    python3 scripts/eval/bench.py --corpus data/book.txt --vocab-size 2000
"""

import argparse
import base64
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ALPHA = 2.5  # Renyi order, per the step-11 plan


# ---------------------------------------------------------------------------
# Metrics -- one implementation, used for every tokenizer
# ---------------------------------------------------------------------------


def renyi(ids: list[int], declared_vocab: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Return (raw H_alpha in nats, efficiency H_alpha / log|V|).

    "Am I using the vocabulary I paid for?" Spread evenly across the vocabulary gives 1.0;
    a distribution dominated by a handful of tokens tends to 0.

    The denominator is the DECLARED vocabulary size, not the number of distinct tokens
    actually observed. That matters: measured on `data/fineweb.ids` (5M ids), log|V|
    converges by 25k tokens while observed-types has still not converged at 5M, because
    the observed count keeps climbing with sample size. Using it would report a property
    of the sample rather than of the tokenizer.

    Reported alongside the raw H_alpha because across different vocabulary sizes the
    denominator is a free penalty -- log(200000)/log(50000) = 1.13. At fixed vocab size,
    which is every comparison in this step, that confound does not apply.

    This is a diagnostic and not a target: "Two Counterexamples to Tokenization and the
    Noiseless Channel" (LREC 2024) builds BPE variants where Renyi rises and translation
    quality falls.
    """
    total = len(ids)
    counts = Counter(ids)
    # sum p^alpha, computed over observed tokens only -- unobserved tokens contribute
    # 0^alpha = 0 and are correctly ignored by the sum, while still counting in |V|.
    power_sum = sum((c / total) ** alpha for c in counts.values())
    h_alpha = math.log(power_sum) / (1.0 - alpha)
    return h_alpha, h_alpha / math.log(declared_vocab)


def score(name: str, ids: list[int], high_byte_ids: set[int],
          text: str, declared_vocab: int) -> dict:
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())
    h_alpha, efficiency = renyi(ids, declared_vocab)
    return {
        "name": name,
        "tokens": len(ids),
        # Fertility is the headline because every published reference number is a
        # fertility number -- sarvam-1 at 1.4-2.1, Llama-4 Tamil at 5.88. Headline
        # anything else and the report cannot be compared to the literature.
        #
        # The word rule is whitespace split, and the word count is printed so any number
        # here can be rescaled to another definition. Within one language it is the same
        # constant for all four rows, so it cancels and cannot change the ranking.
        "fertility": len(ids) / n_words,
        "bytes_per_token": n_bytes / len(ids),
        # Ids 0-127 are ASCII -- space, 'e', '.' -- perfectly ordinary tokens. Ids
        # 128-255 never appear alone in valid text; they exist only as pieces of a
        # multi-byte character. Counting `id < 256` instead would sweep in ASCII
        # punctuation and report a coverage problem for English that does not exist.
        "high_byte_rate": sum(1 for i in ids if i in high_byte_ids) / len(ids),
        "renyi_raw": h_alpha,
        "renyi_efficiency": efficiency,
        "distinct": len(set(ids)),
    }


def check_round_trip(name: str, original: str, decoded: str) -> str:
    if decoded == original:
        return "ok"
    for i, (a, b) in enumerate(zip(original, decoded)):
        if a != b:
            return f"VOID (differs at char {i})"
    return f"VOID (length {len(original)} in, {len(decoded)} out)"


def load_ranks(path: Path) -> list[tuple[bytes, int]]:
    out = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line:
            token, rank = line.split()
            out.append((base64.b64decode(token), int(rank)))
    return out


def high_bytes_from_ranks(ranks: list[tuple[bytes, int]]) -> set[int]:
    """Ids whose token is exactly one byte in 128..255."""
    return {rank for token, rank in ranks if len(token) == 1 and token[0] >= 128}


# ---------------------------------------------------------------------------
# One runner per tokenizer: text in, (ids, high-byte ids, declared vocab, decoded) out
# ---------------------------------------------------------------------------


def run_ours(text: str, prefix: Path, binary: Path):
    model = Path(f"{prefix}.model")
    encoded = subprocess.run([str(binary), "encode", "--model", str(model)],
                             input=text.encode("utf-8"), capture_output=True)
    if encoded.returncode != 0:
        raise SystemExit(f"ours: encode failed\n{encoded.stderr.decode()}")
    ids = [int(f) for f in encoded.stdout.split()]

    decoded = subprocess.run([str(binary), "decode", "--model", str(model)],
                             input=encoded.stdout, capture_output=True)

    # Declared vocab = 256 byte tokens + one per merge line. The .model format stores no
    # vocabulary; counting the merge lines is how its size is known without replaying it.
    lines = model.read_bytes().decode("utf-8").split("\n")
    n_special = int(lines[3])
    n_merges = len([l for l in lines[4 + n_special:] if l.strip()])
    # Ids 0-255 are the byte values, by construction: the untrained vocabulary is one
    # token per possible byte, in order, which is why byte-level BPE has no OOV case.
    return ids, set(range(128, 256)), 256 + n_merges, decoded.stdout.decode("utf-8")


def run_rustbpe(text: str, prefix: Path, vocab_size: int, pattern: str):
    """rustbpe has no load, so it is retrained here -- and the retrain is checked.

    Its entire API is train_from_iterator / encode / batch_encode / decode /
    get_mergeable_ranks / get_pattern / vocab_size. There is no way to persist or reload
    a trained tokenizer, so scoring it means training it again.

    That is turned into a free determinism check: the freshly trained ranks must equal
    the .ranks file written at 11.2. If they ever differ, one of the two runs was
    nondeterministic and no number below it is trustworthy.
    """
    import rustbpe
    sys.path.insert(0, str(Path(__file__).parent))
    from train_comparators import corpus_chunks

    tok = rustbpe.Tokenizer()
    tok.train_from_iterator(corpus_chunks(text), vocab_size, pattern=pattern)

    fresh = tok.get_mergeable_ranks()
    saved = load_ranks(Path(f"{prefix}.ranks"))
    if fresh != saved:
        raise SystemExit("rustbpe: retraining produced different ranks than 11.2 saved "
                         "-- training is not deterministic, numbers are not trustworthy")

    ids = tok.encode(text)
    return ids, high_bytes_from_ranks(saved), tok.vocab_size, tok.decode(ids)


def run_hf(text: str, prefix: Path):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(f"{prefix}.json")
    encoding = tok.encode(text, add_special_tokens=False)
    ids = encoding.ids
    return (ids, high_bytes_from_ranks(load_ranks(Path(f"{prefix}.ranks"))),
            tok.get_vocab_size(), tok.decode(ids))


def run_sp(text: str, prefix: Path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=f"{prefix}.model")
    ids = sp.encode(text)
    # Byte-fallback pieces are spelled <0xNN>. They are the direct equivalent of our
    # ids 128..255, just shifted: <unk> takes id 0 and the 256 byte pieces follow.
    byte_piece = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
    high = set()
    for i in range(sp.get_piece_size()):
        m = byte_piece.match(sp.id_to_piece(i))
        if m and int(m.group(1), 16) >= 128:
            high.add(i)
    return ids, high, sp.get_piece_size(), sp.decode(ids)


# ---------------------------------------------------------------------------


def read_pattern(model_path: Path) -> str:
    return model_path.read_bytes().decode("utf-8").split("\n")[1]


def bench_one(corpus: Path, vocab_size: int, models: Path, binary: Path) -> None:
    text = corpus.read_text(encoding="utf-8")
    stem = corpus.stem
    n_bytes = len(text.encode("utf-8"))
    n_words = len(text.split())

    print(f"\n=== {corpus} @ vocab {vocab_size} ===")
    print(f"{n_bytes:,} bytes   {n_words:,} words (whitespace split)\n")

    def prefix(name: str) -> Path:
        return models / f"{name}-{stem}-{vocab_size}"

    ours_model = Path(f"{prefix('ours')}.model")
    if not ours_model.exists():
        raise SystemExit(f"{ours_model} not found -- run train_comparators.py first")
    pattern = read_pattern(ours_model)

    runners = [
        ("ours", lambda: run_ours(text, prefix("ours"), binary)),
        ("rustbpe", lambda: run_rustbpe(text, prefix("rustbpe"), vocab_size, pattern)),
        ("hf", lambda: run_hf(text, prefix("hf"))),
        ("sp", lambda: run_sp(text, prefix("sp"))),
    ]

    header = (f"{'tokenizer':<10} {'tokens':>12} {'fertility':>10} {'bytes/tok':>10} "
              f"{'high-byte':>10} {'distinct':>9} {'H_a':>7} {'Renyi':>7}  round-trip")
    print(header)
    print("-" * len(header))

    rows = []
    for name, run in runners:
        ids, high, declared, decoded = run()
        verdict = check_round_trip(name, text, decoded)
        row = score(name, ids, high, text, declared)
        row["round_trip"] = verdict
        rows.append(row)
        print(f"{name:<10} {row['tokens']:>12,} {row['fertility']:>10.4f} "
              f"{row['bytes_per_token']:>10.4f} {row['high_byte_rate'] * 100:>9.3f}% "
              f"{row['distinct']:>9,} {row['renyi_raw']:>7.4f} "
              f"{row['renyi_efficiency']:>7.4f}  {verdict}")

    # The deliverable is parity, so state the spread rather than leaving it to be eyeballed.
    scored = [r for r in rows if r["round_trip"] == "ok"]
    if len(scored) > 1:
        ours = next(r for r in scored if r["name"] == "ours")
        print()
        for r in scored:
            if r["name"] == "ours":
                continue
            delta = (ours["fertility"] - r["fertility"]) / r["fertility"] * 100
            print(f"  ours vs {r['name']:<8} fertility {delta:+7.3f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, action="append", required=True,
                    help="may be given more than once, paired with --vocab-size")
    ap.add_argument("--vocab-size", type=int, action="append", required=True)
    ap.add_argument("--models", type=Path, default=Path("models/eval"))
    ap.add_argument("--binary", type=Path, default=Path("target/release/naive_bbpe"))
    args = ap.parse_args()

    if len(args.corpus) != len(args.vocab_size):
        raise SystemExit("--corpus and --vocab-size must be given the same number of times")

    for corpus, vocab_size in zip(args.corpus, args.vocab_size):
        bench_one(corpus, vocab_size, args.models, args.binary)


if __name__ == "__main__":
    sys.exit(main())
