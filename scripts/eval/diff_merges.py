#!/usr/bin/env python3
"""Compare our learned merges against rustbpe's and HuggingFace's, token for token.

WHY THIS EXISTS, SEPARATELY FROM bench.py
------------------------------------------
`bench.py` shows that all three tokenizers encode a corpus into the same number of
tokens. That is strong evidence but not proof: two *different* vocabularies could in
principle encode one corpus to the same length by coincidence. This script checks the
vocabularies themselves.

It is also the sharper of the two checks in one specific way. `bench.py` exercises
training AND encoding together, so a failure could come from either. This one reads only
the `.model` file, so it isolates the trainer.

THE REPRESENTATIONS DO NOT MATCH, AND THAT IS THE WHOLE PROBLEM
---------------------------------------------------------------
Our `.model` stores PAIRS OF IDS:

    97 116        <- "merge token 97 with token 116"

rustbpe hands back TOKEN BYTES:

    (b'at', 256)

Those are the same fact written two ways, and neither can be diffed against the other.
The bridge is REPLAY, which works because of two facts about our design:

  1. ids 0-255 ARE the byte values -- the untrained vocabulary is one token per possible
     byte, in order, which is why byte-level BPE has no out-of-vocabulary case;
  2. a merge's new id is `256 + line_index` and is never written down, because ids are
     handed out in merge order.

So replaying is: start with the 256 bytes, then walk the merge lines concatenating. It
terminates correctly because merges arrive in dependency order -- a merge can only name
tokens that already exist, so by line `i` both halves are in the table.

WHY ORDERED AGAINST rustbpe BUT SET AGAINST HUGGINGFACE
--------------------------------------------------------
rustbpe shares our tie-break: highest count, ties to the smallest pair. So an ORDERED
match means we agreed on the winning pair at every single decision, ties included. That
is the strict check and the one this script's exit code is wired to.

HuggingFace breaks ties differently (measured in 11.2: on tiny.txt at 268, ranks 256-259
and 264-267 agree while 260-263 are the same four tokens in a different order). Its
merge ORDER therefore differs from ours even when everything is correct, so comparing it
by order would report a failure that is not one. Its learned SET is what should match.

EXIT CODE
---------
Non-zero ONLY on an ordered mismatch against rustbpe -- that is the correctness claim.
Everything else is reported for a human to read. A set difference against HuggingFace is
expected in small numbers and is explained in the output rather than failed on.

USAGE
-----
    python3 scripts/eval/diff_merges.py --corpus data/book.txt --vocab-size 2000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench import load_ranks  # noqa: E402  -- the .ranks reader already lives there

MODEL_HEADER = "naive_bbpe v1"


def replay_model(path: Path) -> list[tuple[bytes, int]]:
    """Turn our `.model` into the same shape rustbpe's get_mergeable_ranks() returns.

    Returns `[(token_bytes, id), ...]` ordered by id, so the comparison downstream is a
    plain list `==` with no parsing or reimplementation on either side.
    """
    lines = path.read_bytes().decode("utf-8").split("\n")
    if lines[0] != MODEL_HEADER:
        raise SystemExit(f"{path} is not a {MODEL_HEADER} model")
    n_special = int(lines[3])
    merges = [tuple(map(int, l.split())) for l in lines[4 + n_special:] if l.strip()]

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for i, (p0, p1) in enumerate(merges):
        # The new id is implicit. This is the same loop `load` runs in Rust, and the
        # reason the vocabulary is not stored in the file at all: merges are ground truth.
        vocab[256 + i] = vocab[p0] + vocab[p1]
    return [(vocab[i], i) for i in range(len(vocab))]


def report_ordered(ours, theirs, name: str) -> bool:
    """Ordered comparison. Returns True if identical."""
    if ours == theirs:
        print(f"  ours vs {name:<8} ORDERED  IDENTICAL   "
              f"({len(ours) - 256:,} merges, same tokens in the same order)")
        return True

    print(f"  ours vs {name:<8} ORDERED  MISMATCH")
    if len(ours) != len(theirs):
        print(f"      vocabulary sizes differ: {len(ours):,} vs {len(theirs):,}")
    for (a, ra), (b, rb) in zip(ours, theirs):
        if (a, ra) != (b, rb):
            print(f"      first divergence at rank {ra}: ours {a!r}, {name} {b!r}")
            break
    return False


def report_set(ours, theirs, name: str) -> None:
    """Set comparison, which is the right question for a different tie-break."""
    so = {t for t, _ in ours[256:]}
    st = {t for t, _ in theirs[256:]}
    if so == st:
        print(f"  ours vs {name:<8} SET      IDENTICAL   ({len(so):,} learned tokens)")
        return

    only_ours, only_theirs = sorted(so - st), sorted(st - so)
    print(f"  ours vs {name:<8} SET      {len(so & st):,} of {len(so):,} in common "
          f"({len(only_ours)} differ)")
    for label, items in (("only ours", only_ours), (f"only {name}", only_theirs)):
        for tok in items[:6]:
            print(f"      {label:<12} {tok!r}")
        if len(items) > 6:
            print(f"      {label:<12} ... and {len(items) - 6} more")


def diff_one(corpus: Path, vocab_size: int, models: Path) -> bool:
    """Returns True if the correctness claim holds (ordered match against rustbpe)."""
    stem = corpus.stem
    ours_path = models / f"ours-{stem}-{vocab_size}.model"
    if not ours_path.exists():
        raise SystemExit(f"{ours_path} not found -- run train_comparators.py first")

    ours = replay_model(ours_path)
    print(f"\n=== {corpus} @ vocab {vocab_size} ===")
    print(f"replayed {ours_path.name}: {len(ours):,} tokens "
          f"= 256 bytes + {len(ours) - 256:,} merges\n")

    ok = True
    for name in ("rustbpe", "hf"):
        ranks_path = models / f"{name}-{stem}-{vocab_size}.ranks"
        if not ranks_path.exists():
            print(f"  ours vs {name:<8} SKIPPED — {ranks_path.name} not found")
            continue
        theirs = load_ranks(ranks_path)
        if name == "rustbpe":
            # The strict check: same tie-break, so order must agree too.
            ok &= report_ordered(ours, theirs, name)
        else:
            # Different tie-break by design; order is expected to differ, the set is not.
            report_set(ours, theirs, name)
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", type=Path, action="append", required=True,
                    help="may be given more than once, paired with --vocab-size")
    ap.add_argument("--vocab-size", type=int, action="append", required=True)
    ap.add_argument("--models", type=Path, default=Path("models/eval"))
    args = ap.parse_args()

    if len(args.corpus) != len(args.vocab_size):
        raise SystemExit("--corpus and --vocab-size must be given the same number of times")

    all_ok = True
    for corpus, vocab_size in zip(args.corpus, args.vocab_size):
        all_ok &= diff_one(corpus, vocab_size, args.models)

    print()
    if all_ok:
        print("PASS — our merge list is byte-identical to rustbpe's, in learned order.")
        print("       Same counting, same tie-break, same pre-tokenizer, every decision.")
    else:
        print("FAIL — our merges diverge from rustbpe's. This is a correctness bug in")
        print("       the trainer, not a benchmarking difference.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
