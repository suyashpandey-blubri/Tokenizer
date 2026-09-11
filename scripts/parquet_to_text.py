#!/usr/bin/env python3
"""Turn fineweb-edu parquet shards into a plain UTF-8 .txt file.

WHY THIS EXISTS
---------------
`naive_bbpe train` reads one thing: a plain UTF-8 text file. It has no idea what
parquet is, and deliberately so -- adding a parquet reader to a naive BPE tokenizer
would mean pulling in a large dependency for a job that has nothing to do with
tokenization. So the "get the data into a shape the tool understands" step lives
out here in Python, where the HuggingFace tooling already is.

WHY IT STREAMS
--------------
One fineweb-edu shard is ~2.15 GB compressed and several GB decompressed. Reading it
whole would need more memory than the training run that follows. `iter_batches` reads
one row group at a time, so peak memory stays in the tens of MB no matter how big the
shard is, and `--max-bytes` stops as soon as enough text has been written.

WHY DOCUMENTS ARE JOINED WITH A BLANK LINE AND NOT "<|endoftext|>"
-----------------------------------------------------------------
It is tempting to write "<|endoftext|>" between documents. Don't. Training would then
see those 13 characters as ordinary text and happily learn merges *inside* them --
"<|", "endoftext", "|>" -- wasting vocabulary on a marker that is supposed to be a
single indivisible token. Real tokenizers keep the two apart: the marker is registered
as a *special token* (`--special '<|endoftext|>'` at train time) and inserted at encode
time, never learned from the corpus.

USAGE
-----
    python3 scripts/parquet_to_text.py \\
        --input data/fineweb/sample/10BT/000_00000.parquet \\
        --output data/fineweb.txt \\
        --max-bytes 200MB
"""

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


def parse_size(text: str) -> int:
    """Accept '200MB', '1GB', '500000' -- anything a human would type."""
    text = text.strip().upper().replace("_", "")
    for suffix, mult in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10), ("B", 1)):
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * mult)
    return int(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", nargs="+", required=True, help="parquet shard(s)")
    ap.add_argument("--output", required=True, help="the .txt file to write")
    ap.add_argument("--max-bytes", default="200MB",
                    help="stop once this much text is written [default: 200MB]")
    ap.add_argument("--column", default="text", help="column to extract [default: text]")
    ap.add_argument("--min-doc-bytes", type=int, default=200,
                    help="skip documents shorter than this [default: 200]")
    args = ap.parse_args()

    budget = parse_size(args.max_bytes)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    docs = 0
    skipped = 0

    with out_path.open("w", encoding="utf-8") as out:
        for shard in args.input:
            if written >= budget:
                break
            pf = pq.ParquetFile(shard)
            if args.column not in pf.schema_arrow.names:
                print(f"error: no column {args.column!r} in {shard}; "
                      f"available: {pf.schema_arrow.names}", file=sys.stderr)
                return 1

            # One row group at a time: peak memory stays small however big the shard is.
            for batch in pf.iter_batches(batch_size=2000, columns=[args.column]):
                for value in batch.column(0):
                    text = value.as_py()
                    if text is None or len(text) < args.min_doc_bytes:
                        skipped += 1
                        continue

                    # A blank line between documents. Plain text, no marker -- see the
                    # module docstring for why.
                    chunk = text.rstrip("\n") + "\n\n"
                    out.write(chunk)
                    written += len(chunk.encode("utf-8"))
                    docs += 1

                    if written >= budget:
                        break
                if written >= budget:
                    break

    print(f"wrote {out_path}")
    print(f"  {written:,} bytes  ({written / (1 << 20):.1f} MiB)")
    print(f"  {docs:,} documents  ({skipped:,} skipped as too short)")
    if written < budget:
        print(f"  note: ran out of input before reaching {args.max_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
