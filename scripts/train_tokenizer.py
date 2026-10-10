#!/usr/bin/env python3
"""Train a tokenizer with naive_bbpe and export it, into one run folder with a run record.

WHAT IT DOES, IN ORDER
----------------------
1. Refuses anything that would fail later -- before training, not after a multi-day run:
   an existing run folder, a missing input, --vocab-size under 256, a normalizer with
   --formats tiktoken (tiktoken has no normalization stage), a missing Python library.
2. Builds the binary (`cargo build --release`; nothing to do if it is up to date), so the
   git commit recorded in run.json is the code that actually trains.
3. Creates <runs-dir>/<name>/ and writes run.json with status "running".
4. Trains: `naive_bbpe train ... --verbose`, writing ours.model and ours.vocab. Everything
   it prints goes to the terminal AND to train.log.
5. Exports ours.model to the formats asked for, with export_tokenizer.py's own code.
6. Finishes run.json: status "done" (or "failed" and at which step), time per step, and
   the training process's peak memory as the operating system measured it.

It does NOT slice, add special tokens, or encode any text: slicing is
scripts/slice_tokenizer.py and checking on text is the benchmark, both separate commands.

WHAT THE RUN FOLDER HOLDS
-------------------------
    ours.model, ours.vocab                        always -- every export is made from ours.model
    tokenizer.tiktoken, tokenizer.tiktoken.json   --formats tiktoken
    tokenizer.json                                --formats hf
    train.log                                     everything the training binary printed
    run.json                                      the run record

USAGE
-----
    python3 scripts/train_tokenizer.py --name prod-65k --input /tank/suyash/train.txt \\
        --vocab-size 65536 --formats tiktoken \\
        --pattern gpt4 --normalizer none --threads 512 --no-report

A RUN THAT TAKES DAYS: START IT INSIDE tmux
-------------------------------------------
A program started from an ssh login is stopped when that connection closes (laptop
sleeps, network drops, window closed). tmux keeps a session open on the server itself:

    tmux new -s train              # open a session named "train"
    python3 scripts/train_tokenizer.py ...
    # Ctrl-b then d                -> detach: it keeps running; you can disconnect
    tmux attach -t train           # later: reconnect and see the output
"""

import argparse
import datetime
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_tokenizer  # noqa: E402

REPO = export_tokenizer.REPO
BINARY = export_tokenizer.DEFAULT_BINARY
# export_tokenizer.py's names for the formats, and the files each one writes.
FORMAT_FILES = {"tiktoken": ["tokenizer.tiktoken", "tokenizer.tiktoken.json"],
                "hf": ["tokenizer.json"]}


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def write_record(path, record):
    """Write run.json whole or not at all: into a temporary file, then renamed over it."""
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def git(*args):
    # rstrip, not strip: `git status --porcelain` lines START with a status column that can
    # be a space (" M file" = modified, not staged), and strip() would eat it.
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True,
                          text=True).stdout.rstrip("\n")


def refuse_early(a):
    """Every check that can be made before training. Returns a reason, or None."""
    run_dir = Path(a.runs_dir) / a.name
    if run_dir.exists():
        return f"{run_dir} already exists -- a finished run is never overwritten; pick a new --name"
    if not Path(a.input).is_file():
        return f"--input {a.input}: no such file"
    if a.vocab_size < 256:
        return f"--vocab-size {a.vocab_size}: must be at least 256 (the byte tokens)"
    if "tiktoken" in a.formats and a.normalizer != "none":
        return (f"--normalizer {a.normalizer} with --formats tiktoken: tiktoken has no "
                "normalization stage, so the export would encode differently from training")
    if a.threads is not None and a.threads < 1:
        return f"--threads {a.threads}: must be at least 1"
    for fmt, module in (("tiktoken", "tiktoken"), ("hf", "tokenizers")):
        if fmt in a.formats:
            try:
                __import__(module)
            except ImportError:
                return f"--formats {fmt} needs the Python package {module}: pip install {module}"
    return None


def train(cmd, log_path):
    """Run the training binary. Each line it prints goes to the terminal and to train.log.
    Returns (exit code, peak memory in bytes) -- the peak for THIS process alone, from
    `os.wait4`, so the earlier `cargo build` cannot be mistaken for it."""
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()                 # so `tail -f train.log` shows progress as it happens
        _, status, usage = os.wait4(proc.pid, 0)
        proc.returncode = os.waitstatus_to_exitcode(status)   # tell Popen it was reaped
    return proc.returncode, usage.ru_maxrss * 1024            # Linux reports kilobytes


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="the run's name; its folder is <runs-dir>/<name>")
    ap.add_argument("--input", required=True, help="ONE UTF-8 text file to train on")
    ap.add_argument("--vocab-size", type=int, required=True,
                    help="total vocabulary size, including the 256 byte tokens")
    ap.add_argument("--formats", nargs="+", required=True, choices=sorted(FORMAT_FILES),
                    help="which files to export from ours.model (ours.model is always kept)")
    ap.add_argument("--pattern", choices=["gpt4", "gpt2"], default="gpt4")
    ap.add_argument("--normalizer", choices=["none", "nfc", "nfd"], default="none")
    ap.add_argument("--threads", type=int, help="passed to naive_bbpe train (default: all cores, at most 64)")
    ap.add_argument("--no-report", action="store_true", help="passed to naive_bbpe train")
    ap.add_argument("--runs-dir", default=str(REPO / "runs"),
                    help="where run folders go (default: runs/ in the repo, which git ignores)")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    formats = sorted(set(a.formats))

    reason = refuse_early(a)
    if reason:
        raise SystemExit(f"refused: {reason}")

    print("building the binary (cargo build --release)")
    t = time.monotonic()
    build = subprocess.run(["cargo", "build", "--release"], cwd=REPO, capture_output=True, text=True)
    if build.returncode != 0:
        raise SystemExit(f"cargo build failed:\n{build.stderr}")
    build_seconds = round(time.monotonic() - t, 1)

    run_dir = Path(a.runs_dir) / a.name
    run_dir.mkdir(parents=True)
    record_path = run_dir / "run.json"
    record = {
        "name": a.name,
        "status": "running",
        "command": shlex.join([sys.executable, *sys.argv]),
        "input": {"path": str(Path(a.input).resolve()), "bytes": Path(a.input).stat().st_size},
        "settings": {"vocab_size": a.vocab_size, "pattern": a.pattern,
                     "normalizer": a.normalizer, "threads": a.threads,
                     "report": not a.no_report, "formats": formats},
        "code": {"git_commit": git("rev-parse", "HEAD"),
                 # Every path `git status` lists: modified, added, untracked. Empty = clean.
                 "uncommitted_changes": git("status", "--porcelain").splitlines()},
        "started": now(),
        "steps": {"build": {"seconds": build_seconds}},
    }
    write_record(record_path, record)

    step = "train"
    try:
        cmd = [str(BINARY), "train", "--input", a.input, "--vocab-size", str(a.vocab_size),
               "--out", str(run_dir / "ours"), "--pattern", a.pattern,
               "--normalizer", a.normalizer, "--verbose"]
        if a.threads is not None:
            cmd += ["--threads", str(a.threads)]
        if a.no_report:
            cmd.append("--no-report")
        print(f"[1/2] training -> {run_dir}/ours.model   (log: {run_dir}/train.log)")
        t = time.monotonic()
        code, peak = train(cmd, run_dir / "train.log")
        record["steps"]["train"] = {"seconds": round(time.monotonic() - t, 1),
                                    "peak_memory_gb": round(peak / 1e9, 2), "exit_code": code}
        if code != 0:
            raise RuntimeError(f"naive_bbpe train exited with code {code}; see train.log")
        # The thread count the binary actually used, as it printed it.
        for line in (run_dir / "train.log").read_text(encoding="utf-8").splitlines():
            if line.startswith("threads:"):
                record["settings"]["threads"] = int(line.split()[1])
        write_record(record_path, record)

        step = "export"
        print(f"[2/2] exporting -> {', '.join(formats)}")
        t = time.monotonic()
        export_tokenizer.export(run_dir / "ours.model", run_dir, formats)
        record["steps"]["export"] = {"seconds": round(time.monotonic() - t, 1)}
    except BaseException as e:      # BaseException, so Ctrl-C is recorded too
        record.update(status="failed", failed_step=step, error=f"{type(e).__name__}: {e}",
                      finished=now())
        write_record(record_path, record)
        raise SystemExit(f"FAILED at step {step!r}: {e}  (recorded in {record_path})")

    record.update(status="done", finished=now(),
                  outputs=["ours.model", "ours.vocab"] + [f for fmt in formats
                                                          for f in FORMAT_FILES[fmt]])
    write_record(record_path, record)
    print(f"done -- {run_dir}/  ({', '.join(record['outputs'])}, train.log, run.json)")


if __name__ == "__main__":
    main()
