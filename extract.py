#!/usr/bin/env python3
"""Milk every chat turn out of EROS's session JSONLs into one big sorted feed.

Walks ~/.eros/agent/sessions/<bucket>/*.jsonl (top-level journals only,
skipping dotfile lock sidecars), parses every JSONL in filename-timestamp
order, and emits one JSON object per TEXT turn with
{session, seq, timestamp, role, text, parentId, msgId}, sorted globally by
(timestamp, session, seq). Blob refs become [image:sha] placeholders -- no
binary ever touches the output. Non-text entries (model_usage, compaction
markers, model/tool chatter) stay markers in the source, never turns.

Idempotency: --state records per-input (path, mtime, size, sha256); a rerun
with unchanged inputs and matching output leaves --out byte-identical
(does not even touch the file). Dedup key: session+msgId (fallback
session+seq). Stdlib only.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

BLOB_RE = re.compile(r"blob:sha256:(?:[0-9a-f]{64})?")

TEXT_ROLES = {"user", "developer", "assistant", "toolResult"}

def blob_placeholder(m):
    h = m.group(0).split(":", 2)[2]
    return "[image:%s]" % (h if h else "?")

def sha_file(path, mtime=None, size=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path):
    st = os.stat(path)
    return {
        "mtime": st.st_mtime,
        "size": st.st_size,
        "sha": sha_file(path),
    }



def clean_text(s):
    return BLOB_RE.sub(blob_placeholder, s)


def iter_session_files(sessions_dir):
    files = sorted(glob.glob(os.path.join(sessions_dir, "*", "*.jsonl")))
    # skip dotfiles (flock lease sidecars) -- glob already skips dotfiles
    return [f for f in files if not os.path.basename(f).startswith(".")]


def extract_turns(path):
    """Yield (entry_ts, msgId, parentId, role, text) for one session file."""
    try:
        fh = open(path, "r", encoding="utf-8", errors="replace")
    except OSError as e:
        print("warn: cannot open %s: %s" % (path, e), file=sys.stderr)
        return
        yield  # unreachable: keeps this a generator on the error path
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("type") != "message":
                # model_usage, compaction/reset markers, custom, title, etc.
                # stay markers in the source -- never turns.
                continue
            m = o.get("message") or {}
            role = m.get("role")
            if role not in TEXT_ROLES:
                continue
            texts = []
            images = []
            for b in m.get("content") or []:
                bt = b.get("type")
                if bt == "text" and b.get("text"):
                    texts.append(clean_text(b["text"]))
                elif bt == "image" and b.get("data"):
                    d = b["data"]
                    mm = re.search(r"[0-9a-f]{64}", d)
                    images.append("[image:%s]" % (mm.group(0) if mm else d))
            if images:
                texts.append(" ".join(images))
            text = "\n".join(s for s in texts if s).strip()
            if not text:
                continue  # toolCall/thinking/empty-error shells: no text, no turn
            yield (
                o.get("timestamp", ""),
                o.get("id", ""),
                o.get("parentId"),
                role,
                text,
            )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions",
                    default=os.path.expanduser("~/.eros/agent/sessions"))
    ap.add_argument("--out", default=None,
                    help="output file (default: stdout; no state tracking)")
    ap.add_argument("--state", default=None,
                    help="json state file for idempotent reruns")
    args = ap.parse_args()

    files = iter_session_files(args.sessions)
    inputs = {}
    for f in files:
        try:
            inputs[f] = fingerprint(f)
        except OSError as e:
            print("warn: cannot stat %s: %s" % (f, e), file=sys.stderr)

    state = {}
    if args.state and os.path.exists(args.state):
        try:
            with open(args.state, encoding="utf-8") as fh:
                state = json.load(fh)
        except (OSError, ValueError) as e:
            print("warn: unreadable state, rebuilding: %s" % e,
                  file=sys.stderr)

    if args.out and state.get("inputs") == inputs and state.get("files") == files:
        old_sha = state.get("out_sha")
        if old_sha and os.path.exists(args.out):
            if sha_file(args.out) == old_sha:
                print("no-op: %d inputs unchanged, output %s untouched "
                      "(%d turns)" % (len(files), args.out,
                                      state.get("turns", -1)),
                      file=sys.stderr)
                return 0

    # full parse in filename-timestamp order; seq assigned in that order
    rows = []
    per_session_seq = {}
    sessions_seen = set()
    for f in files:
        gen = extract_turns(f)
        # extract_turns is a generator EXCEPT the early-return path... it
        # always returns a generator; session id resolved inside. Instead
        # collect with file tag and resolve session from header below.
        turns = list(gen)
        # session id: re-read header cheaply from first pass info -- the
        # generator already knew it; recover via filename fallback here by
        # peeking: use the same fallback logic through a tiny header read.
        sid = None
        try:
            with open(f, encoding="utf-8", errors="replace") as fh:
                for ln in fh:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        o = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    if o.get("type") == "session" and o.get("id"):
                        sid = o["id"]
                        break
                    if o.get("type") not in ("title", "session"):
                        break
        except OSError:
            pass
        if sid is None:
            base = os.path.basename(f)
            stem = base[:-len(".jsonl")] if base.endswith(".jsonl") else base
            sid = stem.split("_", 1)[1] if "_" in stem else stem
        sessions_seen.add(sid)
        for (ts, mid, pid, role, text) in turns:
            seq = per_session_seq.get(sid, 0)
            per_session_seq[sid] = seq + 1
            rows.append({"session": sid, "seq": seq, "timestamp": ts,
                         "role": role, "text": text,
                         "parentId": pid, "msgId": mid})

    # dedupe (session+msgId, fallback session+seq) + global sort
    seen = set()
    uniq = []
    for r in rows:
        key = (r["session"], r["msgId"]) if r["msgId"] \
            else (r["session"], "seq:%d" % r["seq"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    uniq.sort(key=lambda r: (r["timestamp"], r["session"], r["seq"]))

    out_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in uniq)

    if args.out:
        new_sha = hashlib.sha256(out_text.encode("utf-8")).hexdigest()
        write = True
        if os.path.exists(args.out) and sha_file(args.out) == new_sha:
            write = False  # already identical: do not touch (mtime stable)
        if write:
            tmp = args.out + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(out_text)
            os.replace(tmp, args.out)
            print("wrote %d turns from %d sessions -> %s"
                  % (len(uniq), len(sessions_seen), args.out), file=sys.stderr)
        else:
            print("no-op: output %s already identical (%d turns)"
                  % (args.out, len(uniq)), file=sys.stderr)
        if args.state:
            with open(args.state, "w", encoding="utf-8") as fh:
                json.dump({"inputs": inputs, "files": files,
                           "out_sha": new_sha, "turns": len(uniq),
                           "sessions": len(sessions_seen)}, fh, indent=1)
    else:
        sys.stdout.write(out_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
