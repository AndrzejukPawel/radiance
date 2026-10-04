#!/usr/bin/env python3
# diskstall.py -- what bringing a conversation back from DISK costs: the returning request's time to
# first token, and the stall it puts on a conversation decoding beside it.
#
# A restore out of disk is the one tier move that reads files, and where it reads them decides who
# waits. So this measures both halves: the request that asked (TTFT, and how much of its prompt the
# cache served) and a bystander that asked for nothing (the longest gap between its streamed chunks
# while the restore ran, against its median gap outside that window).
#
#   scripts/diskstall.py [--port 8100] [--session 20000] [--fill 30000]
#
# THE SERVER MUST HAVE A HOST TIER THE FILLER OVERFLOWS, or the session is restored out of host
# memory and nothing here touches the disk -- the run then says so and exits non-zero:
#   --prefix-cache-host-mib 512 --prefix-cache-disk-mib 8192 --prefix-cache-dir <dir>
# holds about 27K tokens on Qwen3.8-Flash-Next, so --session 20000 --fill 30000 puts the session
# on disk. The session must still FIT the host tier: a restore reads through it.
import argparse, json, random, statistics, sys, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--session", type=int, default=20000, help="tokens in the returning conversation")
ap.add_argument("--fill", type=int, default=30000, help="tokens of filler that push it out of RAM")
ap.add_argument("--wait", type=int, default=240, help="seconds to wait for it to reach disk")
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
base = f"http://{a.host}:{a.port}"

WORDS = ["ledger", "valve", "crate", "harbor", "signal", "copper", "lantern", "orchard", "gravel",
         "tunnel", "beacon", "anchor", "quarry", "saddle", "furnace", "meadow", "cobalt", "pylon"]

def text(seed, ntok):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) for _ in range(int(ntok / 1.25)))

def post(path, body, timeout=1800):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(base + path, timeout=30) as r:
        return json.loads(r.read())

def complete(prompt, n=1):
    t0 = time.time()
    d = post("/v1/completions", {"model": "m", "prompt": prompt, "max_tokens": n,
                                 "temperature": 0, "stream": False})
    u = d.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    return time.time() - t0, u.get("prompt_tokens"), cached

def row_for(tokens):
    for s in get("/sessions")["sessions"]:
        if tokens <= s["tokens"] <= tokens + 8:
            return s
    return None

session = text(a.seed, a.session) + "\nName one word above.\nAnswer:"
t, n_prompt, _ = complete(session)
print(f"session   {n_prompt} tokens prefilled in {t:.1f}s")
t, n_fill, _ = complete(text(a.seed + 1, a.fill) + "\nName one word above.\nAnswer:")
print(f"filler    {n_fill} tokens prefilled in {t:.1f}s")

# ON DISK AND NOWHERE ELSE: not in VRAM (the loan takes a clean entry's blocks), not in RAM (the
# filler's copies took the slots, which the disk copy made free to give up).
t0 = time.time()
while True:
    s = row_for(n_prompt)
    if s and s["on_device"] == 0 and s["on_host"] == 0 and s["on_disk"] > 0 and not s["moving"]:
        break
    if time.time() - t0 > a.wait:
        print(f"diskstall: the session never left RAM and VRAM for disk: {s}", file=sys.stderr)
        sys.exit(1)
    time.sleep(1)
print(f"on disk   {s['on_disk']} blocks, snapshots device/host/disk "
      f"{s['snap_device']}/{s['snap_host']}/{s['snap_disk']}, after {time.time()-t0:.0f}s")

# ---- the bystander: a stream that only decodes ----------------------------------------------------
stamps = []
def bystander():
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps({
        "model": "m", "prompt": "Write a long story about a lighthouse keeper.\n\n",
        "max_tokens": 1500, "temperature": 0.7, "seed": 1, "ignore_eos": True,
        "stream": True}).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            if line.startswith(b"data:") and not line.startswith(b"data: [DONE]"):
                stamps.append(time.time())
th = threading.Thread(target=bystander)
th.start()
while len(stamps) < 40:
    time.sleep(0.05)
time.sleep(2.0)

moved0 = get("/sessions")["moved"]
t_ask = time.time()
ttft, n2, cached = complete(session + " the")
t_done = time.time()
moved1 = get("/sessions")["moved"]
time.sleep(2.0)
th.join()

gaps_in, gaps_out = [], []
for x, y in zip(stamps, stamps[1:]):
    (gaps_in if y >= t_ask and x <= t_done else gaps_out).append(y - x)
promoted = moved1["promoted"] - moved0["promoted"]
print(f"restore   {promoted} entries promoted; prompt {n2} cached {cached}; TTFT {ttft*1000:.0f} ms")
print(f"bystander median gap {statistics.median(gaps_out)*1000:.1f} ms outside the restore, "
      f"longest {max(gaps_in)*1000:.1f} ms during it ({len(gaps_in)} chunks)")
if not cached or cached < n_prompt // 2:
    print("diskstall: the returning request was not served from the cache", file=sys.stderr)
    sys.exit(1)
