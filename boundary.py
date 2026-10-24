#!/usr/bin/env python3
"""
boundary.py -- boundary representation of a stream (user's definition).

A boundary is a switch point: where the value changes (e.g. -1 -> +1).
A stream is then stored as

    (initial value, [(position, jump), ...])        jump = new - old

and nothing else. The interior of each stretch carries no information,
because nothing changes there. Reconstruction is the cumulative sum of
an impulse stream -- the discrete difference *is* boundary extraction,
and the cumulative sum *is* boundary reconstruction (the pair from the
Link core: diff / cumsum).

Compared encodings of the same stream:
  dense    : every sample                                N numbers
  interval : (start, end, value) for every run           3 * (switches + 1)
  boundary : initial + (position, jump) per switch       1 + 2 * switches
             (for a +-1 signal the jumps alternate +2/-2,
              so positions alone suffice:                1 + switches)

Exactness: for integer-valued streams the reconstruction is exact.
For general floats the jumps are stored exactly, but re-summing them
accumulates rounding like any cumulative sum; measured, not assumed.

Usage:
    python3 boundary.py encode  < stream.txt   # numbers in, boundary events out
    python3 boundary.py decode  < events.txt   # events in, numbers out
    python3 boundary.py stats   < stream.txt   # compare dense/interval/boundary sizes
Event text format: first token = initial value, then "pos:jump" pairs,
finally "#N" with the stream length, e.g.  -1  100:+2  500:-2  #1000000
"""
import sys

def extract(x):
    """Stream -> (initial, [(pos, jump), ...]). Boundaries = nonzero diffs."""
    if len(x) == 0:
        return (0.0, [])
    events = []
    prev = x[0]
    for i in range(1, len(x)):
        if x[i] != prev:
            events.append((i, x[i] - prev))
            prev = x[i]
    return (x[0], events)

def reconstruct(n, initial, events):
    """Boundary events -> stream, by cumulative sum of impulses."""
    out = [0.0] * n
    ev = dict(events)
    cur = initial
    for i in range(n):
        if i in ev:
            cur = cur + ev[i]
        out[i] = cur
    return out

def intervals(x):
    """Stream -> [(start, end_exclusive, value), ...] (run representation)."""
    if len(x) == 0:
        return []
    runs = []
    start, prev = 0, x[0]
    for i in range(1, len(x)):
        if x[i] != prev:
            runs.append((start, i, prev))
            start, prev = i, x[i]
    runs.append((start, len(x), prev))
    return runs

def sizes(x):
    init, ev = extract(x)
    runs = intervals(x)
    return {
        'dense': len(x),
        'interval': 3 * len(runs),
        'boundary': 1 + 2 * len(ev),
        'boundary_positions_only': 1 + len(ev),
        'switches': len(ev),
    }

def fmt_num(v):
    return str(int(v)) if float(v) == int(v) and abs(v) < 1e16 else repr(v)

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'stats'
    data = sys.stdin.read().split()
    if mode == 'encode':
        x = [float(t) for t in data]
        init, ev = extract(x)
        parts = [fmt_num(init)] + [f'{p}:{fmt_num(j)}' for p, j in ev] + [f'#{len(x)}']
        print(' '.join(parts))
    elif mode == 'decode':
        init = float(data[0])
        n = int(data[-1][1:])
        ev = []
        for t in data[1:-1]:
            p, j = t.split(':')
            ev.append((int(p), float(j)))
        print(' '.join(fmt_num(v) for v in reconstruct(n, init, ev)))
    elif mode == 'stats':
        x = [float(t) for t in data]
        s = sizes(x)
        print(f"dense={s['dense']} interval={s['interval']} "
              f"boundary={s['boundary']} (positions only={s['boundary_positions_only']}) "
              f"switches={s['switches']}")
    else:
        print(f'unknown mode: {mode}', file=sys.stderr)
        return 2
    return 0

if __name__ == '__main__':
    sys.exit(main())
