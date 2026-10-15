"""Coverage-aware visibility cache (the object Filip asked for).

Replaces the ad-hoc npz handling in sched_first.py, whose rebuild trigger
only compared TARGET sets: a rerun with a new date but the same targets
found nothing "missing", skipped the parallel fill entirely, and the
scheduler then fell back to one sequential sgp4+astropy call per 30 s
sample -- the "point by point, takes forever" failure mode. The opposite
waste also existed (one new ToO target -> recompute every target over every
period), and a TLE update silently reused stale geometry.

The cache knows three things and recomputes only what is actually missing:

  - the TLE it was built with: a different TLE invalidates everything
    (eclipse times and visibility both move);
  - which time segments it has eclipse periods for: a new/extended horizon
    computes eclipses only for the uncovered gaps;
  - which targets each eclipse period was sampled with: a period is
    refilled only if it intersects the requested window AND lacks some
    requested target. A refill covers the union of old and new targets and
    first purges the period's old samples -- fill_times' recursive
    bisection picks different timestamps for different target sets, so
    partial results cannot be merged within one period (see the note in
    sched_first.py), but distinct periods are independent.

Every fill goes through fill_eclipses_parallel, so the parallel path is
structurally guaranteed to run whenever any computation happens at all.

File format (npz, version 2): tle, segments, periods, period_pks, allvis.
Legacy v1 files (allvis/periods/targs_pks, no TLE) are treated as STALE by
default -- we cannot know what TLE built them, and this project has already
been burned once by a silently mislabeled cache; pass trust_legacy=True
(and the current TLE is then assumed) to adopt one anyway.
"""

import logging
from datetime import datetime, timedelta

import numpy as np

logger = logging.getLogger(__name__)


def _norm_tle(tle: str) -> str:
    """The two element lines, whitespace-normalized; the name line and
    formatting don't affect the propagation."""
    lines = [ln.strip() for ln in (tle or '').strip().splitlines() if ln.strip()]
    return '\n'.join(lines[-2:])


def _naive(dt):
    return dt.replace(tzinfo=None) if getattr(dt, 'tzinfo', None) else dt


class VisibilityCache:

    FILE_VERSION = 2

    def __init__(self, path: str, tle: str, trust_legacy: bool = False):
        self.path = path
        self.tle = _norm_tle(tle)
        self.segments = []      # [(t_a, t_b)] horizon segments with known eclipses
        self.periods = []       # [(t0, t1)] eclipse periods, sorted
        self.period_pks = []    # parallel to periods: set of sampled target pks
        self.allvis = {}        # iso-stamp -> visible pk collection
        self._load(trust_legacy)
        from sgp4 import api
        tlin=self.tle.split('\n')
        if len(tlin)>=2:
            self.sat = api.Satrec.twoline2rv(tlin[0],tlin[1])
        else:
            self.sat = None
    # ---------------------------------------------------------- persistence

    def _load(self, trust_legacy: bool):
        import os
        if not os.path.exists(self.path):
            return
        f = np.load(self.path, allow_pickle=True)
        if 'version' in f.files and int(f['version']) >= 2:
            stored_tle = str(f['tle'])
            if _norm_tle(stored_tle) != self.tle:
                print(f"vis cache {self.path}: built for a different TLE, "
                      "discarding all cached visibility")
                return
            self.segments = [(_naive(a), _naive(b)) for a, b in f['segments']]
            self.periods = [(_naive(a), _naive(b)) for a, b in f['periods']]
            self.period_pks = [set(int(x) for x in pks) for pks in f['period_pks']]
            self.allvis = f['allvis'].reshape(1,)[0]
            logger.info(f"{self.path}: {len(self.periods)} eclipse periods, "
                        f"{len(self.allvis)} samples, "
                        f"{len(self.segments)} horizon segments")
        elif 'allvis' in f.files:   # legacy v1
            if not trust_legacy:
                print(f"vis cache {self.path}: legacy file without a TLE stamp "
                      "-- treating as stale (pass trust_legacy=True to adopt it)")
                return
            self.allvis = f['allvis'].reshape(1,)[0]
            self.periods = [(_naive(a), _naive(b)) for a, b in f['periods']]
            pks = set(int(x) for x in f['targs_pks']) \
                if 'targs_pks' in f.files else set()
            self.period_pks = [set(pks) for _ in self.periods]
            if self.periods:
                self.segments = [(self.periods[0][0], self.periods[-1][1])]
            logger.warning(f"{self.path}: adopted legacy cache on trust, "
                           f"assuming the current TLE")

    def save(self):
        np.savez(self.path,
                 version=self.FILE_VERSION,
                 tle=self.tle,
                 segments=np.array(self.segments, dtype=object),
                 periods=np.array(self.periods, dtype=object),
                 period_pks=np.array([sorted(p) for p in self.period_pks],
                                     dtype=object),
                 allvis=self.allvis)

    # -------------------------------------------------------------- merging

    def copy(self):
        """Independent copy (no file access, nothing recomputed)."""
        new = object.__new__(type(self))
        new.path = self.path
        new.tle = self.tle
        new.sat = self.sat
        new.segments = list(self.segments)
        new.periods = list(self.periods)
        new.period_pks = [set(p) for p in self.period_pks]
        new.allvis = {k: list(v) for k, v in self.allvis.items()}
        return new

    def __add__(self, other):
        """cache_a + cache_b -> new merged cache (path taken from cache_a);
        see merge()."""
        if not isinstance(other, VisibilityCache):
            return NotImplemented
        return self.copy().merge(other)

    def __iadd__(self, other):
        if not isinstance(other, VisibilityCache):
            return NotImplemented
        return self.merge(other)

    def merge(self, other, tol: float = 2.):
        """Merge `other` into this cache in place and return self.

        Both caches must be built for the same TLE. Periods of `other`
        that do not overlap any of ours are copied over with their samples.
        Overlapping periods (the same eclipse, possibly with boundaries
        differing by truncation at a horizon segment edge) are fused into
        one period spanning both; its sample stamps are the union of both
        grids and at every stamp the visible set is the union of what each
        cache sees there. A cache that has no sample at that exact stamp
        contributes its nearest sample within the same period -- the same
        nearest-key lookup the scheduler uses -- since fill_times' grids
        differ between target sets and cannot be matched one-to-one.
        """
        if other.tle != self.tle:
            raise ValueError("cannot merge visibility caches built for "
                             "different TLEs")
        if other is self:
            return self
        dtol = timedelta(seconds=tol)

        def stamps(cache):
            return sorted((datetime.fromisoformat(k), k) for k in cache.allvis)

        def samples(stamped, p0, p1):
            """Sorted [(datetime, iso-stamp)] of samples in [p0, p1]."""
            return [s for s in stamped if p0 - dtol <= s[0] <= p1 + dtol]

        our_stamped, their_stamped = stamps(self), stamps(other)

        def nearest(smp, t):
            """Iso-stamp of the sample in smp closest to t (smp non-empty)."""
            import bisect
            i = bisect.bisect_left([s[0] for s in smp], t)
            cand = [j for j in (i - 1, i) if 0 <= j < len(smp)]
            return smp[min(cand, key=lambda j: abs(smp[j][0] - t))][1]

        for (q0, q1), qpks in zip(other.periods, other.period_pks):
            over = [i for i, (p0, p1) in enumerate(self.periods)
                    if p0 < q1 + dtol and p1 > q0 - dtol]
            theirs = samples(their_stamped, q0, q1)
            if not over:
                self.periods.append((q0, q1))
                self.period_pks.append(set(qpks))
                for t, k in theirs:
                    self.allvis[k] = sorted(set(self.allvis.get(k, ()))
                                            | set(other.allvis[k]))
                our_stamped = stamps(self)
                continue

            # fuse all our overlapping periods with the incoming one
            spans = [self.periods[i] for i in over]
            n0 = min([q0] + [a for a, _ in spans])
            n1 = max([q1] + [b for _, b in spans])
            npks = set(qpks).union(*(self.period_pks[i] for i in over))
            ours = sorted({s for a, b in spans
                           for s in samples(our_stamped, a, b)})
            ourvis = {k: self.allvis[k] for _, k in ours}

            merged = {}
            for t, k in sorted(set(ours) | set(theirs)):
                vis = set()
                for smp, vdict, spans_ in ((ours, ourvis, spans),
                                           (theirs, other.allvis, [(q0, q1)])):
                    if smp and any(a - dtol <= t <= b + dtol
                                   for a, b in spans_):
                        vis |= set(vdict[k] if k in vdict
                                   else vdict[nearest(smp, t)])
                merged[k] = sorted(vis)
            for _, k in ours:
                del self.allvis[k]
            self.allvis.update(merged)
            our_stamped = stamps(self)

            for i in sorted(over, reverse=True):
                del self.periods[i]
                del self.period_pks[i]
            self.periods.append((n0, n1))
            self.period_pks.append(npks)

        order = np.argsort([p[0] for p in self.periods])
        self.periods = [self.periods[i] for i in order]
        self.period_pks = [self.period_pks[i] for i in order]
        for a, b in other.segments:
            self._add_segment(a, b)
        return self

    def targcount(self, around):
        from datetime import datetime
        from collections import Counter
        pertamp=np.array([p[0].timestamp() for p in self.periods])
        vistamp=np.array([datetime.fromisoformat(q).timestamp() for q in self.allvis])
        pid=np.sum(around.timestamp()>pertamp)
        vid=np.sum(pertamp[pid]>vistamp)
        vid2=np.sum(pertamp[pid+1]-100>vistamp) if pid<len(pertamp) else None
        alstam=sorted(self.allvis.keys())
        cts=Counter(self.allvis[alstam[vid-1]])
        for v in range(vid,vid2):
            cts+=Counter(self.allvis[alstam[v]])
        return cts
    # ------------------------------------------------------------ geometry

    def _gaps(self, t0, t1):
        """Parts of [t0, t1] not inside any covered segment."""
        gaps, cursor = [], t0
        for a, b in sorted(self.segments):
            if b <= cursor:
                continue
            if a > t1:
                break
            if a > cursor:
                gaps.append((cursor, min(a, t1)))
            cursor = max(cursor, b)
            if cursor >= t1:
                break
        if cursor < t1:
            gaps.append((cursor, t1))
        return gaps

    def _add_segment(self, a, b):
        segs = sorted(self.segments + [(a, b)])
        merged = [segs[0]]
        for s in segs[1:]:
            if s[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], s[1]))
            else:
                merged.append(s)
        self.segments = merged

    def eclipses(self, t0, t1):
        """Cached eclipse periods intersecting [t0, t1] (call ensure first)."""
        return [(a, b) for a, b in self.periods if a < t1 and b > t0]

    # ------------------------------------------------------------- filling

    def ensure(self, t0, t1, needed_pks, margin: float = 1.05):
        """Guarantee [t0, t1] x needed_pks is covered; compute only what's
        missing, always through the parallel fill. Returns
        (eclipse periods intersecting [t0, t1], the allvis dict)."""
        import visibility_loop as vl
        if self.sat!=None: vl.our_sat=self.sat
        t0, t1 = _naive(t0), _naive(t1)
        needed = set(int(p) for p in needed_pks)
        t1e = t0 + timedelta(seconds=(t1 - t0).total_seconds() * margin)

        # 1. eclipse periods for uncovered horizon gaps
        for a, b in self._gaps(t0, t1e):
            span = (b - a).total_seconds()
            if span < 60:
                continue
            print(f"vis cache: computing eclipses for uncovered {a} .. {b}")
            new = vl.get_eclipses(self.tle, span, t0=a)
            known = {p0 for p0, _ in self.periods}
            for p0, p1 in new:
                p0, p1 = _naive(p0), _naive(p1)
                # a gap can re-find a boundary period already known
                if any(abs((p0 - k).total_seconds()) < 2 for k in known):
                    continue
                self.periods.append((p0, p1))
                self.period_pks.append(set())
            self._add_segment(a, b)
        order = np.argsort([p[0] for p in self.periods])
        self.periods = [self.periods[i] for i in order]
        self.period_pks = [self.period_pks[i] for i in order]

        # 2. periods in the window whose target coverage is short
        to_fill = []
        for i, (p0, p1) in enumerate(self.periods):
            if p0 >= t1 or p1 <= t0:
                continue
            if needed - self.period_pks[i]:
                to_fill.append(i)

        if to_fill:
            fill_pks = sorted(set.union(needed,
                                        *(self.period_pks[i] for i in to_fill)))
            print(f"vis cache: parallel fill of {len(to_fill)} of "
                  f"{len(self.periods)} known periods for {len(fill_pks)} "
                  f"targets ({len(self.periods) - len(to_fill)} already covered)")
            # purge old samples of the refilled periods: fill_times' sampling
            # grid depends on the target set, mixed grids within one period
            # are not mergeable
            for i in to_fill:
                p0, p1 = self.periods[i]
                drop = [k for k in self.allvis
                        if p0 <= datetime.fromisoformat(k) <= p1]
                for k in drop:
                    del self.allvis[k]
            vl.allvis = {}
            fresh = vl.fill_eclipses_parallel(
                [self.periods[i] for i in to_fill], fill_pks)
            self.allvis.update(fresh)
            for i in to_fill:
                self.period_pks[i] |= set(fill_pks)
            self.save()
        else:
            print(f"vis cache: complete for this window "
                  f"({len(self.eclipses(t0, t1))} periods, "
                  f"{len(needed)} targets) -- nothing to compute")

        return len(self.eclipses(t0, t1)), len(self.allvis)
