#!/usr/bin/env python3
"""Populate Target.infield and Target.nearby from catalog geometry.

infield: PKs of other targets close enough to share one FoV pointing with
    this target. Default radius 0.5 deg = the pair fits the 0.9 x 0.5 deg
    FoV at any spacecraft roll when pointing between them; pass 0.9 only if
    the roll can be steered to align the pair with the long axis. (The
    scheduler's co-field grouping points at the PRIMARY target and therefore
    uses the half-extent 0.25/0.45 -- infield's looser pair-wise criterion
    is the right one once centroid pointing exists.)
nearby: PKs within the wider --nearby-radius, for next-pointing selection
    (see Target.get_near / catalog tools).

Also reports exact duplicates (separation ~0): the catalog has a few
targets entered twice under different names, which double-book exposures.
"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'schedule.settings'

import argparse

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--infield-radius', type=float, default=0.5,
                    help="max pair separation (deg) to fit one pointing (default %(default)s)")
parser.add_argument('--nearby-radius', type=float, default=5.0,
                    help="radius (deg) for the 'nearby' list (default %(default)s)")
parser.add_argument('--dry-run', action='store_true',
                    help="report what would be written without saving")
args = parser.parse_args()

import django
django.setup()
import numpy as np
from mockat.models import Target

targets = [t for t in Target.objects.all()
           if t.pRA is not None and t.pdec is not None]
n = len(targets)
ra = np.radians([t.pRA for t in targets])
dec = np.radians([t.pdec for t in targets])
u = np.stack([np.cos(dec) * np.cos(ra),
              np.cos(dec) * np.sin(ra),
              np.sin(dec)], axis=1)
sep = np.degrees(np.arccos(np.clip(u @ u.T, -1.0, 1.0)))
np.fill_diagonal(sep, np.inf)

dups = [(targets[i].name.strip(), targets[j].name.strip(), sep[i, j])
        for i, j in zip(*np.where(sep < 0.005)) if i < j]
if dups:
    print(f"WARNING: {len(dups)} near-duplicate pairs (<0.005 deg) -- "
          "likely the same object entered twice:")
    for a, b, d in dups:
        print(f"  {a} == {b}  ({d*3600:.1f} arcsec)")

changed = 0
for i, t in enumerate(targets):
    infield_ids = [targets[j].pk for j in np.where(sep[i] <= args.infield_radius)[0]]
    nearby_ids = [targets[j].pk for j in
                  np.where((sep[i] > args.infield_radius)
                           & (sep[i] <= args.nearby_radius))[0]]

    def pack(ids, limit):
        s = ','.join(str(x) for x in ids)
        while len(s) > limit and ids:   # CharField length caps (200/400)
            ids = ids[:-1]
            s = ','.join(str(x) for x in ids)
        return s

    infield = pack(infield_ids, 200)
    nearby = pack(nearby_ids, 400)
    if infield != (t.infield or '') or nearby != (t.nearby or ''):
        changed += 1
        if not args.dry_run:
            t.infield = infield
            t.nearby = nearby
            t.save(update_fields=['infield', 'nearby'])

with_infield = sum(1 for i in range(n) if (sep[i] <= args.infield_radius).any())
print(f"{n} targets: {with_infield} have at least one in-field companion "
      f"(<= {args.infield_radius} deg); {changed} rows "
      f"{'would be' if args.dry_run else ''} updated")
