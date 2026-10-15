#!/usr/bin/env python3
"""Import recent transients from Astro-COLIBRI as schedulable objectives.

Pipeline piece (cron-able, like fill_infield.py): queries the
latest_transients endpoint, keeps events localized well enough to point at
(err <= --max-err), dedupes multi-observatory alerts of the same event by
best localization, and creates Target + Objective rows:

  - Target.group gets 'transient' plus the event type (an 'FRB' type lands
    in the group string, triggering the scheduler's existing FRB boost);
  - Target.prior = --prior (transients outrank the prior<=1 survey targets);
  - Objective.notes['camp'] = --camp, chosen to match sched_first's
    --camp prefix filter, and notes['before'] = event time + --fade-days,
    which the scheduler's time-based priority already understands
    (late windows are soft-deprioritized, not forbidden).

Re-running is idempotent: an existing target of the same name is updated,
an existing open objective for it is left alone.
"""

import os
os.environ['DJANGO_SETTINGS_MODULE'] = 'schedule.settings'

import argparse
import json
import urllib.request
from datetime import datetime, timedelta, timezone

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--uid', default=os.environ.get('ACOLIBRI_UID', ''),
                    help="Astro-COLIBRI user id (default: $ACOLIBRI_UID)")
parser.add_argument('--days', type=float, default=3.0,
                    help="look-back window in days (default %(default)s)")
parser.add_argument('--max-err', type=float, default=0.25, dest='max_err',
                    help="max localization error radius in deg; larger error boxes "
                         "can't be usefully pointed at with a 0.9x0.5 FoV (default %(default)s)")
parser.add_argument('--types', default='grb,ot_sn,burst',
                    help="comma-separated event types to import (default %(default)s); "
                         "FRBs deliberately excluded -- not a science case for this mission")
parser.add_argument('--expo', type=float, default=500.0)
parser.add_argument('--repeat', type=int, default=2)
parser.add_argument('--prior', type=float, default=3.0,
                    help="Target.prior for imported transients (default %(default)s)")
parser.add_argument('--fade-days', type=float, default=5.0, dest='fade_days',
                    help="notes['before'] deadline = event time + this (default %(default)s)")
parser.add_argument('--camp', default='schedme_too',
                    help="objective campaign tag; keep the 'schedme' prefix so the "
                         "default sched_first run picks it up (default %(default)s)")
parser.add_argument('--select', default='',
                    help="comma-separated event names to import regardless of the err/type "
                         "filters (hand-picked representative events)")
parser.add_argument('--dry-run', action='store_true')
args = parser.parse_args()

if not args.uid:
    raise SystemExit("no uid: pass --uid or set ACOLIBRI_UID")

now = datetime.now(timezone.utc)
body = {"uid": args.uid,
        "time_range": {"min": (now - timedelta(days=args.days)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                       "max": now.strftime('%Y-%m-%dT%H:%M:%SZ')}}
data=json.dumps(body).encode()
print(data)
req = urllib.request.Request("https://astro-colibri.science/latest_transients",
                             data=data,headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=60) as r:
    events = json.loads(r.read())
if isinstance(events, dict):
    events = events.get('voevents', [])
print(f"{len(events)} events in the last {args.days:g} days")

wanted = {t.strip() for t in args.types.split(',') if t.strip()}


def parse_time(s):
    s = (s or '').strip().replace(' ', 'T').rstrip('Z')
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


selected = {n.strip() for n in args.select.split(',') if n.strip()}

# best localization per event name
best = {}
for ev in events:
    name = (ev.get('source_name') or '').strip() or \
           (ev.get('astro_colibri_id') or '').strip()
    if not name:
        continue
    ra, dec, err = ev.get('ra'), ev.get('dec'), ev.get('err', -1.0)
    if ra is None or dec is None:
        continue
    if name not in selected:
        if selected:
            continue   # hand-picked mode: only the named events
        if ev.get('type') not in wanted:
            continue
        if err is None or err < 0 or err > args.max_err:
            continue
    tm = parse_time(ev.get('time'))
    if name not in best or err < best[name]['err']:
        best[name] = dict(name=name, ra=float(ra), dec=float(dec), err=float(err),
                          type=ev.get('type'), time=tm,
                          observatory=ev.get('observatory', ''))

print(f"{len(best)} usable events (types {sorted(wanted)}, err <= {args.max_err} deg):")
for ev in best.values():
    print(f"  {ev['type']:6s} {ev['name']:28s} ra={ev['ra']:8.3f} dec={ev['dec']:8.3f} "
          f"err={ev['err']*3600:6.1f}\" {ev['observatory']}")

if args.dry_run:
    raise SystemExit(0)

import django
django.setup()
from astropy.coordinates import SkyCoord, GeocentricMeanEcliptic
import astropy.units as u
from mockat.models import Target, Objective

created_t = created_o = 0
for ev in best.values():
    target = Target.objects.filter(name=ev['name']).first()
    if target is None:
        target = Target(name=ev['name'], group=f"transient {ev['type']}",
                        notes={})
        created_t += 1
    target.pRA, target.pdec = ev['ra'], ev['dec']
    target.prior = max(target.prior or 0, args.prior)
    notes = target.notes if isinstance(target.notes, dict) else {}
    notes['type'] = ev['type']
    notes['colibri_observatory'] = ev['observatory']
    # [lon, lat] ecliptic degrees, the DB convention (e.g. Crab [84, -1]):
    # without this, sched_first's select_targets() cannot apply its sun/moon
    # avoidance and a sun-close transient sails straight into the schedule
    ecl = SkyCoord(ev['ra'] * u.deg, ev['dec'] * u.deg
                   ).transform_to(GeocentricMeanEcliptic())
    notes['eclipt_bl'] = [round(ecl.lon.degree), round(ecl.lat.degree)]
    target.notes = notes
    target.save()

    if Objective.objects.filter(target=target, scheduled_line__isnull=True,
                                notes__camp__startswith='schedme').exists():
        continue  # already queued
    notes = {'camp': args.camp, 'colibri': True}
    if ev['time'] is not None:
        notes['before'] = (ev['time'] + timedelta(days=args.fade_days)
                           ).replace(tzinfo=None).isoformat()
    Objective.objects.create(target=target, expo=args.expo, repeat=args.repeat,
                             delay=0, notes=notes)
    created_o += 1

print(f"created {created_t} targets, {created_o} objectives (camp {args.camp!r}, "
      f"prior {args.prior}, expo {args.expo:g}x{args.repeat})")
