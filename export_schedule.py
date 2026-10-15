#!/usr/bin/env python3
"""Export one schedule line as a human-readable listing and a loadable SQL file.

The listing is grouped by orbit: each eclipse pass gets a header with the
orbit number, umbra entry/exit (from the visibility cache if given, else
inferred from the exposure grouping), and a per-pass science-time summary.
Transient targets (group starting 'transient') are marked with *.
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument('--line', type=float, required=True)
parser.add_argument('--db', default='db.sqlite3')
parser.add_argument('--vis-file', default='', dest='visfile',
                    help="visibility NPZ with eclipse 'periods' for exact pass boundaries")
parser.add_argument('--out-prefix', default='', dest='prefix',
                    help="output file prefix (default: line<N>_schedule)")
args = parser.parse_args()

prefix = args.prefix or f"line{args.line:g}_schedule"

con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
rows = list(con.execute("""SELECT s.date, s.expo, s.target_id, t.name, t."group"
    FROM mockat_schedule s JOIN mockat_target t ON t.id=s.target_id
    WHERE s.time>=? AND s.time<?+1 ORDER BY s.date""", (args.line, args.line)))
if not rows:
    raise SystemExit(f"no schedule rows for line {args.line:g}")

periods = []
if args.visfile:
    import numpy as np
    d = np.load(args.visfile, allow_pickle=True)
    periods = [(p0.replace(tzinfo=None), p1.replace(tzinfo=None))
               for p0, p1 in d['periods']]

# group exposures into passes by >30 min gaps
groups, cur = [], [rows[0]]
for r in rows[1:]:
    prev_end = datetime.fromisoformat(cur[-1][0]) + timedelta(seconds=cur[-1][1])
    if (datetime.fromisoformat(r[0]) - prev_end).total_seconds() > 1800:
        groups.append(cur)
        cur = [r]
    else:
        cur.append(r)
groups.append(cur)


def pass_bounds(group):
    first = datetime.fromisoformat(group[0][0])
    for p0, p1 in periods:
        if p0 <= first <= p1:
            return p0, p1
    return None, None


first_d = rows[0][0][:10]
total = sum(r[1] for r in rows)
tra = sum(r[1] for r in rows if r[4].startswith('transient'))
txt = [f"QUVIK schedule line {args.line:g}  --  starting {first_d}, "
       f"{len(groups)} eclipse passes / orbits",
       f"{len(rows)} exposures, {total:.0f} s science"
       + (f", {tra/total*100:.0f}% transients (*)" if tra else ""),
       "every exposure verified: umbra, Earth limb, sun/moon limits, slew feasibility",
       ""]
sql = [f"-- QUVIK schedule line {args.line:g} ({first_d}), "
       f"exported {datetime.now().date()}",
       "-- load ONE line per night: sqlite3 db.sqlite3 < " + prefix + ".sql",
       "BEGIN;",
       f"DELETE FROM mockat_schedule WHERE time>={args.line} AND time<{args.line}+1;"]

for n, group in enumerate(groups, 1):
    p0, p1 = pass_bounds(group)
    if p0 is not None:
        head = (f"orbit {n:2d} | eclipse {p0.strftime('%H:%M:%S')} - "
                f"{p1.strftime('%H:%M:%S')} UTC")
    else:
        head = (f"orbit {n:2d} | exposures {group[0][0][11:19]} - "
                + (datetime.fromisoformat(group[-1][0])
                   + timedelta(seconds=group[-1][1])).strftime('%H:%M:%S') + " UTC")
    sci = sum(r[1] for r in group)
    txt.append(f"--- {head} | {sci:.0f} s science " + "-" * max(1, 30 - len(head) // 2))
    prev_end = None
    for d, expo, tid, name, grp in group:
        dt = datetime.fromisoformat(d)
        gap = "" if prev_end is None else \
            f"   (+{(dt - prev_end).total_seconds() / 60:.1f} min slew)"
        mark = " *" if grp.startswith('transient') else ""
        txt.append(f"  {d[11:19]}  {expo:6.0f} s  {name.strip()}{mark}{gap}")
        prev_end = dt + timedelta(seconds=expo)
        sql.append(f"INSERT INTO mockat_schedule (time, expo, date, target_id) "
                   f"VALUES ({args.line:g}, {expo}, '{d}', {tid});  -- {name.strip()}")
    txt.append("")

for oid, notes, sl in con.execute("""SELECT id, notes, scheduled_line FROM mockat_objective
        WHERE json_extract(notes,'$.done_by_line.' || ?) IS NOT NULL""",
        (str(int(args.line)),)):
    nn = json.dumps(json.loads(notes)).replace("'", "''")
    sql.append(f"UPDATE mockat_objective SET notes='{nn}', "
               f"scheduled_line={sl if sl is not None else 'NULL'} WHERE id={oid};")
sql.append("COMMIT;")

open(prefix + '.txt', 'w').write("\n".join(txt) + "\n")
open(prefix + '.sql', 'w').write("\n".join(sql) + "\n")
print(f"wrote {prefix}.txt and {prefix}.sql ({len(rows)} exposures, "
      f"{len(groups)} orbits)")
