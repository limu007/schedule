"""Independent post-run verification of a generated schedule.

Recomputes everything from first principles with sgp4 + astropy only --
deliberately NOT via visibility_loop, whose eclipse-anchoring bug
(OrbitModel pinned at import time instead of t0, shifting every eclipse
boundary by (t0 - now) mod the orbital period) is what motivated this gate.
The umbra test here was cross-checked against skyfield's is_sunlit() with
the JPL de421 ephemeris.

Checks, per exposure (at start / mid / end where relevant):
  - satellite fully inside the umbra (conical shadow, spherical Earth)
  - target above the Earth limb by the configured earth_limb margin
  - sun and moon separations respect the configured limits
  - consecutive exposures don't overlap and leave room for the slew
    (settle + trapezoidal transit, the same model the adapter enforces)

Called by sched_first.py between scheduling and saving; a non-empty
problem list means the schedule must not be saved.
"""

import math
import warnings
from datetime import timedelta

import numpy as np

R_EQ = 6378.137     # km
R_SUN = 695700.0    # km


def verify(schedules, tle: str, adapter) -> list:
    """Return a list of human-readable problems; empty list = schedule OK."""
    problems = []
    if not schedules:
        return problems

    from sgp4.api import Satrec, jday
    from astropy.time import Time
    from astropy.coordinates import get_sun, get_body, SkyCoord
    import astropy.units as u

    lines = [ln.strip() for ln in tle.strip().splitlines() if ln.strip()]
    sat = Satrec.twoline2rv(lines[-2], lines[-1])
    calc = adapter.visibility_calc

    def naive(dt):
        return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

    def sat_r(dt):
        dt = naive(dt)
        jd, fr = jday(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                      dt.second + dt.microsecond * 1e-6)
        err, r, _ = sat.sgp4(jd, fr)
        if err:
            raise RuntimeError(f"sgp4 error {err} at {dt}")
        return np.array(r)

    def in_umbra(dt):
        r = sat_r(dt)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            s = get_sun(Time(naive(dt))).cartesian.xyz.to(u.km).value
        rn = np.linalg.norm(r)
        d = s - r
        dn = np.linalg.norm(d)
        theta_e = math.asin(min(1.0, R_EQ / rn))
        theta_s = math.asin(min(1.0, R_SUN / dn))
        theta = math.acos(np.clip(np.dot(-r, d) / (rn * dn), -1.0, 1.0))
        return theta < theta_e - theta_s

    def above_limb(dt, coord):
        r = sat_r(dt)
        rn = np.linalg.norm(r)
        tgt = coord.cartesian.xyz.value
        gamma = math.degrees(math.acos(np.clip(np.dot(tgt, -r / rn), -1.0, 1.0)))
        rho = math.degrees(math.asin(R_EQ / rn))
        return gamma > rho + calc.earth_limb

    prev = None
    for sch in schedules:
        name = (sch.target.name or '?').strip()
        # true exposure, not the slice-coverage quantum stored in expo
        dur = float(getattr(sch, '_chunk_duration', sch.expo))
        t0 = sch.date
        t1 = t0 + timedelta(seconds=dur)
        mid = t0 + timedelta(seconds=dur / 2)
        tag = f"{name} @{naive(t0).isoformat()}"

        if dur <= 0:
            problems.append(f"{tag}: non-positive duration {dur}")
            continue
        coord = SkyCoord(sch.target.pRA * u.deg, sch.target.pdec * u.deg)

        if not all(in_umbra(t) for t in (t0, mid, t1)):
            problems.append(f"{tag}: exposure not fully inside the umbra")
        if not all(above_limb(t, coord) for t in (t0, mid, t1)):
            problems.append(f"{tag}: target below Earth limb "
                            f"(limit {calc.earth_limb} deg)")
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tmid = Time(naive(mid))
            sun_sep = get_sun(tmid).separation(coord).degree
            moon_sep = get_body('moon', tmid).separation(coord).degree
        if sun_sep < calc.sun_limit:
            problems.append(f"{tag}: sun separation {sun_sep:.1f} deg "
                            f"< limit {calc.sun_limit}")
        if moon_sep < calc.moon_limit:
            problems.append(f"{tag}: moon separation {moon_sep:.1f} deg "
                            f"< limit {calc.moon_limit}")

        if prev is not None:
            pname, pend, pcoord = prev
            gap = (t0 - pend).total_seconds()
            if gap < 0:
                problems.append(f"{pname} -> {name}: exposures overlap "
                                f"by {-gap:.0f}s")
            elif gap < 3000:  # same-pass neighbours; cross-pass gaps are huge
                sep = pcoord.separation(coord).degree
                need = 0.0 if sep < 1e-6 else \
                    adapter.slew_settle + adapter._slew_move_seconds(sep)
                if gap + 1e-6 < need:
                    problems.append(f"{pname} -> {name}: slew of {sep:.1f} deg "
                                    f"needs {need:.0f}s, only {gap:.0f}s given")
        prev = (name, t1, coord)

    return problems
