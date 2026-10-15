# --- constants ---------------------------------------------------------
R_EQ = 6378.137          # Earth equatorial radius, km (WGS-72/84)
R_POL = 6356.752314245   # Earth polar radius, km
R_SUN = 695700.0         # Sun radius, km
AU = 149597870.7         # km
 
import math
import numpy as np
from sgp4.api import Satrec, jday, SGP4_ERRORS
from datetime import datetime, timedelta, timezone

tn=datetime.now(tz=timezone.utc)

tle_vzlu='''TRANSPORTER-12 OBJECT G 
1 62615U 25009G   26263.86389899  .00003966  00000+0  15330-3 0  9993
2 62615  97.3878 342.8058 0005859   6.0294 354.1014 15.26806629 93399'''


earth_avoidance=5
#these 2 are not used now - applied only when selecting targets (for short periods the sun/moon position doesn't change much)
sun_avoidance=50
moon_avoidance=10

global gdebug
gdebug=0

# --- Sun ephemeris (Vallado, "Fundamentals of Astrodynamics", Alg. 29) --
def sun_vector(jd: float, fr: float) -> tuple[float, float, float]:
    """Geocentric Sun position in km, mean equator and equinox of date.
    Accuracy ~0.01 deg, which is ~0.2 s of eclipse timing in LEO."""
    T = (jd - 2451545.0 + fr) / 36525.0
    lam_m = math.radians(280.460 + 36000.771 * T)
    M = math.radians(357.5291092 + 35999.05034 * T)
    lam = lam_m + math.radians(1.914666471 * math.sin(M)
                               + 0.019994643 * math.sin(2 * M))
    r = (1.000140612 - 0.016708617 * math.cos(M)
         - 0.000139589 * math.cos(2 * M)) * AU
    eps = math.radians(23.439291 - 0.0130042 * T)
    return (r * math.cos(lam),
            r * math.cos(eps) * math.sin(lam),
            r * math.sin(eps) * math.sin(lam))
 
def jd_of(t: datetime) -> tuple[float, float]:
    return jday(t.year, t.month, t.day,
                t.hour, t.minute, t.second + t.microsecond * 1e-6)
  
def mean_sun_ra(jd: float, fr: float) -> float:
    """Right ascension of the fictitious mean sun, degrees in [0,360)."""
    d = (jd - 2451545.0) + fr
    return (280.4606184 + 0.9856473662862 * d + 2.9e-13 * d * d) % 360.0

def true_sun_ra(jd: float, fr: float) -> float:
    """Apparent right ascension of the true Sun, degrees (low precision)."""
    T = ((jd - 2451545.0) + fr) / 36525.0
    lam_m = 280.460 + 36000.771 * T
    M = math.radians(357.5291092 + 35999.05034 * T)
    lam = math.radians(lam_m + 1.914666471 * math.sin(M)
                       + 0.019994643 * math.sin(2 * M))
    eps = math.radians(23.439291 - 0.0130042 * T)
    return math.degrees(math.atan2(math.cos(eps) * math.sin(lam),
                                   math.cos(lam))) % 360.0
 
def gmst_deg(jd: float, fr: float) -> float:
    """Greenwich mean sidereal time, degrees in [0,360)."""
    d = (jd - 2451545.0) + fr
    return (280.46061837 + 360.98564736629 * d
            + 3.87933e-4 * (d / 36525.0) ** 2) % 360.0

 
# --- shadow geometry ---------------------------------------------------
def shadow_functions(r, s, oblate: bool = True) -> tuple[float, float]:
    """Umbra and penumbra shadow functions, in radians.
 
    Work in the satellite's sky: let theta_e be the apparent angular radius
    of the Earth, theta_s that of the Sun, and theta the apparent separation
    of their centres.  Then
 
        theta < theta_e - theta_s   the Sun's disc is fully hidden  (umbra)
        theta < theta_e + theta_s   part of the disc is hidden   (penumbra)
 
    so the two returned values are negative inside the umbra and inside the
    penumbra respectively.  Because theta_s > 0, f_umb > f_pen always, i.e.
    the umbra is guaranteed to nest inside the penumbra.  Both functions are
    smooth everywhere, including across the terminator plane, so they are
    safe to bisect on.  (A cone-surface distance test is not: sunward of the
    terminator, being inside the umbral cone does not mean being in shadow.)
 
    With oblate=True the z axis of both vectors is stretched by R_eq/R_pol,
    a linear map that carries the reference ellipsoid onto a sphere of
    radius R_eq and carries tangent lines to tangent lines.  It shifts event
    times by a few seconds relative to a spherical Earth.  The only
    approximation is a <0.4% distortion of the Sun's apparent shape.
    """
    if oblate:
        k = R_EQ / R_POL
        r = (r[0], r[1], r[2] * k)
        s = (s[0], s[1], s[2] * k)
 
    r_mag = math.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
    d = (s[0] - r[0], s[1] - r[1], s[2] - r[2])          # satellite -> Sun
    d_mag = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
 
    theta_e = math.asin(min(1.0, R_EQ / r_mag))           # Earth's disc
    theta_s = math.asin(min(1.0, R_SUN / d_mag))          # Sun's disc
 
    # separation between the direction to Earth's centre (-r) and to the Sun
    cos_t = -(r[0] * d[0] + r[1] * d[1] + r[2] * d[2]) / (r_mag * d_mag)
    theta = math.acos(max(-1.0, min(1.0, cos_t)))
 
    return theta - (theta_e - theta_s), theta - (theta_e + theta_s)

MU = 398600.4418        # km^3/s^2
J2 = 1.08262668e-3
# Nodal drift a sun-synchronous orbit must match: 360 deg per tropical year
SSO_RATE = 0.9856473663     # deg/day

def z_at(sat: Satrec, jd0: float, fr0: float, t: float) -> tuple[float, float]:
    """(z, vz) t seconds after the given epoch."""
    frac = fr0 + t / 86400.0
    w = math.floor(frac)
    e, r, v = sat.sgp4(jd0 + w, frac - w)
    if e != 0:
        raise RuntimeError(f"SGP4 error {e} at t={t:.1f}s")
    return r[2], v[2]

def next_ascending_node(sat: Satrec, jd0: float, fr0: float,
                        step: float = 30.0, span: float = 8640.0) -> float:
    """Seconds from epoch to the next northbound equator crossing."""
    t, z_prev = 0.0, z_at(sat, jd0, fr0, 0.0)[0]
    while t < span:
        t_next = t + step
        z_next = z_at(sat, jd0, fr0, t_next)[0]
        if z_prev < 0.0 <= z_next:                 # ascending crossing
            a, b = t, t_next
            for _ in range(80):                     # bisect to ~1 us
                if b - a < 1e-6:
                    break
                m = 0.5 * (a + b)
                if z_at(sat, jd0, fr0, m)[0] < 0.0:
                    a = m
                else:
                    b = m
            return 0.5 * (a + b)
        t, z_prev = t_next, z_next
    raise RuntimeError("no ascending node found in the search span")
 
 
def node_right_ascension(sat: Satrec, jd0: float, fr0: float,
                         t: float) -> float:
    frac = fr0 + t / 86400.0
    w = math.floor(frac)
    _, r, _ = sat.sgp4(jd0 + w, frac - w)
    return math.degrees(math.atan2(r[1], r[0])) % 360.0
    

from scipy.optimize import bisect

class OrbitModel:
    def __init__(self, satrec: Satrec, t0: datetime, oblate: bool = True):
        self.sat = satrec
        self.t0 = t0
        self.oblate = oblate
        self.jd0, self.fr0 = jday(t0.year, t0.month, t0.day,
                                  t0.hour, t0.minute,
                                  t0.second + t0.microsecond * 1e-6)
 
    def at(self, t: float) -> tuple[float, float]:
        """Shadow functions t seconds after the start epoch."""
        frac = self.fr0 + t / 86400.0
        whole = math.floor(frac)
        e, r, _ = self.sat.sgp4(self.jd0 + whole, frac - whole)
        if e != 0:
            raise RuntimeError(f"SGP4 error {e} at t={t:.1f}s "
                               "(orbit decayed or TLE out of range)")
        return shadow_functions(r, sun_vector(self.jd0 + whole, frac - whole),
                                self.oblate)
 
    def umbra(self, t: float) -> float:
        return self.at(t)[0]
 
    def penumbra(self, t: float) -> float:
        return self.at(t)[1]

def find_crossings(f, span: float, step: float, label: str,
                   rescan: bool = True) -> list[tuple[float, str]]:
    """Locate every sign change of f on [0, span]."""
    n = int(math.ceil(span / step))
    ts = [min(i * step, span) for i in range(n + 1)]
    vals = [f(t) for t in ts]
    events: list[tuple[float, str]] = []
 
    def scan(ts, vals):
        for i in range(len(ts) - 1):
            if (vals[i] < 0.0) != (vals[i + 1] < 0.0):
                t = bisect(f, ts[i], ts[i + 1])
                kind = "entry" if vals[i] > 0.0 else "exit"
                events.append((t, f"{label} {kind}"))
 
    scan(ts, vals)

    if rescan:
        # A short pass entirely between two grid points leaves no sign
        # change.  Rescan any positive local minimum that dips close to the
        # boundary, using a 20x finer grid.
        for i in range(1, len(ts) - 1):
            if (vals[i] > 0.0 and vals[i] <= vals[i - 1]
                    and vals[i] <= vals[i + 1]
                    and vals[i] < 0.5 * max(abs(vals[i - 1] - vals[i]),
                                            abs(vals[i + 1] - vals[i]))):
                fine = step / 20.0
                m = int((ts[i + 1] - ts[i - 1]) / fine)
                sub = [ts[i - 1] + j * fine for j in range(m + 1)]
                scan(sub, [f(t) for t in sub])
 
    events.sort()
    return events

def load_tle(text: str) -> tuple[str, Satrec]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) >= 3 and not lines[0].startswith("1 "):
        name, l1, l2 = lines[0], lines[1], lines[2]
    else:
        name, l1, l2 = "satellite", lines[0], lines[1]
    return name, Satrec.twoline2rv(l1, l2)
 
 
def tle_epoch(sat: Satrec) -> datetime:
    jd = sat.jdsatepoch + sat.jdsatepochF
    # Julian date -> UTC datetime
    z = math.floor(jd + 0.5)
    f = (jd + 0.5) - z
    alpha = math.floor((z - 1867216.25) / 36524.25)
    a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    di = int(day)
    secs = (day - di) * 86400.0
    return (datetime(int(year), int(month), di, tzinfo=timezone.utc)
            + timedelta(seconds=secs))

def eclipse_periods(events,t0=None):
    '''turn the flat entry/exit list from get_eclipses() into (start,end)
    datetime pairs, one per full eclipse (a leading exit or trailing entry
    with no matching partner is dropped).'''
    if t0 is None:
        t0=tn
    periods=[]
    t_in=None
    for t,kind in events:
        if kind.endswith('entry'):
            t_in=t
        elif kind.endswith('exit') and t_in is not None:
            periods.append((t0+timedelta(seconds=t_in),t0+timedelta(seconds=t)))
            t_in=None
    return periods

def get_eclipses(tle,span=2e4,step=10, t0=None):
    sat=load_tle(tle)[1]
    model=OrbitModel(sat,tn)

    jd0=sat.jdsatepoch
    fr0=sat.jdsatepochF    
    epoch = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(days=(jd0 - 2451545.0) + fr0)
    umbras = find_crossings(model.umbra, span, step, "umbra")

    if t0==None: t0=models.t0
    return eclipse_periods(umbras, t0)



#checking visibility -------------------------------------------------------------------------------------------

from astropy.utils import iers

import astropy.units as u
from astropy.coordinates import (TEME, ITRS, GCRS, AltAz, EarthLocation,
                                 CartesianRepresentation,
                                 CartesianDifferential)
from astropy.time import Time


def teme_state(sat: Satrec, t: Time) -> TEME:
    """SGP4 state at Time t as a TEME frame object, with velocity attached."""
    jd1, jd2 = t.jd1, t.jd2          # astropy already splits the JD for us
    err, r, v = sat.sgp4(jd1, jd2)
    if err != 0:
        raise RuntimeError(SGP4_ERRORS.get(err, f"SGP4 error {err}"))
    return TEME(CartesianRepresentation(
                    np.asarray(r) * u.km,
                    differentials=CartesianDifferential(
                        np.asarray(v) * u.km / u.s)),
                obstime=t)


def satellite_location(sat: Satrec, t: Time) -> EarthLocation:
    """EarthLocation of the satellite at time t.

    Valid ONLY for obstime == t: EarthLocation is Earth-fixed, so reusing
    it at another epoch drags the satellite around with the Earth's spin
    (~0.25 deg/min, about 29 km/min at LEO altitude).
    """
    itrs = teme_state(sat, t).transform_to(ITRS(obstime=t))
    return EarthLocation.from_geocentric(*itrs.cartesian.xyz)


def satellite_observer(sat: Satrec, t: Time) -> GCRS:
    """A GCRS frame centred on the satellite, carrying its true velocity.

    Transform a target into this frame to get the direction as actually
    seen from the spacecraft.  Unlike an EarthLocation, this knows the
    orbital velocity, so stellar aberration is right.
    """
    g = teme_state(sat, t).transform_to(GCRS(obstime=t)).cartesian
    return GCRS(obstime=t,
                obsgeoloc=g.without_differentials(),
                obsgeovel=g.differentials["s"].to_cartesian())


def limb_geometry(sat: Satrec, t: Time, targets, margin=0) -> dict:
    """Earth-occultation geometry, done in inertial vectors.

    Returns the nadir angle to the target, the angular radius of the Earth
    as seen from the satellite, and whether the target is blocked.  Do this
    with geocentric vectors, not with AltAz altitude: the geodetic vertical
    is tilted from the geocentric radius by up to ~0.19 deg, which matters
    for a limb test.
    """
    state = teme_state(sat, t).transform_to(GCRS(obstime=t))
    r = state.cartesian.without_differentials().xyz.to_value(u.km)
    r_mag = float(np.linalg.norm(r))

    nadir = -r / r_mag
    limb = np.degrees(np.arcsin(R_EQ / r_mag))
    replist=[]
    if not np.iterable(targets): targets=[targets]
    for target in targets:
        d = target.transform_to(GCRS(obstime=t)).cartesian.xyz.to_value()
        d = d / np.linalg.norm(d)
        nadir_angle = np.degrees(np.arccos(np.clip(nadir @ d, -1.0, 1.0)))
        repdict={"nadir_angle_deg": float(nadir_angle),
            "earth_angular_radius_deg": float(limb),
            "occulted": bool(nadir_angle < limb+margin),
            # the limb sits this far BELOW the local horizontal plane
            "limb_elevation_deg": float(limb - 90.0),
            "altitude_km": r_mag - 6378.137}
        replist.append(repdict)
    return replist

global our_sat
our_sat = None

def visible(tm,targs):
    from mockat.models import Target
    targ_objs=list(Target.objects.in_bulk(targs).values())
    allrep=limb_geometry(our_sat,Time(tm),[tg.get_pos() for tg in targ_objs],margin=earth_avoidance)
    visrep=[targ_objs[i].pk for i in range(len(targs)) if allrep[i]['occulted']==False]
    if gdebug==3: print(tm.isoformat(),len(visrep))
    return visrep
    
#---------------------------------------------------------------
targs=[]

def intermix(tls1,tls2,tim1,tim2,tmid=None):
    #from mockat.tools import visible
    tcom=list(set.intersection(set(tls1),set(tls2)))
    #print('{%i}'%(len(tcom))),
    if tmid==None:tmid=tim1+(tim2-tim1)/2
    tdif1=set(tls1)-set(tls2)
    if len(tdif1)>0:
        tls1p=visible(tmid,targs=list(tdif1))
        tcom+=tls1p
        #print('{%i}'%(len(tls1p))),
    tdif2=set(tls2)-set(tls1)
    if len(tdif2)>0:
        tls2p=visible(tmid,targs=list(tdif2))
        tcom+=tls2p
        #print('{%i}'%(len(tls2p))),
    return tcom,tmid#list(set(tcom)),tmid

def recumix(tls1,tls2,ts1,ts2,lev=3): #recursive down to given level
    lm,tm=intermix(tls1,tls2,ts1,ts2)
    if lev>0: return recumix(tls1,lm,ts1,tm,lev=lev-1)+[lm]+recumix(lm,tls2,tm,ts2,lev=lev-1)
    else: return [lm]

#allvis={}
def get_vis(tm):
    stamp=tm.isoformat()[:19]
    
    if not stamp in allvis:
        allvis[stamp]=visible(tm,targs=targs)
    return allvis[stamp]

def fill_times(ts1,ts2,dt=60,loud=0):
    '''using recursive splitting of a time interval
    writing visibilities in a database (dict)'''
    teclips=(ts2-ts1).seconds
    #if not 'allvis' in locals():allvis={}
    ex2=int(np.floor(np.log2((teclips)//dt)))
    if ex2>1:
        nb_int=teclips/(2**(ex2-1)*dt)
        tslot=timedelta(0,2**(ex2-1)*dt)
        if int(np.floor(nb_int))==3:
            tpnts=[ts1,ts1+tslot,ts1+tslot*2,ts1+tslot*3,ts2]
        else:
            tpnts=[ts1,ts1+tslot,ts1+tslot*2,ts2]
    else:
        tslot=timedelta(0,2**ex2*dt)
        tpnts=[ts1,ts1+tslot,ts2]

    zlev=int(np.log2(tslot.seconds//dt))-1
    for k in range(len(tpnts)-1):
        tls1=get_vis(tpnts[k])#model.t0 + timedelta(seconds=umb[0][0])
        tls2=get_vis(tpnts[k+1])
        if loud>1: print(f'running from {tpnts[k]} till {tpnts[k+1]}*******:')
        if len(set(tls2)-set(tls1))>1 or len(set(tls1)-set(tls2))>1: #sources are changing between these times
            allev=recumix(tls1,tls2,tpnts[k],tpnts[k+1],lev=zlev)
            #print('||',len(allev))
            tm=tpnts[k]+timedelta(0,dt)
            for a in allev:
                stamp=tm.isoformat()[:19]
                allvis[stamp]=a
                tm=tm+timedelta(0,dt)
                if loud>0: print(stamp,':',len(a))
        else:
            print('skipping no. ',k)

    if loud>0: print(f'remaining {teclips}-{tslot.seconds} sec')


#parallel processing of several eclipses --------------------------------------------------------


def _fill_one_eclipse(args):
    '''runs in a worker process: fills a private `allvis` cache for a single
    eclipse and hands it back, so workers never share state.'''
    ts1,ts2,targs_local,dt=args
    global targs,allvis
    targs=targs_local
    allvis={}
    fill_times(ts1,ts2,dt)
    return allvis


def fill_eclipses_parallel(eclipses,targs_list,dt=60,max_workers=None,debug=0,sat=None):
    '''compute visibility lists for several eclipse periods in parallel,
    one worker process per eclipse, and merge the results into `allvis`.

    eclipses    - list of (ts1,ts2) datetime pairs, e.g. from eclipse_periods()
    targs_list  - target list to search (becomes the module-level `targs`)
    dt          - sampling step in seconds, passed through to fill_times
    max_workers - defaults to one process per eclipse, capped at cpu_count()

    Requires the (default, on Linux) 'fork' multiprocessing start method,
    since targs_list holds Django model instances that were fetched once in
    the parent process and workers rely on inheriting the already-configured
    Django app registry rather than re-running django.setup() themselves.
    '''
    import os,multiprocessing
    from concurrent.futures import ProcessPoolExecutor,as_completed
    #if sat!=None: sat=
        
    max_workers=max_workers or min(len(eclipses),os.cpu_count() or 4)
    jobs=[(ts1,ts2,targs_list,dt) for ts1,ts2 in eclipses]
    ctx=multiprocessing.get_context('fork')

    merged={}
    with ProcessPoolExecutor(max_workers=max_workers,mp_context=ctx) as ex:
        futs={ex.submit(_fill_one_eclipse,j):j for j in jobs}
        for fut in as_completed(futs):
            ts1,ts2,_,_=futs[fut]
            try:
                partial=fut.result()
            except Exception as e:
                print(f'eclipse {ts1} - {ts2} failed: {e}')
                continue
            merged.update(partial)
            #print(f'done {ts1} - {ts2}: {len(partial)} stamps')

    allvis.update(merged)
    return merged
