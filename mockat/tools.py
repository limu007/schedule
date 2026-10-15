from .models import *

def sun_check(targs,lskip=70):
    if targs==None: targs=Target.objects.all()
    from astropy import coordinates as coor
    for t in targs:
        bl=t.get_pos().transform_to(coor.GeocentricTrueEcliptic())
        if abs(bl.lat.degree>lskip): continue
        t.notes['eclipt_bl']=[int(bl.lon.degree),int(bl.lat.degree)]
        t.save()

def sun_close():
    import numpy as np
    calc=Target.objects.filter(notes__eclipt_bl__gt=0)
    coor=np.array([c.notes['eclipt_bl'] for c in calc]).T
    return coor

def sun_pos(date): 
    #rough (1 arcmin) Sun ecliptical position
    #day measured since equinox
    from math import cos,pi,floor
    if hasattr(date,'year'):
        from datetime import datetime,timezone
        if date.tzinfo==None:
            date=datetime.fromtimestamp(date.timestamp(),tz=timezone.utc)
        dref=datetime(2022,3,21,tzinfo=timezone.utc)
        day=(date-dref).days
    else:
        day=date
    year=365.2425
    day-=floor(day//year)*year
    p=[-1.82554891,  1.91369249,  0.26383126]
    cor=p[0]+p[1]*cos(day/year*2*pi-p[2])
    return day/year*360+cor


tle='''QUVIK
1 51085U 22002DF  22038.96554158  .00002307  00000-0  13979-3 0  9997
2 51085  97.4300 270.0000 0014600 156.2113 203.9840 15.11467589  3830'''

def sat_pos(time,ret=1):
    import predict
    qth=(15,50,200)
    if ret<0: glopos=predict.observe(tle, qth, at=time)
    else: glopos=predict.observe(tle, qth, at=time.timestamp())
    if ret==0: return glopos['longitude'],glopos['latitude'],glopos['altitude']
    from astropy import coordinates as coor
    location=coor.EarthLocation.from_geodetic(glopos['longitude'],glopos['latitude'],glopos['altitude']*1e3)
    return location

    
def visible(tm,delta=600,sun=45,moon=20,limb=-5,rep=0,targs=[],add_dist=False):
    '''get list of targets visible at time tm (+delta duration)
    with sun/moon limiting angle '''
    from datetime import timedelta,timezone
    from astropy import coordinates as coor
    from astropy.time import Time
    from numpy import array,where
    if len(targs)==0: targs=Target.objects.all()
    actpos=0
    pos=sat_pos(tm)
    azpos=[t.get_pos().transform_to(coor.AltAz(obstime=Time(tm),location=pos)) for t in targs]
    azarr=array([(a.alt.degree,a.az.degree) for a in azpos]).T
    if delta>0: #earth limb avoidance
        tm2=tm+timedelta(0,delta)
        pos2=sat_pos(tm2)
        azpos=[t.get_pos().transform_to(coor.AltAz(obstime=Time(tm2),location=pos2)) for t in targs] 
        azarr2=array([(a.alt.degree,a.az.degree) for a in azpos]).T
        sel=(azarr[0]>limb)*(azarr2[0]>limb)
    else:
        sel=(azarr[0]>limb)
    
    if sun>0:
        from numpy import sin,cos,deg2rad,arccos,pi
        cdis=lambda b0,l0:cos(deg2rad(b0))*cos(deg2rad(l0))#great circle distance from 0,0#sin(b1)*sin(b0)+cos(b1)*cos(b0)*cos(l0)
        b,l=array([t.notes.get('eclipt_bl',[0,90]) for t in targs])[sel].T
        actpos=sun_pos(tm)
        sundis=b-actpos
        sundis[sundis<-360]=sundis[sundis<-360]+360
        sunsel=abs(sundis)<sun
        tdis=cdis(sundis[sunsel],l[sunsel])
        sunsel2=tdis>cos(deg2rad(sun))
        sid=where(sel)[0][sunsel][sunsel2]
        if rep==2: return sid
        sel[sid]=False #too close to Sun
    if moon>90: #TB implemented
        lunpos=coor.get_moon(tm,pos)
    if rep==1: return sel
    tlist=[targs[int(i)] for i in where(sel)[0]]
    if sun>0 and add_dist:
        for t in tlist:
            b,l=t.notes.get('eclipt_bl',[0,90])
            tdis=cdis(b-actpos,l)
            t.sundist=arccos(tdis)/pi*180
    return tlist
    
def candidates(atti,targs):
    '''sort targets starting with nearest
    helper function for a naive scheduling approach
    ''' 
    for t in targs:
        t.dist=t.get_pos().separation(atti).degree
    from numpy import argsort
    return [targs[i] for i in argsort([t.dist for t in targs])] 
    
def chain_ver1(nn=None):
    from datetime import datetime
    if nn==None: nn=datetime.now()
    ame=visible(nn,add_dist=True)
    good=[a for a in ame if a.sundist>110 and a.sundist<130]
    return good
