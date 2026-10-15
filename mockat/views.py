from django.shortcuts import render

from django.http import HttpResponse
from django.views import generic

from .models import *

groups=['star:BA',
     'star:opacity',
     'star:warps',
     'pulsar:Magnetar',
     'star:Be',
     'bin:',
     'agn:',
     'merg:',
     'pulsar:',
     'pulsar:FRB',
     'star:windinst']

def index(request):
    context={'targets':[]}
    #labfile="templates/catalog/STM_labels.csv"
    labfile="/home/limu/Public/alphive/mockat/STM_labels.csv"
    lines=[a.strip().split(';') for a in open(labfile).readlines()]
    groups=[a[2] for a in lines]
    base=Target.objects
    if 'limmag' in request.GET:
        base=base.filter(mag0__lte=float(request.GET['limmag']))
    if 'grp' in request.GET:
        context={'grpid':request.GET['grp']}
        tarlist=list(base.filter(group=request.GET['grp']))
        if request.GET['grp'] in groups:
            il=groups.index(request.GET['grp'])
            context["grpname"]=lines[il][0]
            context["desc"]=lines[il][3]
        if 'limmag' in request.GET: context["extra"]="&limmag="+request.GET['limmag']
        if 'mags' in request.GET:
            context["table"]=True
            try:
                Usel=[t.mag1 for t in tarlist if t.mag1>0]
                if len(Usel)==0: Usel=[0]
                maxlims=[max(Usel)]
                maxlims+=[max([t.mag0 for t in tarlist if t.mag0>0])]
                maxlims+=[max([t.notes['mag_V'] for t in tarlist if 'mag_V' in t.notes])]
            except:
                maxlims=[]
            else:
                context['lims']=maxlims
        for t in tarlist:
            if 'mag_V' in t.notes: t.magV=t.notes['mag_V']
        context['targets']=tarlist
        #context['grpid']=
    else:
        context['cntall']=base.count()
        grlist=[(l[2],l[0],base.filter(group=l[2]).count(),l[1]) for l in lines]
        context['census']=[l for l in grlist if l[2]>0]
        if 'stat' in request.GET:
            grtemp={}
            for l in context['census']:
                grtemp[l[0]]=sum([a.all_plans() for a in base.filter(group=l[0])])
            context['stats']=grtemp
            context['census']=[l+(grtemp[l[0]],) for l in grlist if l[2]>0]

    return render(request,"catalog/index.html",context)
    #return HttpResponse("Hello, world. You're at trigger index. Please have a look at <a href=\"month/9/\">last month</a> GCNs")

def observe(request, pk):
    from django.core.exceptions import ObjectDoesNotExist
    import os
    try:
        src=Target.objects.get(pk=pk)
    except ObjectDoesNotExist:
        HttpResponse("Target <"+pk+"> not found")
        return
    context={'src':src}
    if 'time' in request.GET:
        from datetime import datetime
        tstring=request.GET['time'].replace('T',' ')
        try:
            tm=datetime.strptime(tstring,"%Y-%m-%d %H-%M-%S")
        except:
            tm=datetime.strptime(tstring,"%Y-%m-%d")
        context['time']=tm
        from .tools import visible
        #context['near']=visible(tm)
    return render(request,"catalog/detail.html",context)

def simbad_mod(grname,bibcode="1978ApJ...226..379W",maglim=0):
    '''list of targets referenced in an article'''
    from astropy.coordinates import Angle
    from astroquery.simbad import Simbad
    Simbad.add_votable_fields("sptype")
    Simbad.add_votable_fields("flux(U)")
    #tab = Simbad.query_region(coord.GalacticCoordinates(l=31.0087, b=14.0627, unit=(u.deg, u.deg)), radius='0d0m2s')
    targ=[]
    tab=Simbad.query_bibobj(bibcode)
    frm=tab.to_pandas()
    if maglim>0:
        frm=frm[frm.FLUX_U>maglim]
    for i in range(len(frm)):
        r=frm.iloc[i]
        t=Target(name=r['MAIN_ID'].replace("* ",""),pRA=Angle(r.RA+'h').degree,pdec=Angle(r.DEC+'d').degree,mag1=r.FLUX_U,group=grname)
        t.notes['spectype']=r.SP_TYPE
        targ.append(t)
    return targ

def schedule(request,pk):
    from .tools import tle
    if pk==0: #overall page
        from collections import Counter
        cts=Counter([p.time for p in Schedule.objects.all()])
        clines=['<li><a href="../%i/"> observation schedule %i</a> (%i pointings)</li>'%(c,c,cts[c])  for c in sorted(cts.keys())]
        url='https://github.com/nsat/pypredict'
        foot="\n<br/>calculation done using <a href='%s'>PyPredict</a> and this TLE<br/><pre>"%url+tle+"</pre>"
        return HttpResponse(''.join(clines)+foot)
    olist=Schedule.objects.filter(time=pk)
    dif=olist[len(olist)-1].end()-olist[0].date
    context={'line':pk}
    headhash={'Content-Disposition': 'attachment; filename="observation%i.csv"'%pk}
    from .tools import sat_pos
    if 'alt' in request.GET:
        from astropy.time import Time
        import astropy.coordinates as coor
        from datetime import timedelta
        olist=[o for o in olist]
        context['alt']=1
        for o in olist:
            tm2=o.date+timedelta(0,o.expo/2) #middle of exposure
            pos2=sat_pos(tm2)
            o.position=[pos2.lon.degree,pos2.lat.degree,pos2.height.value/1e3]
            o.altaz=o.target.get_pos().transform_to(coor.AltAz(obstime=Time(tm2),location=pos2))
        if 'raw' in request.GET:
            head='#RA;Dec;start timestamp;target;sat_lon;sat_lat;sat_alt\n'
            mask='%10.5f;%10.5f;%i;%s;%10.5f;%10.5f;%10.5f;\n'
            data="".join([mask%((p.target.pRA,p.target.pdec,p.date.timestamp(),p.target.name)+tuple(p.position)) for p in olist])
            return HttpResponse(head+data,content_type="text/csv",headers=headhash)#"text/plain")

    if 'raw' in request.GET:
        head='#RA;Dec;start timestamp;target\n'
        data="".join(['%10.5f;%10.5f;%i;%s\n'%(p.target.pRA,p.target.pdec,p.date.timestamp(),p.target.name) for p in olist])
        return HttpResponse(head+data,content_type="text/csv",headers=headhash)#"text/plain")
    context['obs']=olist
    context['length']=dif.days*24*60+dif.seconds/60
    return render(request,"catalog/schedule.html",context)

def plot(request,pk,mode='object'):
    from plotly.offline import plot
    import plotly.graph_objects as go

    context={}
    layout = {
        'title': 'Observation plan %i'%int(pk),
        'xaxis_title': 'Right Ascension',
        'yaxis_title': 'Declination',
        'geo': {'showcoastlines': False,'showland':False, 'projection_type':'orthographic','lataxis_showgrid':True,'lonaxis_showgrid':True} ,
        'height': 620, 'width': 960,
    }
    graphs=[]
    #if 'sched' in request.GET:
    olist=Schedule.objects.filter(time=pk)
    context['olen']=olist.count()
    if len(olist)>1:
        x=[p.target.pRA for p in olist]
        y=[p.target.pdec for p in olist]
        import numpy as np
        if mode=='express':
            import plotly.express as px
            from plotly import io
            fig = px.line_geo(lon=x,lat=y)
            fig.update_geos(projection_type="natural earth2", lataxis_showgrid=True, lonaxis_showgrid=True)# orthogonal
            fig.update_layout(height=300, margin={"r":0,"t":0,"l":0,"b":0})
            context['plot_div']=io.show(fig)
        else:
            #labs=['%i [%s]'%(i,p.target.name) for i,p in enumerate(olist)]
            dlabs=np.array([[i+1,p.target.name] for i,p in enumerate(olist)])
            graphs.append(go.Scattergeo(lon=x,lat=y,mode='markers+lines',name='plan %i'%pk,
                                        customdata=dlabs,hovertemplate="%{customdata[0]} [%{customdata[1]}]"))
            #hovertext=labs))#,hovertemplate="%{x:$.4f}")
            from astropy.time import Time
            import astropy.coordinates as coor
            sunpos=coor.get_sun(Time(olist[0].date))
            lunpos=[coor.get_body('moon',Time(olist[i].date)) for i in [0,len(olist)-1]]

            graphs.append(go.Scattergeo(lon=[sunpos.ra.degree],lat=[sunpos.dec.degree],mode='markers',hovertext=['Sun'],name='Sun'))
            graphs.append(go.Scattergeo(lon=[m.ra.degree for m in lunpos],lat=[m.dec.degree for m in lunpos],
                                        mode='markers+lines',hovertext=['start','end'],name='Moon'))
            #r2,d2=mb(sunpos.ra.degree,sunpos.dec.degree)
            context['plot_div']=plot({'data': graphs, 'layout': layout}, output_type='div')
    return render(request,"catalog/plot.html",context)
