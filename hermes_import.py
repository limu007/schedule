
import requests
#fetching data from HERMES using api https://raw.githubusercontent.com/LCOGT/hermes/main/hermes/.

BASE_URL = "https://hermes.lco.global/api/v0/query/"
MESSAGE_URL = "https://hermes.lco.global/api/v0/query/message/{uuid}/"


def fetch_circular_list(start, end, topic="gcn.circulars", limit=200, session=None):
    """Page through /api/v0/query/ and return all message summaries in [start, end)."""
    session = session or requests.Session()
    params = {"topic": topic, "start": start, "end": end, "limit": limit}
    results = []
    page = None
    while True:
        q = dict(params)
        if page:
            q["page"] = page
        resp = session.get(BASE_URL, params=q, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data["messages"])
        page = data.get("next")
        if not page:
            break
    return results


def fetch_full_text(uuid, session=None):
    """Fetch the full circular (subject, body, circular id, ...) by its uuid."""
    session = session or requests.Session()
    resp = session.get(MESSAGE_URL.format(uuid=uuid), timeout=30)
    try:
        #resp.raise_for_status()
        return resp.json()
    except:
        return []


#-----------------------------------------------------------
def get_wxt(start="2026-05-05",end="2026-05-10"):
    lst3=fetch_circular_list(start,end,"gcn.notices.einstein_probe.wxt.alert") #"2026-08-02","2026-08-24"
    uids=[a['annotations']["con_text_uuid"] for a in lst3]
    sess=requests.Session()
    ftep=[fetch_full_text(u,sess)['message'] for u in uids]
    print([f['image_snr'] for f in ftep])
    return ftep

def get_transients(start="2026-05-05",end="2026-05-10"):
    lst2=fetch_circular_list(start,end,'gcn.circulars.x-ray_transient')
    tats=[a['annotations'] for a in lst2]
    trans={}
    for a in tats:
        if a['title'].find(':')<1: continue
        try:
            b,c=a['title'].split(':')[:2]
        except:
            print(a['title'])
            continue
        b=b.replace(' ','')
        if b.find('/'): b=b.split('/')[0]
        c=c.strip()
        uid=a["con_text_uuid"]
        if b in trans:
            trans[b].append([c,uid])
        else:
            trans[b]=[[c,uid]]
    import numpy as np
    tkeys=np.sort(list(trans.keys()))
    return [trans[y] for y in tkeys]


def make_target(circ,prefix='EP',band='xray'):
    idame=prefix+'_'+str(circ['id'][0])
    tsel=Target.objects.filter(name=idame)
    if tsel.count()==0:
        tg=Target(name=idame,pRA=circ['ra'],pdec=circ['dec'],notes={band:circ['image_snr']})
    else:
        tg=tsel[0]
        tg.pRA=circ['ra']
        tg.pdec=circ['dec']
    tg.notes['ra_dec_error']=circ['ra_dec_error']
    tg.notes['trigger']=circ['trigger_time'][:19]#datetime.fromisoformat(circ['trigger_time'][:19])
    tg.save()
    return tg.pk

