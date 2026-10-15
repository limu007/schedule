from django.db import models
from django.db.models import F, FloatField, Sum
from django.db.models.functions import Coalesce


def emptyjson():
    return {"type": "none"}


class Target(models.Model):
    """A catalogue source, with its photometry and neighbourhood."""

    name = models.CharField(max_length=100)
    pRA = models.FloatField('RAsc', blank=True, null=True, default = 0)
    pdec = models.FloatField('Decl', blank=True, null=True, default = 0)
    mag1 = models.FloatField('magNUV1', blank=True, null=True, default = 999)
    mag2 = models.FloatField('magNUV2', blank=True, null=True, default = 999)
    mag3 = models.FloatField('magFUV', blank=True, null=True, default = 999)
    mag0 = models.FloatField('magB', blank=True, null=True, default = 999)
    group = models.CharField(max_length=20)
    prior = models.FloatField('priority', blank=True, default=0)
    notes = models.JSONField('notes', blank=True, default=emptyjson)
    # Comma-separated Target PKs. A ManyToManyField('self', symmetrical=False)
    # would be the cleaner model (referential integrity, no length ceiling),
    # but that needs a data migration -- see the note in the summary.
    infield = models.CharField('targets close', max_length=200, blank=True)
    nearby = models.CharField('targets near', max_length=400, blank=True)

    class Meta:
        ordering = ["name"]
        get_latest_by = "name"

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @staticmethod
    def _parse_ids(field):
        """PK list from a comma-separated string, skipping blanks and junk."""
        ids = []
        for part in (field or '').split(','):
            part = part.strip()
            if not part:
                continue
            try:
                ids.append(int(part))
            except ValueError:
                continue
        return ids

    def get_coor(self, style='hmsdms'):
        """Coordinates as a sexagesimal string."""
        pos = self.get_pos()
        if pos is None:
            return ''
        return pos.to_string(style)

    def get_pos(self):
        """Position in astropy format (None if not yet set)."""
        if self.pRA is None or self.pdec is None:
            return None
        from astropy import coordinates as coord, units as u
        if not hasattr(self,'pos') or self.pos==None:
            self.pos = coord.ICRS(ra=self.pRA * u.degree, dec=self.pdec * u.degree)
        return self.pos

    def set_pos(self, pos):
        self.pRA = pos.ra.degree
        self.pdec = pos.dec.degree
        self.save()

    def get_near(self, dist=0):
        """List of nearby targets.

        dist=0: only in FoV
        dist>0: also around (next pointing)
        """
        ids = self._parse_ids(self.infield)
        if dist > 0:
            ids += self._parse_ids(self.nearby)
        if not ids:
            return []
        return list(Target.objects.filter(pk__in=set(ids)))

    def get_dist(self,targ):
        if type(targ)==Target:
            tacoor=targ.get_pos()
        else:
            tacoor=targ #direct source
        if tacoor==None: return None
        return self.get_pos().separation(tacoor).degree

    def get_dist2(self,moon,td=None):
        import numpy as np
        if moon==None:
            from astropy.time import Time
            moon=coor.get_body('moon',Time(td))
        coos=lambda a:np.array([a.x.value,a.y.value,a.z.value])
        norm=lambda a:a/np.sqrt((a**2).sum())
        return np.rad2deg(np.arccos(norm(coos(self.get_pos().cartesian))@norm(coos(moon.cartesian))))

    def all_plans(self, unit='min'):
        """Total planned duration over all objectives."""
        total = self.objective_set.aggregate(
            total=Coalesce(
                Sum(F('expo') * F('repeat'), output_field=FloatField()),
                0.0,
                output_field=FloatField(),
            )
        )['total']
        if not total:
            return 0
        unit = (unit or 's').lower()
        if unit.startswith('min') or unit == 'm':
            total /= 60
        elif unit.startswith('h'):
            total /= 3600
        return total

    def expos(self, line=0, tol=1e-6):
        """Total exposure time already scheduled.

        line>0: only the block starting at that time (within tol).
        """
        sel = self.schedule_set
        if line > 0:
            sel = sel.filter(time__gte=line - tol, time__lte=line + tol)
        return sel.aggregate(
            total=Coalesce(Sum('expo'), 0.0, output_field=FloatField())
        )['total'] or 0


class Schedule(models.Model):
    """One planned exposure of one target."""

    time = models.FloatField('line_id', blank=True, null=True, default = 0)
    expo = models.FloatField('exposure',default = 0)
    date = models.DateTimeField('planned')
    target = models.ForeignKey(
        Target, blank=True, null=True, on_delete=models.CASCADE
    )

    class Meta:
        ordering = ["date"]
        get_latest_by = "date"

    def __repr__(self):
        return self.date.strftime("%y-%m-%d %H:%M")

    def __str__(self):
        return "%s: %s" % (self.date.strftime("%y-%m-%d %H:%M"),
                           self.target or '-')

    def end(self):
        from datetime import timedelta
        return self.date + timedelta(0, self.expo)

    def orient(self, horiz=True):
        """Set pointing and calculate relevant values (Alt/Az, sun distance)."""
        from datetime import timedelta
        import numpy as np
        from tools import sat_pos, sun_pos

        tm = self.date + timedelta(0, self.expo / 2)
        pos = sat_pos(tm)

        # Ecliptic coordinates of the target; the Sun sits at latitude 0,
        # so cos(sep) = cos(b) * cos(l - l_sun).
        l,b = self.target.notes.get('eclipt_bl', [0, 90])
        cosdis = np.cos(np.deg2rad(b)) * np.cos(np.deg2rad(l - sun_pos(tm)))
        self.sundis = np.rad2deg(np.arccos(np.clip(cosdis, -1.0, 1.0)))

        if not horiz:
            return self.sundis

        from astropy import coordinates as coor
        from astropy.time import Time

        loc = coor.AltAz(obstime=Time(tm), location=pos)
        self.altaz = self.target.get_pos().transform_to(loc)
        return self.altaz


class Objective(models.Model):
    """An observation requirement for a target."""

    expo = models.FloatField('exposure')
    delay = models.FloatField('delay', default=0)
    target = models.ForeignKey(
        Target, blank=True, null=True, on_delete=models.CASCADE
    )
    repeat = models.IntegerField('repeat', default=1)
    notes = models.JSONField('notes', blank=True, default=emptyjson)
    scheduled_line = models.IntegerField(
        'scheduled generation cycle', null=True, blank=True, default=None,
        help_text="sched_line of the Schedule this objective was placed into, "
                   "or NULL if not yet scheduled. Cleared when that line is regenerated."
    )

    def __repr__(self):
        return "%s x%d" % (self.target or '-', self.repeat)

    def __str__(self):
        return self.__repr__()

    def totime(self):
        """Total duration."""
        return self.expo * self.repeat
    
    def when(self):
        '''finds scheduled times
        filter is not optimal if multiple objectives exist for a given target
        (probably some notes entry should be added in such case) 
        '''
        if self.scheduled_line==None: return []
        return [s for s in Schedule.objects.filter(time=self.scheduled_line).order_by('date') if s.target.pk==self.target.pk]
        #Schedule.objects.filter(time=self.scheduled_line,target=self.target)
    
    def filled(self):
        """Is required time scheduled completely?"""
        return sum([s.expo for s in self.when()])>=self.totime()
