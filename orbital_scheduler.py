"""
orbital_scheduler.py - Adapter connecting mockat satellite observations with sch CP-SAT scheduler.

The key difference from ground observatories: the observer (satellite) moves,
so visibility windows must be computed by sampling along the orbit.
"""

import sys
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Tuple

import numpy as np

# Add sch to path
# sys.path.insert(0, '/home/filip/muni/sch')

from kernel.intervals import Intervals
from kernel.reservation import Reservation, CompoundReservation
from kernel.solver import CPScheduler

from mockat.models import Target, Objective, Schedule
from mockat.tools import visible, sat_pos, sun_pos

logger = logging.getLogger(__name__)


class OrbitalVisibilityCalculator:
    """
    Computes visibility windows for targets from a moving satellite.

    Unlike ground observatories with fixed positions, we must sample
    the orbit to find when each target becomes visible/invisible.
    """

    def __init__(self,
                 sun_limit: float = 45.0,
                 moon_limit: float = 20.0,
                 earth_limb: float = -5.0,
                 sample_interval: int = 30,
                 eclipse_margin: float = 0.0):
        """
        Args:
            sun_limit: Minimum angular distance from Sun (degrees)
            moon_limit: Minimum angular distance from Moon (degrees)
            earth_limb: Minimum altitude above Earth limb (degrees, negative = below horizon)
            sample_interval: How often to sample visibility (seconds)
            eclipse_margin: Trim this many seconds off both edges of every
                eclipse window before allowing observations. Guards against
                TLE staleness and eclipse-boundary model error (a mislabeled
                boundary already burned us once) at the cost of
                2*margin/2100 of each pass.
        """
        self.sun_limit = sun_limit
        self.moon_limit = moon_limit
        self.earth_limb = earth_limb
        self.sample_interval = sample_interval
        self.eclipse_margin = eclipse_margin

        # Cache for visibility calculations
        #self._visibility_cache: Dict[datetime, set] = {}
        self._visibility_cache: Dict[str, set] = {}
        self._period_cache: List = []

    def _get_visible_at_time(self, tm: datetime) -> set:
        """Get set of visible target PKs at a specific time.

        Prefers an exact/nearest match against a precomputed cache (as
        loaded from the sgp4-based visibility_loop .npz by the caller),
        since that cache is keyed by full-precision timestamps tied to
        actual satellite pass boundaries rather than to our own sampling
        grid. Falls back to a live sgp4 computation only for genuine
        cache misses.
        """
        cache_key = tm.replace(microsecond=0)
        cache_key = cache_key.replace(second=(cache_key.second // self.sample_interval) * self.sample_interval)
        cache_stamp = cache_key.timestamp()
        # _period_cache has shape (2, N): row 0 is all period-start
        # timestamps, row 1 is all period-end timestamps (same layout
        # show_schedules() relies on). The periods are eclipse windows,
        # which are the *only* times observation is allowed
        if len(self._period_cache):
            # eclipse_margin shrinks each window symmetrically: a time counts
            # as inside only if it is margin seconds past the entry and margin
            # seconds before the exit
            n_in = int(np.sum(self._period_cache[0] + self.eclipse_margin < cache_stamp))
            n_out = int(np.sum(self._period_cache[1] - self.eclipse_margin < cache_stamp))
            if n_in == n_out: return set()  # outside every (trimmed) eclipse window

        nearest_key = self._nearest_cache_key(cache_stamp)
        if nearest_key is not None:
            return self._visibility_cache[nearest_key]
        print("key not found for ",cache_stamp)
        #visible_pks = self._compute_visible_live(cache_key)
        visible_pks=[]
        self._visibility_cache[cache_key.isoformat()[:19]] = visible_pks
        return visible_pks

    def _nearest_cache_key(self, cache_stamp: float, tolerance: Optional[float] = None) -> Optional[str]:
        """Find the precomputed cache entry closest to cache_stamp, within
        tolerance seconds (default: max(sample_interval, 60), matching
        visibility_loop's default sampling step). Returns None if the
        cache is empty or nothing is close enough."""
        if not self._visibility_cache:
            return None
        if tolerance is None:
            tolerance = max(self.sample_interval, 60)
        if getattr(self, '_cache_keys_len', -1) != len(self._visibility_cache):
            keys = sorted(self._visibility_cache.keys())
            self._cache_keys_sorted = keys
            self._cache_ts_sorted = np.array([datetime.fromisoformat(k).timestamp() for k in keys])
            self._cache_keys_len = len(self._visibility_cache)
        ts = self._cache_ts_sorted
        if len(ts) == 0:
            return None
        idx = int(np.searchsorted(ts, cache_stamp))
        candidates = [i for i in (idx - 1, idx) if 0 <= i < len(ts)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(ts[i] - cache_stamp))
        if abs(ts[best] - cache_stamp) <= tolerance:
            return self._cache_keys_sorted[best]
        return None

    def _compute_visible_live(self, tm: datetime) -> set:
        """Live sgp4-based visibility computation for a cache miss outside
        the precomputed range. Requires visibility_loop.our_sat to be set
        by the caller (sched_first.py sets it from the mission TLE)."""
        import visibility_loop as vl
        if vl.our_sat is None:
            logger.warning(
                f"No precomputed visibility for {tm.isoformat()} and "
                "visibility_loop.our_sat is not set -- treating as not visible"
            )
            return set()
        targ_pks = list(Target.objects.values_list('pk', flat=True))
        return set(vl.visible(tm, targ_pks))

    def attach_cache(self,ext_cache):
        self._visibility_cache=ext_cache.allvis
        self._period_cache=np.array([[q.timestamp() for q in p] for p in ext_cache.periods]).T
        
    def compute_windows(self,
                        target: Target,
                        start_time: datetime,
                        end_time: datetime,
                        min_window_duration: float = 60.0) -> Intervals:
        """
        Compute visibility windows for a target over a time range.

        Samples the orbit at regular intervals to detect when the target
        enters/exits visibility constraints (sun, moon, earth limb).

        Args:
            target: Target to compute windows for
            start_time: Start of planning horizon
            end_time: End of planning horizon
            min_window_duration: Minimum window length to keep (seconds)

        Returns:
            Intervals object with visibility windows
        """
        self.windows = []
        step = timedelta(seconds=self.sample_interval)

        current = start_time
        window_start = None
        was_visible = False

        while current <= end_time:
            visible_pks = self._get_visible_at_time(current)
            is_visible = target.pk in visible_pks

            if is_visible and not was_visible:
                # Target just became visible
                window_start = current
            elif not is_visible and was_visible and window_start is not None:
                # Target just became invisible
                window_duration = (current - window_start).total_seconds()
                if window_duration >= min_window_duration:
                    self.windows.append((window_start, current))
                window_start = None

            was_visible = is_visible
            current += step

        # Close final window if still open
        if was_visible and window_start is not None:
            window_duration = (end_time - window_start).total_seconds()
            if window_duration >= min_window_duration:
                self.windows.append((window_start, end_time))

        return Intervals(self.windows)

    def compute_windows_batch(self,
                              targets: List[Target],
                              start_time: datetime,
                              end_time: datetime,
                              min_window_duration: float = 60.0) -> Dict[int, Intervals]:
        """
        Compute visibility windows for multiple targets efficiently.

        More efficient than calling compute_windows repeatedly because
        it shares the visibility cache across all targets.

        Returns:
            Dict mapping target.pk to Intervals
        """
        # Pre-populate cache by sampling the full time range once
        step = timedelta(seconds=self.sample_interval)
        current = start_time
        while current <= end_time:
            self._get_visible_at_time(current)
            current += step

        # Now compute windows for each target using cached data
        result = {}
        for target in targets:
            result[target.pk] = self.compute_windows(
                target, start_time, end_time, min_window_duration
            )

        return result

    def clear_cache(self):
        """Clear the visibility cache."""
        self._visibility_cache.clear()


class MockatRequest:
    """
    Wrapper to provide request-like interface expected by sch scheduler.

    This allows the scheduler to apply time-based priority adjustments
    and airmass-like optimizations (in our case, sun distance optimization).
    """

    # Multiplier applied to windows starting after a target's notes['before']
    # deadline has passed. A soft de-prioritization rather than a hard
    # cutoff: a late window still beats not observing the target at all.
    LATE_PRIORITY_FACTOR = 0.3

    def __init__(self, target: Target, optimization_type: str = 'TIME'):
        self.target = target
        self.optimization_type = optimization_type
        self.request_group_id = target.group

    def calculate_time_based_priority(self, start_time: datetime) -> float:
        """
        Calculate priority multiplier based on observation time.

        Targets with a 'before' deadline in their notes (e.g. fading
        transients) get full priority for windows at or before that
        deadline, and LATE_PRIORITY_FACTOR after it -- so the solver
        prefers an earlier slot but can still use a late one if that's
        all that's available.
        """
        before = (self.target.notes or {}).get('before')
        if not before:
            return 1.0
        try:
            deadline = datetime.fromisoformat(before)
        except (TypeError, ValueError):
            logger.warning(f"Target {self.target.pk} has unparseable 'before' note: {before!r}")
            return 1.0

        compare_start = start_time
        if deadline.tzinfo is not None and compare_start.tzinfo is None:
            compare_start = compare_start.replace(tzinfo=deadline.tzinfo)
        elif deadline.tzinfo is None and compare_start.tzinfo is not None:
            deadline = deadline.replace(tzinfo=compare_start.tzinfo)

        return 1.0 if compare_start <= deadline else self.LATE_PRIORITY_FACTOR

    def get_airmasses_within_kernel_windows(self, resource: str) -> dict:
        """
        Return airmass-equivalent data for optimization.

        For satellites, we could use sun distance or earth limb altitude
        as an analog to airmass optimization.
        """
        return {'times': [], 'airmasses': []}


class OrbitalSchedulerAdapter:
    """
    Main adapter connecting mockat models with sch CP-SAT scheduler.

    Handles:
    - Converting Objectives to Reservations
    - Computing orbital visibility windows
    - Running the scheduler
    - Converting results back to Schedule objects
    """

    RESOURCE_NAME = "QUVIK"  # Single satellite resource

    # Slew model shared with catalog/catools.py's next_aim(): time to slew is
    # a linear function of angular distance, transit = dist/slewspeed + slewsettle.
    DEFAULT_SLEW_SPEED = 1.2   # degrees/second
    DEFAULT_SLEW_SETTLE = 60.0  # fixed settle time, seconds
    DEFAULT_SLEW_ACC = 0.83/60 # in deg/sec^2

    def __init__(self,
                 start_time: datetime,
                 end_time: datetime,
                 slice_size: int = 60,
                 sun_limit: float = 45.0,
                 moon_limit: float = 20.0,
                 earth_limb: float = -5.0,
                 sample_interval: int = 30,
                 slew_speed: float = DEFAULT_SLEW_SPEED,
                 slew_settle: float = DEFAULT_SLEW_SETTLE,
                 slew_penalty_weight: float = 0.0,
                 max_chunk_seconds: float = 600.0,
                 remainder_boost: float = 2.0,
                 free_split: bool = False,
                 cofield_radius: float = 0.25,
                 eclipse_margin: float = 30.0,
                 mip_gap: float = 0.005,
                 scarcity_cap: float = 3.0):
        """
        Args:
            start_time: Planning window start (UTC)
            end_time: Planning window end (UTC)
            slice_size: Scheduling granularity in seconds
            sun_limit: Minimum sun angle (degrees)
            moon_limit: Minimum moon angle (degrees)
            earth_limb: Earth limb avoidance (degrees)
            sample_interval: Visibility sampling interval (seconds)
            slew_speed: Slew rate in degrees/second (set to None or 0 to disable
                slew-time modeling entirely, restoring the old back-to-back behavior)
            slew_settle: Fixed settle/overhead time added to every slew, in seconds
            slew_penalty_weight: Tunable >=0, off by default. Soft preference (on
                top of the hard "a slew must physically fit" constraint that
                slew_speed/slew_settle already enforce) for scheduling pairs of
                targets with smaller angular separation next to each other --
                e.g. within the same eclipse pass. 0 disables it, reproducing
                prior behaviour exactly; raise it to trade off scheduled priority
                against shorter slews between neighbours. See CPScheduler's
                slew_penalty_weight docstring for the exact mechanism.
            max_chunk_seconds: Objectives with delay == 0 don't require their
                total exposure in one contiguous block, so they are split into
                independently schedulable chunks of at most this length (0 or
                None disables splitting). Chunks respect the objective's expo
                as the atomic sub-exposure where possible: repeat exposures are
                grouped into chunks of floor(max_chunk/expo) repeats; a single
                long exposure (repeat == 1) is cut into near-equal pieces.
                Chunks of the same target cost no slew between them, so the
                solver can still place them contiguously when that's best.
            remainder_boost: Priority multiplier for chunks of objectives that
                already have exposure scheduled in an earlier (still current)
                line. A partially observed target's data is not usable until
                its total is reached, so finishing it should outrank starting
                a fresh objective of equal priority. 1.0 disables the boost.
            free_split: When True, delay == 0 objectives may be cut anywhere,
                not only on repeat boundaries: chunks are max_chunk-sized with
                one remainder piece. Choose max_chunk so an integer number of
                chunks plus slews tiles the eclipse pass (with 2100 s passes
                and ~100 s slews, ~620 s x 3 is near-optimal). Only valid if
                sub-exposure boundaries are flexible, i.e. the requirement is
                genuinely just the accumulated total.
            cofield_radius: Targets within this angular radius (degrees) of a
                pointing's primary target are observed simultaneously by the
                same exposure, and every group member's exposure account is
                credited. Default 0.25 = the FoV short half-axis (0.9 x 0.5
                deg), which fits at any spacecraft roll; raise towards 0.45
                only if the roll can be steered to align a pair with the FoV
                long axis. 0 disables grouping.
            eclipse_margin: Seconds trimmed off both edges of every eclipse
                window (see OrbitalVisibilityCalculator). Default 30 s.
            mip_gap: Solver relative optimality tolerance. These models solve
                fast; the old 0.05 could silently give away 5% of the science.
            scarcity_cap: Upper bound on the scarcity priority multiplier
                n_passes_in_horizon / n_windows_of_target, which makes targets
                visible in only a few passes outrank ubiquitous ones (they
                cannot be deferred to another pass). 1.0 disables.
        """
        self.start_time = start_time
        self.end_time = end_time
        self.slice_size = slice_size
        self.slew_speed = slew_speed
        # NOTE: acceleration/braking overhead is NOT folded into slew_settle
        # anymore -- for dist < speed^2/acc (~104 deg with the defaults!) the
        # slew never reaches full speed and the correct transit time is the
        # triangular profile 2*sqrt(dist/acc), much shorter than
        # dist/speed + speed/acc. See _slew_move_seconds().
        self.slew_settle = slew_settle
        self.slew_penalty_weight = slew_penalty_weight
        self.max_chunk = max_chunk_seconds or 0
        self.remainder_boost = remainder_boost
        self.free_split = free_split
        self.cofield_radius = cofield_radius
        self.mip_gap = mip_gap
        self.scarcity_cap = scarcity_cap
        self._n_passes = 1
        self._slew_cache: Dict[Tuple[int, int], float] = {}

        self.visibility_calc = OrbitalVisibilityCalculator(
            sun_limit=sun_limit,
            moon_limit=moon_limit,
            earth_limb=earth_limb,
            sample_interval=sample_interval,
            eclipse_margin=eclipse_margin
        )

        # Storage for mapping back to mockat objects: resID -> list of
        # (objective, target) pairs sharing the pointing, primary first
        self._reservation_map: Dict[int, List[Tuple[Objective, Target]]] = {}
        self.windows_map = {}

    @staticmethod
    def _objective_total_seconds(objective: Objective) -> float:
        """Total exposure an objective asks for, including repeat delays."""
        total = objective.expo * max(objective.repeat, 1)
        if objective.repeat > 1 and objective.delay > 0:
            total += objective.delay * (objective.repeat - 1)
        return total

    @staticmethod
    def _objective_done_seconds(objective: Objective) -> float:
        """Exposure already scheduled for this objective in previous
        generation cycles, from the notes['done_by_line'] bookkeeping that
        save_schedules() maintains (line_id -> seconds). Regenerating a line
        removes its entry (see sched_first.py), so this always reflects the
        lines that are still current."""
        done_by_line = (objective.notes or {}).get('done_by_line')
        if not isinstance(done_by_line, dict):
            return 0.0
        return float(sum(done_by_line.values()))

    def _chunk_durations(self, objective: Objective,
                         remaining: Optional[float] = None) -> List[float]:
        """Split an objective's still-unscheduled exposure into independently
        schedulable chunk durations.

        Only delay == 0 objectives are split: a nonzero delay encodes a
        cadence between repeats (monitoring), which chunking would destroy.
        Repeat exposures are the natural atomic unit, so chunks are whole
        groups of them; a single exposure longer than max_chunk is cut into
        near-equal pieces instead (the exposure time is accumulated across
        orbits anyway, per the observing requirements).

        remaining defaults to the full total; callers pass total - done to
        schedule only the residual of a partially covered objective.
        """
        expo, repeat, delay = objective.expo, objective.repeat, objective.delay
        total = self._objective_total_seconds(objective)
        if remaining is None:
            remaining = total
        remaining = min(remaining, total)

        if remaining <= 0:
            return []
        if not self.max_chunk or delay > 0 or remaining <= self.max_chunk:
            return [remaining]

        if self.free_split:
            # cut anywhere: full-size chunks plus one remainder piece, so
            # max_chunk can be tuned to tile the eclipse pass exactly
            full = int(remaining // self.max_chunk)
            rest = remaining - full * self.max_chunk
            chunks = [self.max_chunk] * full
            if rest > 1e-6:
                if chunks and rest < 120:
                    chunks[-1] += rest  # avoid unschedulably small fragments
                else:
                    chunks.append(rest)
            return chunks

        # remaining is a whole number of repeats unless a repeat==1 exposure
        # was time-split; keep chunking on repeat boundaries when it is
        if repeat > 1 and abs(remaining / expo - round(remaining / expo)) < 1e-6:
            n_rep = int(round(remaining / expo))
            per_chunk = max(1, int(self.max_chunk // expo))
            full, rest = divmod(n_rep, per_chunk)
            return [expo * per_chunk] * full + ([expo * rest] if rest else [])

        n = math.ceil(remaining / self.max_chunk)
        base = remaining / n
        # keep chunk boundaries on whole seconds; last chunk takes the remainder
        chunks = [float(int(base))] * (n - 1)
        return chunks + [remaining - sum(chunks)]

    def _cofield_groups(self, objectives: List[Objective]) -> List[List[Objective]]:
        """Greedy star clusters of objectives observable with one pointing.

        The member with the largest remaining exposure becomes the primary
        (the pointing target); any other objective whose target lies within
        cofield_radius of the primary joins the group and is exposed
        simultaneously. Objectives sharing the same target always merge.
        Star-shaped clustering around the primary keeps every member within
        the FoV of a pointing centred on it, which chained pairwise linking
        would not guarantee.
        """
        if not self.cofield_radius:
            return [[obj] for obj in objectives]
        remaining = {
            obj.pk: max(self._objective_total_seconds(obj)
                        - self._objective_done_seconds(obj), 0.0)
            for obj in objectives}
        order = sorted(objectives, key=lambda o: -remaining[o.pk])
        used = set()
        groups = []
        for obj in order:
            if obj.pk in used:
                continue
            used.add(obj.pk)
            group = [obj]
            for other in order:
                if other.pk in used:
                    continue
                dist = obj.target.get_dist(other.target)
                if dist is not None and dist <= self.cofield_radius:
                    used.add(other.pk)
                    group.append(other)
            if len(group) > 1:
                logger.info(
                    f"Co-field pointing on {obj.target.name}: also exposes "
                    + ", ".join(m.target.name.strip() for m in group[1:]))
            groups.append(group)
        return groups

    def _group_to_reservations(self,
                               members: List[Objective],
                               visibility_windows: Intervals) -> List[CompoundReservation]:
        """
        Convert one pointing group (primary objective first, see
        _cofield_groups) to sch CompoundReservations, one per chunk (see
        _chunk_durations). Chunk durations cover the largest remaining
        exposure in the group -- every member accumulates simultaneously, so
        smaller members simply finish early. Returns [] if no valid windows
        exist or nothing remains to observe.
        """
        if visibility_windows.is_empty():
            return []

        primary = members[0]
        target = primary.target

        totals = [self._objective_total_seconds(m) for m in members]
        if max(totals) <= 0:
            # A zero/negative duration (typically objective.expo == 0) makes
            # kernel.solver.get_slices() generate an empty possible-start slice,
            # which later crashes with IndexError in PossibleStart.__init__.
            logger.warning(
                f"Skipping objective {primary.pk} ({target.name}): "
                f"non-positive duration (expo={primary.expo}, "
                f"repeat={primary.repeat}, delay={primary.delay})"
            )
            return []

        dones = [self._objective_done_seconds(m) for m in members]
        remaining = max(t - d for t, d in zip(totals, dones))
        chunks = self._chunk_durations(primary, remaining=remaining)
        if not chunks:
            # fully covered by still-current lines; scheduled_line should
            # normally exclude it from the queryset before we get here
            logger.info(f"Objective {primary.pk} ({target.name}) already "
                        f"complete ({dones[0]:.0f}/{totals[0]:.0f}s scheduled)")
            return []
        remainder_of_started = any(d > 0 for d in dones)
        if remainder_of_started:
            logger.info(f"Objective {primary.pk} ({target.name}): scheduling "
                        f"remainder {remaining:.0f}s")

        # Group priority: every member gets photons from the same exposure,
        # so a multi-target pointing is worth the sum of its members
        priority = 0.0
        for member in members:
            member_priority = member.target.prior if member.target.prior > 0 else 1.0
            # Boost priority for certain groups if desired
            # e.g., transient sources might get higher priority
            if 'merg' in member.target.group or 'FRB' in member.target.group:
                member_priority *= 2.0
            priority += member_priority

        # Finishing a partially observed objective beats starting a new one
        # of equal priority -- its data is unusable until the total is reached
        if remainder_of_started:
            priority *= self.remainder_boost

        # Scarcity: fewer visibility windows than passes in the horizon means
        # fewer chances to defer -- boost proportionally, capped
        n_windows = len(visibility_windows.toTupleList())
        if n_windows and self.scarcity_cap > 1.0:
            scarcity = min(self.scarcity_cap, self._n_passes / n_windows)
            if scarcity > 1.0:
                priority *= scarcity

        member_pairs = [(m, m.target) for m in members]
        compound_reservations = []
        for duration in chunks:
            reservation = Reservation(
                priority=priority,
                duration=duration,
                possible_windows_dict={self.RESOURCE_NAME: visibility_windows},
                request=MockatRequest(target),
                request_group_id=target.group
            )
            # Store mapping for later extraction
            self._reservation_map[reservation.resID] = member_pairs
            # Wrap in CompoundReservation (required by scheduler)
            compound_reservations.append(
                CompoundReservation([reservation], cr_type='single'))
        return compound_reservations

    def build_compound_reservations(self,
                                     objectives: Optional[List[Objective]] = None
                                     ) -> List[CompoundReservation]:
        """
        Build CompoundReservation list from Objectives.

        Args:
            objectives: List of objectives, or None to use all from DB

        Returns:
            List of CompoundReservations ready for scheduler
        """
        if objectives is None:
            objectives = list(Objective.objects.select_related('target').all())

        if not objectives:
            logger.warning("No objectives to schedule")
            return []

        # Get unique targets
        targets = list({obj.target for obj in objectives})
        logger.info(f"Computing visibility for {len(targets)} targets...")
        print(f"Computing visibility for {len(targets)} targets...")
        
        # Batch compute visibility windows
        min_duration = min(obj.expo for obj in objectives)
        self.windows_map = self.visibility_calc.compute_windows_batch(
            targets,
            self.start_time,
            self.end_time,
            min_window_duration=min_duration
        )

        # Passes in the horizon, for the scarcity priority factor: a target
        # visible in 1 of N passes cannot be deferred, so its chunks should
        # outrank targets visible in every pass
        opc = self.visibility_calc._period_cache
        if len(opc):
            t0s, t1s = self.start_time.timestamp(), self.end_time.timestamp()
            self._n_passes = max(1, int(np.sum((opc[0] < t1s) & (opc[1] > t0s))))
        else:
            self._n_passes = max(
                [len(w.toTupleList()) for w in self.windows_map.values()] or [1])

        # Convert pointing groups to reservations (one per chunk); the
        # group's visibility is the primary target's -- members sit within
        # cofield_radius, far below the visibility computation's resolution
        groups = self._cofield_groups(objectives)
        compound_reservations = []
        skipped = 0

        for group in groups:
            windows = self.windows_map.get(group[0].target.pk, Intervals([]))
            crs = self._group_to_reservations(group, windows)

            if crs:
                compound_reservations.extend(crs)
            else:
                skipped += 1
                logger.debug(f"Skipped {group[0].target.name}: no visibility windows")

        logger.info(f"Created {len(compound_reservations)} chunk reservations "
                    f"from {len(groups) - skipped} pointing groups "
                    f"({len(objectives)} objectives), skipped {skipped}")
        return compound_reservations

    def _slew_move_seconds(self, dist_deg: float) -> float:
        """Transit time for a slew of dist_deg, trapezoidal velocity profile.

        Below speed^2/acc (~104 deg with the defaults) the slew never reaches
        full speed: triangular profile, 2*sqrt(dist/acc). Beyond that, cruise
        at full speed plus the acc+brake overhead speed/acc.
        """
        v, a = self.slew_speed, self.DEFAULT_SLEW_ACC
        if dist_deg >= v * v / a:
            return dist_deg / v + v / a
        return 2.0 * math.sqrt(dist_deg / a)

    def _slew_time_seconds(self, request_a: Optional['MockatRequest'],
                            request_b: Optional['MockatRequest']) -> float:
        """Slew-time model: fixed settle time + trapezoidal transit time.

        Passed to CPScheduler as slew_time_fn -- see its docstring. Chunks of
        the same target need no repointing at all (0.0), so the solver may
        place them contiguously. Targets with no known position (get_dist
        returning None) don't constrain each other; that data problem is
        caught earlier (visibility computation needs coordinates too).

        Results are cached per target-pair: this is called O(candidates^2)
        times from _add_slew_constraints and astropy separations are far too
        slow for that.
        """
        if request_a is None or request_b is None:
            return 0.0
        pk_a, pk_b = request_a.target.pk, request_b.target.pk
        if pk_a == pk_b:
            return 0.0
        key = (pk_a, pk_b) if pk_a < pk_b else (pk_b, pk_a)
        cached = self._slew_cache.get(key)
        if cached is not None:
            return cached
        dist_deg = request_a.target.get_dist(request_b.target)
        result = 0.0 if dist_deg is None else \
            self.slew_settle + self._slew_move_seconds(dist_deg)
        self._slew_cache[key] = result
        return result

    def run(self,
            objectives: Optional[List[Objective]] = None,
            timelimit: int = 180) -> List[Schedule]:
        """
        Run the scheduler and return mockat Schedule objects.

        Args:
            objectives: Objectives to schedule (None = all from DB)
            timelimit: Solver time limit in seconds

        Returns:
            List of Schedule objects (not yet saved to DB)
        """
        # Build reservations
        compound_reservations = self.build_compound_reservations(objectives)

        if not compound_reservations:
            return []

        # Global windows = full planning period
        global_windows = Intervals([(self.start_time, self.end_time)])
        globally_possible = {self.RESOURCE_NAME: global_windows}

        # Create scheduler
        logger.info(f"Running CP-SAT scheduler (timelimit={timelimit}s)...")
        if self.slew_speed:
            slew_kwargs = dict(
                slew_time_fn=self._slew_time_seconds,
                max_slew_seconds=self.slew_settle + self._slew_move_seconds(180.0),
                slew_penalty_weight=self.slew_penalty_weight,
            )
        else:
            slew_kwargs = {}
        self.scheduler = CPScheduler(
            compound_reservation_list=compound_reservations,
            globally_possible_windows_dict=globally_possible,
            slice_size_seconds=self.slice_size,
            timelimit=timelimit,
            mip_gap=self.mip_gap,
            **slew_kwargs
        )

        # Run scheduler
        schedule_dict = self.scheduler.schedule_all()

        # Extract results
        return self._extract_schedules(self.scheduler)

    def _extract_schedules(self, scheduler: CPScheduler) -> List[Schedule]:
        """Convert scheduler results to mockat Schedule objects."""
        schedules = []

        for reservation in scheduler.reservation_list:
            if not reservation.scheduled:
                continue

            # Look up original mockat objects
            if reservation.resID not in self._reservation_map:
                logger.warning(f"Unknown reservation ID: {reservation.resID}")
                continue

            members = self._reservation_map[reservation.resID]
            objective, target = members[0]

            # Create Schedule object
            # kernel/solver.py works with naive datetimes throughout (see
            # get_slices()), but self.start_time/end_time are documented as
            # UTC and Django's Schedule.date is a DateTimeField with
            # USE_TZ=True -- so attach the UTC label here, at the boundary,
            # rather than threading tz-awareness through the naive scheduler.
            scheduled_start = reservation.scheduled_start
            if scheduled_start.tzinfo is None:
                scheduled_start = scheduled_start.replace(tzinfo=timezone.utc)

            sched = Schedule(
                time=0,  # Will be set when saving
                expo=reservation.scheduled_quantum,
                date=scheduled_start.replace(tzinfo=timezone.utc),
                target=target
            )
            # Not a Django field -- Schedule has no FK to Objective -- just
            # a way to carry the link through to save_schedules(), which
            # credits every co-field member and stamps scheduled_line on
            # completion. _source_objective (the primary) kept for callers
            # that predate co-field grouping.
            sched._source_objective = objective
            sched._source_objectives = [m for m, _ in members]
            # True exposure of this chunk, for the remainder bookkeeping:
            # sched.expo holds the slice-coverage quantum, which overstates
            # the real exposure by up to one slice and would inflate 'done'.
            sched._chunk_duration = reservation.duration
            schedules.append(sched)

        # Sort by observation time
        schedules.sort(key=lambda s: s.date)

        scheduled_count = len(schedules)
        total_count = len(scheduler.reservation_list)
        logger.info(f"Scheduled {scheduled_count}/{total_count} observations")

        return schedules

    def save_schedules(self,
                       schedules: List[Schedule],
                       line_id: Optional[int] = None) -> int:
        """
        Save schedules to database.

        Args:
            schedules: List of Schedule objects to save
            line_id: Schedule line ID (auto-generated if None)

        Returns:
            The line_id used
        """
        if not schedules:
            return 0

        if line_id is None:
            from django.db.models import Max
            max_line = Schedule.objects.aggregate(Max('time'))['time__max']
            line_id = (max_line or 0) + 1

        for sched in schedules:
            sched.time = line_id
            sched.save()

        # Remainder bookkeeping: accumulate the true chunk durations saved in
        # this line into notes['done_by_line'][line_id], and stamp
        # scheduled_line only once the objective's total is fully covered --
        # a partially covered objective keeps scheduled_line NULL so the next
        # generation cycle schedules just its remainder (see
        # _objective_to_reservations). sched_first.py drops a line's
        # done_by_line entries before regenerating it, so re-running a line
        # never double-counts.
        added_per_objective: Dict[int, float] = {}
        objectives: Dict[int, Objective] = {}
        for sched in schedules:
            members = getattr(sched, '_source_objectives', None)
            if members is None:
                single = getattr(sched, '_source_objective', None)
                members = [single] if single is not None else []
            chunk = getattr(sched, '_chunk_duration', sched.expo)
            # every co-field member is exposed for the whole chunk
            for objective in members:
                added_per_objective[objective.pk] = \
                    added_per_objective.get(objective.pk, 0.0) + chunk
                objectives[objective.pk] = objective

        for pk, added in added_per_objective.items():
            objective = objectives[pk]
            notes = objective.notes if isinstance(objective.notes, dict) else {}
            done_by_line = notes.setdefault('done_by_line', {})
            done_by_line[str(line_id)] = done_by_line.get(str(line_id), 0.0) + added
            done = float(sum(done_by_line.values()))
            total = self._objective_total_seconds(objective)
            objective.notes = notes
            if done >= total - 1e-3:
                objective.scheduled_line = line_id
                logger.info(f"Objective {pk} ({objective.target.name}) complete: "
                            f"{done:.0f}/{total:.0f}s")
            else:
                logger.info(f"Objective {pk} ({objective.target.name}) partial: "
                            f"{done:.0f}/{total:.0f}s, {total - done:.0f}s remaining "
                            f"for a later line")
            objective.save(update_fields=['notes', 'scheduled_line'])

        logger.info(f"Saved {len(schedules)} observations to schedule line {line_id}")
        return line_id

    def show_schedules(self,
                      schedules: Optional[List[Schedule]] = None,
                      line_id: Optional[int] = None) -> int:
        if schedules==None: schedules=Schedule.objects.filter(time=line_id)
        oldper=[0,0]
        oldsch=None
        for sch in schedules:
            cnper=np.sum(self.visibility_calc._period_cache<sch.date.timestamp(),1)
            if oldper[0]<cnper[0]:
                oldper=cnper
                enter=datetime.fromtimestamp(self.visibility_calc._period_cache[0][cnper[0]-1],sch.date.tzinfo).strftime("%H:%M:%S") #time entering eclipse
                print('------------ eclipse ',cnper[0],f' /{enter}/')
            print(sch.date,f'/{sch.expo//60} min/ ',sch.target,f' [{sch.target.pRA:.4f}, {sch.target.pdec:.4f}]')
            if (oldsch!=None):
                dis=sch.target.get_dist(oldsch.target)
                # total_seconds, not .seconds: a negative gap (overlap!) must
                # show as negative, not wrap to ~1439 min
                print('  slew %.1f deg'%(dis))
                #print('  slew %.1f deg (%.1f min)'%(dis,(sch.date-oldsch.end()).total_seconds()/60))
            oldsch=sch
        return len(schedules)

global adapter
adapter=None

def run_orbital_scheduler(
    start_time: datetime,
    end_time: datetime,
    objectives: Optional[List[Objective]] = None,
    slice_size: int = 60,
    sun_limit: float = 45.0,
    moon_limit: float = 20.0,
    earth_limb: float = -5.0,
    timelimit: int = 180,
    save: bool = False
) -> List[Schedule]:
    """
    Convenience function to run the orbital scheduler.

    Example:
        from datetime import datetime, timezone, timedelta
        from orbital_scheduler import run_orbital_scheduler

        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=24)

        schedules = run_orbital_scheduler(start, end, timelimit=120, save=True)
        for s in schedules:
            print(f"{s.date} - {s.target.name} ({s.expo}s)")
    """
    global adapter
    adapter = OrbitalSchedulerAdapter(
        start_time=start_time,
        end_time=end_time,
        slice_size=slice_size,
        sun_limit=sun_limit,
        moon_limit=moon_limit,
        earth_limb=earth_limb
    )

    schedules = adapter.run(objectives=objectives, timelimit=timelimit)

    if save and schedules:
        adapter.save_schedules(schedules)

    return schedules


if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)

    import django
    django.setup()

    start = datetime.now(timezone.utc)
    end = start + timedelta(hours=24)

    print(f"Scheduling observations from {start} to {end}")

    schedules = run_orbital_scheduler(
        start_time=start,
        end_time=end,
        timelimit=60,
        save=False  # Set True to save to DB
    )

    print(f"\nGenerated {len(schedules)} observations:")
    for s in schedules:
        print(f"  {s.date.strftime('%H:%M:%S')} - {s.target.name} ({s.expo:.0f}s)")
