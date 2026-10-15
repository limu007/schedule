"""
kernel/solver.py - Simplified CP-SAT Solver
Merges the functionality of fullscheduler_cp_model.py and slicedipscheduler_v2.py
Removes all the dead code and experimental methods.
"""

import logging
import math
import numpy as np
from datetime import timedelta, datetime
from collections import defaultdict

from ortools.sat.python import cp_model

from kernel.scheduler_base import BaseScheduler
from kernel.reservation import OptimizationType

logger = logging.getLogger(__name__)


class PossibleStart:
    """Represents a possible start time for a reservation."""

    def __init__(self, resource, slice_starts, internal_start, airmass_coeff=1.0):
        self.resource = resource
        self.first_slice_start = slice_starts[0]
        self.all_slice_starts = slice_starts
        self.internal_start = internal_start
        self.airmass_coefficient = airmass_coeff

    def __lt__(self, other):
        return self.first_slice_start < other.first_slice_start


class CPScheduler(BaseScheduler):
    """
    Constraint Programming scheduler using OR-Tools CP-SAT solver.
    Handles time-sliced scheduling across multiple resources.
    """

    def __init__(self, compound_reservation_list, globally_possible_windows_dict,
                 slice_size_seconds, timelimit=180, mip_gap=0.05,
                 celestial_data=None, slice_centers=None,
                 moon_min_distance=10.0, moon_penalty_range=30.0,
                 telescope_preference=None,
                 slew_time_fn=None, max_slew_seconds=None,
                 slew_penalty_weight=0.0):
        """
        Initialize the CP scheduler.

        Args:
            compound_reservation_list: List of compound reservations to schedule
            globally_possible_windows_dict: Global visibility windows per resource
            slice_size_seconds: Time slice size in seconds
            timelimit: Maximum solver time in seconds
            mip_gap: Relative optimality gap
            celestial_data: Pre-calculated sun and moon positions
            slice_centers: Time points for celestial calculations
            moon_min_distance: Minimum distance from moon (degrees)
            moon_penalty_range: Distance where moon penalties apply (degrees)
            telescope_preference: Dict mapping telescope names to preference multipliers
            slew_time_fn: Optional callable(request_a, request_b) -> seconds, giving the
                minimum time needed between finishing request_a's exposure and starting
                request_b's, on the same resource (e.g. a linear function of the angular
                distance between the two targets). When None, slewing is not modeled and
                back-to-back scheduling is allowed as before.
            max_slew_seconds: Upper bound on any value slew_time_fn can return. Used only
                to prune the pairwise search in _add_slew_constraints(); if omitted, no
                pruning is done (still correct, just potentially slower to build).
            slew_penalty_weight: Tunable >=0 controlling how strongly the objective
                prefers small angular separation between candidates that could end up
                scheduled back-to-back (same resource, gap below max_slew_seconds --
                the same pairs _add_slew_constraints() already considers, since a third
                candidate scheduled in between would need a larger gap than that, per
                its docstring). 0 (default) disables this soft preference entirely,
                reproducing prior behaviour. The penalty magnitude for a given pair is
                slew_time_fn(request_a, request_b) itself (seconds, monotonic in
                angular separation for the linear slew model this is built for) --
                a fixed settle-time offset baked into that value is a harmless constant
                across all candidate pairs and doesn't change which pairing the
                objective prefers. This is a soft preference on top of the hard
                feasibility constraint slew_time_fn already enforces, not a substitute
                for it.
        """
        super().__init__(
            compound_reservation_list,
            globally_possible_windows_dict,
            []  # No contractual obligations
        )

        self.slice_size_seconds = slice_size_seconds
        self.timelimit = timelimit
        self.mip_gap = mip_gap

        # Moon penalty parameters
        self.celestial_data = celestial_data
        self.slice_centers = slice_centers
        self.moon_min_distance = moon_min_distance
        self.moon_penalty_range = moon_penalty_range

        # Telescope preference multipliers
        self.telescope_preference = telescope_preference if telescope_preference is not None else {}
        if self.telescope_preference:
            logger.info(f"Telescope preferences: {self.telescope_preference}")

        # Sequence-dependent slew time between consecutive observations
        self.slew_time_fn = slew_time_fn
        self.max_slew_seconds = max_slew_seconds if max_slew_seconds is not None else float('inf')
        self.slew_penalty_weight = slew_penalty_weight
        self._slew_penalty_terms = []  # [(adjacency BoolVar, penalty magnitude), ...]

        # Data structures for the solver
        self.Yik = []  # Maps idx -> [resID, window_idx, priority, resource]
        self.Yik_times = []  # Parallel to Yik: (start_dt, end_dt, request) per entry
        self.aikt = {}  # Maps slice -> list of Yik indices
        self.hashes = set()  # Track slice hashes

        # Initialize time slicing for each resource
        self.time_slicing_dict = {}
        for r in self.resource_list:
            self.time_slicing_dict[r] = [0, slice_size_seconds]

    def schedule_all(self):
        """Run the scheduling algorithm."""
        if not self.reservation_list:
            logger.warning("No reservations to schedule")
            return self.schedule_dict

        logger.info(f"Starting scheduler with {len(self.reservation_list)} reservations")

        # Build data structures
        self._build_data_structures()

        # DEBUG: Check if we have valid scheduling options
        logger.info(f"Created {len(self.Yik)} scheduling options")
        logger.info(f"Active time slices: {len(self.aikt)}")

        if len(self.Yik) == 0:
            logger.error("No valid scheduling options created!")
            return self.schedule_dict

        # Create and solve the CP model
        model = cp_model.CpModel()

        # Create decision variables
        is_scheduled = self._create_variables(model)

        # Add constraints
        self._add_constraints(model, is_scheduled)

        # Set objective
        self._set_objective(model, is_scheduled)

        # Solve
        solver = cp_model.CpSolver()
        if self.timelimit > 0:
            solver.parameters.max_time_in_seconds = self.timelimit
        solver.parameters.log_search_progress = False # True for debugging

        logger.info("Starting solver...")
        status = solver.Solve(model)
        # NOTE: solver.StatusName() is broken in ortools==9.15.6755 (it calls
        # self._checked_response.status() but .status is a property there, not
        # a method), so use the enum already returned by Solve() instead.
        status_name = status.name
        logger.info(f"Solver finished with status: {status_name}")

        # Process results
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            scheduled_count = sum(solver.BooleanValue(is_scheduled[idx]) for idx in range(len(self.Yik)))
            logger.info(f"Solver scheduled {scheduled_count} reservations")
            self._process_solution(solver, is_scheduled)
        else:
            logger.warning(f"No solution found. Status: {status_name}")

        return self.schedule_dict

    def _build_data_structures(self):
        """Build the data structures needed for the solver."""
        debug_count = 0  # Only log first few applications
        for r in self.reservation_list:
            r.possible_starts = []
            r.Yik_entries = []

            # Calculate all possible starts for all resources
            for resource in sorted(r.free_windows_dict.keys()):
                r.possible_starts.extend(self.get_slices(
                    r.free_windows_dict[resource],
                    resource,
                    r.duration,
                    r.request
                ))

            # Sort by time
            r.possible_starts.sort()

            # Calculate priorities and build Yik entries
            w_idx = 0
            for ps in r.possible_starts:
                Yik_idx = len(self.Yik)
                r.Yik_entries.append(Yik_idx)

                # Set warm start (if available)
                scheduled = 0
                if (r.previous_solution_reservation and
                    r.previous_solution_reservation.scheduled_start == ps.internal_start and
                    r.previous_solution_reservation.scheduled_resource == ps.resource):
                    scheduled = 1

                # FIXED: Use original priority calculation
                if r.request and r.request.optimization_type == OptimizationType.AIRMASS:
                    base_priority = r.priority / ps.airmass_coefficient
                else:
                    # TIME optimization (GRBs): strongly prefer earlier windows
                    # Use multiplicative bonus that decreases with window index
                    # First window gets full priority, later windows get progressively less
                    early_time_factor = 10.0 / (w_idx + 1.0)  # 10x for first, 5x for second, etc.
                    base_priority = r.priority * early_time_factor

                base_priority *= r.duration  # fair handling of long requests

                if r.request:
                    final_priority = base_priority * r.request.calculate_time_based_priority(ps.internal_start)
                else:
                    final_priority = base_priority

                # Apply telescope preference multiplier
                telescope_multiplier = self.telescope_preference.get(ps.resource, 1.0)
                if debug_count < 5 and telescope_multiplier != 1.0:
                    logger.debug(f"Applying {telescope_multiplier}x multiplier for {ps.resource}: "
                                f"priority {final_priority:.2f} -> {final_priority * telescope_multiplier:.2f}")
                    debug_count += 1
                final_priority *= telescope_multiplier

                self.Yik.append([r.resID, w_idx, final_priority, ps.resource, scheduled])
                self.Yik_times.append(
                    (ps.internal_start, ps.internal_start + timedelta(seconds=r.duration), r.request)
                )
                w_idx += 1

                # Build aikt with proper hashing
                for s in ps.all_slice_starts:
                    key, exists = self.hash_slice(s, ps.resource, self.time_slicing_dict[ps.resource][1])
                    if exists:
                        self.aikt[key].append(Yik_idx)
                    else:
                        self.aikt[key] = [Yik_idx]

    def hash_slice(self, start, resource, slice_length):
        """Hash slice with collision detection (restored from original)."""
        string = f"resource_{resource}_start_{repr(start)}_length_{repr(slice_length)}"
        exists = string in self.hashes
        self.hashes.add(string)
        return string, exists

    def get_slices(self, intervals, resource, duration, request):
        """Calculate possible start times (restored from original logic)."""
        ps_list = []
        if resource in self.time_slicing_dict:
            slice_alignment = self.time_slicing_dict[resource][0]
            slice_length = self.time_slicing_dict[resource][1]
            slices = []
            internal_starts = []

            # CRITICAL: Process timepoint dictionaries (not tuples!)
            for t in intervals.toDictList():
                if t['type'] == 'start':
                    # Handle slice alignment properly
                    if isinstance(t['time'], datetime) and isinstance(slice_alignment, int):
                        slice_alignment = datetime.fromtimestamp(slice_alignment)

                    if t['time'] <= slice_alignment:
                        start = slice_alignment
                        internal_start = slice_alignment
                    else:
                        time_diff = (t['time'] - slice_alignment).total_seconds()
                        start = slice_alignment + timedelta(seconds=int(time_diff / slice_length) * slice_length)
                        internal_start = t['time']

                elif t['type'] == 'end':
                    if isinstance(t['time'], datetime) and isinstance(slice_alignment, int):
                        slice_alignment = datetime.fromtimestamp(slice_alignment)

                    if t['time'] < slice_alignment:
                        continue

                    # Generate all possible starts in this window. Gate on
                    # room for the TRUE occupied span [internal_start,
                    # internal_start+duration), not room from the floored
                    # `start` -- otherwise the first (possibly sub-slice-
                    # offset) candidate in a window could be accepted even
                    # though it would actually run past the window's end.
                    while (t['time'] - internal_start).total_seconds() >= duration:
                        # internal_start may sit partway into the first slice
                        # (start is floored to the slice grid, internal_start
                        # is the true, possibly sub-slice-precision window
                        # start). Size coverage from internal_start, not from
                        # the floored start, so the reserved slices actually
                        # contain the true occupied span [internal_start,
                        # internal_start+duration) -- otherwise the floor can
                        # eat into the buffer that ceil() added, leaving the
                        # tail end under-covered by up to one slice.
                        offset = (internal_start - start).total_seconds()
                        num_slices = math.ceil((offset + duration) / slice_length)
                        tmp = [start + timedelta(seconds=i*slice_length) for i in range(num_slices)]
                        slices.append(tmp)
                        internal_starts.append(internal_start)
                        start += timedelta(seconds=slice_length)
                        internal_start = start

            if internal_starts:  # Only proceed if we have valid start times
                if request and request.optimization_type == OptimizationType.AIRMASS:
                    airmasses_at_times = request.get_airmasses_within_kernel_windows(resource)
                    if airmasses_at_times['times']:  # Check if we have airmass data
                        mid_times = [t + timedelta(seconds=duration/2) for t in internal_starts]
                        interpolated_airmasses = np.interp(
                            [t.timestamp() for t in mid_times],
                            [t.timestamp() for t in airmasses_at_times['times']],
                            airmasses_at_times['airmasses']
                        )
                    else:
                        interpolated_airmasses = np.ones(len(internal_starts))
                else:
                    interpolated_airmasses = np.ones(len(internal_starts))

                for idx, w in enumerate(slices):
                    ps_list.append(PossibleStart(resource, w, internal_starts[idx], interpolated_airmasses[idx]))

        return ps_list

    def _create_variables(self, model):
        """Create CP-SAT variables."""
        is_scheduled = {}

        for idx in range(len(self.Yik)):
            resID, start_idx, _, _, _ = self.Yik[idx]
            is_scheduled[idx] = model.NewBoolVar(f'sched_{resID}_{start_idx}')

        return is_scheduled

    def _add_constraints(self, model, is_scheduled):
        """Add all constraints to the model."""

        # One-of constraints
        for i, oneof in enumerate(self.oneof_constraints):
            constraint_vars = []
            for r in oneof:
                constraint_vars.extend(is_scheduled[idx] for idx in r.Yik_entries)
                r.skip_constraint2 = True
            model.Add(sum(constraint_vars) <= 1).WithName(f'oneof_{i}')

        # And constraints (all or nothing)
        for i, andconstraint in enumerate(self.and_constraints):
            and_var = model.NewBoolVar(f'and_{i}')
            for r in andconstraint:
                for idx in r.Yik_entries:
                    model.Add(is_scheduled[idx] == and_var)

        # No more than one reservation per time slice
        for slice_key, indices in self.aikt.items():
            model.Add(sum(is_scheduled[idx] for idx in indices) <= 1).WithName(f'slice_{slice_key}')

        # Each reservation scheduled at most once
        for r in self.reservation_list:
            if not hasattr(r, 'skip_constraint2'):
                model.Add(sum(is_scheduled[idx] for idx in r.Yik_entries) <= 1).WithName(f'once_{r.resID}')

        # Slew time between consecutive observations on the same resource
        self._add_slew_constraints(model, is_scheduled)

    def _add_slew_constraints(self, model, is_scheduled):
        """Forbid pairs of candidates that don't leave enough time to slew
        from one target to the next.

        For a given resource, whichever candidate ends up immediately before
        another in the final schedule must leave a gap of at least
        slew_time_fn(prev.request, next.request) seconds. We don't know the
        eventual neighbours in advance, but it's enough to forbid every pair
        (i, j) whose *raw* gap (ignoring anything that might be slotted
        between them) is already too small: inserting a third candidate k
        between i and j only ever requires a larger total gap (duration(k)
        plus its own two slews), never a smaller one -- true whenever
        slew_time_fn is non-negative and non-decreasing in angular distance,
        which holds for the linear model this is built for. So this pairwise
        check is a valid necessary condition regardless of what else gets
        scheduled in between, without needing explicit ordering variables.
        """
        if self.slew_time_fn is None:
            return

        by_resource = defaultdict(list)
        for idx, row in enumerate(self.Yik):
            resID, _, _, resource, _ = row
            start_dt, end_dt, request = self.Yik_times[idx]
            by_resource[resource].append((start_dt, end_dt, request, resID, idx))

        added = 0
        for resource, entries in by_resource.items():
            entries.sort(key=lambda e: e[0])  # by start time
            n = len(entries)
            for a in range(n):
                start_a, end_a, req_a, resid_a, idx_a = entries[a]
                for b in range(a + 1, n):
                    start_b, end_b, req_b, resid_b, idx_b = entries[b]
                    if resid_a == resid_b:
                        continue  # same reservation: already mutually exclusive

                    gap = (start_b - end_a).total_seconds()
                    if gap >= self.max_slew_seconds:
                        break  # entries are start-sorted, so gap only grows from here
                    if gap < 0:
                        continue  # overlapping candidates; already forbidden via aikt

                    required = self.slew_time_fn(req_a, req_b)
                    if gap < required:
                        model.Add(is_scheduled[idx_a] + is_scheduled[idx_b] <= 1)
                        added += 1
                    elif (self.slew_penalty_weight > 0 and required > 0
                          and self.max_slew_seconds != float('inf')):
                        # Feasible neighbour pair: let _set_objective() reward
                        # this pairing via an AND-linearized adjacency variable
                        # (standard: adj<=a, adj<=b, adj>=a+b-1 forces
                        # adj == a AND b for booleans a,b). The BONUS magnitude
                        # max_slew - required grows as the angular separation
                        # shrinks, so packing next to a CLOSE target gains
                        # objective value while spreading candidates further
                        # than max_slew apart gains nothing. (A penalty
                        # formulation -weight*required*adj was tried first and
                        # does not work: the solver simply spaces candidates
                        # beyond max_slew where no adjacency var exists,
                        # avoiding all penalties without ever grouping close
                        # targets.) adj <= a and adj <= b keep the bonus from
                        # being claimed when either candidate is unscheduled.
                        adj_var = model.NewBoolVar(f'adj_{idx_a}_{idx_b}')
                        model.Add(adj_var <= is_scheduled[idx_a])
                        model.Add(adj_var <= is_scheduled[idx_b])
                        model.Add(adj_var >= is_scheduled[idx_a] + is_scheduled[idx_b] - 1)
                        self._slew_penalty_terms.append(
                            (adj_var, self.max_slew_seconds - required))

        logger.info(f"Added {added} slew-time exclusion constraints")
        if self._slew_penalty_terms:
            logger.info(f"Added {len(self._slew_penalty_terms)} slew-time penalty terms "
                        f"(weight={self.slew_penalty_weight})")

    def _set_objective(self, model, is_scheduled):
        """Set the optimization objective."""
        objective = sum(self.Yik[idx][2] * is_scheduled[idx] for idx in range(len(self.Yik)))
        if self._slew_penalty_terms:
            # adjacency bonus, see _add_slew_constraints: rewards scheduling
            # angularly close targets within max_slew_seconds of each other
            objective += self.slew_penalty_weight * sum(
                magnitude * adj_var for adj_var, magnitude in self._slew_penalty_terms
            )
        model.Maximize(objective)

    def _process_solution(self, solver, is_scheduled):
        """Process the solver solution and commit scheduled reservations."""
        for idx in range(len(self.Yik)):
            if solver.BooleanValue(is_scheduled[idx]):
                resID, start_idx, _, resource, _ = self.Yik[idx]
                reservation = self.get_reservation_by_ID(resID)
                start = reservation.possible_starts[start_idx].internal_start

                # Calculate actual quantum, anchored to `start` (the true
                # scheduled instant, stored as-is in Schedule.date) rather
                # than all_slice_starts[0] (floored to the slice grid --
                # can be up to one slice earlier than `start`). Anchoring
                # to the floor made `date + expo` overstate the reserved
                # coverage's real end by that same floor offset.
                quantum = (reservation.possible_starts[start_idx].all_slice_starts[-1] +
                          timedelta(seconds=self.time_slicing_dict[resource][1]) -
                          start)

                # Schedule and commit
                reservation.schedule(start, quantum.total_seconds(), resource, 'CPScheduler')
                #reservation.schedule(start, quantum, resource, 'CPScheduler')
                self.commit_reservation_to_schedule(reservation)

# the following is an idea of a revamp of get_slices, but as it is it cannot be used:

    def _get_possible_starts(self, intervals, resource, duration, request):
        """Calculate possible start times within given intervals."""
        possible_starts = []
        slice_length = self.time_slicing[resource][1]

        for start_time, end_time in intervals.toTupleList():
            # Align to slice boundaries
            if isinstance(start_time, datetime):
                aligned_start = self._align_to_slice(start_time, slice_length)
            else:
                aligned_start = start_time

            # Generate all possible starts
            current = aligned_start
            while (end_time - current).total_seconds() >= duration:
                num_slices = math.ceil(duration / slice_length)
                slice_starts = [
                    current + timedelta(seconds=i * slice_length)
                    for i in range(num_slices)
                ]

                # Calculate airmass coefficient if needed
                airmass_coeff = 1.0
                if request and request.optimization_type == OptimizationType.AIRMASS:
                    airmass_coeff = self._get_airmass_coefficient(
                        request, resource, current, duration
                    )

                possible_starts.append(
                    PossibleStart(resource, slice_starts, current, airmass_coeff)
                )

                current += timedelta(seconds=slice_length)

        return possible_starts

    def _calculate_priority(self, reservation, possible_start, window_idx):
        """Calculate priority for a reservation at a specific start time."""
        base_priority = reservation.priority

        if reservation.request:
            if reservation.request.optimization_type == OptimizationType.AIRMASS:
                # Apply airmass weighting
                base_priority /= possible_start.airmass_coefficient
            else:
                # Prefer earlier windows
                base_priority += 10 / (window_idx + 1.0)

            # Fair handling of long requests
            base_priority *= reservation.duration

            # Apply time-based priority
            base_priority *= reservation.request.calculate_time_based_priority(
                possible_start.internal_start
            )

            # Apply moon penalty (cached lookup, very fast)
            moon_penalty = self._get_moon_penalty(
                reservation.request, possible_start.resource,
                possible_start.internal_start, reservation.duration
            )
            base_priority *= moon_penalty

        return base_priority

    def _get_moon_penalty(self, request, resource, start_time, duration):
        """Get moon penalty coefficient for optimization."""
        if not hasattr(request, 'moon_penalty_data'):
            return 1.0

        if resource not in request.moon_penalty_data:
            return 1.0

        data = request.moon_penalty_data[resource]
        if not data['times']:
            return 1.0

        mid_time = start_time + timedelta(seconds=duration/2)

        # Find penalty for observation midpoint (same logic as airmass)
        times = data['times']
        penalties = data['penalties']

        # Find closest time
        time_diffs = [abs((t - mid_time).total_seconds()) for t in times]
        closest_idx = time_diffs.index(min(time_diffs))

        return penalties[closest_idx]

    def _get_airmass_coefficient(self, request, resource, start_time, duration):
        """Get airmass coefficient for optimization."""
        if not hasattr(request, 'airmass_data'):
            return 1.0

        if resource not in request.airmass_data:
            return 1.0

        data = request.airmass_data[resource]
        if not data['times']:
            return 1.0

        # Find the airmass at the midpoint of the observation
        mid_time = start_time + timedelta(seconds=duration/2)

        # Interpolate
        times_ts = [t.timestamp() for t in data['times']]
        mid_ts = mid_time.timestamp()

        airmass = np.interp(mid_ts, times_ts, data['airmasses'])
        return max(airmass, 1.0)  # Ensure it's at least 1

    def _align_to_slice(self, time, slice_size):
        """Align a datetime to slice boundaries."""
        epoch = datetime(1970, 1, 1)
        seconds_since_epoch = (time - epoch).total_seconds()
        aligned_seconds = (seconds_since_epoch // slice_size) * slice_size
        return epoch + timedelta(seconds=aligned_seconds)

    def _hash_slice(self, start, resource):
        """Create a hash key for a time slice."""
        if isinstance(start, datetime):
            start_str = start.isoformat()
        else:
            start_str = str(start)
        return f"{resource}_{start_str}_{self.slice_size}"

