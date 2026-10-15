"""
Base scheduler class with essential functionality only.
"""

from collections import defaultdict
from kernel.intervals import Intervals


class BaseScheduler:
    """Base class for schedulers with minimal essential functionality."""

    def __init__(self, compound_reservation_list, globally_possible_windows_dict,
                 contractual_obligation_list):
        self.compound_reservation_list = compound_reservation_list
        self.contractual_obligation_list = contractual_obligation_list

        # Remove empty windows
        self.globally_possible_windows_dict = {
            resource: windows
            for resource, windows in globally_possible_windows_dict.items()
            if not windows.is_empty()
        }

        self.resource_list = list(self.globally_possible_windows_dict.keys())

        # Initialize schedule storage
        self.schedule_dict = defaultdict(list)
        self.schedule_dict_busy = defaultdict(lambda: Intervals([], 'busy'))
        self.schedule_dict_free = self.globally_possible_windows_dict.copy()

        # Initialize constraints
        self.and_constraints = []
        self.oneof_constraints = []

        # Convert compound to simple reservations
        self.reservation_list, self.reservation_dict = self._convert_compound_to_simple()
        self.unscheduled_reservation_list = list(self.reservation_list)

    def _convert_compound_to_simple(self):
        """Convert compound reservations to simple ones with constraints."""
        reservation_list = []
        reservation_dict = {}

        for cr in self.compound_reservation_list:
            valid_reservations = [
                res for res in cr.reservation_list
                if self._check_against_global_windows(res)
            ]

            if not valid_reservations:
                continue

            if cr.issingle():
                res = valid_reservations[0]
                reservation_list.append(res)
                reservation_dict[res.resID] = res

            elif cr.isoneof():
                reservation_list.extend(valid_reservations)
                reservation_dict.update({r.resID: r for r in valid_reservations})
                self.oneof_constraints.append(valid_reservations)

            elif cr.isand():
                if len(valid_reservations) == len(cr.reservation_list):
                    reservation_list.extend(valid_reservations)
                    reservation_dict.update({r.resID: r for r in valid_reservations})
                    self.and_constraints.append(valid_reservations)

        return reservation_list, reservation_dict

    def _check_against_global_windows(self, reservation):
        """Check if reservation has valid windows."""
        result = {}

        for resource, windows in reservation.free_windows_dict.items():
            if resource not in self.globally_possible_windows_dict:
                continue

            global_windows = self.globally_possible_windows_dict[resource]
            intersection = windows.intersect([global_windows])

            if not intersection.is_empty():
                result[resource] = intersection

        if result:
            reservation.free_windows_dict = result
            return True
        return False

    def get_reservation_by_ID(self, ID):
        """Get reservation by ID."""
        return self.reservation_dict.get(ID)

    def commit_reservation_to_schedule(self, reservation):
        """Commit a scheduled reservation."""
        if not reservation.scheduled:
            raise ValueError("Attempting to commit unscheduled reservation")

        resource = reservation.scheduled_resource
        timepoints = reservation.scheduled_timepoints

        self.schedule_dict[resource].append(reservation)
        self.schedule_dict_busy[resource].add(timepoints)
        self.schedule_dict_free[resource] = self.schedule_dict_free[resource].subtract(
            Intervals(timepoints)
        )

        try:
            self.unscheduled_reservation_list.remove(reservation)
        except ValueError:
            raise ValueError(f"Reservation {reservation.resID} not in unscheduled list")
