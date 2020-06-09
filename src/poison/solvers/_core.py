from typing import Callable, Dict

from .base import PoisonedDrinksSolver
from poison.solvers.strategy import PoisonedGroupStrategies


class SolverTwoLevels(PoisonedDrinksSolver):
    """
    The two-stage solver has basically two levels:
    1) Top level: evaluate a group of drinks, with no prior knowledge.
    2) Bottom level: scrutinize all elements of the a group of drinks, knowing at least one is poisoned.
    """

    def __init__(self, top_group_strategy: Callable, poisoned_group_strategy: Callable):
        super().__init__()
        self.top_group_strategy = top_group_strategy
        self.poisoned_group_strategy = poisoned_group_strategy

    @property
    def params(self) -> Dict[str, str]:
        return {"Solver type": "Two-level",
                "Top group strategy": self.top_group_strategy.display_name,
                "Poisoned group strategy": self.poisoned_group_strategy.display_name}

    def _solve_core(self) -> None:
        # Determine the group size according the minimum expected tests.
        top_level_group_size = self.top_group_strategy(n=self.n, p=self.p)
        # Check each group [i_start, i_end) to see if one glass in it is poisoned.
        for i_start in range(0, super().n, top_level_group_size):
            i_end = min([i_start + top_level_group_size, super().n])
            if super().check_for_poison(range(i_start, i_end)):
                self.poisoned_group_strategy(solver=self, i_start=i_start, i_end=i_end)


class SolverDeepDive(SolverTwoLevels):
    """
    An extension of the two-levels solver, where the bottom level
    (scrutinizing a poisoned group) is done in a deep dive manner.
    """

    def __init__(self, top_group_strategy: Callable, poisoned_group_split_strategy: Callable):
        super().__init__(top_group_strategy=top_group_strategy,
                         poisoned_group_strategy=SolverDeepDive._deep_dive_poisoned_group)
        self.top_group_strategy = top_group_strategy
        self.poisoned_group_split_strategy = poisoned_group_split_strategy

    @property
    def params(self) -> Dict[str, str]:
        return {"Solver type": "Deep-dive",
                "Top group strategy": self.top_group_strategy.display_name,
                "Poisoned group strategy": f"Split on {self.poisoned_group_split_strategy.display_name}"}

    @staticmethod
    def _deep_dive_poisoned_group(solver, i_start: int, i_end: int):
        # If there is only one glass, it has to contain poison.
        if (i_end - i_start) == 1:
            solver.mark_as_poison(i_start)
        elif (i_end - i_start) > 1:
            # If we don't know, we split the search. This is under the assumption, that in most time,
            # we will only dive into one subset of the search space.
            i_split = solver.poisoned_group_split_strategy(i_start=i_start, i_end=i_end, p=solver.p)
            # If we couldn't find a split point, perform Round-robin+.
            if not i_split:
                PoisonedGroupStrategies.round_robin_plus(solver=solver, i_start=i_start, i_end=i_end)
            else:
                # We first check if there's poison in the left half.
                found_poison_left = solver.check_for_poison(range(i_start, i_split))
                if found_poison_left:
                    SolverDeepDive._deep_dive_poisoned_group(solver=solver, i_start=i_start, i_end=i_split)
                # If there's still a chance there's poison in the right half, we have to check it as well.
                if not found_poison_left or solver.check_for_poison(range(i_split, i_end)):
                    SolverDeepDive._deep_dive_poisoned_group(solver=solver, i_start=i_split, i_end=i_end)


class SolverEarlyStop(PoisonedDrinksSolver):

    def __init__(self, top_group_strategy: Callable, poisoned_group_split_strategy: Callable):
        super().__init__()
        self.top_group_strategy = top_group_strategy
        self.poisoned_group_split_strategy = poisoned_group_split_strategy

    @property
    def params(self) -> Dict[str, str]:
        return {"Solver type": "Early-stop",
                "Top group strategy": self.top_group_strategy.display_name,
                "Poisoned group strategy": f"Split on {self.poisoned_group_split_strategy.display_name}"}

    def _solve_core(self) -> None:
        # Determine the group size according the minimum expected tests.
        top_level_group_size = self.top_group_strategy(n=self.n, p=self.p)
        # Check each group [i_start, i_end) to see if one glass in it is poisoned.
        i_start = 0
        while i_start < super().n:
            i_end = min([i_start + top_level_group_size, super().n])
            if super().check_for_poison(range(i_start, i_end)):
                # If found poison, we should continue with the glass right after the poisoned one.
                i_end = self._early_stop_deep_dive_poisoned_group(i_start=i_start, i_end=i_end) + 1
            i_start = i_end

    def _early_stop_deep_dive_poisoned_group(self, i_start: int, i_end: int) -> int:
        """ Deep diving into a group of drinks knowing that there's at least one poisoned drink.
            Stops once we have found the first poisoned drink, and returns its location. """
        # If there is only one glass, it has to contain poison.
        if (i_end - i_start) == 1:
            super().mark_as_poison(i_start)
            return i_start
        if (i_end - i_start) > 1:
            # If we don't know, we split the search. This is under the assumption, that in most time,
            # we will only dive into one subset of the search space.
            i_split = self.poisoned_group_split_strategy(i_start=i_start, i_end=i_end, p=self.p)
            if not i_split:
                return self._early_stop_round_robin_plus(i_start=i_start, i_end=i_end)
            else:
                # We first check if there's poison in the left half.
                found_poison_left = super().check_for_poison(range(i_start, i_split))
                if found_poison_left:
                    return self._early_stop_deep_dive_poisoned_group(i_start=i_start, i_end=i_split)
                # If there's still a chance there's poison in the right half, we have to check it as well.
                if not found_poison_left or super().check_for_poison(range(i_split, i_end)):
                    return self._early_stop_deep_dive_poisoned_group(i_start=i_split, i_end=i_end)

    def _early_stop_round_robin_plus(self, i_start: int, i_end: int) -> int:
        """ Greedy round robin marks the last glass as poisoned if all previous glasses were not poisoned. """
        # If found poison in group element, check each individual glass in the group.
        last_poison_index = None
        for i in range(i_start, i_end):
            if (last_poison_index is None and i == (i_end - 1)) or self.check_for_poison([i]):
                self.mark_as_poison(i)
                return i
        return last_poison_index
