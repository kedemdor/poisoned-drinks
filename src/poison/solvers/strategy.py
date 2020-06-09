import numpy as np
from poison.solvers.base import PoisonedDrinksSolver
from poison.stats import StatsCalculatorFactory


def display_name(name):
    """ A decorator to add a display name to each strategy. """
    def decorator(func):
        func.display_name = name
        return func
    return decorator


class TopGroupSizeStrategies:
    """ This class contains multiple approaches for determining the size of the top group to inspect. """

    @staticmethod
    @display_name(name="YouTube")
    def youtube(n: int, p: float) -> int:
        """ Gets group size that reduce the expected number of tests according to timestamp 5:37 in the video. """
        optimal_group_size = np.argmin([((1 - (1 - p) ** i) * n + n / i)
                                        for i in range(1, n + 1)]) + 1
        return 1 if optimal_group_size >= n else optimal_group_size

    @staticmethod
    @display_name(name="1/p")
    def inverse_p(n: int, p: float) -> int:
        """ Returns 1/p as the group size. """
        optimal_group_size = int(np.round(1 / p))
        return min([optimal_group_size, n])

    @staticmethod
    @display_name(name="Bernoulli trial 50%")
    def bernoulli_trial(n: int, p: float) -> int:
        """ Returns the number of tests which, according to the Bernoulli trial,
            should give you roughly 50% chance of having a poisoned glass in the group.
            More details in https://en.wikipedia.org/wiki/Bernoulli_trial. """
        optimal_group_size = 0
        chance_of_no_poison = 1
        while chance_of_no_poison > 0.5:
            optimal_group_size += 1
            chance_of_no_poison *= (1 - p)
        return min([optimal_group_size, n])

    @staticmethod
    @display_name(name="min(E(#tests))")
    def minimum_total_expected_tests(n: int, p: float) -> int:
        """ Based on a full analysis of expected number of tests. """
        bernoulli_calculator = StatsCalculatorFactory().get_bernoulli(p)
        return max([1, bernoulli_calculator.get_group_size_statistics(n)[1]])


class PoisonedGroupStrategies:

    @staticmethod
    @display_name(name="Round-robin")
    def round_robin(solver: PoisonedDrinksSolver, i_start: int, i_end: int):
        # If the group size is 1, no need to test again - we know this glass is poisoned.
        if i_end - i_start == 1:
            solver.mark_as_poison(i_start)
        else:
            # Else, iterate over the poisoned group elements.
            for i in range(i_start, i_end):
                if solver.check_for_poison([i]):
                    solver.mark_as_poison(i)

    @staticmethod
    @display_name(name="Round-robin plus")
    def round_robin_plus(solver: PoisonedDrinksSolver, i_start: int, i_end: int):
        """ Greedy round robin marks the last glass as poisoned if all previous glasses were not poisoned. """
        # If found poison in group element, check each individual glass in the group.
        found_poison = False
        for i in range(i_start, i_end):
            if (not found_poison and i == (i_end - 1)) or solver.check_for_poison([i]):
                solver.mark_as_poison(i)
                found_poison = True


class PoisonedGroupSplitStrategies:

    @staticmethod
    @display_name(name="Middle")
    def middle(i_start: int, i_end: int, p: float):
        """ Splits the data groups in the middle, between i_start and i_end. """
        return (i_start + i_end) // 2

    @staticmethod
    @display_name(name="min(E(# tests))")
    def minimum_expected_tests_complete(i_start: int, i_end: int, p: float):
        stats_calculator = StatsCalculatorFactory().get_bernoulli(p)
        split_delta = stats_calculator.get_poisoned_group_size_split_point(n=i_end - i_start)
        # If split_delta = 0, it means the best strategy for the group is greedy round robin, rather than splitting.
        return None if not split_delta else i_start + split_delta
