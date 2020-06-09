from functools import lru_cache
from typing import Tuple

import numpy as np
from poison.utils import SingletonMeta


class BernoulliTrialCalculator:
    """ A statistics calculator used to compute some probabilities and expected number of tests
        for a series of trials based on the Bernoulli distribution. """

    def __init__(self, p: float):
        self.p = p
        self.max_n_searched = 0
        self.poisoned_group_size_statistics = dict()

    # region Probability estimators.

    @lru_cache(maxsize=None)
    def prob_at_least_one_poison(self, n: int) -> float:
        p = self.p
        return 1 - (1 - p) ** n

    @lru_cache(maxsize=None)
    def prob_exactly_one_poison(self, n: int) -> float:
        p = self.p
        return n * ((p ** 1) * ((1-p) ** (n-1)))

    @lru_cache(maxsize=None)
    def prob_at_least_one_poison_but_not_in_first_x(self, n: int, x: int) -> float:
        p = self.p
        return ((1-p) ** x) * self.prob_at_least_one_poison(n=n-x)

    def prob_at_least_one_poison_but_not_in_first_x_array(self, n: int) -> np.array:
        return np.array([self.prob_at_least_one_poison_but_not_in_first_x(n=n, x=x) for x in range(n+1)])

    @lru_cache(maxsize=None)
    def prob_no_poison_in_first_x_given_at_least_one_poison_array(self, n: int) -> np.array:
        return self.prob_at_least_one_poison_but_not_in_first_x_array(n=n) / self.prob_at_least_one_poison(n=n)

    # endregion Probability estimators.

    # region Expected number of tests for a poisoned group size.

    def compute_poisoned_group_size_statistics(self, n: int):
        if n > self.max_n_searched:
            for i in range(self.max_n_searched + 1, n):
                self.poisoned_group_size_statistics[i] = self.get_poisoned_group_size_statistics(i)
            self.max_n_searched = n

    def get_poisoned_group_size_split_point(self, n: int) -> int:
        self.compute_poisoned_group_size_statistics(n)
        return self.get_poisoned_group_size_statistics(n)[1]

    def get_poisoned_group_size_expected_number_of_tests(self, n: int) -> float:
        self.compute_poisoned_group_size_statistics(n)
        return self.get_poisoned_group_size_statistics(n)[0]

    @lru_cache(maxsize=None)
    def get_poisoned_group_size_statistics(self, n) -> Tuple[float, int]:
        p_start_has_no_poison = self.prob_no_poison_in_first_x_given_at_least_one_poison_array(n)
        expected_number_of_tests_on_split = (
            # Either we go with strategy 0 (round robin, and maybe get lucky to not test the last).
            [((p_start_has_no_poison[n - 1]) * (n - 1) + (1 - p_start_has_no_poison[n - 1]) * n, 0)] +
            # Or we try and split.
            [(
                (
                    # Split on x and test the first x.
                    1 +
                    # If there isn't a poison first in the first x, continue to scrutinize the rest.
                    (p_start_has_no_poison[x] * self.get_poisoned_group_size_statistics(n-x)[0]) +
                    # If there is a poison in first in the first x.
                    ((1 - p_start_has_no_poison[x]) * (
                        # We need to scrutinize the first x.
                        self.get_poisoned_group_size_statistics(x)[0] +
                        # We have to spend another test to check if there is poison in remaining.
                        1 + (
                            # If there wasn't, we're in luck. No need to test further.
                            (1 - self.prob_at_least_one_poison(n-x)) * 0 +
                            # Otherwise, we have to scrutinize the rest.
                            (self.prob_at_least_one_poison(n-x)) * (self.get_poisoned_group_size_statistics(n-x)[0])
                        )
                    ))
                ),
             x) for x in range(1, n)])
        # Identify the best split point which minimizes the expected number of tests.
        optimal_strategy = expected_number_of_tests_on_split[np.argmin(expected_number_of_tests_on_split, axis=0)[0]]
        return optimal_strategy

    # endregion Calculate expected number of tests for a group size.

    # region Expected number of tests for a top-level group size.

    def get_group_size_statistics(self, n: int) -> Tuple[float, int]:
        self.compute_poisoned_group_size_statistics(n)
        expected_num_tests = [(
                # Testing each group size, and if found poison, scrutinize based on our best strategy.
                np.floor(n / group_size) * (
                        1
                        + (self.prob_at_least_one_poison(n=group_size) *
                           self.get_poisoned_group_size_expected_number_of_tests(n=group_size)))
                # If the last group is smaller than n.
                + (0.0 if not n % group_size
                    else (1 + (self.prob_at_least_one_poison(n=n % group_size) *
                               self.get_poisoned_group_size_expected_number_of_tests(n=(n % group_size))))),
                group_size)
            for group_size in range(1, n+1)]
        minimal_expected_num_tests = expected_num_tests[np.argmin(expected_num_tests, axis=0)[0]]
        return minimal_expected_num_tests

    # endregion Expected number of tests for a top-level group size.


class StatsCalculatorFactory(metaclass=SingletonMeta):
    """ A factory class used to build calculators. """

    def __init__(self):
        self._bernoulli_calculators = dict()

    def get_bernoulli(self, p: float) -> BernoulliTrialCalculator:
        """
        Gets a Bernoulli trial poisoned drinks calculator for the given probability.
        :param p: The probability calculator.
        :return: The statistical calculator.
        """
        if p not in self._bernoulli_calculators:
            self._bernoulli_calculators[p] = BernoulliTrialCalculator(p=p)
        return self._bernoulli_calculators[p]
