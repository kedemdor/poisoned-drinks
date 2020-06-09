import abc
import numpy as np
from typing import Iterable, Dict

from poison.experiment import PoisonedDrinksExperiment


class PoisonedDrinksSolver(abc.ABC):
    """ The base class for a poisoned drinks solver. """

    def __init__(self):
        """ Creates a new instance of the poisoned drinks solver. """
        self.__experiment = None  # Hiding the experiment so extending classes won't check for it.
        self.__num_tests = 0  # Hiding the number of tests so that extending classes won't update it.
        self.__solution = None

    # region Properties

    @property
    def name(self) -> str:
        """ Gets the name of the poisoned drinks solver. """
        return f"{self.params['Solver type']} - " \
               f"top: [{self.params['Top group strategy']}], " \
               f"poisoned: [{self.params['Poisoned group strategy']}]"

    @property
    def n(self) -> int:
        """ Gets the number of glasses in the experiment. """
        return self.__experiment.num_glasses

    @property
    def p(self) -> float:
        """ Gets the probability a glass is poisoned in the experiment. """
        return self.__experiment.poison_chance

    @property
    def num_tests(self) -> int:
        """ Gets the number of tests. """
        return self.__num_tests

    @property
    @abc.abstractmethod
    def params(self) -> Dict[str, str]:
        pass

    # endregion Properties

    # region Solver base functionality

    def load_experiment(self, experiment: PoisonedDrinksExperiment):
        self.__experiment = experiment  # Hiding the experiment so extending classes won't check for it.
        self.__num_tests = 0  # Hiding the number of tests so that extending classes won't update it.
        self.__solution = np.zeros([self.__experiment.num_glasses])

    def check_for_poison(self, glasses_indices: Iterable[int]) -> bool:
        """ Checks if at least one of the given set of glasses is poisoned. """
        self.__num_tests += 1  # Keep track that a test was taken.
        return self.__experiment.check_poison(glasses_indices)

    def mark_as_poison(self, glass_index: int) -> None:
        """ Marks a given glass as poisoned. """
        self.__solution[glass_index] = 1

    # endregion Solver base functionality

    # region Solving the problem

    def solve(self) -> None:
        """ Solves the poisoned drinks problem. """
        self._solve_core()
        if not self.__experiment.check_solution(self.__solution):
            raise ArithmeticError("The solution is incorrect. Fix your solver.")

    @abc.abstractmethod
    def _solve_core(self) -> None:
        """ The implementation of the solve function of the extender class. """
        pass

    # endregion Solving the problem
