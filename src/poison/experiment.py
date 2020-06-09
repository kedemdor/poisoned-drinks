import numpy as np
from typing import Iterable


class PoisonedDrinksExperiment:
    """ A single experiment of the poisoned drinks problem. """

    def __init__(self, num_glasses: int, poison_chance: float, seed: int = 0):
        """ Creates an instance of the poisoned drinks problem. """
        self.num_glasses = num_glasses
        self.poison_chance = poison_chance
        # Poison some glasses... 0 is healthy, 1 is poisoned.
        np.random.seed(seed)
        self.__glasses = (np.random.random([num_glasses]) <= poison_chance) * 1

    def check_poison(self, glasses_indices: Iterable[int]) -> bool:
        """ Checks whether a group of glasses are poisoned. """
        return np.any(self.__glasses[glasses_indices])

    def check_solution(self, solution: np.array) -> bool:
        """ Checks whether a found solution to the experiment is correct. """
        return np.all(self.__glasses == solution)
