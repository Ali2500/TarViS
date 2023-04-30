from typing import Callable, Dict, Any
from time import time as current_time

import contextlib


@contextlib.contextmanager
def timed_event(print_fn: Callable, label: str, enable: bool):
    start_time = current_time()
    yield None
    end_time = current_time()
    if enable:
        print_fn(f"{label}: {end_time - start_time:.3f} sec")


class ETAEstimator:
    def __init__(self, total_iterations: int, num_initial_iterations_to_discard: int):
        self._total_duration_elapsed = 0
        self._last_timestamp = None
        self._latest_iteration_num = 0
        self._initial_iters_discarded = False
        self._is_started = False

        self.num_initial_iterations_to_discard = num_initial_iterations_to_discard
        self.total_iterations = total_iterations

    @property
    def elapsed_time(self):
        return self._total_duration_elapsed

    def start(self):
        assert self.total_iterations > 0
        self._is_started = True
        self._last_timestamp = current_time()
            
    def tick(self, step: int = 1):
        if self.num_initial_iterations_to_discard > 0:
            if self._latest_iteration_num == self.num_initial_iterations_to_discard and not self._initial_iters_discarded:
                self._total_duration_elapsed = 0
                self._latest_iteration_num = 0
                self._initial_iters_discarded = True

        time_now = current_time()
        self._total_duration_elapsed += (time_now - self._last_timestamp)
        self._last_timestamp = time_now
        self._latest_iteration_num += step

    def get_eta(self, as_string=True):
        # assert self._train_start_time is not None
        assert self._is_started
        avg_time_per_iter = self._total_duration_elapsed / float(self._latest_iteration_num)
        eta = float(self.total_iterations - self._latest_iteration_num) * avg_time_per_iter
        if not as_string:
            return eta, avg_time_per_iter

        days, rem = divmod(eta, 3600*24)
        hours, rem = divmod(rem, 3600)
        minutes, seconds = divmod(rem, 60)
        return "%02d-%02d:%02d:%02d" % (int(days), int(hours), int(minutes), int(seconds)), avg_time_per_iter

    def state_dict(self):
        return {'_total_duration_elapsed': self._total_duration_elapsed,
                '_latest_iteration_num': self._latest_iteration_num,
                'num_initial_iterations_to_discard': self.num_initial_iterations_to_discard,
                '_initial_iters_discarded': self._initial_iters_discarded,
                'total_iterations': self.total_iterations
                }

    def load_state_dict(self, d):
        for key in d:
            assert key in self.__dict__, f"Invalid parameter '{key}' in state dict"
        self.__dict__.update(d)

    @classmethod
    def create(cls,
               total_iterations: int,
               num_iterations_to_discard: int = 0):
        return cls(
            total_iterations=total_iterations,
            num_initial_iterations_to_discard=num_iterations_to_discard
        )

    @classmethod
    def restore_from_checkpoint(cls,
                                state_dict: Dict[str, Any]):
        inst = cls(0, 0)
        inst.load_state_dict(state_dict)
        return inst
