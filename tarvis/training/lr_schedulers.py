from typing import List, Optional, Union

import math
import torch.optim.lr_scheduler as lr_schedulers


def get_warmup_lr(current_iter: int, num_warmup_iters: int, warmup_lr_init: float, base_lr: float) -> float:
    assert current_iter <= num_warmup_iters

    lr_range = base_lr - warmup_lr_init
    return warmup_lr_init + ((current_iter / num_warmup_iters) * lr_range)


class WarmupLR(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, warmup_iters: int, warmup_lr_init: float, last_epoch: Optional[int] = -1,
                 verbose: Optional[bool] = False):
        assert warmup_iters > 0

        self.warmup_iters = warmup_iters
        self.warmup_lr_init = warmup_lr_init

        super(WarmupLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            return [
                get_warmup_lr(self.last_epoch, self.warmup_iters, self.warmup_lr_init, base_lr)
                for base_lr in self.base_lrs
            ]
        else:
            return [group['lr'] for group in self.optimizer.param_groups]


class StepLR(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, step_iters: List[int], decay_rate: Union[float, List[float]],
                 warmup_iters: Optional[int] = 0, warmup_lr_init: Optional[int] = 0.,
                 last_epoch: Optional[int] = -1, verbose: Optional[bool] = False):

        if isinstance(decay_rate, float):
            self.decay_rates = [decay_rate] * len(step_iters)
        else:
            assert len(decay_rate) == len(step_iters)
            self.decay_rates = decay_rate

        assert warmup_iters < step_iters[0]
        self.warmup_iters = warmup_iters
        self.warmup_lr_init = warmup_lr_init

        self.step_iters = step_iters
        self.current_step_index = 0

        super(StepLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters and self.warmup_iters > 0:
            return [
                get_warmup_lr(self.last_epoch, self.warmup_iters, self.warmup_lr_init, base_lr)
                for base_lr in self.base_lrs
            ]

        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch == self.step_iters[self.current_step_index]:
            new_lrs = [current_lr * self.decay_rates[self.current_step_index] for current_lr in current_lrs]

            if self.current_step_index < len(self.step_iters) - 1:
                self.current_step_index += 1
            return new_lrs

        else:
            return current_lrs


class PolynomialLR(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, decay_factor: float, num_decay_iters: int, power: float,
                 warmup_iters: Optional[int] = 0, warmup_lr_init: Optional[int] = 0.,
                 last_epoch: Optional[int] = -1, verbose: Optional[bool] = False):

        self.num_decay_iters = num_decay_iters
        self.power = power

        self.warmup_iters = warmup_iters
        self.warmup_lr_init = warmup_lr_init

        super(PolynomialLR, self).__init__(optimizer, last_epoch, verbose)
        self.end_lrs = [base_lr * decay_factor for base_lr in self.base_lrs]

    def _compute_lr(self, base_lr, end_lr):
        lr_range = base_lr - end_lr
        progress_frac = float(self.last_epoch - self.warmup_iters) / float(self.num_decay_iters)
        return base_lr - ((progress_frac ** self.power) * lr_range)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters and self.warmup_iters > 0:
            return [
                get_warmup_lr(self.last_epoch, self.warmup_iters, self.warmup_lr_init, base_lr)
                for base_lr in self.base_lrs
            ]

        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        if self.last_epoch > (self.warmup_iters + self.num_decay_iters):
            return current_lrs

        return [self._compute_lr(base_lr, end_lr) for base_lr, end_lr in zip(self.base_lrs, self.end_lrs)]


class LinearLR(PolynomialLR):
    def __init__(self, optimizer, decay_factor: float, num_decay_iters: int,
                 warmup_iters: Optional[int] = 0, warmup_lr_init: Optional[int] = 0.,
                 last_epoch: Optional[int] = -1, verbose: Optional[bool] = False):

        super(LinearLR, self).__init__(optimizer, decay_factor=decay_factor, num_decay_iters=num_decay_iters,
                                       power=1.0, warmup_iters=warmup_iters, warmup_lr_init=warmup_lr_init,
                                       last_epoch=last_epoch, verbose=verbose)


class ExponentialLR(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, decay_factor: float, decay_start: int, decay_steps: int,
                 warmup_iters: Optional[int] = 0, warmup_lr_init: Optional[int] = 0.,
                 last_epoch: Optional[int] = -1, verbose: Optional[bool] = False):

        assert decay_start > warmup_iters

        self.decay_start = decay_start
        self.decay_steps = decay_steps
        self.gamma = math.exp(math.log(decay_factor) / float(decay_steps))

        self.warmup_iters = warmup_iters
        self.warmup_lr_init = warmup_lr_init

        super(ExponentialLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters and self.warmup_iters > 0:
            return [
                get_warmup_lr(self.last_epoch, self.warmup_iters, self.warmup_lr_init, base_lr)
                for base_lr in self.base_lrs
            ]

        current_lrs = [group['lr'] for group in self.optimizer.param_groups]

        if self.decay_start <= self.last_epoch < self.decay_start + self.decay_steps:
            return [lr * self.gamma for lr in current_lrs]
        else:
            return current_lrs


class CosineLR(lr_schedulers._LRScheduler):
    def __init__(self, optimizer, cycle_length: int, cycle_end_lr: Optional[float] = 0.0, t_mul: Optional[float] = 1.0,
                 warmup_iters: Optional[int] = 0, warmup_lr_init: Optional[int] = 0.,
                 last_epoch: Optional[int] = -1, verbose: Optional[bool] = False):

        self.cycle_length = cycle_length - 1
        self.cycle_end_lr = cycle_end_lr
        self.t_mul = t_mul

        self.warmup_iters = warmup_iters
        self.warmup_lr_init = warmup_lr_init

        self.current_cycle_index = 0

        super(CosineLR, self).__init__(optimizer, last_epoch, verbose)
        self.current_cycle_start_lrs = self.base_lrs

    def interp_cosine(self):
        assert self.current_cycle_index <= self.cycle_length
        factor = float(self.current_cycle_index) / self.cycle_length
        cosine = math.cos(factor * math.pi * 0.5)
        return [self.cycle_end_lr + ((start_lr - self.cycle_end_lr) * cosine) for start_lr in self.current_cycle_start_lrs]

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters and self.warmup_iters > 0:
            return [
                get_warmup_lr(self.last_epoch, self.warmup_iters, self.warmup_lr_init, base_lr)
                for base_lr in self.base_lrs
            ]

        # current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        new_lrs = self.interp_cosine()
        self.current_cycle_index += 1

        if self.current_cycle_index > self.cycle_length:
            self.current_cycle_index = 0
            self.current_cycle_start_lrs = [lr * self.t_mul for lr in self.current_cycle_start_lrs]

        return new_lrs