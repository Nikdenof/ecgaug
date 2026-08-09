from typing import Callable

import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


class AddGaussianNoise:
    def __init__(self, mu: float = 0.0, sigma: float = 0.01, p: float = 0.5):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, signal):
        if np.random.random() < self.p:
            noise = np.random.normal(self.mu, self.sigma, signal.shape)
            signal = signal.copy() + noise

        return signal


class RandomYShift:
    """
    I think this augmentation should be applied in latter line of any aug pipeline
    as it can potentially mess up underlying lead relations.
    """

    def __init__(self, min_shift: float = -0.5, max_shift: float = 0.5, p: float = 0.5):
        self.min_shift = min_shift
        self.max_shift = max_shift
        self.p = p

    def __call__(self, signal, verbose=False):
        y_shift_value = np.random.uniform(self.min_shift, self.max_shift)
        if verbose:
            print(y_shift_value)

        if np.random.random() < self.p:
            signal = signal.copy() + y_shift_value

        return signal


class Compose:
    def __init__(self, augs: list[Callable]):
        self.augs = augs

    def __call__(self, signal):
        if not len(self.augs):
            print("Warning: No augs")

        for aug in self.augs:
            signal = aug(signal)

        return signal


def main():
    ecg12: pd.DataFrame = nk.ecg_simulate(
        duration=11, method="multileads", sampling_rate=500
    )

    ecg12[0:5000].plot(subplots=True)
    plt.show(block=False)

    ecg = ecg12[0:5000].to_numpy()

    aug = AddGaussianNoise(p=0.0)
    ecg_augmented = aug(ecg)
    np.array_equal(ecg, ecg_augmented)

    aug = AddGaussianNoise(p=1.0)
    ecg_augmented = aug(ecg)
    np.array_equal(ecg, ecg_augmented)

    aug = RandomYShift(p=1.0)
    ecg_augmented = aug(ecg, verbose=True)
    np.array_equal(ecg, ecg_augmented)

    ecg[:10, 0]
    ecg_augmented[:10, 0]


    ecg[0, 0]
    ecg_augmented[0, 0]

    aug_pipeline = Compose(
        [
            AddGaussianNoise(mu=0.0, sigma=0.03, p=1.0),
            RandomYShift(-0.3, 0.3, p=1.0),
        ]
    )
    ecg_augmented = aug_pipeline(ecg)
    # Lead II comparisson
    plt.plot(ecg[:, 0], label='base')
    plt.plot(ecg_augmented[:, 0], label='aug')
    plt.legend()
    plt.show()

    print(type(ecg_augmented), ecg_augmented.shape)
    pd.DataFrame(ecg_augmented).plot(subplots=True)
    plt.show()


    # ecg_tf = ecg_base
    # ecg_torch = np.swapaxes(ecg_base, 1, 0)  # or transpose


if __name__ == "__main__":
    main()
