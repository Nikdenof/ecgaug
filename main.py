import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class AddNoise:
    mu = 0
    sigma = 0.01

    def __call__(self, signal):
        noise = np.random.normal(self.mu, self.sigma, signal.shape)
        augmented_signal = signal + noise
        return augmented_signal


def main():
    ecg12: pd.DataFrame = nk.ecg_simulate(
        duration=11, method="multileads", sampling_rate=500
    )

    ecg12[0:5000].plot(subplots=True)
    plt.show(block=False)

    ecg = ecg12[0:5000].to_numpy()

    aug = AddNoise()
    ecg_augmented = aug(ecg)

    ecg[:10, 0]
    ecg_augmented[:10, 0]


    print(type(ecg_augmented), ecg_augmented.shape)
    pd.DataFrame(ecg_augmented).plot(subplots=True)
    plt.show()

    # ecg_tf = ecg_base
    # ecg_torch = np.swapaxes(ecg_base, 1, 0)  # or transpose


if __name__ == "__main__":
    main()
