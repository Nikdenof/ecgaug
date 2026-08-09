import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


class AddNoise:
    def __init__(self, mu: float = 0.0, sigma: float = 0.01, p: float = 0.5):
        self.mu = mu
        self.sigma = sigma
        self.p = p

    def __call__(self, signal):
        if np.random.random() < self.p:
            noise = np.random.normal(self.mu, self.sigma, signal.shape)
            signal = signal.copy() + noise

        return signal


def main():
    ecg12: pd.DataFrame = nk.ecg_simulate(
        duration=11, method="multileads", sampling_rate=500
    )

    ecg12[0:5000].plot(subplots=True)
    plt.show(block=False)

    ecg = ecg12[0:5000].to_numpy()

    aug = AddNoise(p=0.0)
    ecg_augmented = aug(ecg)
    np.array_equal(ecg, ecg_augmented)

    aug = AddNoise(p=1.0)
    ecg_augmented = aug(ecg)
    np.array_equal(ecg, ecg_augmented)

    ecg[0, 0]
    ecg_augmented[0, 0]

    print(type(ecg_augmented), ecg_augmented.shape)
    pd.DataFrame(ecg_augmented).plot(subplots=True)
    plt.show()

    # ecg_tf = ecg_base
    # ecg_torch = np.swapaxes(ecg_base, 1, 0)  # or transpose


if __name__ == "__main__":
    main()
