import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np
import pandas as pd


def main():
    ecg12 = nk.ecg_simulate(duration=11, method="multileads", sampling_rate=500)

    ecg12[0:5000].plot(subplots=True)
    plt.show()

    ecg12.shape

    ecg_base = ecg12[0:5000].to_numpy()

    mu, sigma = 0, 0.05
    noise = np.random.normal(mu, sigma, ecg_base.shape)
    ecg_augmented = ecg_base + noise

    print(type(ecg_augmented), ecg_augmented.shape)
    pd.DataFrame(ecg_augmented).plot(subplots=True)
    plt.show()

    # ecg_tf = ecg_base
    # ecg_torch = np.swapaxes(ecg_base, 1, 0)  # or transpose


if __name__ == "__main__":
    main()
