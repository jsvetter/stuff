# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%

ecg_df = pd.read_csv("../assets/ECG_SegData.csv")

# turn into numpy array
ecg_data = ecg_df.to_numpy()
print(ecg_data.shape)

# %%
good_ecgs_to_plot = [2000]  # , 2021, 2042, 2063, 2105]
also_good_ecgs = [1, 22, 43, 64, 106]

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))
    for i in good_ecgs_to_plot:
        ax.plot(ecg_data[i, :], color="C0")

    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig("../outputs/ecg_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

# %%
# %%
good_ecgs_to_plot = [2000, 2021, 2042, 2063, 2105]
colors = ["C0", "royalblue", "cornflowerblue", "dodgerblue", "blue"]

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))
    for i, color in zip(good_ecgs_to_plot, colors):
        ax.plot(ecg_data[i, :], color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig("../outputs/ecgs_blue_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()


also_good_ecgs = [1, 22, 43, 64, 106]
colors = ["C3", "firebrick", "tomato", "orangered", "maroon"]

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))
    for i, color in zip(also_good_ecgs, colors):
        ax.plot(ecg_data[i, :], color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig("../outputs/ecgs_reds_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

# %%
eegs = np.load("../assets/EEG_data_diffusion_paper.npy")
disentangle_for_plotting = (
    20 * np.arange(eegs.shape[1])[None, :, None]
)  # shape (1, n_channels, 1)
eegs = eegs + disentangle_for_plotting  # shape (n_samples, n

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(eegs[0, :, :].T, color="C0")
    ax.plot(np.arange(100, 260), eegs[0, 20, 100:], color="white")
    ax.plot(np.arange(100, 260), eegs[0, 21, 100:], color="white")
    ax.plot(np.arange(100, 260), eegs[0, 22, 100:], color="white")
    ax.plot(np.arange(100, 260), eegs[0, 23, 100:], color="white")
    ax.plot(np.arange(100, 260), eegs[0, 24, 100:], color="white")
    ax.plot(np.arange(100, 260), eegs[0, 25, 100:], color="white")

    ax.plot(eegs[0, 19, :], color="C0")
    ax.plot(eegs[0, 26, :], color="C0")
    # for i in also_good_ecgs:
    #     ax.plot(ecg_data[i, :])

    # Remove tick labels (only show units)
    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    # draw 10% of old x_lim as new bounds
    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.2
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )
    # ax.set_xlim(0, 500)

    fig.savefig("../outputs/eeg_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()

print(eegs.shape)


# %%
with open(
    "../assets/allen_support_files/ephys_cell_569469018_sweep_number_44.pkl", "rb"
) as f:
    trace = pickle.load(f)

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))

    ax.plot(trace[0].flatten(), color="C0")
    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig("../outputs/spike_plot.pdf", bbox_inches="tight", transparent=True)
    plt.show()


# %%
synth_traces = []
for i in range(2, 3):
    with open(f"../assets/allen_support_files/synthetic_obs_{i}.pkl", "rb") as f:
        trace = pickle.load(f)
        synth_traces.append(trace["data"].flatten())

with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))

    for trace in synth_traces:
        ax.plot(trace, color="black")

    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig(
        "../outputs/synthetic_spike_plot.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()

# %%
synth_traces = []
for i in range(1, 9):
    with open(f"../assets/allen_support_files/synthetic_obs_{i}.pkl", "rb") as f:
        trace = pickle.load(f)
        synth_traces.append(trace["data"].flatten())

colors = [
    "silver",
    "dimgrey",
    "slategrey",
    "darkslategrey",
    "grey",
    "black",
    "lightslategrey",
    "darkgray",
]


with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(4, 2))

    for trace, color in zip(synth_traces, colors):
        ax.plot(trace, color=color)

    ax.set_xticks([])
    ax.set_yticks([])
    x_lim_low, x_lim_high = ax.get_xlim()
    y_lim_low, y_lim_high = ax.get_ylim()

    ax.spines["left"].set_bounds(
        low=y_lim_low, high=y_lim_low + (y_lim_high - y_lim_low) * 0.4
    )
    ax.spines["bottom"].set_bounds(
        low=x_lim_low, high=x_lim_low + (x_lim_high - x_lim_low) * 0.2
    )

    fig.savefig(
        "../outputs/synthetic_spikes_plot.pdf", bbox_inches="tight", transparent=True
    )
    plt.show()


# %%
np.random.seed(0)
res = np.random.uniform(0, 1, size=8)
ras = np.random.uniform(0, 1, size=8)


with plt.rc_context(fname="../matplotlibrc"):
    fig, ax = plt.subplots(figsize=(2.5, 2.5))

    ax.scatter(res, ras, color=colors, s=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig("../outputs/param_scatter.pdf", bbox_inches="tight", transparent=True)
    plt.show()
