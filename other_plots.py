import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv

def pad_list(lst, l):
    if len(lst) < l:
        num_to_pad = l - len(lst)
        padded_list = lst + [lst[-1]] * num_to_pad
        return padded_list
    else:
        return lst

def plot(source_1, source_2, task_category, destination, num_exp):

    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    results_1 = []
    results_2 = []

    for idx, formula in enumerate(formulas):
        if idx in task_category:
            print(formula[2])
            path_1 = os.path.join(source_1, f"task{idx+1}")
            path_2 = os.path.join(source_2, formula[2])

            for exp in range(num_exp):
                with open("{}/train_rewards_{}.txt".format(path_1, exp), "r") as f:
                    lines_1 = f.readlines()
                lines_1 = [float(line.strip()) for line in lines_1]
                lines_1 = pad_list(lines_1, 10000)
                lines_1 = np.convolve(lines_1, np.ones(100)/100, mode='valid')
                results_1.append(lines_1)

                with open("{}/train_rewards_{}.txt".format(path_2, exp), "r") as f:
                    lines_2 = f.readlines()
                lines_2 = [float(line.strip()) for line in lines_2]
                lines_2 = np.convolve(lines_2, np.ones(100)/100, mode='valid')
                results_2.append(lines_2)
    
    results_1 = np.array(results_1)
    results_2 = np.array(results_2)

    df1 = None
    df2 = None

    df1 = pd.DataFrame(results_1).melt()
    df2 = pd.DataFrame(results_2).melt()

    sns.lineplot(x="variable", y="value", data=df1, label = "NRM+A2C")
    sns.lineplot(x="variable", y="value", data=df2, label = "RNN+A2C")

    plt.title("Image enviroment, second task class", fontsize=17)
    plt.axhline(y=max_reward, color='r', linestyle='--')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Rewards", fontsize=17)
    plt.legend(loc = "upper left", fontsize=16)
    plt.savefig(destination+"/second_class_nrm_vs_rnn_image.png")
    plt.clf()

def triple_plot(
    source_1,
    source_2,
    source_3,
    task_category,
    destination,
    num_exp,
    max_len=10000,
    smooth_window=100,
    stride=1,  # set >1 (e.g. 5 or 10) to further speed up / declutter
):
    """
    Plot smoothed training rewards for three methods over multiple tasks.

    Parameters
    ----------
    source_1, source_2, source_3 : str
        Base directories for the three methods (NRM+A2C, RNN+A2C, RM+A2C).
    task_category : iterable of int
        Indices of `formulas` to plot.
    destination : str
        Folder where the plots will be saved.
    num_exp : int
        Number of seeds / experiments.
    max_len : int
        Pad / truncate reward curves to this length before smoothing.
    smooth_window : int
        Size of the moving average window.
    stride : int
        Keep one every `stride` points when plotting (for speed). 1 = no downsample.
    """

    # environment / max reward as in your original code
    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    smooth_kernel = np.ones(smooth_window, dtype=float) / smooth_window

    def load_smoothed_rewards(base_path: str) -> np.ndarray:
        """Load, pad, smooth all runs for a given method."""
        all_results = []

        for exp in range(num_exp):
            reward_path = os.path.join(base_path, f"train_rewards_{exp}.txt")

            # load as 1D float array
            rewards = np.loadtxt(reward_path, dtype=float)
            rewards = np.atleast_1d(rewards).astype(float)

            # pad / truncate to max_len
            if rewards.size < max_len:
                rewards = np.pad(rewards, (0, max_len - rewards.size), mode="edge")
            elif rewards.size > max_len:
                rewards = rewards[:max_len]

            # moving average smoothing
            smoothed = np.convolve(rewards, smooth_kernel, mode="valid")
            all_results.append(smoothed)

        return np.vstack(all_results)  # shape: [num_exp, T]

    title_count = 0
    correct_indices = [1,2,3,5,7,8,9,10]

    for idx, formula in enumerate(formulas):
        if idx not in task_category:
            continue

        print(formula[2])

        path_1 = os.path.join(source_1, formula[2])
        path_2 = os.path.join(source_2, formula[2])
        path_3 = os.path.join(source_3, formula[2])

        results_1 = load_smoothed_rewards(path_1)  # [num_exp, T]
        results_2 = load_smoothed_rewards(path_2)
        results_3 = load_smoothed_rewards(path_3)

        # all have same T because we pad + same smoothing
        T = results_1.shape[1]
        x = np.arange(T)

        # average + std over seeds
        mean1, std1 = results_1.mean(axis=0), results_1.std(axis=0)
        mean2, std2 = results_2.mean(axis=0), results_2.std(axis=0)
        mean3, std3 = results_3.mean(axis=0), results_3.std(axis=0)

        # optional downsampling for speed/clarity
        if stride > 1:
            x = x[::stride]
            mean1, std1 = mean1[::stride], std1[::stride]
            mean2, std2 = mean2[::stride], std2[::stride]
            mean3, std3 = mean3[::stride], std3[::stride]

        plt.figure()
        # method 1
        plt.plot(x, mean1, label="NRM+A2C")
        plt.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.2)

        # method 2
        plt.plot(x, mean2, label="RNN+A2C")
        plt.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.2)

        # method 3
        plt.plot(x, mean3, label="RM+A2C")
        plt.fill_between(x, mean3 - std3, mean3 + std3, alpha=0.2)

        plt.title(f"Image enviroment, task{title_count+1}", fontsize=17)
        plt.axhline(y=max_reward, color="r", linestyle="--")
        plt.tick_params(axis="both", which="both", labelsize=12)
        plt.xlabel("Episodes", fontsize=17)
        plt.ylabel("Rewards", fontsize=17)
        plt.legend(loc="lower right", fontsize=16)

        out_path = os.path.join(destination, f"task{correct_indices[title_count]}_nrm_vs_rnn_vs_rm_image.png")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()

        title_count += 1
    
def plot_sequence(
    source_1,
    category,
    destination,
    num_exp,
    max_len=10000,
    smooth_window=30,
    stride=1,  # set e.g. to 5 or 10 to speed up and declutter
):
    """
    Plot smoothed reward prediction accuracy over episodes, aggregated across
    tasks in the chosen category and across seeds.

    Parameters
    ----------
    source_1 : str
        Base directory containing task folders (task1, task2, ...).
    category : str
        "first" or "second" (selects which tasks to aggregate).
    destination : str
        Directory where the output plot is saved.
    num_exp : int
        Number of seeds / experiments per task.
    max_len : int
        Pad / truncate accuracy curves to this length before smoothing.
    smooth_window : int
        Moving average window size.
    stride : int
        Keep one every `stride` points when plotting.
    """

    if category == "first":
        task_category = [0, 1, 2, 3]
    elif category == "second":
        task_category = [4, 5, 6, 7]
    else:
        raise ValueError(f"Unknown category: {category}")

    # env / max_reward as in your original code
    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    smooth_kernel = np.ones(smooth_window, dtype=float) / smooth_window
    all_curves = []

    for idx, formula in enumerate(formulas):
        if idx not in task_category:
            continue

        print(formula[2])
        path_1 = os.path.join(source_1, formula[2])

        for exp in range(num_exp):
            acc_path = os.path.join(path_1, f"sequence_classification_accuracy_{exp}.txt")

            # Load as float array
            acc = np.loadtxt(acc_path, dtype=float)
            acc = np.atleast_1d(acc).astype(float)

            # pad / truncate to max_len
            if acc.size < max_len:
                acc = np.pad(acc, (0, max_len - acc.size), mode="edge")
            elif acc.size > max_len:
                acc = acc[:max_len]

            # moving average smoothing
            smoothed = np.convolve(acc, smooth_kernel, mode="valid")
            all_curves.append(smoothed)

    if not all_curves:
        print("No curves found for the selected category; nothing to plot.")
        return

    results_1 = np.vstack(all_curves)  # shape: [num_curves, T]
    T = results_1.shape[1]
    x = np.arange(T)

    mean_acc = results_1.mean(axis=0)
    std_acc = results_1.std(axis=0)

    # Optional downsampling
    if stride > 1:
        x = x[::stride]
        mean_acc = mean_acc[::stride]
        std_acc = std_acc[::stride]

    plt.figure()
    plt.plot(x, mean_acc, label="Mean accuracy")
    plt.fill_between(x, mean_acc - std_acc, mean_acc + std_acc, alpha=0.2)

    plt.title(f"Image enviroment, {category} task class", fontsize=17)
    plt.axhline(y=max_reward, color="r", linestyle="--")
    plt.tick_params(axis="both", which="both", labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Reward prediction accuracy", fontsize=17)
    # no legend originally, but we could enable it if you like
    # plt.legend(loc="lower right", fontsize=16)

    out_path = os.path.join(destination, f"{category}_class_sequence_classification_image.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

SOURCE_PATH_NRM  = "Results/Results_NRM_Three_NoBuff"
SOURCE_PATH_RNN  = "Results/Results_RNN_Three_NoBuff"
SOURCE_PATH_RM   = "Results/Results_RM_Three_NoBuff"
DESTINATION_PATH = "Plots/"

TASKS = [0, 1, 2, 3, 4, 5, 6, 7]

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH)

triple_plot(SOURCE_PATH_NRM, SOURCE_PATH_RNN, SOURCE_PATH_RM, TASKS, DESTINATION_PATH, 5)
plot_sequence(SOURCE_PATH_NRM, "first", DESTINATION_PATH, 5)
plot_sequence(SOURCE_PATH_NRM, "second", DESTINATION_PATH, 5)