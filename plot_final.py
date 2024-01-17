import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from new_LTL_tasks import formulas
from RL.Env.Environment import GridWorldEnv

def plot(source_1, source_2, task_category, destination, num_exp):

    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    results_1 = []
    results_2 = []

    for idx, formula in enumerate(formulas):
        if idx in task_category:
            print(formula[2])
            path_1 = os.path.join(source_1, formula[2])
            path_2 = os.path.join(source_2, formula[2])

            for exp in range(num_exp):
                with open("{}/train_rewards_{}.txt".format(path_1, exp), "r") as f:
                    lines_1 = f.readlines()
                lines_1 = [float(line.strip()) for line in lines_1]
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

    plt.title("Map enviroment, first task class", fontsize=17)
    plt.axhline(y=max_reward, color='r', linestyle='--')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Rewards", fontsize=17)
    plt.legend(loc = "lower right", fontsize=16)
    plt.savefig(destination+"/first_class_nrm_vs_rnn_map.png")
    plt.clf()

def plot_sequence(source_1, task_category, destination, num_exp):

    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    results_1 = []

    for idx, formula in enumerate(formulas):
        if idx in task_category:
            print(formula[2])
            path_1 = os.path.join(source_1, formula[2])

            for exp in range(num_exp):
                with open("{}/sequence_classification_accuracy_{}.txt".format(path_1, exp), "r") as f:
                    lines_1 = f.readlines()
                lines_1 = [float(line.strip()) for line in lines_1]
                lines_1 = np.convolve(lines_1, np.ones(30)/30, mode='valid')
                results_1.append(lines_1)

    results_1 = np.array(results_1)

    df1 = None

    df1 = pd.DataFrame(results_1).melt()

    sns.lineplot(x="variable", y="value", data=df1)

    plt.title("Map enviroment, second task class", fontsize=17)
    plt.axhline(y=max_reward, color='r', linestyle='--')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Reward prediction accuracy", fontsize=17)
    # plt.legend(loc = "lower right", fontsize=16)
    plt.savefig(destination+"/second_class_sequence_classification_map.png")
    plt.clf()

SOURCE_PATH_RNN_LOC = "Results/RNN_MAP/"
SOURCE_PATH_GROUND = "Results/GROUNDED_MAP"
DESTINATION_PATH = "Plots/"

TASKS_1_2_3_5 = [0, 1, 2, 4]
TASKS_7_8_9_10 = [6, 7, 8, 9]

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH+"/")

# plot(SOURCE_PATH_GROUND, SOURCE_PATH_RNN_LOC, TASKS_1_2_3_5, DESTINATION_PATH, 5)
plot_sequence(SOURCE_PATH_GROUND, TASKS_7_8_9_10, DESTINATION_PATH, 5)
    