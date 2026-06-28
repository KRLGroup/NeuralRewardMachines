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

    plt.title("Image environment, second task class", fontsize=17)
    plt.axhline(y=max_reward, color='r', linestyle='--')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Rewards", fontsize=17)
    plt.legend(loc = "upper left", fontsize=16)
    plt.savefig(destination+"/second_class_nrm_vs_rnn_image.png")
    plt.clf()

def triple_plot(source_1, source_2, source_3, task_category, destination, num_exp):

    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    title_count = 1
    for idx, formula in enumerate(formulas):
        if idx in task_category:
            print(formula[2])
            
            results_1 = []
            results_2 = []
            results_3 = []

            path_1 = os.path.join(source_1, formula[2])
            path_2 = os.path.join(source_2, formula[2])
            path_3 = os.path.join(source_3, formula[2])

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
                lines_2 = pad_list(lines_2, 10000)
                lines_2 = np.convolve(lines_2, np.ones(100)/100, mode='valid')
                results_2.append(lines_2)

                with open("{}/train_rewards_{}.txt".format(path_3, exp), "r") as f:
                    lines_3 = f.readlines()
                lines_3 = [float(line.strip()) for line in lines_3]
                lines_3 = pad_list(lines_3, 10000)
                lines_3 = np.convolve(lines_3, np.ones(100)/100, mode='valid')
                results_3.append(lines_3)
        
            results_1 = np.array(results_1)
            results_2 = np.array(results_2)
            results_3 = np.array(results_3)

            df1 = None
            df2 = None
            df3 = None

            df1 = pd.DataFrame(results_1).melt()
            df2 = pd.DataFrame(results_2).melt()
            df3 = pd.DataFrame(results_3).melt()

            sns.lineplot(x="variable", y="value", data=df1, label = "NRM+A2C")
            sns.lineplot(x="variable", y="value", data=df2, label = "RNN+A2C")
            sns.lineplot(x="variable", y="value", data=df3, label = "RM+A2C")

            plt.title(f"Image environment, task{title_count}", fontsize=17)
            plt.axhline(y=max_reward, color='r', linestyle='--')
            plt.tick_params(axis='both', which='both', labelsize=12)
            plt.xlabel("Episodes", fontsize=17)
            plt.ylabel("Rewards", fontsize=17)
            plt.legend(loc = "lower right", fontsize=16)
            plt.savefig(destination+f"/task{title_count}_nrm_vs_rnn_vs_rm_image.png")
            plt.clf()

            title_count+=1

def plot_sequence(source_1, task_category, destination, num_exp):

    env = GridWorldEnv(formulas[0], "rgb_array", "symbolic", use_dfa_state=False, train=False)
    max_reward = env.max_reward

    results_1 = []

    for idx, formula in enumerate(formulas):
        if idx in task_category:
            print(formula[2])
            path_1 = os.path.join(source_1, f"task{idx+1}")

            for exp in range(num_exp):
                with open("{}/sequence_classification_accuracy_{}.txt".format(path_1, exp), "r") as f:
                    lines_1 = f.readlines()
                lines_1 = [float(line.strip()) for line in lines_1]
                lines_1 = pad_list(lines_1, 10000)
                lines_1 = np.convolve(lines_1, np.ones(30)/30, mode='valid')
                results_1.append(lines_1)

    results_1 = np.array(results_1)

    df1 = None

    df1 = pd.DataFrame(results_1).melt()

    sns.lineplot(x="variable", y="value", data=df1)

    plt.title("Image environment, second task class", fontsize=17)
    plt.axhline(y=max_reward, color='r', linestyle='--')
    plt.tick_params(axis='both', which='both', labelsize=12)
    plt.xlabel("Episodes", fontsize=17)
    plt.ylabel("Reward prediction accuracy", fontsize=17)
    # plt.legend(loc = "lower right", fontsize=16)
    plt.savefig(destination+"/second_class_sequence_classification_image.png")
    plt.clf()

SOURCE_PATH_RNN = "Results/RNN_CNN/"
SOURCE_PATH_GROUND = "Results/GROUNDED_CNN"
SOURCE_PATH_RM = "Results/RM_CNN"
DESTINATION_PATH = "Plots/Single_Tasks"

TASKS = [0, 1, 2, 4, 6, 7, 8, 9]
# TASKS_1_2_3_5 = [0, 1, 2, 4, 6, 7, 8, 9]
# TASKS_7_8_9_10 = []

if not os.path.exists(DESTINATION_PATH):
    os.makedirs(DESTINATION_PATH+"/")

# plot(SOURCE_PATH_GROUND, SOURCE_PATH_RNN_LOC, TASKS_7_8_9_10, DESTINATION_PATH, 5)
# plot_sequence(SOURCE_PATH_GROUND, TASKS_7_8_9_10, DESTINATION_PATH, 5)
triple_plot(SOURCE_PATH_GROUND, SOURCE_PATH_RNN, SOURCE_PATH_RM, TASKS, DESTINATION_PATH, 5)