import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from statistics import mean

from .NN_models import ActorCritic, RNN, Net
from .NRM.NeuralRewardMachine import NeuralRewardMachine
from .NRM.utils import eval_acceptance
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
print(device)
torch.autograd.set_detect_anomaly(True)


# max number of episodes
max_episodes   = 10000


# output of rnn
rnn_outputs = 5

# layers of rnn
num_layers  = 2

# hyper params:
hidden_size = 120 #of a2c
rnn_hidden_size = 50 #of rnn

# slidind window
slide_wind = 100

lr=0.0004

# we train the policy every num_steps
num_steps = 5
TT_policy = 5
TT_grounder = 120
grounder_epochs = 100

# we plot the graph every TTT episode
TTT = 10

# Compute the returns (of the rewards) for one episode
def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        m = masks[step].to(device)
        A = rewards[step].to(device)
        B = gamma * R * m
        R = A + B
        returns.insert(0, R)
    return returns

def pad_list(lst, desired_length):
    if len(lst) < desired_length and lst:
        lst.extend([lst[-1]] * (desired_length - len(lst)))
    return lst

def prepare_dataset(sequence_accuracy, image_trajectory, info_trajectory, TT):
    indices = list(np.argsort(sequence_accuracy))
    indices.reverse()
    indices = indices[:int(TT/2)]

    worst_trajectories = [image_trajectory[i] for i in indices]
    worst_related_info = [info_trajectory[i] for i in indices]
    return worst_trajectories, worst_related_info


def recurrent_A2C(env, path, experiment, method, feature_extraction):
    #recurrency =
    #       - 'rnn'     (rnn+A2C)
    #       - 'nrm'     (grounding+A2C)
    #       - 'rm'    (reward machines)

    #################### reinitialize files
    f = open(path+"/train_rewards_"+str(experiment)+".txt", "w")
    f.close()

    #################### env dimensions
    # number of actions
    num_outputs = env.action_space.n
    # size of the state vector
    params = []
    if feature_extraction:
        cnn = Net().to(device)
        cnn.double()
        CNN_output_size = 16
        num_inputs = CNN_output_size
        params += list(cnn.parameters())
    else:
        num_inputs = env.state_space_size

    # Initializing the Actor critic model
    if method == "rnn":
        model = ActorCritic(rnn_hidden_size, num_outputs, hidden_size).to(device)
    else:#"nrm" o "rm"
        model = ActorCritic(num_inputs + env.automaton.num_of_states, num_outputs, hidden_size).to(device)
    params += list(model.parameters())
    model.double()

    if method == "rnn":
        rnn=RNN(num_inputs, rnn_hidden_size, num_layers).to(device)
        rnn.double()
        params += list(rnn.parameters())
    elif method == "nrm":
        f = open(path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "w")
        f.close()
        f = open(path + "/image_classification_accuracy_" + str(experiment) + ".txt", "w")
        f.close()

        num_of_states, num_of_symbols, num_automaton_outputs, transition_function, automaton_rewards = env.get_automaton_specs()
        if env.state_type == "symbolic":
            dataset = "minecraft_location"
        elif env.state_type == "image":
            dataset = "minecraft_image"
        grounder = NeuralRewardMachine(num_of_states, num_of_symbols, num_automaton_outputs, num_exp=experiment,
                                       log_dir=path, dataset=dataset)
        grounder.deepAutoma.initFromDfa(transition_function, automaton_rewards)
        grounder.deepAutoma.double()
        grounder.deepAutoma.to(device)
        grounder.classifier.double()
        grounder.classifier.to(device)

        image_traj = []
        rew_traj = []
        sum_rew_traj = []
        info_traj = []

        sequence_accuracy = []
        image_accuracy = []

    optimizer = optim.Adam(params, lr=lr)

    # re-initialize episodes
    episode_idx = 0


    advantage_cat = torch.tensor([]).to(device)
    log_probs_cat = torch.tensor([]).to(device)


    all_mean_rewards = []
    all_mean_rewards_averaged = []
    while episode_idx < max_episodes:
        episode_rewards = []
        done = False
        truncated = False
        obs, reward, info = env.reset()

        if method == "rm":
            state_dfa = torch.DoubleTensor(obs[0]).to(device)
            state_env = torch.DoubleTensor(obs[1]).to(device).unsqueeze(0)
            if feature_extraction:
                state_env = cnn(state_env.view(-1, 3, 64, 64))
            state = torch.cat((state_env, state_dfa.unsqueeze(0)), 1).squeeze()
        else:
            state = torch.DoubleTensor(obs).to(device)

            if method == "rnn":
                # Initialize hidden and cell states
                h_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).double()
                c_0 = torch.zeros(num_layers, rnn_hidden_size).to(device).double()
            elif method == "nrm":
                # initialize deep automa state
                state_automa = np.zeros( num_of_states)
                state_automa[0] = 1.0
                state_automa = torch.tensor(state_automa).to(device)

            raw_state = state
            if feature_extraction:
                state = cnn(state.view(-1, 3, 64, 64))
                state = state.squeeze()

        #first step with RNN or dfa
        if method == "rnn":
            out, (h_0, c_0) = rnn(state.unsqueeze(0), h_0, c_0)
            state = out
        elif method == "nrm":
            state_grounding = grounder.classifier(raw_state.unsqueeze(0))
            next_state_automa, reward_automa = grounder.deepAutoma.step(state_automa.unsqueeze(0), state_grounding, 1.0)
            state_automa = next_state_automa

            state = torch.cat((state.unsqueeze(0), state_automa), dim=-1)

            state = state.squeeze()

        if method == "nrm":
                curr_traj = []
                curr_rew = []
                curr_info = []

                curr_traj.append(raw_state)
                curr_rew.append(reward)
                curr_info.append(info)
        while not (done or truncated):
            log_probs = []
            values = []
            rewards = []
            masks = []
            entropy = 0

            # rollout trajectory
            for _ in range(num_steps):
                #state = torch.tensor(state, dtype=torch.float32)
                state = torch.unsqueeze(state, 0)

                state = state.to(device)

                #elif method == "nrm":
                #    grounder..
                #    deep_automa

                dist, value = model(state)
                action = dist.sample()

                next_state, reward, done, truncated, info = env.step(action.item())

                if method == "rm":
                    state_dfa = torch.DoubleTensor(next_state[0]).to(device)
                    state_env = torch.DoubleTensor(next_state[1]).unsqueeze(0).to(device)
                    if feature_extraction:
                        state_env = cnn(state_env.view(-1, 3, 64, 64))
                    next_state = torch.cat((state_env, state_dfa.unsqueeze(0)), 1).squeeze()
                else:
                    next_state = torch.DoubleTensor(next_state).to(device)

                    # if method == "rm":
                    #   ...dividi next_state in stato ambiente da stato dfa
                    #   if feature extraction:
                    #       stato_env = cnn(stato_env)
                    #   next_state = concat(stato_env, stato_dfa)
                    raw_state = next_state
                    if feature_extraction:
                       next_state = cnn(next_state.view(-1, 3, 64, 64))
                       next_state = next_state.squeeze()

                    # first step with RNN or dfa
                    if method == "rnn":
                        out, (h_0, c_0) = rnn(next_state.unsqueeze(0), h_0, c_0)
                        next_state = out
                    elif method == "nrm":
                        state_grounding = grounder.classifier(raw_state.unsqueeze(0))

                        next_state_automa, reward_automa = grounder.deepAutoma.step(state_automa, state_grounding, 1.0)
                        state_automa = next_state_automa

                        next_state = torch.cat((next_state.unsqueeze(0), next_state_automa), dim=-1)
                        next_state = state.squeeze()

                        curr_traj.append(raw_state)
                        curr_rew.append(reward)
                        curr_info.append(info)

                state = next_state

                # now we store the values
                log_prob = dist.log_prob(action)
                entropy += dist.entropy().mean()
                log_prob = torch.unsqueeze(log_prob, 0)
                log_probs.append(log_prob)

                # storing the value retrieved from the Critic
                values.append(value)

                # storing the reward retrived from the enbv
                reward = float(reward)
                episode_rewards.append(reward)
                reward = np.expand_dims(reward, axis=0)
                reward = np.expand_dims(reward, axis=0)
                reward = torch.tensor(reward)
                rewards.append(reward)

                formask = 1 if done is True else 0
                formask = np.expand_dims(formask, axis=0)
                formask = np.expand_dims(formask, axis=0)
                formask = torch.tensor(formask)

                masks.append(formask)

                if done or truncated:
                    break

            dist, next_value = model(next_state)

            returns = compute_returns(next_value, rewards, masks)

            log_probs = torch.cat(log_probs)
            returns = torch.cat(returns)
            values = torch.cat(values)
            values = values.reshape((values.size()[0], 1))
            log_probs = log_probs.to(device)
            returns = returns.to(device)

            advantage = returns - values

            log_probs_cat = torch.cat((log_probs_cat, log_probs), 0)
            advantage_cat = torch.cat((advantage_cat, advantage), 0)

            torch.cuda.empty_cache()

        #EPISODIO FINITO
        episode_idx +=1
        all_mean_rewards.append(np.sum(np.array(episode_rewards)))
        all_mean_rewards_averaged.append(mean(all_mean_rewards[-slide_wind:]))

        if method == "nrm":
            # calcolare le accuracy su curr_tray curr_info e scriverla su un file
            curr_traj_t = torch.stack(curr_traj).unsqueeze(0)
            curr_info_t = torch.LongTensor(curr_info).unsqueeze(0)
            acc = eval_acceptance(grounder.classifier, grounder.deepAutoma, env.automaton.alphabet,
                                  ([curr_traj_t], [curr_info_t]), automa_implementation="logic_circuit")
            sequence_accuracy.append(acc)
            with open(path + "/sequence_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(acc))
            if env.state_type == "symbolic":
                acc, _ = grounder.eval_image_classification()
            else:
                acc = 0
            image_accuracy.append(acc)
            with open(path + "/image_classification_accuracy_" + str(experiment) + ".txt", "a") as f:
                f.write("{}\n".format(acc))

            curr_traj = pad_list(curr_traj, env.max_num_steps + 1)
            curr_rew = pad_list(curr_rew, env.max_num_steps + 1)
            curr_info = pad_list(curr_info, env.max_num_steps + 1)

            image_traj.append(curr_traj)
            rew_traj.append(curr_rew)
            info_traj.append(curr_info)

        if episode_idx % TT_policy == 0:
            log_probs_cat = torch.unsqueeze(log_probs_cat, dim=1)
            actor_loss = -(log_probs_cat * advantage_cat).mean()
            critic_loss = advantage_cat.pow(2).mean()
            loss = 0.3 * actor_loss + 0.5 * critic_loss - 0.0001 * entropy

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            log_probs_cat = torch.tensor([]).to(device)
            advantage_cat = torch.tensor([]).to(device)
            #h_0 = h_0.detach()

        if method == "nrm":
            if episode_idx % TT_grounder == 0:
                worst_trajectories, worst_related_info = prepare_dataset(sequence_accuracy[-TT_grounder:], image_traj,
                                                                         info_traj, TT_grounder)
                grounder.set_dataset(worst_trajectories, worst_related_info)

                grounder.train_symbol_grounding(grounder_epochs)

                image_traj = []
                rew_traj = []
                info_traj = []
                sum_rew_traj = []

        if episode_idx % TTT == 0 and len(all_mean_rewards) >= 100:
            ## plot rewards
            plt.plot([i for i in range(len(all_mean_rewards))], all_mean_rewards_averaged)
            plt.axhline(y=env.max_reward, color='r', linestyle='--')
            plt.xlabel("episode")
            plt.ylabel("mean episode rewards")
            plt.savefig(path + "/ImageEnvMeanRewardsReal_" + str(experiment) + ".png")
            plt.clf()
            plt.close()

        # else:
        ep_reward = all_mean_rewards[-1]
        f = open(path + "/train_rewards_" + str(experiment) + ".txt", "a")
        f.write(str(ep_reward) + "\n")
        f.close()
        if episode_idx % 100 == 0:
            print("Mean cumulative reward in the last {} episodes: {}".format(slide_wind,
                                                                              mean(all_mean_rewards[-slide_wind:])))

        if len(all_mean_rewards) >= 100 and all_mean_rewards_averaged[-1] == 100:
            episode_idx = max_episodes




