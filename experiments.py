from LTL_tasks import formulas
import absl.flags
import absl.app
import os
from RL.NRM.utils import set_seed
from RL.Env.Environment import GridWorldEnv
from RL.A2C import recurrent_A2C
from plot import plot

#flags
absl.flags.DEFINE_string("METHOD", "rnn", "Method to test, one in ['rnn', 's4', 'nrm', 'rm'], default= 'rnn' ")
absl.flags.DEFINE_string("ENV", "map_env", "Environment to test, one in ['map_env', 'image_env'], default= 'map_env' ")
absl.flags.DEFINE_string("LOG_DIR", "Results/", "path where to save the results, default='Results/'")
absl.flags.DEFINE_integer("NUM_EXPERIMENTS", 5, "num of runs for each test, default= 5")
absl.flags.DEFINE_boolean("USE_REPLAY_BUFFER", False, "for 'nrm', train the grounder via balanced replay buffers (True) or on the worst recent trajectories (False), default= False")
absl.flags.DEFINE_string("REWARD", "three_value_acceptance", "reward scheme of the Moore machine, one in ['distance', 'acceptance', 'three_value_acceptance'], default= 'three_value_acceptance'")


FLAGS = absl.flags.FLAGS


def launch_experiments(path, formula, experiment, env_type, method, use_replay_buffer=True, reward="three_value_acceptance"):
    set_seed(experiment)

    if env_type == 'map_env':
        state_type = "symbolic"
        feature_extraction = False
    elif env_type == 'image_env':
        state_type = "image"
        feature_extraction = True

    if method == 'rnn':
        use_dfa_state = False
    elif method == 's4':
        use_dfa_state = False
    elif method == 'nrm':
        use_dfa_state = False
    elif method == 'rm':
        use_dfa_state = True

    env = GridWorldEnv(formula, "human", state_type=state_type, use_dfa_state=use_dfa_state, train=False, reward=reward)
    if not os.path.exists(path):
        os.makedirs(path)

    recurrent_A2C(env, path, experiment, method, feature_extraction, use_replay_buffer=use_replay_buffer)


def config_recap():
    # short, filesystem-friendly summary of the run configuration
    recap = f"{FLAGS.METHOD}_{FLAGS.ENV}_{FLAGS.REWARD}"
    if FLAGS.METHOD == "nrm":
        recap += "_buffer" if FLAGS.USE_REPLAY_BUFFER else "_nobuffer"
    return recap


def main(argv):
    base_dir = os.path.join(FLAGS.LOG_DIR, config_recap())
    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)
    for formula_idx, formula in enumerate(formulas):
        for experiment in range(FLAGS.NUM_EXPERIMENTS):
            print(f"Experiment {experiment} on formula {formula[2]}")
            path = os.path.join(base_dir, str(formula[2]))

            launch_experiments(path, formula, experiment, FLAGS.ENV, FLAGS.METHOD, FLAGS.USE_REPLAY_BUFFER, FLAGS.REWARD)
        plot(path, FLAGS.NUM_EXPERIMENTS, formula, 100)


if __name__ == '__main__':
    absl.app.run(main)
