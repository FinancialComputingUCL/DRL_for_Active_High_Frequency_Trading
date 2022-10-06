import multiprocessing

import yaml
from pathos.multiprocessing import ProcessPool
from stable_baselines3 import PPO

from data_manager import *
from environment import StockEnv


def run_single_env(j, day, args, model):
    '''
    Run a single environment for evaluation
    @param j: What iteration of a specific environment is evaluated
    @param day: What day to evaluate
    @param model: The model to evaluate
    @return: Stats from how the agent acted
    '''

    data = Data(args)
    file_name, test_files = data.load_test_file(day)

    n_files = len(os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data')))

    env = StockEnv(test_files[0], test_files[1], True, args)
    state = env.reset()

    env_steps = env_reward = env_pos = 0
    profit_per_trade = []

    while True:
        env_steps += 1

        action, _ = model.predict(state)
        state, reward, done, obs = env.step(action)
        env_pos += obs['closed']
        env_reward += reward
        if obs['closed']:
            profit_per_trade += [[reward, obs['open_pos'], obs['closed_pos'], obs['position'], obs['action']]]

        if done:
            break

    return [[file_name, j, env_steps, env_pos, profit_per_trade, env_reward]], n_files * j + day


def eval_agent(args, save_directory):
    save_dir = 'runs_results/' + save_directory
    os.makedirs(save_dir + '/', exist_ok=True)

    model = PPO.load(os.path.join('runs/' + save_directory, 'agent'), device='cpu')

    with open(os.path.join(save_dir, 'parameters.yaml'), 'w') as file:
        yaml.dump(args._get_kwargs(), file)

    n_test_files = len(os.listdir(os.path.join(os.getcwd(), args.data_dir, 'test_data')))
    jobs_to_run = n_test_files * args.eval_runs_per_env
    pool = ProcessPool(multiprocessing.cpu_count())

    for ret, n in pool.uimap(run_single_env, np.reshape([[i] * n_test_files for i in range(args.eval_runs_per_env)],
                                                        jobs_to_run), [*range(n_test_files)] * args.eval_runs_per_env,
                             [args] * jobs_to_run, [model] * jobs_to_run):
        np.save(save_dir + f'/eval{n}', np.array(ret, dtype=object), allow_pickle=True)
