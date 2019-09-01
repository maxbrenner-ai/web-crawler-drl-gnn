import time

from PPO_agent import PPOAgent
from environments.nervenet import Environment
from utils import *
from models.nervenet import NerveNet_GNN
from models.no_stucture import NoStructure_baseline
from models.fully_connected import FullyConnected_baseline


def make_model(model_type, episode_C, model_C, goal_C, agent_C, other_C, device):
    if model_type == 'nervenet':
        return NerveNet_GNN(model_C['node_feat_size'], model_C['node_hidden_size'],
                     model_C['message_size'], model_C['output_size'],
                     goal_C['goal_size'], goal_C['goal_opt'], agent_C['critic_agg_weight'],
                     device).to(device)
    elif model_type == 'no_structure':
        return NoStructure_baseline(model_C['node_feat_size'], model_C['node_hidden_size'],
                                    goal_C['goal_size'], goal_C['goal_opt'], agent_C['critic_agg_weight'],
                                    device).to(device)
    elif model_type == 'fully_connected':
        return FullyConnected_baseline(model_C['node_feat_size'], model_C['node_hidden_size'],
                                       model_C['message_size'], goal_C['goal_size'], goal_C['goal_opt'],
                                       agent_C['critic_agg_weight'], device).to(device)
    assert True == False, 'didnt put in a valid model type'


# target for multiprocess
def train(id, shared_gnn, optimizer, rollout_counter, args):
    episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges = args
    local_gnn = make_model(model_C['model_type'], episode_C, model_C, goal_C, agent_C, other_C, device)
    agent = PPOAgent(args, Environment(args), shared_gnn, local_gnn, optimizer)
    train_step = 0
    rollout_times, batch_times, pred_times = [], [], []
    #     for r in range(episode_C['num_train_rollouts']+1):
    r = 0
    while True:
        agent.train_rollout(train_step)
        r += 1
        rollout_counter.increment()
        # print('Agent {} finished its rollout {} which is gobal rollout {}'.format(id, r, rollout_counter.get()))
        if rollout_counter.get() >= episode_C['num_train_rollouts'] + 1:
            return


# target for multiprocess
def eval(shared_gnn, rollout_counter, args, df):
    time_start = time.time()
    episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges = args
    local_gnn = make_model(model_C['model_type'], episode_C, model_C, goal_C, agent_C, other_C, device)
    agent = PPOAgent(args, Environment(args), shared_gnn, local_gnn, None)
    last_eval = 0
    run_info = {}  # based on the avg steps taken
    run_info['eval_ach_goal_perc'] = -1.; run_info['eval_avg_opt_steps'] = -1.; run_info['eval_avg_steps_taken'] = float('inf')
    while True:
        curr_r = rollout_counter.get()
        if curr_r % episode_C['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            avg_rew, max_rew, min_rew, ach_perc, avg_opt_steps, avg_steps_taken = agent.eval_episodes()
            # if df is None:  # Only print each one if df is none which means this isn't a hyp param search run
            print(
                'Testing summary at rollout {}: Avg ep rew: {:.2f}  Max ep rew: {}  Min ep rew: {}  Achieved goal percent: {:.2f}  Avg opt steps: {:.2f}  Avg steps taken: {:.2f}\n'.format(
                    curr_r, avg_rew, max_rew, min_rew, ach_perc, avg_opt_steps, avg_steps_taken))
            if avg_steps_taken < run_info['eval_avg_steps_taken']:
                run_info['eval_ach_goal_perc'] = ach_perc
                run_info['eval_avg_opt_steps'] = avg_opt_steps
                run_info['eval_avg_steps_taken'] = avg_steps_taken
        if curr_r >= episode_C['num_train_rollouts'] + 1:
            # Add run info to df
            if df is not None:
                def add_hyp_param_dict(append_letter, dic):
                    for k, v in list(dic.items()):
                        run_info[append_letter + '_' + k] = v
                add_hyp_param_dict('E', episode_C)
                add_hyp_param_dict('M', model_C)
                add_hyp_param_dict('G', goal_C)
                add_hyp_param_dict('A', agent_C)
                add_hyp_param_dict('O', other_C)
                time_end = time.time()
                time_taken = (time_end - time_start) / 60.
                run_info['total_time_taken(m)'] = time_taken
                df = df.append(run_info, ignore_index=True)
                df.to_excel('run-data.xlsx', index=False)

                print(
                    'Best run summary: Achieved goal percent: {:.2f}  Avg opt steps: {:.2f}  Avg steps taken: {:.2f}'.format(
                        run_info['eval_ach_goal_perc'], run_info['eval_avg_opt_steps'], run_info['eval_avg_steps_taken']))
            return


def run_nervenet(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges, df):
    shared_gnn = make_model(model_C['model_type'], episode_C, model_C, goal_C, agent_C, other_C, device)
    shared_gnn.share_memory()
    optimizer = torch.optim.Adam(shared_gnn.parameters(), agent_C['learning_rate'])
    #     optimizer.share_memory()
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []

    args = (episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges)
    # Run eval agent
    p = mp.Process(target=eval, args=(shared_gnn, rollout_counter, args, df))
    p.start()
    processes.append(p)
    # Run training agents
    for i in range(other_C['num_agents']):
        p = mp.Process(target=train, args=(i, shared_gnn, optimizer, rollout_counter, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
