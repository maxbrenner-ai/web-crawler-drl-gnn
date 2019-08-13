# %matplotlib inline
import networkx as nx
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time
import pickle
from collections import OrderedDict, deque, defaultdict
from nervenet_PPO import NerveNet_GNN
from copy import deepcopy
import random
import json
import pandas as pd
from utils import *
import torch.multiprocessing as mp


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PPOAgent:
    def __init__(self, env, shared_gnn, optimizer):
        self.gnn = NerveNet_GNN(model_C['node_feat_size'], model_C['node_hidden_size'],
                                model_C['message_size'], model_C['output_size'],
                                goal_C['goal_size'], goal_C['goal_opt'], agent_C['critic_agg_weight'],
                                device).to(device)
        self.shared_gnn = shared_gnn
        self.env = env
        self.state = self.env.reset()
        self.ep_step = 0
        #         self.opt = torch.optim.Adam(self.gnn.parameters(), agent_C['learning_rate'])
        self.opt = optimizer
        self.gnn.eval()

    def _eval_episode(self, test_step):
        state = self.env.reset()
        shortest_path_length = state[4]
        ep_rew = 0
        for step in range(episode_C['max_ep_steps']):
            prediction = self.env.propogate_multi(self.gnn, [state])
            action = prediction['a'].cpu().numpy()[0]
            state = deepcopy(state)
            next_state, reward, done, achieved_goal = self.env.step(action, step, state)
            if achieved_goal: assert done
            ep_rew += reward
            test_step += 1
            if done:
                break
            state = deepcopy(next_state)
        return test_step, ep_rew, achieved_goal, shortest_path_length - 1, step + 1

    def eval_episodes(self):
        self.gnn.load_state_dict(self.shared_gnn.state_dict())
        test_step = 0
        test_info = {}
        test_info['all ep rew'] = []
        test_info['max ep rew'] = float('-inf')
        test_info['min ep rew'] = float('inf')
        test_info['achieved goal'] = []
        test_info['opt steps'] = []
        test_info['steps taken'] = []
        for ep in range(episode_C['eval_num_eps']):
            test_step, ep_rew, achieved_goal, opt_steps, steps_taken = self._eval_episode(test_step)
            test_info['all ep rew'].append(ep_rew)
            test_info['max ep rew'] = max(test_info['max ep rew'], ep_rew)
            test_info['min ep rew'] = min(test_info['min ep rew'], ep_rew)
            test_info['achieved goal'].append(achieved_goal)
            test_info['opt steps'].append(opt_steps)
            test_info['steps taken'].append(steps_taken)
        return (np.array(test_info['max ep rew']).mean(),
                test_info['max ep rew'],
                test_info['min ep rew'],
                np.array(test_info['achieved goal']).sum() / ep,
                np.array(test_info['opt steps']).mean(),
                np.array(test_info['steps taken']).mean())

    def train_rollout(self, total_step):
        storage = Storage(episode_C['rollout_length'])
        state = Environment._copy_state(*self.state)
        step_times = []
        # Sync.
        self.gnn.load_state_dict(self.shared_gnn.state_dict())
        for rollout_step in range(episode_C['rollout_length']):
            start_step_time = time.time()
            prediction = self.env.propogate_multi(self.gnn, [state])
            action = prediction['a'].cpu().numpy()[0]
            next_state, reward, done, achieved_goal = self.env.step(action, self.ep_step, state)

            self.ep_step += 1
            if done:
                # Sync local model with shared model at start of each ep
                self.gnn.load_state_dict(self.shared_gnn.state_dict())
                self.ep_step = 0

            storage.add(prediction)
            storage.add({'r': tensor(reward, device).unsqueeze(-1).unsqueeze(-1),
                         'm': tensor(1 - done, device).unsqueeze(-1).unsqueeze(-1),
                         's': state})

            state = Environment._copy_state(*next_state)

            total_step += 1

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)

        self.state = Environment._copy_state(*state)

        prediction = self.env.propogate_multi(self.gnn, [state])
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((1, 1)), device)
        returns = prediction['v'].detach()
        for i in reversed(range(episode_C['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + agent_C['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + agent_C['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
            advantages = advantages * agent_C['gae_tau'] * agent_C['discount'] * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()

        # print(returns.shape, td_error.shape, advantages.shape, storage.adv[-1].shape, storage.ret[-1].shape)

        actions, log_probs_old, returns, advantages = storage.cat(['a', 'log_pi_a', 'ret', 'adv'])
        states = [storage.s[i] for i in range(storage.size)]

        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        # Train
        self.gnn.train()
        batch_times = []
        train_pred_times = []
        for _ in range(agent_C['optimization_epochs']):
            # Sync. at start of each epoch TODO: TEST IF THIS IS OKAY!!
            self.gnn.load_state_dict(self.shared_gnn.state_dict())
            sampler = random_sample(np.arange(len(states)), agent_C['minibatch_size'])
            for batch_indices in sampler:
                start_batch_time = time.time()

                batch_indices_tensor = tensor(batch_indices, device).long()

                # Important Node: these are tensors but dont have a grad
                sampled_states = [states[i] for i in batch_indices]
                sampled_actions = actions[batch_indices_tensor]
                sampled_log_probs_old = log_probs_old[batch_indices_tensor]
                sampled_returns = returns[batch_indices_tensor]
                sampled_advantages = advantages[batch_indices_tensor]

                start_pred_time = time.time()
                prediction = self.env.propogate_multi(self.gnn, sampled_states, sampled_actions)
                end_pred_time = time.time()
                train_pred_times.append(end_pred_time - start_pred_time)

                # Calc. Loss
                #                 self.gnn.train()
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - agent_C['ppo_ratio_clip'],
                                          1.0 + agent_C['ppo_ratio_clip']) * sampled_advantages

                # policy loss and value loss are scalars
                policy_loss = -torch.min(obj, obj_clipped).mean() - agent_C['entropy_weight'] * prediction['ent'].mean()

                value_loss = agent_C['value_loss_coef'] * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                if agent_C['clip_grads']:
                    nn.utils.clip_grad_norm_(self.gnn.parameters(), agent_C['gradient_clip'])
                ensure_shared_grads(self.gnn, self.shared_gnn)
                #                 self.gnn.graph_grads()
                self.opt.step()
                #                 self.gnn.eval()
                end_batch_time = time.time()
                batch_times.append(end_batch_time - start_batch_time)
        self.gnn.eval()
        return total_step, np.array(step_times).mean(), np.array(batch_times).mean(), np.array(train_pred_times).mean()


# TODO: Right now its just a collection of static methods, but I might wanna make it an actually useful class and
# give it internal memory
class Environment:
    def __init__(self):
        ...

    @staticmethod
    def step(selected_node_rel, ep_step, state):
        state_copy = Environment._copy_state(*state)
        G_curr, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors = state_copy[:]

        selected_node_abs = list(current_nodes.keys())[selected_node_rel]
        achieved_goal = Environment.add_children(G_curr, selected_node_abs, goal_node, current_nodes)
        # Done if goal found or end of ep
        done = True if (ep_step == episode_C['max_ep_steps'] - 1) or achieved_goal else False
        reward = Environment.reward_func(done, achieved_goal, shortest_path_length, ep_step + 1)
        predecessors = Environment.get_predecessors(G_curr, current_nodes)

        if not done:
            next_state = Environment._copy_state(G_curr, current_nodes, goal_node, goal_feats, shortest_path_length,
                                                 predecessors)
        else:
            next_state = Environment.reset()

        return next_state, reward, done, achieved_goal

    # Deep copy everything before it goes into a state
    @staticmethod
    def _copy_state(G_curr, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors):
        G_curr_copy = G_curr.copy()
        current_nodes_copy = current_nodes.copy()
        goal_feats_copy = goal_feats.copy()
        predecessors_copy = deepcopy(predecessors)

        return (G_curr_copy, current_nodes_copy, goal_node, goal_feats_copy, shortest_path_length, predecessors_copy)

    # For now this is -1 per timestep +5 on terminal for reaching goal, -5 on terminal for not reaching goal
    # And when it reaches the goal, give add (shortest_path_length - 1) - num actions taken (neg number)
    @staticmethod
    def reward_func(terminal, reach_goal, shortest_path_length, num_actions_taken):
        rew = -1
        if terminal:
            if reach_goal:
                assert num_actions_taken >= (shortest_path_length - 1)
                rew += other_C['reach_goal_rew']
                rew += ((shortest_path_length - 1) - num_actions_taken)  # optimal num actions - num actions taken
            else:
                rew += other_C['not_reach_goal_rew']
        return rew

    @staticmethod
    def reset():
        current_try = 0
        while True:
            current_try += 1
            #         if current_try >= 50:
            #              print('Current try for initialize ep is at: {}'.format(current_try))
            init_node = random.randint(0, model_C['num_nodes'] - 1)
            goal_node = random.randint(0, model_C['num_nodes'] - 1)
            # restart if goal node is init node, or no path
            if init_node == goal_node or not nx.has_path(G_whole, init_node, goal_node):
                continue
            # restart if shortest path is too long or too short
            shortest_path_length = nx.shortest_path_length(G_whole, init_node, goal_node)
            if shortest_path_length < episode_C['shortest_path_range_allowed_MIN'] or shortest_path_length > episode_C[
                'shortest_path_range_allowed_MAX']:
                continue
            break

        # Get goal feats
        goal_feats = node_feats[goal_node]
        assert goal_feats.shape == (model_C['node_feat_size'],)
        # Make init graph
        G_init = nx.DiGraph()
        G_init.add_node(init_node)
        current_nodes = OrderedDict({init_node: 0})  # Init current nodes dict
        got_goal = Environment.add_children(G_init, init_node, goal_node, current_nodes)
        assert sorted(list(current_nodes.values())) == list(current_nodes.values())
        assert not got_goal
        predecessors = Environment.get_predecessors(G_init, current_nodes)
        return (G_init, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors)

    @staticmethod
    def add_children(G_curr, node_indx, goal_node_indx, current_nodes):
        achieved_goal = False
        # Check the children of the node to see if they need to be added to the current graph
        children = G_whole.successors(node_indx)
        for child in children:
            # Add child if not in G and check if goal
            if child not in G_curr:
                G_curr.add_node(child)
                current_nodes.update({child: len(current_nodes)})
                if child == goal_node_indx:
                    achieved_goal = True
            # If the edge doesnt exist add it
            if not G_curr.has_edge(node_indx, child):
                G_curr.add_edge(node_indx, child)
        assert sorted(list(current_nodes.values())) == list(current_nodes.values())
        return achieved_goal

    # current_nodes: ordereddict with keys as abs node indices, values as rel node indices (rel to the ordered dict)
    @staticmethod
    def get_predecessors(G_curr, current_nodes):
        all_preds = []  # List of lists
        for node in current_nodes.keys():
            preds_abs = G_curr.predecessors(node)  # abs to all nodes, keys to the dict
            preds_rel = [current_nodes[x] for x in preds_abs]
            all_preds.append(preds_rel)
        return all_preds  # Returns a list of lists with the values being tth rel node indices

    @staticmethod
    def _unpack_states(states, node_feats_tensor):
        num_nodes_all = []
        goal_feats_all = []
        predecessors_all = []
        node_states_all = []
        total_num_nodes = 0
        for state in states:
            G_curr, current_nodes, goal_node, goal_feats, shortest_path, predecessors = state[:]
            num_nodes_all.append(len(current_nodes))
            total_num_nodes += num_nodes_all[-1]
            goal_feats_tensor = torch.tensor(goal_feats, device=device, requires_grad=True, dtype=torch.float)
            goal_feats_all.append(goal_feats_tensor)
            predecessors_all.append(predecessors)
            node_states_all.append(node_feats_tensor[list(current_nodes.keys())])
        node_states_all_tensor = torch.cat(node_states_all, dim=0)
        goal_feats_all_tensor = torch.stack(goal_feats_all, dim=0)

        #         print(node_states_all_tensor.shape, goal_feats_all_tensor.shape, len(num_nodes_all), len(predecessors_all), total_num_nodes)
        #         print(node_states_all_tensor, goal_feats_all_tensor, num_nodes_all, predecessors_all)

        return node_states_all_tensor, goal_feats_all_tensor, num_nodes_all, predecessors_all, total_num_nodes

    @staticmethod
    def _stack_goals(goal_tensors_all, num_nodes_all):
        stacked_goal_embeds_all = []
        for i in range(len(num_nodes_all)):
            goal = goal_tensors_all[i]
            num_nodes = num_nodes_all[i]
            stacked_goal_embeds_all.append(torch.stack([goal] * num_nodes))
        stacked_goal_emebds_all_tensor = torch.cat(stacked_goal_embeds_all, dim=0)
        return stacked_goal_emebds_all_tensor

    @staticmethod
    def propogate_multi(gnn, states, actions=None):
        node_feats_tensor = torch.tensor(node_feats, device=device, requires_grad=True, dtype=torch.float)
        node_states_all, goal_feats_all, num_nodes_all, predecessors_all, total_num_nodes = Environment._unpack_states(
            states, node_feats_tensor)
        # If goal_input_layer is True then embed the goal by sending it into the input layer
        if goal_C['goal_input_layer']:
            goal_embeddings = gnn.input_model(goal_feats_all)
            assert goal_embeddings.shape == (len(states), model_C['node_hidden_size'])
            stacked_goal_embeds = Environment._stack_goals(goal_embeddings, num_nodes_all)
            assert stacked_goal_embeds.shape == (total_num_nodes, model_C['node_hidden_size'])
        else:
            stacked_goal_embeds = Environment._stack_goals(goal_feats_all, num_nodes_all)
            assert stacked_goal_embeds.shape == (total_num_nodes, model_C['node_feat_size'])

        for p in range(episode_C['num_props']):
            node_states_all, prediction = gnn(node_states_all, p == 0, p == episode_C['num_props'] - 1,
                                              predecessors_all, stacked_goal_embeds, num_nodes_all, actions)
        # assert node_states.shape == (num_nodes, model_C['node_hidden_size'])

        return prediction


# target for multiprocess
def train(id, shared_gnn, optimizer, rollout_counter, args):
    global episode_C;
    global model_C;
    global goal_C;
    global agent_C;
    global other_C;
    global device;
    global G_whole;
    global pages;
    global node_feats;
    global edges
    episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges = args
    agent = PPOAgent(Environment(), shared_gnn, optimizer)
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
def eval(shared_gnn, rollout_counter, args):
    global episode_C;
    global model_C;
    global goal_C;
    global agent_C;
    global other_C;
    global device;
    global G_whole;
    global pages;
    global node_feats;
    global edges
    episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges = args
    agent = PPOAgent(Environment(), shared_gnn, None)
    last_eval = 0
    while True:
        curr_r = rollout_counter.get()
        if curr_r % episode_C['eval_freq'] == 0 and last_eval != curr_r:
            last_eval = curr_r
            avg_rew, max_rew, min_rew, ach_perc, avg_opt_steps, avg_steps_taken = agent.eval_episodes()
            print(
                'Testing summary at rollout {}: Avg ep rew: {:.2f}  Max ep rew: {}  Min ep rew: {}  Achieved goal percent: {:.2f}  Avg opt steps: {:.2f}  Avg steps taken: {:.2f}\n'.format(
                    curr_r, avg_rew, max_rew, min_rew, ach_perc, avg_opt_steps, avg_steps_taken))
        if curr_r >= episode_C['num_train_rollouts'] + 1:
            return


def run(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges):
    shared_gnn = NerveNet_GNN(model_C['node_feat_size'], model_C['node_hidden_size'],
                              model_C['message_size'], model_C['output_size'],
                              goal_C['goal_size'], goal_C['goal_opt'], agent_C['critic_agg_weight'],
                              device).to(device)
    shared_gnn.share_memory()
    optimizer = torch.optim.Adam(shared_gnn.parameters(), agent_C['learning_rate'])
    #     optimizer.share_memory()
    rollout_counter = Counter()  # To keep track of all the rollouts amongst agents
    processes = []

    args = (episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges)
    # Run eval agent
    p = mp.Process(target=eval, args=(shared_gnn, rollout_counter, args))
    p.start()
    processes.append(p)
    # Run training agents
    NUM_AGENTS = 4
    for i in range(NUM_AGENTS):
        p = mp.Process(target=train, args=(i, shared_gnn, optimizer, rollout_counter, args))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def run_normal():
    # Load constants
    constants = load_constants('constants/PPO/constants.json')

    # global episode_C; global model_C; global goal_C; global agent_C; global other_C

    episode_C, model_C, goal_C, agent_C, other_C = constants['episode_C'], constants['model_C'], constants['goal_C'], \
                                                   constants['agent_C'], constants['other_C']
    # Fill in missing values
    fill_in_missing_hyp_params(model_C, goal_C, len(pages), len(edges), node_feats.shape[1])

    exp_start = time.time()
    #     _ = run()
    run(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges)
    exp_end = time.time()
    print('Time taken (m): {:.2f}'.format((exp_end - exp_start) / 60.))

    # plt.show()


if __name__ == '__main__':
    device = torch.device('cpu')
    G_whole, pages, node_feats, edges = load_data_make_graph(
        'data/animals-D3-small-30K-nodes40-edges202-max10-minout2-minin3_w_features.pkl')
    print('Num cores: {}'.format(mp.cpu_count()))
    run_normal()
