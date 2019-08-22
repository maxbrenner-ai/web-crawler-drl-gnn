from nervenet import NerveNet_GNN
from deepmind import Deepmind_GNN
from copy import deepcopy
import numpy as np
from utils import Storage, tensor, random_sample, ensure_shared_grads
import time
import torch
import torch.nn as nn


class PPOAgent:
    def __init__(self, args, env, shared_gnn, optimizer):
        if args[1]['nervenet']:
            self.episode_C, self.model_C, self.goal_C, self.agent_C, self.other_C, self.device, self.G_whole, self.pages, \
            self.node_feats, self.edges = args

            self.gnn = NerveNet_GNN(self.model_C['node_feat_size'], self.model_C['node_hidden_size'],
                                    self.model_C['message_size'], self.model_C['output_size'],
                                    self.goal_C['goal_size'], self.goal_C['goal_opt'], self.agent_C['critic_agg_weight'],
                                    self.device).to(self.device)
        else:
            self.episode_C, self.model_C, self.goal_C, self.agent_C, self.other_C, self.device, self.G_whole, self.pages, \
            self.node_feats, self.edge_feats, self.edges = args

            self.gnn = Deepmind_GNN(self.model_C['node_feat_size'], self.model_C['edge_feat_size'],
                                    self.model_C['node_hidden_size'], self.model_C['edge_hidden_size'],
                                    self.goal_C['goal_size'], self.goal_C['goal_opt'],
                                    self.agent_C['critic_agg_weight'],
                                    self.device).to(self.device)

        self.shared_gnn = shared_gnn
        self.env = env
        self.state = self.env.reset()
        self.ep_step = 0
        self.opt = optimizer
        self.gnn.eval()

    def _eval_episode(self, test_step):
        state = self.env.reset()
        if self.model_C['nervenet']:
            shortest_path_length = state[4]
        else:
            shortest_path_length = state[5]
        ep_rew = 0
        for step in range(self.episode_C['max_ep_steps']):
            prediction = self.env.propagate(self.gnn, [state])
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
        for ep in range(self.episode_C['eval_num_eps']):
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
        storage = Storage(self.episode_C['rollout_length'])
        state = self.env._copy_state(*self.state)
        step_times = []
        # Sync.
        self.gnn.load_state_dict(self.shared_gnn.state_dict())
        for rollout_step in range(self.episode_C['rollout_length']):
            start_step_time = time.time()
            prediction = self.env.propagate(self.gnn, [state])
            action = prediction['a'].cpu().numpy()[0]
            next_state, reward, done, achieved_goal = self.env.step(action, self.ep_step, state)

            self.ep_step += 1
            if done:
                # Sync local model with shared model at start of each ep
                self.gnn.load_state_dict(self.shared_gnn.state_dict())
                self.ep_step = 0

            storage.add(prediction)
            storage.add({'r': tensor(reward, self.device).unsqueeze(-1).unsqueeze(-1),
                         'm': tensor(1 - done, self.device).unsqueeze(-1).unsqueeze(-1),
                         's': state})

            state = self.env._copy_state(*next_state)

            total_step += 1

            end_step_time = time.time()
            step_times.append(end_step_time - start_step_time)

        self.state = self.env._copy_state(*state)

        prediction = self.env.propagate(self.gnn, [state])
        storage.add(prediction)
        storage.placeholder()

        advantages = tensor(np.zeros((1, 1)), self.device)
        returns = prediction['v'].detach()
        for i in reversed(range(self.episode_C['rollout_length'])):
            # Disc. Return
            returns = storage.r[i] + self.agent_C['discount'] * storage.m[i] * returns
            # GAE
            td_error = storage.r[i] + self.agent_C['discount'] * storage.m[i] * storage.v[i + 1] - storage.v[i]
            advantages = advantages * self.agent_C['gae_tau'] * self.agent_C['discount'] * storage.m[i] + td_error
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
        for _ in range(self.agent_C['optimization_epochs']):
            # Sync. at start of each epoch
            self.gnn.load_state_dict(self.shared_gnn.state_dict())
            sampler = random_sample(np.arange(len(states)), self.agent_C['minibatch_size'])
            for batch_indices in sampler:
                start_batch_time = time.time()

                batch_indices_tensor = tensor(batch_indices, self.device).long()

                # Important Node: these are tensors but dont have a grad
                sampled_states = [states[i] for i in batch_indices]
                sampled_actions = actions[batch_indices_tensor]
                sampled_log_probs_old = log_probs_old[batch_indices_tensor]
                sampled_returns = returns[batch_indices_tensor]
                sampled_advantages = advantages[batch_indices_tensor]

                start_pred_time = time.time()
                prediction = self.env.propagate(self.gnn, sampled_states, sampled_actions)
                end_pred_time = time.time()
                train_pred_times.append(end_pred_time - start_pred_time)

                # assert True == False

                # Calc. Loss
                #                 self.gnn.train()
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()

                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.agent_C['ppo_ratio_clip'],
                                          1.0 + self.agent_C['ppo_ratio_clip']) * sampled_advantages

                # policy loss and value loss are scalars
                policy_loss = -torch.min(obj, obj_clipped).mean() - self.agent_C['entropy_weight'] * prediction['ent'].mean()

                value_loss = self.agent_C['value_loss_coef'] * (sampled_returns - prediction['v']).pow(2).mean()

                self.opt.zero_grad()
                (policy_loss + value_loss).backward()
                if self.agent_C['clip_grads']:
                    nn.utils.clip_grad_norm_(self.gnn.parameters(), self.agent_C['gradient_clip'])
                ensure_shared_grads(self.gnn, self.shared_gnn)
                #                 self.gnn.graph_grads()
                self.opt.step()
                #                 self.gnn.eval()
                end_batch_time = time.time()
                batch_times.append(end_batch_time - start_batch_time)
        self.gnn.eval()
        return total_step, np.array(step_times).mean(), np.array(batch_times).mean(), np.array(train_pred_times).mean()
