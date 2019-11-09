import torch.nn as nn
import torch
import numpy as np
from utils import plot_grad_flow, layer_init_filter
import torch.nn.functional as F


# This is so the hidden size doesnt need to be the same size as the feature size
# Input: (N, f)
# Output: (N, h)
class InputModel(nn.Module):
    def __init__(self, feat_size, hidden_size):
        super(InputModel, self).__init__()
        self.name = 'input'
        # self.num_nodes = num_nodes
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.feat_size, self.hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(self.hidden_size)
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes):
        # assert nodes.shape == (self.num_nodes, self.feat_size)
        hidden_states = self.model(nodes)
        # assert hidden_states.shape == (self.num_nodes, self.hidden_size)
        return hidden_states


# Input: (N, h)  updated node hidden states
# Output: (N, o)  outputs for each node (softmax on classes)
# If goal_opt == 2 then send in goal
class ActorModel(nn.Module):
    def __init__(self, hidden_size, goal_size=None):
        super(ActorModel, self).__init__()
        self.name = 'actor'
        if goal_size:
            input_size = hidden_size + goal_size
            self.use_goal = True
        else:
            input_size = hidden_size
            self.use_goal = False

        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes, goal):
        if self.use_goal:
            concat = torch.cat([nodes, goal], dim=1)
        else:
            concat = nodes
        outputs = self.model(concat)
        return outputs


# Input: (N, h)  updated node hidden states
# Output: ()  state value
# If goal_opt == 2 then send in goal
class CriticModel(nn.Module):
    def __init__(self, hidden_size, weight, goal_size=None, model=False):
        super(CriticModel, self).__init__()
        self.name = 'actor'
        self.weight = weight
        if goal_size:
            input_size = hidden_size + goal_size
            self.use_goal = True
        else:
            input_size = hidden_size
            self.use_goal = False

        if not model:
            self.model = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )
            self.model.apply(layer_init_filter)
        else:
            self.model = model

    def forward(self, nodes, goal, num_nodes):
        if self.use_goal:
            concat = torch.cat([nodes, goal], dim=1)
        else:
            concat = nodes
        outputs_all = self.model(concat).flatten()
        # Cut up by num nodes per state
        values = []
        start_indx = 0
        for n in num_nodes:
            outputs = outputs_all[start_indx:start_indx + n]
            start_indx += n
            # Get the mean and max of the outputs
            mean_out = outputs.mean()
            max_out = outputs.max()
            # Weight the value between the mean and max
            value = max_out * self.weight + mean_out * (1. - self.weight)
            values.append(value)
            assert value.shape == (), 'shape: {}'.format(value.shape)
        values_tensor = torch.stack(values)
        assert values_tensor.shape == (len(num_nodes),), values_tensor.shape
        return values_tensor


class NoStructure_baseline(nn.Module):
    def __init__(self, feat_size, hidden_size, goal_size, goal_opt, critic_agg_weight,
                 combined_a_c, device):
        super(NoStructure_baseline, self).__init__()

        self.device = device

        output_goal_size = None
        if goal_opt == 2:
            output_goal_size = goal_size

        self.input_model = InputModel(feat_size, hidden_size).to(device)
        self.actor_model = ActorModel(hidden_size, output_goal_size).to(device)
        self.critic_model = CriticModel(hidden_size, critic_agg_weight, output_goal_size,
                                        self.actor_model.model if combined_a_c else None).to(device)

        self.models = [self.input_model, self.actor_model, self.critic_model]

    def _gather_dist_values(self, logits, action):
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, action

    # Propogate: assumes that the feats are sent in every episode/epoch
    def forward(self, inputs, send_input, get_output, predecessors, goal, num_nodes, actions=None):
        # Get initial hidden states ------
        if send_input:
            node_states = self.input_model(inputs)
        else:
            node_states = inputs

        # print('================')
        # print(node_states.shape)
        # print(goal.shape)
        # print(num_nodes)

        # Get outputs if need to ------
        if get_output:
            v = self.critic_model(node_states, goal, num_nodes).unsqueeze(-1)

            logits_all = self.actor_model(node_states, goal).flatten()
            assert logits_all.shape == (inputs.shape[0],)
            log_prob_all, entropy_all, actions_all = [], [], []
            start_indx = 0
            for i, n in enumerate(num_nodes):
                logits = logits_all[start_indx:start_indx + n]
                action = actions[i] if actions is not None else None
                lp, e, a = self._gather_dist_values(logits, action)
                log_prob_all.append(lp)
                entropy_all.append(e)
                actions_all.append(a)
                start_indx += n
            actions_tensor = torch.stack(actions_all)
            log_prob_tensor = torch.stack(log_prob_all).unsqueeze(-1)
            entropy_tensor = torch.stack(entropy_all).unsqueeze(-1)

            assert actions_tensor.shape == (len(num_nodes),)
            assert log_prob_tensor.shape == (len(num_nodes), 1)
            assert entropy_tensor.shape == (len(num_nodes), 1)
            assert v.shape == (len(num_nodes), 1)

            return node_states, {'a': actions_tensor, 'log_pi_a': log_prob_tensor, 'ent': entropy_tensor, 'v': v}
        return node_states, None

    def graph_grads(self):
        layers, avg_grads, max_grads = [], [], []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plot_grad_flow(layers, avg_grads, max_grads)

    # def print_layer_weights(self, which='all'):
    #     if which == 'all':
    #         models = self.models
    #     elif which == 'input':
    #         models = [self.input_model]
    #     elif which == 'message':
    #         models = [self.message_model]
    #     elif which == 'update':
    #         models = [self.update_model]
    #     elif which == 'actor':
    #         models = [self.actor_model]
    #     elif which == 'critic':
    #         models = [self.critic_model]
    #
    #     weights = []
    #     for model in models:
    #         print('Model: ' + model.name)
    #         for n, p in model.named_parameters():
    #             if (p.requires_grad) and ("bias" not in n):
    #                 print(str(n) + ': ' + str(p[0][:10]))
