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
            nn.ReLU()
            # nn.BatchNorm1d(self.hidden_size)
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes):
        # assert nodes.shape == (self.num_nodes, self.feat_size)
        hidden_states = self.model(nodes)
        # assert hidden_states.shape == (self.num_nodes, self.hidden_size)
        return hidden_states


# Input: (N, h) which is all nodes hidden states
# Outut: (N, m) all nodes messages
class MessageModel(nn.Module):
    def __init__(self, hidden_size, message_size):
        super(MessageModel, self).__init__()
        self.name = 'message'
        # self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.message_size),
            nn.ReLU(),
            # nn.BatchNorm1d(message_size)
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes):
        # assert nodes.shape == (self.num_nodes, self.hidden_size)
        messages = self.model(nodes)
        # assert messages.shape == (self.num_nodes, self.message_size)
        return messages


# Input: (N, m + h) agg messages and hidden states
# Output: (N, h) new hidden states for nodes
# If goal_opt is 1 then add goal to update model input
class UpdateModel(nn.Module):
    def __init__(self, message_size, hidden_size, goal_size=None):
        super(UpdateModel, self).__init__()
        self.name = 'update'
        if goal_size:
            input_size = message_size + hidden_size + goal_size
            self.use_goal = True
        else:
            input_size = message_size + hidden_size
            self.use_goal = False
            
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size)
        )
        self.model.apply(layer_init_filter)

    def forward(self, messages, hidden_states, goal):
        # Concat
        if self.use_goal:
            concat = torch.cat([messages, hidden_states, goal], dim=1)
        else:
            concat = torch.cat([messages, hidden_states], dim=1)
        updates = self.model(concat)
        return updates


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
    def __init__(self, hidden_size, weight, goal_size=None, model=None):
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
            outputs = outputs_all[start_indx:start_indx+n]
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


class NerveNet_GNN(nn.Module):
    def __init__(self, feat_size, hidden_size, message_size, output_size, goal_size, goal_opt, critic_agg_weight,
                 combined_a_c, device):
        super(NerveNet_GNN, self).__init__()
        
        self.device = device
        
        # self.num_nodes = num_nodes
        # self.feat_size = feat_size
        # self.hidden_size = hidden_size
        self.message_size = message_size
        # self.output_size = output_size
        
        update_goal_size = None
        output_goal_size = None
        if goal_opt == 1:
            update_goal_size = goal_size
        elif goal_opt == 2:
            output_goal_size = goal_size
        
        self.input_model = InputModel(feat_size, hidden_size).to(device)
        self.message_model = MessageModel(hidden_size, message_size).to(device)
        self.update_model = UpdateModel(message_size, hidden_size, update_goal_size).to(device)
        # If combined_actor_critic == True then send the model made in actor to critic
        self.actor_model = ActorModel(hidden_size, output_goal_size).to(device)
        self.critic_model = CriticModel(hidden_size, critic_agg_weight, output_goal_size,
                                        self.actor_model.model if combined_a_c else None).to(device)
        
        self.models = [self.input_model, self.message_model, self.update_model, self.actor_model, self.critic_model]

    def _aggregate_all(self, predecessors, messages):
        # Cut up messages
        num_states = len(predecessors)
        stacks = []
        start_indx = 0
        for preds in predecessors:
            num_nodes = len(preds)
            mess = messages[start_indx:start_indx+num_nodes]
            start_indx += num_nodes
            agg_stack = self._aggregate(preds, mess)
            stacks.append(agg_stack)
        stack = torch.cat(stacks)

        return stack

    # Input: (N x m)
    # Output: (N x m)
    def _aggregate(self, predecessors, messages):
        agg = []
        # Collect all in predecessors for each node, if a node has no preds then just 0s for it
        for preds in predecessors:
            if len(preds) > 0:
                in_mess = messages[preds, :]
                assert in_mess.shape == (len(preds), self.message_size) or in_mess.shape == (self.message_size,)  # if one in-node
                agg_in_mess = torch.sum(in_mess, dim=0)
                assert agg_in_mess.shape == (self.message_size,)
                agg.append(agg_in_mess)
            else:
                agg.append(torch.zeros(self.message_size, device=self.device, requires_grad=True, dtype=torch.float))
        stack = torch.stack(agg)
        
        # assert stack.shape == (self.num_nodes, self.message_size)
        return stack

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

        # Get messages of each node ----
        messages = self.message_model(node_states)
        # Aggregate pred. edges -----
        aggregates = self._aggregate_all(predecessors, messages)
        # Get Updates for each node hidden state ---------
        updates = self.update_model(aggregates, node_states, goal)

        # Get outputs if need to ------
        if get_output:
            v = self.critic_model(updates, goal, num_nodes).unsqueeze(-1)
            logits_all = self.actor_model(updates, goal).flatten()

            assert logits_all.shape == (inputs.shape[0],)
            log_prob_all, entropy_all, actions_all = [], [], []
            start_indx = 0
            for i, n in enumerate(num_nodes):
                logits = logits_all[start_indx:start_indx+n]
                action = actions[i] if actions is not None else None
                lp, e, a = self._gather_dist_values(logits, action)
                log_prob_all.append(lp); entropy_all.append(e); actions_all.append(a)
                start_indx += n
            actions_tensor = torch.stack(actions_all)
            log_prob_tensor = torch.stack(log_prob_all).unsqueeze(-1)
            entropy_tensor = torch.stack(entropy_all).unsqueeze(-1)

            assert actions_tensor.shape == (len(num_nodes),)
            assert log_prob_tensor.shape == (len(num_nodes), 1)
            assert entropy_tensor.shape == (len(num_nodes), 1)
            assert v.shape == (len(num_nodes), 1)

            return updates, {'a': actions_tensor, 'log_pi_a': log_prob_tensor, 'ent': entropy_tensor, 'v': v}
        return updates, None

    def graph_grads(self):
        layers, avg_grads, max_grads = [], [], []
        for n, p in self.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plot_grad_flow(layers, avg_grads, max_grads)
       
    def print_layer_weights(self, which='all'):
        if which == 'all':
            models = self.models
        elif which == 'input':
            models = [self.input_model]
        elif which == 'message':
            models = [self.message_model]
        elif which == 'update':
            models = [self.update_model]
        elif which == 'actor':
            models = [self.actor_model]
        elif which == 'critic':
            models = [self.critic_model]
            
        weights = []
        for model in models:
            print('Model: ' + model.name)
            for n, p in model.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    print(str(n) + ': ' + str(p[0][:10]))
