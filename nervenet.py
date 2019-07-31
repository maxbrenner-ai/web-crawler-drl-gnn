import torch.nn as nn
import torch
import numpy as np
from utils import plot_grad_flow, layer_init
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
            nn.BatchNorm1d(self.hidden_size)
        )
        self.model.apply(layer_init)

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
            nn.BatchNorm1d(message_size)
        )
        self.model.apply(layer_init)

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
            nn.BatchNorm1d(hidden_size)
        )
        self.model.apply(layer_init)

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
class OutputModel(nn.Module):
    def __init__(self, hidden_size, output_size, goal_size=None):
        super(OutputModel, self).__init__()
        self.name = 'output'
        if goal_size:
            input_size = hidden_size + goal_size
            self.use_goal = True
        else:
            input_size = hidden_size
            self.use_goal = False
            
        self.model = nn.Sequential(
            nn.Linear(input_size, output_size)
        )
        self.model.apply(layer_init)

    def forward(self, nodes, goal):
        if self.use_goal:
            concat = torch.cat([nodes, goal], dim=1)
        else:
            concat = nodes
        outputs = self.model(concat)
        return outputs


class NerveNet_GNN(nn.Module):
    def __init__(self, feat_size, hidden_size, message_size, output_size, goal_size, goal_opt, device):
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
        self.output_model = OutputModel(hidden_size, output_size, output_goal_size).to(device)
        
        self.models = [self.input_model, self.message_model, self.update_model, self.output_model]
        
#         self.loss = nn.MSELoss()

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

    # Propogate: assumes that the feats are sent in every episode/epoch
    def forward(self, inputs, send_input, get_output, predecessors, goal):
        # Get initial hidden states ------
        if send_input:
            node_states = self.input_model(inputs)
        else:
            node_states = inputs
        
        # Get messages of each node ----
        messages = self.message_model(node_states)
        # Aggregate pred. edges -----
        aggregates = self._aggregate(predecessors, messages)
        # Get Updates for each node hidden state ---------
        updates = self.update_model(aggregates, node_states, goal)
        # Get outputs if need to ------
        if get_output:
            outputs = self.output_model(updates, goal)
            return updates, outputs
        return updates, None

    def _graph_grads(self):
        layers, avg_grads, max_grads = [], [], []
        for n, p in self.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plot_grad_flow(layers, avg_grads, max_grads)
       
    def print_layer_weights(self, which='all'):
        if which == 'all':
            models = [self.input_model, self.message_model, self.update_model, self.output_model]
        elif which == 'input':
            models = [self.input_model]
        elif which == 'message':
            models = [self.message_model]
        elif which == 'update':
            models = [self.update_model]
        elif which == 'output':
            models = [self.output_model]
            
        weights = []
        for model in models:
            print('Model: ' + model.name)
            for n, p in model.named_parameters():
                if(p.requires_grad) and ("bias" not in n):
                    print(str(n) + ': ' + str(p[0][:10]))
    
    # Output: (mb size, 1)  the outputs of the mb
    # Targets: (mb size, 1)  the targets of the mb
    def backward(self, outputs, targets):
        # Huber loss
        loss = F.smooth_l1_loss(outputs, targets)
        loss.backward()
        # Graph gradient flow
        self._graph_grads()
        return loss.detach().cpu().numpy()