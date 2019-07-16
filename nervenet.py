import torch.nn as nn
import torch
import numpy as np
from gnn import GNN


# This is so the hidden size doesnt need to be the same size as the feature size
# Input: (N, f)
# Output: (N, h)
class InputModel(nn.Module):
    def __init__(self, num_nodes, feat_size, hidden_size):
        super(InputModel, self).__init__()
        self.num_nodes = num_nodes
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.feat_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, nodes):
        assert nodes.shape == (self.num_nodes, self.feat_size)
        hidden_states = self.model(nodes)
        assert hidden_states.shape == (self.num_nodes, self.hidden_size)
        return hidden_states


# Input: (N, h) which is all nodes hidden states
# Outut: (N, m) all nodes messages
class MessageModel(nn.Module):
    def __init__(self, num_nodes, hidden_size, message_size):
        super(MessageModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.message_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, nodes):
        assert nodes.shape == (self.num_nodes, self.hidden_size)
        messages = self.model(nodes)
        assert messages.shape == (self.num_nodes, self.message_size)
        return messages


# Input: (N, m + h) agg messages and hidden states
# Output: (N, h) new hidden states for nodes
class UpdateModel(nn.Module):
    def __init__(self, num_nodes, message_size, hidden_size):
        super(UpdateModel, self).__init__()
        self.num_nodes = num_nodes
        self.message_size = message_size
        self.hidden_size = hidden_size
        self.input_size = message_size + hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, messages, hidden_states):
        assert messages.shape == (self.num_nodes, self.message_size)
        assert hidden_states.shape == (self.num_nodes, self.hidden_size)
        # Concat
        concat = torch.cat([messages, hidden_states], dim=1)
        assert concat.shape == (self.num_nodes, self.input_size)
        updates = self.model(concat)
        assert updates.shape == (self.num_nodes, self.hidden_size)
        return updates


# Input: (N, h)  updated node hidden states
# Output: (N, o)  outputs for each node (softmax on classes)
class OutputModel(nn.Module):
    def __init__(self, num_nodes, hidden_size, output_size):
        super(OutputModel, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, nodes):
        assert nodes.shape == (self.num_nodes, self.hidden_size)
        outputs = self.model(nodes)
        assert outputs.shape == (self.num_nodes, self.output_size)
        return outputs


class NerveNet_GNN(GNN):
    def __init__(self, num_nodes, feat_size, hidden_size, message_size, output_size):
        super(NerveNet_GNN, self).__init__()

        self.num_nodes = num_nodes
        # self.feat_size = feat_size
        # self.hidden_size = hidden_size
        self.message_size = message_size
        # self.output_size = output_size

        self.input_model = InputModel(num_nodes, feat_size, hidden_size)
        self.message_model = MessageModel(num_nodes, hidden_size, message_size)
        self.update_model = UpdateModel(num_nodes, message_size, hidden_size)
        self.output_model = OutputModel(num_nodes, hidden_size, output_size)

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
                agg.append(torch.zeros(self.message_size))
        stack = torch.stack(agg)
        assert stack.shape == (self.num_nodes, self.message_size)
        return stack

    # Propogate: assumes that the feats are sent in every episode/epoch
    def forward(self, inputs, send_input, get_output, predecessors):
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
        updates = self.update_model(aggregates, node_states)
        # Get outputs if need to ------
        if get_output:
            outputs = self.output_model(updates)
            return updates, outputs
        return updates, None

    # Outputs: (N_train, o) tensor
    # Targets: (N_train,) tensor of the classes
    def backward(self, outputs, targets):
        # assert outputs.shape == (num_train, o)
        # assert targets.shape == (num_train,)
        loss = self.loss(outputs, targets)
        loss.backward()
        # Graph gradient flow
        self.graph_grads([self.input_model,
                          self.message_model,
                          self.update_model,
                          self.output_model])
        return loss.data.tolist()