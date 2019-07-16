import torch.nn as nn
import torch
import numpy as np
from gnn import GNN


# Input: (N, f_n)
# Output: (N, h_n)
class NodeInputModel(nn.Module):
    def __init__(self, num_nodes, node_feat_size, node_hidden_size):
        super(NodeInputModel, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_size = node_feat_size
        self.node_hidden_size = node_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.node_feat_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, nodes):
        assert nodes.shape == (self.num_nodes, self.node_feat_size)
        hidden_states = self.model(nodes)
        assert hidden_states.shape == (self.num_nodes, self.node_hidden_size)
        return hidden_states


# Input: (E, f_e)
# Output: (E, h_e)
class EdgeInputModel(nn.Module):
    def __init__(self, num_edges, edge_feat_size, edge_hidden_size):
        super(EdgeInputModel, self).__init__()
        self.num_edges = num_edges
        self.edge_feat_size = edge_feat_size
        self.edge_hidden_size = edge_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.edge_feat_size, self.edge_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, edges):
        assert edges.shape == (self.num_edges, self.edge_feat_size)
        hidden_states = self.model(edges)
        assert hidden_states.shape == (self.num_edges, self.edge_hidden_size)
        return hidden_states


# Input: (E, h_n + h_n + h_e)  # in node, out node and edge hidden states
# Output: (E, h_e)
class EdgeUpdateModel(nn.Module):
    def __init__(self, num_edges, node_hidden_size, edge_hidden_size):
        super(EdgeUpdateModel, self).__init__()
        self.num_edges = num_edges
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.input_shape_cols = 2 * self.node_hidden_size + self.edge_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.input_shape_cols, self.edge_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, send_nodes, rec_nodes, edges):
        assert send_nodes.shape == (self.num_edges, self.node_hidden_size)
        assert rec_nodes.shape == (self.num_edges, self.node_hidden_size)
        assert edges.shape == (self.num_edges, self.edge_hidden_size)
        concat = torch.cat([send_nodes, rec_nodes, edges], dim=1)
        assert concat.shape == (self.num_edges, self.input_shape_cols)
        new_edges = self.model(concat)
        assert new_edges.shape == (self.num_edges, self.edge_hidden_size)
        return new_edges


# Input: (N, h_n + h_e)  # node, in-edge sum agg (ie out-node)
# Output: (N, h_n)
class NodeUpdateModel(nn.Module):
    def __init__(self, num_nodes, node_hidden_size, edge_hidden_size):
        super(NodeUpdateModel, self).__init__()
        self.num_nodes = num_nodes
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.input_size = self.node_hidden_size + self.edge_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.node_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, nodes, in_edges):
        assert nodes.shape == (self.num_nodes, self.node_hidden_size)
        assert in_edges.shape == (self.num_nodes, self.edge_hidden_size)
        concat = torch.cat([nodes, in_edges], dim=1)
        assert concat.shape == (self.num_nodes, self.input_size)
        new_nodes = self.model(concat)
        assert new_nodes.shape == (self.num_nodes, self.node_hidden_size)
        return new_nodes


# Input: (N, h_n)  updated node hidden states
# Output: (N, o)  outputs for each node (softmax on classes)
class OutputModel(nn.Module):
    def __init__(self, num_nodes, node_hidden_size, output_size):
        super(OutputModel, self).__init__()
        self.num_nodes = num_nodes
        self.node_hidden_size = node_hidden_size
        self.output_size = output_size
        self.model = nn.Sequential(
            nn.Linear(self.node_hidden_size, self.output_size)
        )

    def forward(self, nodes):
        assert nodes.shape == (self.num_nodes, self.node_hidden_size)
        outputs = self.model(nodes)
        assert outputs.shape == (self.num_nodes, self.output_size)
        return outputs


class Deepmind_GNN(GNN):
    def __init__(self, num_nodes, num_edges, node_feat_size, edge_feat_size,
                 node_hidden_size, edge_hidden_size, output_size):
        super(Deepmind_GNN, self).__init__()

        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size
        self.output_size = output_size

        self.node_input_model = NodeInputModel(num_nodes, node_feat_size, node_hidden_size)
        self.edge_input_model = EdgeInputModel(num_edges, edge_feat_size, edge_hidden_size)
        self.edge_update_model = EdgeUpdateModel(num_edges, node_hidden_size, edge_hidden_size)
        self.node_update_model = NodeUpdateModel(num_nodes, node_hidden_size, edge_hidden_size)
        self.output_model = OutputModel(num_nodes, node_hidden_size, output_size)

    # edge_tuples contains a list of all edges in terms of in-node indx, and out-node indx
    def _update_edges(self, edge_tuples, node_states, edge_states):
        # Collect in and out nodes
        in_agg, out_agg = [], []
        for e_indx, edge in enumerate(edge_tuples):
            in_node = edge[0]
            out_node = edge[1]
            in_agg.append(node_states[in_node])
            out_agg.append(node_states[out_node])
        in_stack = torch.stack(in_agg)
        out_stack = torch.stack(out_agg)
        return self.edge_update_model(in_stack, out_stack, edge_states)

    # in_edges_list is a list of lists of the edge indxs of in-edges to a node, if empty then append all zeros
    def _update_nodes(self, in_edges_list, node_states, edge_states):
        agg = []
        assert len(in_edges_list) == self.num_nodes
        for in_edges in in_edges_list:
            if len(in_edges) > 0:
                in_edge_states = edge_states[in_edges, :]
                assert in_edge_states.shape == (len(in_edges), self.edge_hidden_size) or in_edge_states.shape == (self.edge_hidden_size,)  # if one
                agg_in_edge_states = torch.sum(in_edge_states, dim=0)
                assert agg_in_edge_states.shape == (self.edge_hidden_size,)
                agg.append(agg_in_edge_states)
            else:
                agg.append(torch.zeros(self.edge_hidden_size))
        stack = torch.stack(agg)
        assert stack.shape == (self.num_nodes, self.edge_hidden_size)
        return self.node_update_model(node_states, stack)

    # Propogate: assumes feats sent in at start of episode/epoch
    def forward(self, node_inputs, edge_inputs, edge_tuples, in_edge_list, send_input, get_output):
        # Get initial hidden states ------
        if send_input:
            node_states = self.node_input_model(node_inputs)
            edge_states = self.edge_input_model(edge_inputs)
        else:
            node_states = node_inputs
            edge_states = edge_inputs
        # Update edges ------
        edge_updates = self._update_edges(edge_tuples, node_states, edge_states)
        # Update nodes ------
        node_updates = self._update_nodes(in_edge_list, node_states, edge_updates)  # Send in updated edge updates
        # Get outputs if need to ------
        if get_output:
            outputs = self.output_model(node_updates)
            return node_updates, edge_updates, outputs
        return node_updates, edge_updates, None

    # Outputs: (N_train, o) tensor
    # Targets: (N_train,) tensor of the classes
    def backward(self, outputs, targets):
        # assert outputs.shape == (num_train, o)
        # assert targets.shape == (num_train,)
        loss = self.loss(outputs, targets)
        loss.backward()
        # Graph gradient flow
        self.graph_grads([self.node_input_model,
                          self.edge_input_model,
                          self.edge_update_model,
                          self.node_update_model,
                          self.output_model])
        return loss.data.tolist()