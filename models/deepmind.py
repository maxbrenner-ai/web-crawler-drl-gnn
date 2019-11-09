import torch.nn as nn
import torch
import numpy as np
from utils import plot_grad_flow, layer_init_filter


# Input: (N, f_n)
# Output: (N, h_n)
class NodeInputModel(nn.Module):
    def __init__(self, node_feat_size, node_hidden_size):
        super(NodeInputModel, self).__init__()
        self.name = 'node_input'
        self.model = nn.Sequential(
            nn.Linear(node_feat_size, node_hidden_size),
            nn.ReLU()
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes):
        hidden_states = self.model(nodes)
        return hidden_states


# Input: (E, f_e)
# Output: (E, h_e)
class EdgeInputModel(nn.Module):
    def __init__(self, edge_feat_size, edge_hidden_size):
        super(EdgeInputModel, self).__init__()
        self.name = 'edge_input'
        self.model = nn.Sequential(
            nn.Linear(edge_feat_size, edge_hidden_size),
            nn.ReLU()
        )
        self.model.apply(layer_init_filter)

    def forward(self, edges):
        hidden_states = self.model(edges)
        return hidden_states


# Input: (E, h_n + h_n + h_e)  # in node, out node and edge hidden states
# Output: (E, h_e)
class EdgeUpdateModel(nn.Module):
    def __init__(self, node_hidden_size, edge_hidden_size):
        super(EdgeUpdateModel, self).__init__()
        self.name = 'edge_update'
        self.input_shape_cols = 2 * node_hidden_size + edge_hidden_size
        self.model = nn.Sequential(
            nn.Linear(self.input_shape_cols, edge_hidden_size),
            nn.ReLU(),
        )
        self.model.apply(layer_init_filter)

    def forward(self, send_nodes, rec_nodes, edges):
        concat = torch.cat([send_nodes, rec_nodes, edges], dim=1)
        new_edges = self.model(concat)
        return new_edges


# Input: (N, h_n + h_e)  # node, in-edge sum agg (ie out-node)
# Output: (N, h_n)
class NodeUpdateModel(nn.Module):
    def __init__(self, node_hidden_size, edge_hidden_size, goal_size=None):
        super(NodeUpdateModel, self).__init__()
        self.name = 'node_update'
        if goal_size:
            input_size = node_hidden_size + edge_hidden_size + goal_size
            self.use_goal = True
        else:
            input_size = node_hidden_size + edge_hidden_size
            self.use_goal = False
        self.model = nn.Sequential(
            nn.Linear(input_size, node_hidden_size),
            nn.ReLU(),
        )
        self.model.apply(layer_init_filter)

    def forward(self, nodes, in_edges, goal):
        if self.use_goal:
            concat = torch.cat([nodes, in_edges, goal], dim=1)
        else:
            concat = torch.cat([nodes, in_edges], dim=1)
        new_nodes = self.model(concat)
        return new_nodes


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
        self.name = 'critic'
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


class Deepmind_GNN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, node_hidden_size, edge_hidden_size, goal_size, goal_opt,
                 critic_agg_weight, combined_a_c, device):
        super(Deepmind_GNN, self).__init__()

        self.device = device

        # self.node_feat_size = node_feat_size
        # self.edge_feat_size = edge_feat_size
        # self.node_hidden_size = node_hidden_size
        self.edge_hidden_size = edge_hidden_size

        update_goal_size = None
        output_goal_size = None
        if goal_opt == 1:
            update_goal_size = goal_size
        elif goal_opt == 2:
            output_goal_size = goal_size

        self.node_input_model = NodeInputModel(node_feat_size, node_hidden_size).to(device)
        self.edge_input_model = EdgeInputModel(edge_feat_size, edge_hidden_size).to(device)
        self.edge_update_model = EdgeUpdateModel(node_hidden_size, edge_hidden_size).to(device)
        self.node_update_model = NodeUpdateModel(node_hidden_size, edge_hidden_size, update_goal_size).to(device)
        self.actor_model = ActorModel(node_hidden_size, output_goal_size).to(device)
        self.critic_model = CriticModel(node_hidden_size, critic_agg_weight, output_goal_size,
                                        self.actor_model.model if combined_a_c else None).to(device)

        self.models = [self.node_input_model, self.edge_input_model, self.edge_update_model, self.node_update_model, self.actor_model, self.critic_model]

    # edge tuples is a list of lists of all edges in terms if in-node indx, and out-node indx
    # node_states is a stack of node states fr each state (graph)
    # edge_states similarly is a stack of edge states for each state (graph)
    # num_nodes is a list of the number of nodes in each state
    def _update_edges_all(self, edge_tuples, node_states, edge_states, num_nodes):

        # print('Checking update edges all...')
        # print('edge tuples: {}'.format(edge_tuples))
        # print('node states: {}'.format(node_states))
        # print('edge states: {}'.format(edge_states))
        # print('num nodes: {}'.format(num_nodes))

        num_states = len(edge_tuples)
        assert len(edge_tuples) == len(num_nodes)
        in_stacks, out_stacks = [], []
        start_indx = 0
        for edge_t, N in zip(edge_tuples, num_nodes):
            nodes = node_states[start_indx:start_indx+N]
            agg_stack_in, agg_stack_out = self._update_edges(edge_t, nodes)
            in_stacks.append(agg_stack_in)
            out_stacks.append(agg_stack_out)
            start_indx += N
        in_stack = torch.cat(in_stacks)
        out_stack = torch.cat(out_stacks)

        # print('in stach shape: {}'.format(in_stack.shape))
        # print('in stack: {}'.format(in_stacks))
        # print('out stack shape: {}'.format(out_stack.shape))
        # print('out stack: {}'.format(out_stacks))
        # print('...done with edge updates all ')

        return self.edge_update_model(in_stack, out_stack, edge_states)

    # edge_tuples contains a list of all edges in terms of in-node indx, and out-node indx
    def _update_edges(self, edge_tuples, node_states):
        # Collect in and out nodes
        in_agg, out_agg = [], []
        for e_indx, edge in enumerate(edge_tuples):
            in_node = edge[0]
            out_node = edge[1]
            in_agg.append(node_states[in_node])
            out_agg.append(node_states[out_node])
        in_stack = torch.stack(in_agg)
        out_stack = torch.stack(out_agg)
        return in_stack, out_stack

    # in_edges_list is a list of list of lists of edge indices
    # node_states is a list of each states (graphs) node states
    # edge_states is a list of each states (graphs) edge states
    def _update_nodes_all(self, in_edges_list, node_states, edge_states, goal, num_edges):

        # print('start update nodes all...')
        # print('edge states: {}'.format(edge_states))
        # print('in edges list: {}'.format(in_edges_list))

        num_states = len(in_edges_list)
        assert len(in_edges_list) == len(num_edges)
        stacks = []
        start_indx = 0
        for in_edges, E in zip(in_edges_list, num_edges):
            edges = edge_states[start_indx:start_indx+E]
            agg_stack = self._update_nodes(in_edges, edges)
            stacks.append(agg_stack)
            start_indx += E
        edge_stack = torch.cat(stacks)

        # print('edge stack shape: {}'.format(edge_stack.shape))
        # print('edge stack: {}'.format(edge_stack))

        return self.node_update_model(node_states, edge_stack, goal)

    # in_edges_list is a list of lists of the edge indxs of in-edges to a node, if empty then append all zeros
    def _update_nodes(self, in_edges_list, edge_states):
        agg = []
        for in_edges in in_edges_list:
            if len(in_edges) > 0:
                in_edge_states = edge_states[in_edges, :]
                assert in_edge_states.shape == (len(in_edges), self.edge_hidden_size) or in_edge_states.shape == (self.edge_hidden_size,)  # if one
                agg_in_edge_states = torch.sum(in_edge_states, dim=0)
                assert agg_in_edge_states.shape == (self.edge_hidden_size,)
                agg.append(agg_in_edge_states)
            else:
                agg.append(torch.zeros(self.edge_hidden_size, device=self.device, requires_grad=True, dtype=torch.float))
        stack = torch.stack(agg)
        return stack

    def _gather_dist_values(self, logits, action):
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, action

    # Propogate: assumes feats sent in at start of episode/epoch
    def forward(self, node_inputs, edge_inputs, edge_tuples, in_edge_list, send_input, get_output, goal, num_nodes, num_edges, actions=None):
        assert len(num_nodes) == len(num_nodes)

        # print('\n\n\n\n----------------------------')
        # print('num states: {}'.format(len(num_nodes)))
        # print('num nodes: {}'.format(num_nodes))
        # print('num edges: {}'.format(num_edges))
        # print('input node states: {}'.format(node_inputs.shape))
        # print('input edge states: {}'.format(edge_inputs.shape))

        # Get initial hidden states ------
        if send_input:
            node_states = self.node_input_model(node_inputs)
            edge_states = self.edge_input_model(edge_inputs)
        else:
            node_states = node_inputs
            edge_states = edge_inputs

        # print('node states: {}'.format(node_states.shape))
        # print('edge states: {}'.format(edge_states.shape))

        # Update edges ------
        edge_updates = self._update_edges_all(edge_tuples, node_states, edge_states, num_nodes)

        # print('edge updates: {}'.format(edge_updates.shape))

        # Update nodes ------
        node_updates = self._update_nodes_all(in_edge_list, node_states, edge_updates, goal, num_edges)  # Send in updated edge updates

        # print('node updates: {}'.format(node_updates.shape))

        # Get outputs if need to ------
        if get_output:
            v = self.critic_model(node_updates, goal, num_nodes).unsqueeze(-1)

            # print('v: {}'.format(v.shape))

            logits_all = self.actor_model(node_updates, goal).flatten()

            # print('logits all: {}'.format(logits_all.shape))

            assert logits_all.shape == (node_inputs.shape[0],)
            log_prob_all, entropy_all, actions_all = [], [], []
            start_indx = 0
            for i, n in enumerate(num_nodes):
                logits = logits_all[start_indx:start_indx + n]
                action = actions[i] if actions is not None else None
                lp, e, a = self._gather_dist_values(logits, action)
                log_prob_all.append(lp);entropy_all.append(e);actions_all.append(a)
                start_indx += n
            actions_tensor = torch.stack(actions_all)
            log_prob_tensor = torch.stack(log_prob_all).unsqueeze(-1)
            entropy_tensor = torch.stack(entropy_all).unsqueeze(-1)
            #
            # print('actions: {}'.format(actions_tensor.shape))
            # print('log prob: {}'.format(log_prob_tensor.shape))
            # print('entropy: {}'.format(entropy_tensor.shape))

            assert actions_tensor.shape == (len(num_nodes),)
            assert log_prob_tensor.shape == (len(num_nodes), 1)
            assert entropy_tensor.shape == (len(num_nodes), 1)
            assert v.shape == (len(num_nodes), 1)

            # assert True == False

            return node_updates, edge_updates, {'a': actions_tensor, 'log_pi_a': log_prob_tensor, 'ent': entropy_tensor, 'v': v}

        return node_updates, edge_updates, None

    def graph_grads(self):
        layers, avg_grads, max_grads = [], [], []
        for n, p in self.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        plot_grad_flow(layers, avg_grads, max_grads)

    def print_layer_weights(self, which='all'):
        if which == 'all':
            models = self.models
        elif which == 'node_input':
            models = [self.node_input_model]
        elif which == 'edge_input':
            models = [self.edge_input_model]
        elif which == 'node_update':
            models = [self.node_update_model]
        elif which == 'edge_update':
            models = [self.edge_update_model]
        elif which == 'actor':
            models = [self.actor_model]
        elif which == 'critic':
            models = [self.critic_model]

        for model in models:
            print('Model: ' + model.name)
            for n, p in model.named_parameters():
                if (p.requires_grad) and ("bias" not in n):
                    print(str(n) + ': ' + str(p[0][:10]))
