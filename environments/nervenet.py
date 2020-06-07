from copy import deepcopy
import random
import networkx as nx
from collections import OrderedDict
import torch


class Environment:
    def __init__(self, args):
        self.episode_C, self.model_C, self.goal_C, self.agent_C, self.other_C, self.device, self.G_whole, self.pages, \
        self.node_feats, self.edges = args

    def step(self, selected_node_rel, ep_step, state):
        state_copy = self._copy_state(*state)
        G_curr, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors = state_copy[:]

        selected_node_abs = list(current_nodes.keys())[selected_node_rel]
        achieved_goal = self.add_children(G_curr, selected_node_abs, goal_node, current_nodes)
        # Done if goal found or end of ep
        done = True if (ep_step == self.episode_C['max_ep_steps'] - 1) or achieved_goal else False
        reward = self.reward_func(done, achieved_goal, shortest_path_length, ep_step + 1)
        predecessors = self.get_predecessors(G_curr, current_nodes)

        if not done:
            next_state = self._copy_state(G_curr, current_nodes, goal_node, goal_feats, shortest_path_length,
                                                 predecessors)
        else:
            next_state = self.reset()

        return next_state, reward, done, achieved_goal

    # Deep copy everything before it goes into a state
    def _copy_state(self, G_curr, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors):
        G_curr_copy = G_curr.copy()
        current_nodes_copy = current_nodes.copy()
        goal_feats_copy = goal_feats.copy()
        predecessors_copy = deepcopy(predecessors)

        return (G_curr_copy, current_nodes_copy, goal_node, goal_feats_copy, shortest_path_length, predecessors_copy)

    # For now this is -1 per timestep +5 on terminal for reaching goal, -5 on terminal for not reaching goal
    # And when it reaches the goal, give add (shortest_path_length - 1) - num actions taken (neg number)
    def reward_func(self, terminal, reach_goal, shortest_path_length, num_actions_taken):
        rew = -1
        if terminal:
            if reach_goal:
                assert num_actions_taken >= (shortest_path_length - 1)
                rew += self.other_C['reach_goal_rew']
                rew += ((shortest_path_length - 1) - num_actions_taken)  # optimal num actions - num actions taken
            else:
                rew += self.other_C['not_reach_goal_rew']
        return rew

    def reset(self):
        current_try = 0
        while True:
            current_try += 1
            init_node = random.randint(0, self.model_C['num_nodes'] - 1)
            goal_node = random.randint(0, self.model_C['num_nodes'] - 1)
            # restart if goal node is init node, or no path
            if init_node == goal_node or not nx.has_path(self.G_whole, init_node, goal_node):
                continue
            # restart if shortest path is too long or too short
            shortest_path_length = nx.shortest_path_length(self.G_whole, init_node, goal_node)
            if shortest_path_length < self.episode_C['shortest_path_range_allowed_MIN'] or shortest_path_length > self.episode_C[
                'shortest_path_range_allowed_MAX']:
                continue
            break

        # Get goal feats
        goal_feats = self.node_feats[goal_node]
        assert goal_feats.shape == (self.model_C['node_feat_size'],)
        # Make init graph
        G_init = nx.DiGraph()
        G_init.add_node(init_node)
        current_nodes = OrderedDict({init_node: 0})  # Init current nodes dict
        got_goal = self.add_children(G_init, init_node, goal_node, current_nodes)
        assert sorted(list(current_nodes.values())) == list(current_nodes.values())
        assert not got_goal
        predecessors = self.get_predecessors(G_init, current_nodes)
        return (G_init, current_nodes, goal_node, goal_feats, shortest_path_length, predecessors)

    def add_children(self, G_curr, node_indx, goal_node_indx, current_nodes):
        achieved_goal = False
        # Check the children of the node to see if they need to be added to the current graph
        children = self.G_whole.successors(node_indx)
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
    def get_predecessors(self, G_curr, current_nodes):
        all_preds = []  # List of lists
        for node in current_nodes.keys():
            preds_abs = G_curr.predecessors(node)  # abs to all nodes, keys to the dict
            preds_rel = [current_nodes[x] for x in preds_abs]
            all_preds.append(preds_rel)
        return all_preds  # Returns a list of lists with the values being tth rel node indices

    def _unpack_states(self, states, node_feats_tensor):
        num_nodes_all = []
        goal_feats_all = []
        predecessors_all = []
        node_states_all = []
        total_num_nodes = 0
        for state in states:
            G_curr, current_nodes, goal_node, goal_feats, shortest_path, predecessors = state[:]
            num_nodes_all.append(len(current_nodes))
            total_num_nodes += num_nodes_all[-1]
            goal_feats_tensor = torch.tensor(goal_feats, device=self.device, requires_grad=True, dtype=torch.float)
            goal_feats_all.append(goal_feats_tensor)
            predecessors_all.append(predecessors)
            node_states_all.append(node_feats_tensor[list(current_nodes.keys())])
        node_states_all_tensor = torch.cat(node_states_all, dim=0)
        goal_feats_all_tensor = torch.stack(goal_feats_all, dim=0)
        return node_states_all_tensor, goal_feats_all_tensor, num_nodes_all, predecessors_all, total_num_nodes

    def _stack_goals(self, goal_tensors_all, num_nodes_all):
        stacked_goal_embeds_all = []
        for i in range(len(num_nodes_all)):
            goal = goal_tensors_all[i]
            num_nodes = num_nodes_all[i]
            stacked_goal_embeds_all.append(torch.stack([goal] * num_nodes))
        stacked_goal_emebds_all_tensor = torch.cat(stacked_goal_embeds_all, dim=0)
        return stacked_goal_emebds_all_tensor

    def propagate(self, gnn, states, actions=None):
        node_feats_tensor = torch.tensor(self.node_feats, device=self.device, requires_grad=True, dtype=torch.float)
        node_states_all, goal_feats_all, num_nodes_all, predecessors_all, total_num_nodes = self._unpack_states(
            states, node_feats_tensor)
        # If goal_input_layer is True then embed the goal by sending it into the input layer
        if self.goal_C['goal_input_layer']:
            goal_embeddings = gnn.input_model(goal_feats_all)
            assert goal_embeddings.shape == (len(states), self.model_C['node_hidden_size'])
            stacked_goal_embeds = self._stack_goals(goal_embeddings, num_nodes_all)
            assert stacked_goal_embeds.shape == (total_num_nodes, self.model_C['node_hidden_size'])
        else:
            stacked_goal_embeds = self._stack_goals(goal_feats_all, num_nodes_all)
            assert stacked_goal_embeds.shape == (total_num_nodes, self.model_C['node_feat_size'])

        for p in range(self.episode_C['num_props']):
            node_states_all, prediction = gnn(node_states_all, p == 0, p == self.episode_C['num_props'] - 1,
                                              predecessors_all, stacked_goal_embeds, num_nodes_all, actions)
        return prediction
