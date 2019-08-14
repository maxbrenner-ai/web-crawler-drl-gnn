import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
import torch.nn as nn
import torch
import random
import torch.multiprocessing as mp


# This assigns this agents grads to the shared grads at the very start
def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad


class Counter:
    """enable the chief to access worker's total number of updates"""

    def __init__(self):
        self.val = mp.Value("i", 0)
        self.lock = mp.Lock()

    def get(self):
        # used by chief
        with self.lock:
            return self.val.value

    def increment(self):
        # used by workers
        with self.lock:
            self.val.value += 1

    def reset(self):
        # used by chief
        with self.lock:
            self.val.value = 0

            
class Page:
    def __init__(self, text, links):
        self.text = text
        self.links = links  # out-links
        self.in_links = []
        self.indx = None  # Relative to the ordered dict below
        self.feats = None
        

def load_data_make_graph(datapath):
    # Load the wiki-dict i want
    with open(datapath, 'rb') as f:
        pages = pickle.load(f)
    # Convert to ordered dict so i can use indices to refer to pages
    # Convert pages to ordered dict
    pages = OrderedDict(pages)
    # Add indices and get feats for each page
    node_feats = []
    for indx, (title, obj) in enumerate(pages.items()):
        obj.indx = indx
        node_feats.append(obj.feats)
    node_feats = np.stack(node_feats)
    # Make edges for graph generation
    edges = []
    for title, obj in pages.items():
        for link in obj.links:
            in_node = obj.indx
            out_node = pages[link].indx
            edges.append((in_node, out_node))
    # Make whole graph
    G_whole = nx.DiGraph()
    G_whole.add_edges_from(edges)
    
    return G_whole, pages, node_feats, edges


# For seeing how long paths tend to be in a graph
def print_paths(num_nodes, init_node, goal_node, G, model_C, G_whole):
    arr = []
    for _ in range(200):
        init_node = random.randint(0, model_C['num_nodes']-1)
        goal_node = random.randint(0, model_C['num_nodes']-1)
        if not nx.has_path(G_whole, init_node, goal_node) or init_node == goal_node:
            continue
        shortest_path_length = nx.shortest_path_length(G_whole, init_node, goal_node)
        if shortest_path_length == 1:
            continue
        arr.append(shortest_path_length)
    np.array(arr).mean()

    
def cos_sim(a, b):
    return cosine_similarity(a, b)[:, 0]
    
    
# Works for both single contants and lists for grid
def load_constants(filepath):
    with open(filepath) as f:
        data = json.load(f)
    return data


# Some of the constants from the constants file need to be filled in 
def fill_in_missing_hyp_params(model_C, goal_C, num_nodes, num_edges, num_node_feats):
    model_C['num_nodes'] = num_nodes
    model_C['num_edges'] = num_edges
    model_C['node_feat_size'] = num_node_feats
    assert model_C['nervenet'], 'Deepmind arch not implemented yet!'
    model_C['edge_feat_size'] = None
    model_C['edge_hidden_size'] = None
    
    if goal_C['goal_input_layer']:
        goal_size = model_C['node_hidden_size']
    else:
        goal_size = model_C['node_feat_size']
    goal_C['goal_size'] = goal_size
    

def select_hyp_params(grid):
    def select_params(dic):
        return_dic = {}
        for name, values in list(dic.items()):
            return_dic[name] = random.sample(values, 1)[0]
        return return_dic
    episode_C = select_params(grid['episode_C'])
    model_C = select_params(grid['model_C'])
    goal_C = select_params(grid['goal_C'])
    agent_C = select_params(grid['agent_C'])
    other_C = select_params(grid['other_C'])
    return episode_C, model_C, goal_C, agent_C, other_C
    

def refresh_excel(filepath):
    df = pd.read_excel(filepath)
    df.drop(df.index.tolist(), inplace=True)
    df.to_excel(filepath, index=False)
    
    
# Vis. an episodes graphs
def vis_ep(ep_graphs):
    for G in ep_graphs:
        nx.draw_kamada_kawai(G, with_labels=True)
        plt.show()
    
    
class Storage:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'q', 'pi', 'log_pi', 'ent',
                       'adv', 'ret', 'q_a', 'log_pi_a',
                       'mean']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self):
        for k in self.keys:
            v = getattr(self, k)
            if len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def cat(self, keys):
        data = [getattr(self, k)[:self.size] for k in keys]
        return map(lambda x: torch.cat(x, dim=0), data)
 

def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=device, dtype=torch.float32)
    return x
  
    
def random_sample(indices, batch_size):
    indices = np.asarray(np.random.permutation(indices))
    batches = indices[:len(indices) // batch_size * batch_size].reshape(-1, batch_size)
    for batch in batches:
        yield batch
    r = len(indices) % batch_size
    if r:
        yield indices[-r:]
    

def layer_init_filter(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)
    
    
def layer_init(layer, w_scale=1.0):
    nn.init.xavier_uniform_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

    
def plot_grad_flow(layers, ave_grads, max_grads):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems. '''
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
