# Parent class (abstract)
import torch.nn as nn
import torch
import numpy as np
from utils import plot_grad_flow


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.loss = nn.CrossEntropyLoss()  # Obvs will need to be changed

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *input):
        raise NotImplementedError

    # Get the grads of a model
    def get_layer_grads(self, model):
        layers, avg_grads, max_grads = [], [], []
        for n, p in model.named_parameters():
            if (p.requires_grad) and ("bias" not in n):
                layers.append(n)
                avg_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        return layers, avg_grads, max_grads

    # Graph the grads of given models
    def graph_grads(self, models):
        layers = []
        avg_grads = []
        max_grads = []

        for model in models:
            l, a, m = self.get_layer_grads(model)
            layers.extend(l)
            avg_grads.extend(a)
            max_grads.extend(m)

        plot_grad_flow(layers, avg_grads, max_grads)

    # Just gets the loss for a set (doesnt optimize)
    def get_loss(self, outputs, targets):
        assert outputs.shape[0] == targets.shape[0]
        loss = self.loss(outputs, targets)
        return loss.data.tolist()

    def get_accuracy(self, outputs, targets):
        assert outputs.shape[0] == targets.shape[0]
        outputs_ = np.array(outputs.data.tolist())
        output_preds = np.argmax(outputs_, axis=1)
        targets = np.array(targets.data.tolist())
        acc = ((output_preds == targets).sum()) / targets.shape[0]
        return acc