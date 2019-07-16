import matplotlib.pyplot as plt


def plot_grad_flow(layers, ave_grads, max_grads):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems. '''

    #     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    #     plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    #     plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(left=0, right=len(ave_grads))
    #     plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.legend([Line2D([0], [0], color="c", lw=4),
    #                 Line2D([0], [0], color="b", lw=4),
    #                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, linewidth=1, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
