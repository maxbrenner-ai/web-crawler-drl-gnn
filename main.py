import time
from utils import *
from train_deepmind_parallel import run_deepmind
from train_nervenet_parallel import run_nervenet


def get_data_path(data_type, node_feat_type):
    if data_type == 'very easy':
        dir = 'data/animals-D3-20K-nodes76-avgedges4.0-totaledges305-max10-minout2-minin2/'
    elif data_type == 'easy':
        dir = 'data/animals-D3-20K-nodes77-avgedges3.6-totaledges278-max10-minout1-minin1/'
    elif data_type == 'medium':
        dir = 'data/animals-D3-20K-nodes74-avgedges3.4-totaledges255-max10-minout2-minin2/'
    elif data_type == 'hard':
        dir = 'data/animals-D3-20K-nodes78-avgedges3.0-totaledges235-max10-minout2-minin1/'
    elif data_type == 'very hard':
        dir = 'data/animals-D3-20K-nodes68-avgedges2.5-totaledges167-max10-minout1-minin1/'
    else:
        assert True == False, 'incorrect data tpe input'

    if node_feat_type == 0:
        return dir + 'features_title.pkl'
    elif node_feat_type == 1:
        return dir + 'features_soup.pkl'
    elif node_feat_type == 2:
        return dir + 'features_concat.pkl'
    else:
        assert True == False, 'incorrect node feat type input'


def run_normal(num_experiments=1):
    # Load constants
    constants = load_constants('constants/constants.json')
    episode_C, model_C, goal_C, agent_C, other_C = constants['episode_C'], constants['model_C'], constants['goal_C'], \
                                                   constants['agent_C'], constants['other_C']
    G_whole, pages, node_feats, edge_feats, edges = load_data_make_graph(
        get_data_path(other_C['data'], other_C['node_feat_type']))
    fill_in_missing_hyp_params(model_C, goal_C, len(pages), len(edges), node_feats.shape[1], edge_feats.shape[1])
    for exp in range(num_experiments):
        print(' --- Running experiment {} --- '.format(exp))

        exp_start = time.time()
        if model_C['model_type'] == 'deepmind':
            run_deepmind(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edge_feats,
                         edges, None)
        else:
            run_nervenet(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges, None)

        exp_end = time.time()
        print('Time taken (m): {:.2f}'.format((exp_end - exp_start) / 60.))


# Only works for testing a single variable
# param_set: out of the 5 hyperparm dicts that are made which one is the tested var in
# var: name of var
# values: values to search
def run_grid_search_single_variable(num_repeat_experiment, param_set, var, values):
    constants = load_constants('constants/constants.json')
    episode_C, model_C, goal_C, agent_C, other_C = constants['episode_C'], constants['model_C'], constants['goal_C'], \
                                                   constants['agent_C'], constants['other_C']
    for value in values:
        # Set the value in the constants
        if param_set == 'episode':
            episode_C[var] = value
        elif param_set == 'model':
            model_C[var] = value
        elif param_set == 'goal':
            goal_C[var] = value
        elif param_set == 'agent':
            agent_C[var] = value
        elif param_set == 'other':
            other_C[var] = value

        G_whole, pages, node_feats, edge_feats, edges = load_data_make_graph(
            get_data_path(other_C['data'], other_C['node_feat_type']))
        fill_in_missing_hyp_params(model_C, goal_C, len(pages), len(edges), node_feats.shape[1], edge_feats.shape[1])

        for same_experiment in range(num_repeat_experiment):
            # Load df for saving data
            df = pd.read_excel('run-data.xlsx')

            exp_start = time.time()

            print(' --- Running experiment {}: {} - {} --- '.format(var, value, same_experiment))

            if model_C['model_type'] == 'deepmind':
                run_deepmind(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats,
                             edge_feats,
                             edges, df)
            else:
                run_nervenet(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges,
                             df)

            exp_end = time.time()
            print('Time taken (m): {:.2f}\n'.format((exp_end - exp_start) / 60.))


def run_random_search(num_diff_experiments, num_repeat_experiment):
    # Load grid of constants
    grid = load_constants('constants/constants-grid.json')

    for diff_experiment in range(num_diff_experiments):
        # First pick the hyp params to use
        episode_C, model_C, goal_C, agent_C, other_C = select_hyp_params(grid)
        G_whole, pages, node_feats, edge_feats, edges = load_data_make_graph(
            get_data_path(other_C['data'], other_C['node_feat_type']))
        fill_in_missing_hyp_params(model_C, goal_C, len(pages), len(edges), node_feats.shape[1], edge_feats.shape[1])

        for same_experiment in range(num_repeat_experiment):
            # Load df for saving data
            df = pd.read_excel('run-data.xlsx')

            exp_start = time.time()

            print(' --- Running experiment {}.{} --- '.format(diff_experiment, same_experiment))

            if model_C['model_type'] == 'deepmind':
                run_deepmind(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats,
                             edge_feats,
                             edges, df)
            else:
                run_nervenet(episode_C, model_C, goal_C, agent_C, other_C, device, G_whole, pages, node_feats, edges,
                             df)

            exp_end = time.time()
            print('Time taken (m): {:.2f}\n'.format((exp_end - exp_start) / 60.))



'''

REEMEMBER TO CHANGE BACK THE EVALS!!!!!!!!


'''

'''


'''

if __name__ == '__main__':
    device = torch.device('cpu')

    # print('Num cores: {}'.format(mp.cpu_count()))

    # run_normal(num_experiments=3)

    # refresh_excel('run-data.xlsx')
    # run_random_search(num_diff_experiments=100, num_repeat_experiment=3)

    refresh_excel('run-data.xlsx')
    run_grid_search_single_variable(num_repeat_experiment=3, param_set='model', var='model_type',
                                    values=['deepmind', 'nervenet', 'fully_connected', 'no_structure'])
