""" Group-wise distance learner
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
from groupwise_distance_learning.kstest import kstest_2samp_greater
from groupwise_distance_learning.util_functions import user_grouped_dist
from groupwise_distance_learning.util_functions import user_dist_kstest
from groupwise_distance_learning.util_functions import users_filter_by_weights
from groupwise_distance_learning.util_functions import ldm_train_with_list
from groupwise_distance_learning.util_functions import find_fit_group
from groupwise_distance_learning.util_functions import get_fit_score


def init_embed_list(n):
    """
    """
    ls = []
    for i in range(n):
        ls.append([])
    return ls


def init_dict_list(k):
    """ create dictionary with k items, each
        item is a empty list
    """
    res_dict = {}
    for ii in range(k):
        res_dict[ii] = []
    return res_dict


def groupwise_dist_learning():
    pass


def _groupwise_dist_learning_single(user_ids, user_profiles, user_graph,
                                    n_group, max_iter, tol,
                                    verbose=False, random_state=None):
    """ a single run of groupwise distance learning

    Parameters:
    ----------
    user_ids: list of all user_id

    user_profile: matrix-like of user profiles, records should align with user_ids

    user_graph: networkx.Graph instance stores user_connections information

    n_group: integer, the number of groups to learn

    max_iter: integer, the maximum number of iteration for learning

    tol: float, tolerance for incremental gain in fit score

    verbose: boolean, optional, default value = False
       verbosity mode

    random_state: integer or numpy.RandomState, optional
       the generator used to initialize the group composition. If an integer
       is given it fixes the seed. Defaults to the global numpy random number
       generator

    Returns;
    -------
    """
    pass


class GroupwiseDistLearner(object):


    def __init__(self):
        pass

    def _fit(self):
        pass

    def fit(self):
        pass

    def load_evaluator(self, score_funcs):
        pass

    def get_score(self):
        pass

    def get_groupwise_weights(self):
        pass

    def get_user_cluster(self):
        pass

