""" Group-wise distance learner
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
from groupwise_distance_learning.kstest import kstest_2samp_greater
from groupwise_distance_learning.util_functions import init_embed_list
from groupwise_distance_learning.util_functions import init_dict_list
from groupwise_distance_learning.util_functions import user_grouped_dist
from groupwise_distance_learning.util_functions import user_dist_kstest
from groupwise_distance_learning.util_functions import users_filter_by_weights
from groupwise_distance_learning.util_functions import ldm_train_with_list
from groupwise_distance_learning.util_functions import find_fit_group
from groupwise_distance_learning.util_functions import get_fit_score


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

