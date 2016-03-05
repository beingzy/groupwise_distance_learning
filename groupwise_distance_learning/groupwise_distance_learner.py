""" Group-wise distance learner
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
from datetime import datetime

from groupwise_distance_learning.kstest import kstest_2samp_greater
from groupwise_distance_learning.util_functions import user_grouped_dist
from groupwise_distance_learning.util_functions import user_dist_kstest
from groupwise_distance_learning.util_functions import users_filter_by_weights
from groupwise_distance_learning.util_functions import ldm_train_with_list
from groupwise_distance_learning.util_functions import find_fit_group
from groupwise_distance_learning.util_functions import get_fit_score


def _init_embed_list(n):
    """
    """
    ls = []
    for i in range(n):
        ls.append([])
    return ls


def _init_dict_list(k):
    """ create dictionary with k items, each
        item is a empty list
    """
    res_dict = {}
    for ii in range(k):
        res_dict[ii] = []
    return res_dict


def _validate_user_information(user_ids, user_profiles, user_connections):
    """ validate user-related information
    """
    a_user_ids = list(set([uid for uid, _ in user_connections]))
    b_user_ids = list(set([uid for _, uid in user_connections]))
    uniq_user_ids = list(set(a_user_ids + b_user_ids))
    num_strange_users = len([uid for uid in uniq_user_ids if not uid in user_ids])

    if len(user_ids) != user_profiles.shape[0]:
        raise ValueError("user_profiles has a different number of records than user_ids's!")

    if num_strange_users > 0:
        raise ValueError("strange users are found in user_connections!")


def _update_fit_groups_with_groupwise_dist():
    """update members in fit_group with distance metrics unfit member will be
    sent to unfit group
    """
    pass

def _update_buffergroup():
    """redistribute member in buffer group into fit_group if fit had been found
    """
    pass

def _update_unfit_groups_with_crossgroup_dist():
    """update members in unfit_group with cross-group distance. unfit members
    are kept in buffer_group
    """
    pass


def groupwise_dist_learning(user_ids, user_profiles, user_connections,
                            n_group, max_iter, tol,
                            verbose=False, random_state=None):
    """ groupwise distance learning algorithm to classify users

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

    _validate_user_information(user_ids, user_profiles, user_connections)

    pass


def _groupwise_dist_learning_single(user_profiles, user_connections, user_graph,
                                    dist_matrics, fit_group, fit_pvals, unfit_group,
                                    buffer_group, ks_alpha=0.05,
                                    min_group_size=5, verbose=False, random_state=None):
    """ a single run of groupwise distance learning

    Parameters:
    ----------
    user_ids: list of all user_id

    user_profile: matrix-like of user profiles, records should align with user_ids

    user_connections: list of user id pair representing connections

    user_graph: networkx.Graph store user conenctions

    dist_metrics: dictionary
        distance metrics containers

    fit_group: dictionary,
        members composition in fit groups

    fit_pvals: dictionary,
       the p-value of KS-test with user's in fit_group

    unfit_group: dictionary,
        members is not considerd fit by its group distance metrics

    buffer_group: list
        members are not considered having fit

    ks_alpha: float
        alpha value for ks-test

    min_group_size: integer
        the minimal size requirement for any groups

    verbose: boolean, optional, default value = False
       verbosity mode

    random_state: integer or numpy.RandomState, optional
       the generator used to initialize the group composition. If an integer
       is given it fixes the seed. Defaults to the global numpy random number
       generator

    Returns;
    -------
    """
    start_time = datetime.now()

    # develop a function to ensure user-ids and user_profile match

    n_user, n_feat = user_profiles.shape

    # step 00: learn distance metriccs
    for gg, gg_user_ids in fit_group.items():
        # ldm() optimized distance metrics - weights
        # for selected users
        if len(gg_user_ids) > min_group_size:
            dist_weights = ldm_train_with_list(gg_user_ids, user_profiles, user_connections)
            dist_matrics[gg] = dist_weights
        else:
            if not gg in dist_matrics:
                # intialize default distance metrics weights
                dist_matrics[gg] = [1] * user_profiles.shape[1]

    # step 01: update the member composition in response to
    # newly learned distance metrics weights
    fit_group_copy = fit_group.copy()

    for gg, gg_user_ids in fit_group_copy.items():
        gg_dist = dist_matrics[gg]

        for ii, ii_user_id in enumerate(gg_user_ids):
            sim_dist, diff_dist = user_grouped_dist(ii_user_id, gg_dist, user_profiles, user_graph)
            ii_pval = user_dist_kstest(sim_dist, diff_dist)

            if ii_pval < ks_alpha:
                # remove the user from fit group
                idx = [idx for idx, uid in enumerate(fit_group[gg]) if uid == ii_user_id]
                del fit_group[gg][idx]
                del fit_pvals[gg][idx]
                # add the user into unfit group
                if gg in unfit_group:
                    unfit_group[gg].append(ii_user_id)
                else:
                    unfit_group[gg] = [ii_user_id]
            else:
                # update pvalue for user
                idx = [idx for idx, uid in enumerate(fit_group[gg]) if uid == ii_user_id]
                fit_pvals[gg][idx] = ii_pval

    # step 02: test members in buffer group with all updated distance metrics
    # fit with other distance metrics
    buffer_group_copy = buffer_group.copy()
    if len(buffer_group_copy) > 0:
        for ii, ii_user_id in enumerate(buffer_group_copy):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, dist_matrics,
                                                       user_profiles, user_graph, ks_alpha,
                                                       current_group=None, fit_rayleigh=False)
            if not ii_new_group is None:
                # remove member with fit from buffer_group
                buffer_group.remove(ii_user_id)
                if ii_new_group in fit_group:
                    fit_group[ii_new_group].append(ii_user_id)
                    fit_pvals[ii_new_group].append(ii_new_pval)
                else:
                    fit_group[ii_new_group] = [ii_user_id]
                    fit_pvals[ii_new_group] = [ii_new_pval]

    # step 03: test members in unfit_group with cross-group distance metrics
    unfit_group_copy = unfit_group.copy()
    for gg, gg_user_ids in unfit_group_copy.items():
        # extract cross-group distance metrics dictionary to avoid duplicate
        # test with distance metrics associated with user's group
        other_group_keys = [group_key for group_key in dist_matrics.keys() if not group_key == gg]
        cross_group_dist_metrics = {key: dist_matrics[key] for key in other_group_keys}
        for ii, ii_user_id in enumerate(gg_user_ids):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, cross_group_dist_metrics,
                                                       user_profiles, user_graph, ks_alpha,
                                                       current_group=None, fit_rayleigh=False)
            # redistribute the user based on fit-test
            if not ii_new_group is None:
                # remove member with fit from buffer_group
                if ii_new_group in fit_group:
                    fit_group[ii_new_group].append(ii_user_id)
                    fit_pvals[ii_new_group].append(ii_new_pval)
                else:
                    fit_group[ii_new_group] = [ii_user_id]
                    fit_pvals[ii_new_group] = [ii_new_pval]
            else:
                buffer_group.append(ii_user_id)
    # clean up unfit_group
    unfit_group = {}

    return (dist_matrics, fit_group, fit_pvals, buffer_group)



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

