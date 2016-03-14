""" Group-wise distance learner
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
from datetime import datetime
from pandas import DataFrame
from networkx import Graph
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


def _update_groupwise_dist(dist_metrics, fit_group, user_ids, user_profiles, user_connections,
                           min_group_size=5, random_state=None):
    """ learning gruopwise distnace metrics """
    nfeat = user_profiles.shape[1]
    # restore user_profiles to DataFrame including
    user_profile_df = DataFrame(user_profiles)
    user_profile_df.columns = ["feat_0", "feat_1", "feat_2"]
    user_profile_df["ID"] = user_ids
    # create data container
    new_dist_metrics = dist_metrics.copy()

    for gg, gg_user_ids in fit_group.items():
        # ldm() optimized distance metrics - weights
        # for selected users
        if len(gg_user_ids) > min_group_size:
            single_dist_weights = ldm_train_with_list(gg_user_ids, user_profile_df, user_connections)
            new_dist_metrics[gg] = single_dist_weights
        else:
            if not gg in new_dist_metrics:
                # intialize default distance metrics weights
                new_dist_metrics[gg] = [1] * nfeat
    return new_dist_metrics


def _update_fit_group_with_groupwise_dist(dist_matrics,
                                          fit_group, fit_pvals,
                                          user_ids, user_profiles, user_connections,
                                          ks_alpha=0.05):
    """ return fit_group, fit_pvals, unfit_group by updating members in fit_group
    with distance metrics unfit member will be sent to unfit group.
    (fit_group, fit_pvals, unfit_group)

    Parameters:
    ----------
    dist_metrics: dictionary

    fit_group: dictionary

    fit_pvals: dictionary

    user_profiles: matrix-like (numpy.array)

    user_graph:

    ks_alpha: float, default value = 0.05

    Returns:
    -------
    fit_group, fit_pvals, unfit_group
    """

    # restore user_profiles to DataFrame including
    user_profile_df = DataFrame(user_profiles)
    user_profile_df["ID"] = user_ids

    user_graph = Graph()
    user_graph.add_edges_from(user_connections)

    # create container
    fit_group_copy = fit_group.copy()
    unfit_group = {}

    for gg, gg_user_ids in fit_group_copy.items():
        gg_dist = dist_matrics[gg]

        for ii, ii_user_id in enumerate(gg_user_ids):
            sim_dist, diff_dist = user_grouped_dist(ii_user_id, gg_dist, user_profile_df, user_graph)
            ii_pval = user_dist_kstest(sim_dist, diff_dist)

            if ii_pval < ks_alpha:
                # remove the user from fit group, retreive [0] to ensure slice is integer
                idx = [idx for idx, uid in enumerate(fit_group[gg]) if uid == ii_user_id][0]
                del fit_group[gg][idx]
                del fit_pvals[gg][idx]
                # add the user into unfit group
                if gg in unfit_group:
                    unfit_group[gg].append(ii_user_id)
                else:
                    unfit_group[gg] = [ii_user_id]
            else:
                # update pvalue for user, retreive [0] to ensure slice is integer
                idx = [idx for idx, uid in enumerate(fit_group[gg]) if uid == ii_user_id][0]
                fit_pvals[gg][idx] = ii_pval

    return fit_group, fit_pvals, unfit_group


def _update_buffer_group(dist_metrics, fit_group, fit_pvals, buffer_group,
                         user_ids, user_profiles, user_connections, ks_alpha=0.05):
    """ return fit_group, fit_pvals, buffer_group
        redistribute member in buffer group into fit_group if fit had been found
    """
    # to keep API consistant
    # restore user_profiles to DataFrame including
    user_profile_df = DataFrame(user_profiles)
    user_profile_df["ID"] = user_ids

    user_graph = Graph()
    user_graph.add_edges_from(user_connections)

    buffer_group_copy = buffer_group.copy()
    if len(buffer_group_copy) > 0:
        for ii, ii_user_id in enumerate(buffer_group_copy):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, dist_metrics,
                                                       user_profile_df, user_graph, ks_alpha,
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

    return fit_group, fit_pvals, buffer_group


def _update_unfit_groups_with_crossgroup_dist(dist_metrics, fit_group, fit_pvals, unfit_group, buffer_group,
                                              user_ids, user_profiles, user_connections, ks_alpha=0.05):
    """ update members in unfit_group with cross-group distance. unfit members are kept in buffer_group
    """
    # to keep API consistant
    # restore user_profiles to DataFrame including
    user_profile_df = DataFrame(user_profiles)
    user_profile_df["ID"] = user_ids

    user_graph = Graph()
    user_graph.add_edges_from(user_connections)

    unfit_group_copy = unfit_group.copy()
    for gg, gg_user_ids in unfit_group_copy.items():
        # extract cross-group distance metrics dictionary to avoid duplicate
        # tests with distance metrics associated with user's group
        other_group_keys = [group_key for group_key in dist_metrics.keys() if not group_key == gg]
        cross_group_dist_metrics = {key: dist_metrics[key] for key in other_group_keys}

        for ii, ii_user_id in enumerate(gg_user_ids):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, cross_group_dist_metrics,
                                                       user_profile_df, user_graph, ks_alpha,
                                                       current_group=None, fit_rayleigh=False)
            # redistribute the user based on fit-tests
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

    return fit_group, fit_pvals, buffer_group


def _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                        user_ids, user_profiles, user_connections,
                                        ks_alpha=0.05, min_group_size=5, verbose=False,
                                        random_state=None):
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
       the p-value of KS-tests with user's in fit_group

    unfit_group: dictionary,
        members is not considerd fit by its group distance metrics

    buffer_group: list
        members are not considered having fit

    ks_alpha: float
        alpha value for ks-tests

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
    total_time = 0

    # develop a function to ensure user-ids and user_profile match

    n_user, n_feat = user_profiles.shape

    start_time = datetime.now()
    # step 00: learn distance metriccs
    dist_metrics = _update_groupwise_dist(dist_metrics, fit_group, user_ids, user_profiles, user_connections,
                                          min_group_size)
    if verbose:
        duration = (datetime.now() - start_time).total_seconds()
        total_time += duration
        print( "updating groupwise distance took about %.2f seconds\n" % duration )

    # step 01: update the member composition in response to
    # newly learned distance metrics weights
    start_time = datetime.now()
    fit_group, fit_pvals, unfit_group = _update_fit_group_with_groupwise_dist(dist_metrics, fit_group, fit_pvals,
                                                                              user_ids, user_profiles, user_connections,
                                                                              ks_alpha)
    if verbose:
        duration = (datetime.now() - start_time).total_seconds()
        total_time += duration
        print( "updating fit group with updated groupwise distance took about %.2f seconds\n" % duration )

    # step 02: tests members in buffer group with all updated distance metrics
    # fit with other distance metrics
    start_time = datetime.now()
    fit_group, fit_pvals, buffer_group = _update_buffer_group(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                              user_ids, user_profiles, user_connections, ks_alpha)
    if verbose:
        duration = (datetime.now() - start_time).total_seconds()
        total_time += duration
        print( "updating buffer_group with updated groupwise distance took about %.2f seconds\n" % duration )

    # step 03: tests members in unfit_group with cross-group distance metrics
    start_time = datetime.now()
    fit_group, fit_pvals, buffer_group = _update_unfit_groups_with_crossgroup_dist(dist_metrics, fit_group, fit_pvals,
                                                                                   unfit_group, buffer_group,
                                                                                   user_ids, user_profiles,
                                                                                   user_connections, ks_alpha)
    if verbose:
        duration = (datetime.now() - start_time).total_seconds()
        total_time += duration
        print( "updating unfit_group with updated cross-group distance took about %.2f seconds\n" % duration )

    return dist_metrics, fit_group, fit_pvals, buffer_group


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

    # initiate containers
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

