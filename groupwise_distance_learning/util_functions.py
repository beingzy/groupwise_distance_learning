""" utility functions facilitate the learner class
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/29
"""
import os
import sys
import glob
import numpy as np
import scipy as sp
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import copy

from scipy.stats import rayleigh


from learning_dist_metrics.ldm import LDM
from learning_dist_metrics.dist_metrics import weighted_euclidean
from groupwise_distance_learning.kstest import kstest_2samp_greater
from distance_metrics.GeneralDistanceWrapper import GeneralDistanceWrapper

def zipf_pdf(k, n, s=1):
    """ return the probability of nth rank
        Parameters:
        ----------
        k: {int} kth rank
        n: {int} total number of elemnets
        s: {float} distribution parameter
    """
    num = 1.0 / (k ** s)
    den = sum([1.0 / ((ii + 1) ** s) for ii in range(n)])
    return num / den


def zipf(n, s=1):
    """return zipf distributions"""
    return [zipf_pdf(k, n, s) for k in range(1, n+1)]


def normalize_user_record(a_profile_record, n_feat=None):
    """ convert an array of single row to list """
    if isinstance(a_profile_record, np.ndarray):
        if n_feat is None:
            _, n_feat = a_profile_record.shape

        row_list = a_profile_record.tolist()
        if len(row_list) < n_feat:
            row_list = row_list[0]
        return row_list

    else:
        return a_profile_record


def get_user_friends(targert_user_id, user_connections, is_directed=False):
    """ return a list of user_ids representing users connected with target users
    """
    if is_directed:
        conn_user_ids = [b_uid for a_uid, b_uid in user_connections if a_uid == targert_user_id]
    else:
        conn_user_ids = []
        for a_uid, b_uid in user_connections:
            if a_uid == targert_user_id:
                conn_user_ids.append(b_uid)
            if b_uid == targert_user_id:
                conn_user_ids.append(a_uid)

    return conn_user_ids


def gen_users_pair_signature(a, b, is_directed=False):
    """ generate string to encode user pair"""
    if is_directed:
        a, b = str(a), str(b)
        return "-".join([a, b])
    else:
        if a > b:
            a, b = str(a), str(b)
            return "-".join([b, a])
        else:
            a, b = str(a), str(b)
            return "-".join([a, b])


def user_grouped_dist(user_id, weights,
                      user_ids, user_profiles, user_connections,
                      dist_memory,
                      is_directed=False):
    """ return vector of weighted distance of a user vs. user's conencted users,
    and a vector of weighted distnces of a user vs. user's non-connected users.

    Parameters:
    ----------
    * user_id: {integer}, the target user's ID
    * weights: {vector-like, float}, the vector of feature weights which
        is extracted by LDM().fit(x, y).get_transform_matrix()
    * user_profile: <matrix-like, array>, a matrix of user profile, sorted by user_ids
    * user_connections: {list } a list of user id pairs representing connections
        to store the relationships
    * dist_memory: {dictionary} store key ('id-id'): value ('distance')
    (deprecated: * dist_func: {function} with argument two vector)
    * is_directed: {boolean} True, directed graph

    Returns:
    -------
    * (sim_dist, dissim_dist): list of distance of user-vs-friends, list of distance
    of user-vs-non-friends

    Examples:
    --------
    learned_weights = ldm().fit(df, users_list).get_transform_matrix()
    user_dist = user_grouped_dist(weights = learned_weights, user_id, user_ids,
        user_profiles, user_graph)
    """
    # initiate general distance wrapper to deal with categorical variable
    gd_wrapper = GeneralDistanceWrapper()
    gd_wrapper.fit(user_profiles)
    gd_wrapper.load_weights(weights)

    # create copy of distance metrics
    # dist_memory = copy.deepcopy(dist_memory)

    _, n_feats = user_profiles.shape

    # get the user_id of friends of the target user
    friend_ls = get_user_friends(user_id, user_connections, is_directed)
    non_friends_ls = [u for u in user_ids if u not in friend_ls + [user_id]]

    # retrieve target user's profile
    u_idx = [i for i, uid in enumerate(user_ids) if uid == user_id]
    user_profile = user_profiles[u_idx, :]
    user_profile = normalize_user_record(user_profile, n_feats)

    sim_dist_vec = []
    for f_id in friend_ls:
        f_idx = [i for i, uid in enumerate(user_ids) if uid == f_id]
        pair_sign = gen_users_pair_signature(u_idx, f_idx, is_directed)
        if pair_sign in dist_memory:
            the_dist = dist_memory[pair_sign]
        else:
            friend_profile = normalize_user_record(user_profiles[f_idx, :], n_feats)
            the_dist = gd_wrapper.dist_euclidean(user_profile, friend_profile)
            # insert distance of a new pair
            dist_memory[pair_sign] = the_dist
        # collect distance with friends
        sim_dist_vec.append(the_dist)

    diff_dist_vec = []
    for nf_id in non_friends_ls:
        nf_idx = [i for i, uid in enumerate(user_ids) if uid == nf_id]
        pair_sign = gen_users_pair_signature(u_idx, nf_idx, is_directed)
        if pair_sign in dist_memory:
            the_dist = dist_memory[pair_sign]
        else:
            nonfriend_profile = normalize_user_record(user_profiles[nf_idx, :], n_feats)
            the_dist = gd_wrapper.dist_euclidean(user_profile, nonfriend_profile)
            # insert distance of a new pair
            dist_memory[pair_sign] = the_dist
        # collect distance with friends
        diff_dist_vec.append(the_dist)

    return sim_dist_vec, diff_dist_vec


def user_dist_kstest(sim_dist_vec, diff_dist_vec,
                     fit_rayleigh=False, _n=100):

    """ Test the goodness of a given weights to defferentiate friend distance
        distributions and non-friend distance distributions of a given user.
        The distance distribution can be assumed to follow Rayleigh distribution.

    Parameters:
    ----------
    sim_dist_vec: {vector-like (list), float}, distances between friends
                  and the user
    diff_dist_vec: {vector-like (list), float}, distances between non-fri
                   -ends and the user
    fit_rayleigh: {boolean}, determine if fit data into Rayleigth distri
                  -bution
    _n: {integer}, number of random samples generated from estimated
        distribution

    Returns:
    -------
    * res: {float}: p-value of ks-tests with assumption that distances follow
            Rayleigh distribution.

    Examples:
    ---------
    pval = user_dist_kstest(sim_dist_vec, diff_dist_vec)
    """

    # convert list to numpy.arrray, which can be
    # automatice transfer to R readable objects
    # for R-function, if the proper setting is
    # configured
    sim_dist_vec = np.array(sim_dist_vec)
    diff_dist_vec = np.array(diff_dist_vec)

    if fit_rayleigh:
        friend_param = rayleigh.fit(sim_dist_vec)
        nonfriend_param = rayleigh.fit(diff_dist_vec)

        samp_friend = rayleigh.rvs(friend_param[0], friend_param[1], _n)
        samp_nonfriend = rayleigh.rvs(nonfriend_param[0], nonfriend_param[1], _n)

        # ouput p-value of ks-tests
        test_stat, pval = kstest_2samp_greater(samp_friend, samp_nonfriend)
    else:
        test_stat, pval = kstest_2samp_greater(sim_dist_vec, diff_dist_vec)

    return pval


def users_filter_by_weights(weights, user_ids, user_profiles, user_connections,
                            dist_memory,
                            is_directed=False,
                            pval_threshold=0.5,
                            mutate_rate=0.4,
                            fit_rayleigh=False,
                            _n=1000):
    """ Split users into two groups, "keep" and "mutate", with respect to
        p-value of the ks-tests on the null hypothesis that the distribution of
        friends' weighted distance is not significantly different from the
        couterpart for non-friends. Assume the weighted distances of each group
        follow Rayleigh distribution.

    Parameters:
    ----------
    weights: {vector-like, float}, the vector of feature weights which
        is extracted by LDM().fit(x, y).get_transform_matrix()
    user_ids: {list} all user ids following same order of user_profiles
    user_profiles: {numpy.array, matrix-like}
    user_connecions: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
    is_directed: {boolean, default=False}
        False: consider user_connections as undirected graph
        True: as directed graph
    pval_threshold: {float}, the threshold for p-value to reject hypothesis
    min_friend_cnt: {integer}, drop users whose total of friends is less than
       this minimum count
    mutate_rate: {float}, a float value [0 - 1] determine the percentage of
       bad_fits member sent to mutation
    fit_rayleigh: {boolean}, determine if fit data into Rayleigth distri
                  -bution
    _n: {integer}, number of random samples generated from estimated
        distribution
    is_debug: {boolean}, to control if it yeilds by-product information

    Returns:
    -------
    res: {list} grouped list of user ids
        res[0] stores all users whose null hypothesis does not holds;
        res[1] stores all users whose null hypothesis hold null hypothesis,
        given weights, distance distribution of all friends is significantly
        different from distance distribution of all non-friends

    Examples:
    --------
    weights = ldm().fit(df, friends_list).get_transform_matrix()
    profile_df = users_df[["ID"] + cols]
    grouped_users = users_filter_by_weights(weights,
                       profile_df, friends_df, pval_threshold = 0.10,
                       min_friend_cnt = 10)

    Notes:
    -----
    min_friend_cnt is not implemented
    """
    pvals = []

    for uid in user_ids:
        sim_dist, diff_dist = user_grouped_dist(uid, weights,
                                                user_ids, user_profiles, user_connections,
                                                dist_memory,
                                                is_directed)
        pval = user_dist_kstest(sim_dist, diff_dist, fit_rayleigh, _n)
        pvals.append(pval)

    sorted_id_pval = sorted(zip(user_ids, pvals), key=lambda x: x[1])

    good_fits = [i for i, p in sorted_id_pval if p >= pval_threshold]
    bad_fits = [i for i, p in sorted_id_pval if p < pval_threshold]

    if len(bad_fits) > 0:
        mutate_size = np.ceil(len(bad_fits) * mutate_rate)
        mutate_size = max(int(mutate_size), 1)
        id_retain = good_fits + bad_fits[mutate_size:]
        id_mutate = bad_fits[:mutate_size]
    else:
        id_retain = good_fits
        id_mutate = bad_fits

    return id_retain, id_mutate


def ldm_train_with_list(users_list, user_ids, user_profiles, user_connections, retain_type=1):
    """ learning distance matrics with ldm() instance, provided with selected
        list of users.

    Parameters:
    -----------
    users_list: {vector-like, integer}, the list of user id
    user_ids: {list} all user ids following same order of user_profiles
    user_profiles: {numpy.array, matrix-like}
    user_connecions: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
    retain_type: {integer}, 0, adopting 'or' logic by keeping relationship in
        friends_df if either of entities is in user_list 1, adopting 'and'
        logic

    Returns:
    -------
    res: {vector-like, float}, output of ldm.get_transform_matrix()

    Examples:
    ---------
    new_dist_metrics = ldm_train_with_list(user_list, profile_df, friends_df)
    """
    if retain_type == 0:
        friends = [(a, b) for a, b in user_connections if a in users_list or b in users_list]
    else:
        friends = [(a, b) for a, b in user_connections if a in users_list and b in users_list]

    ldm = LDM()
    ldm.fit(user_ids=user_ids, user_profiles=user_profiles, S=friends)
    weight_vec = ldm.get_transform_matrix()
    return weight_vec


def find_fit_group(uid, dist_metrics,
                   user_ids, user_profiles, user_connections,
                   dist_memory_container,
                   is_directed=False,
                   threshold=0.5, current_group=None, fit_rayleigh=False, _n=1000):
    """ calculate user p-value for the distance metrics of
        each group

    Parameters:
    ----------
    uid: {integer}, user id
    dist_metrics: {dictionary}, all {index: distance_metrics}
    user_ids: {list} all user ids following same order of user_profiles
    user_profiles: {numpy.array, matrix-like}
    user_connecions: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
    is_directed: {boolean, default=False}
        False: consider user_connections as undirected graph
        True: as directed graph
    threshold: {float}, threshold for qualifying pvalue of ks-tests
    current_group: {integer}, group index
    fit_rayleigh: {boolean}

    Resutls:
    --------
    res: {list}, [group_idx, pvalue]
    """
    if current_group is None:
        other_groups = list(dist_metrics.keys())
        # other_dist_metrics = list(dist_metrics.values())
    else:
        other_groups = [group for group in dist_metrics.keys() if group != current_group]
        # other_dist_metrics = [dist for group, dist in dist_metrics.items() if group != current_group]

    if len(other_groups) > 0:
        # only excute this is at least one alternative group
        pvals = []

        for ii, group in enumerate(other_groups):
            dist_weights = dist_metrics[group]
            dist_memory = dist_memory_container[group]
            # loop through all distance metrics and calculate
            # p-value of ks-tests by applying it to the user
            # relationships
            sim_dist, diff_dist = user_grouped_dist(user_id=uid,
                                                    weights=dist_weights,
                                                    user_ids=user_ids, user_profiles=user_profiles,
                                                    user_connections=user_connections,
                                                    dist_memory=dist_memory,
                                                    is_directed=is_directed)
            # update the group's distance metrics
            # dictionary is immutable, no need to re-assign
            # to the parent variable to acquire update value
            # dist_memory_container[group] = dist_memory
            # append pval
            pval = user_dist_kstest(sim_dist_vec=sim_dist, diff_dist_vec=diff_dist,
                                    fit_rayleigh=fit_rayleigh, _n=_n)

            pvals.append(pval)

        # find group whose distance metrics explained a user's existing
        # connections at the best degree.
        max_pval = max(pvals)
        max_idx = [ii for ii, pval in enumerate(pvals) if pval == max_pval][0]
        best_group = other_groups[max_idx]

        if max_pval < threshold:
            # reject null hypothesis
            best_group = None
            max_pval = None

    else:
        best_group = None
        max_pval = None

    return best_group, max_pval
