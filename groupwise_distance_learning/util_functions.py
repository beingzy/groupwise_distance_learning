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

from scipy.stats import rayleigh
from numpy.random import choice
from networkx import Graph

from learning_dist_metrics.ldm import LDM
from learning_dist_metrics.dist_metrics import weighted_euclidean
from groupwise_distance_learning.kstest import kstest_2samp_greater


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


def user_grouped_dist(user_id, weights, profile_df, friend_networkx):
    """ return vector of weighted distance of a user vs. user's conencted users,
    and a vector of weighted distnces of a user vs. user's non-connected users.

    Parameters:
    ----------
    * user_id: {integer}, the target user's ID
    * weights: {vector-like, float}, the vector of feature weights which
        is extracted by LDM().fit(x, y).get_transform_matrix()
    * profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    * friend_networkx: {networkx.Graph()}, Graph() object from Networkx
        to store the relationships informat
    # -- new interface --
    # * weights: <vector-like>, a vector of weights per user profile feature
    # * user_ids: <list> a list of user_ids
    # * user_profile: <matrix-like, array>, a matrix of user profile, sorted by user_ids
    # * user_connections: <matrix-like, array>, a matrix of user connections
    # * user_graph: <networkx.Graph>

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
    cols = [col for col in profile_df.columns if col is not "ID"]
    # get the user profile information of the target users
    user_profile = profile_df.ix[profile_df.ID == user_id, cols].as_matrix()
    # get the user_id of friends of the target user
    friends_ls = friend_networkx.neighbors(user_id)
    all_ids = profile_df["ID"]
    non_friends_ls = [u for u in all_ids if u not in friends_ls + [user_id]]

    sim_dist_vec = []
    for f_id in friends_ls:
        friend_profile = profile_df.ix[profile_df["ID"] == f_id, cols].as_matrix()
        the_dist = weighted_euclidean(user_profile, friend_profile, weights)
        sim_dist_vec.append(the_dist)

    diff_dist_vec = []
    for nf_id in non_friends_ls:
        non_friend_profile = profile_df.ix[profile_df.ID == nf_id, cols].as_matrix()
        the_dist = weighted_euclidean(user_profile, non_friend_profile, weights)
        diff_dist_vec.append(the_dist)

    res = [sim_dist_vec, diff_dist_vec]
    return res


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


def users_filter_by_weights(weights, profile_df, friends_networkx,
                            pval_threshold=0.5,
                            mutate_rate=0.4,
                            users_list=None,
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
    users_list: {vector-like, integer}, the list of user id
    profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    friends_networkx: {networkx.Graph()}, Graph() object from Networkx to store
        the relationships information
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
    # all_users_ids = list(set(profile_df.ID))
    # users_list
    # container for users meeting different critiria
    pvals = []
    if users_list is None:
        users_list = list(profile_df.ix[:, 0])

    for uid in users_list:
        res_dists = user_grouped_dist(uid, weights, profile_df, friends_networkx)
        pval = user_dist_kstest(res_dists[0], res_dists[1], fit_rayleigh, _n)
        pvals.append(pval)

    sorted_id_pval = sorted(zip(users_list, pvals), key=lambda x: x[1])

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

    res = [id_retain, id_mutate]

    return res


def ldm_train_with_list(users_list, profile_df, friends, retain_type=1):
    """ learning distance matrics with ldm() instance, provided with selected
        list of users.

    Parameters:
    -----------
    users_list: {vector-like, integer}, the list of user id
    profile_df: {matrix-like, pandas.DataFrame}, user profile dataframe
        with columns: ["ID", "x0" - "xn"]
    friends: {list of tuple}, each tuple keeps a pair of user id
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
        friends = [(a, b) for a, b in friends if a in users_list or b in users_list]
    else:
        friends = [(a, b) for a, b in friends if a in users_list and b in users_list]

    ldm = LDM()
    ldm.fit(profile_df, friends)
    weight_vec = ldm.get_transform_matrix()
    return weight_vec


def find_fit_group(uid, dist_metrics, profile_df,
                   friend_networkx, threshold=0.5,
                   current_group=None, fit_rayleigh=False, _n=1000):
    """ calculate user p-value for the distance metrics of
        each group

    Parameters:
    ----------
    uid: {integer}, user id
    dist_metrics: {dictionary}, all {index: distance_metrics}
    profile_df: {DataFrame}, user profile includes "ID" column
    friend_networkx: {networkx.Graph}, user relationships
    threshold: {float}, threshold for qualifying pvalue of ks-tests
    current_group: {integer}, group index
    fit_rayleigh: {boolean}

    Resutls:
    --------
    res: {list}, [group_idx, pvalue]
    """
    if current_group is None:
        other_group = list(dist_metrics.keys())
        other_dist_metrics = list(dist_metrics.values())
    else:
        other_group = [group for group in dist_metrics.keys() if group != current_group]
        other_dist_metrics = [dist for group, dist in dist_metrics.items() if group != current_group]

    if len(other_dist_metrics) > 0:
        # only excute this is at least one alternative group
        pvals = []

        for dist in other_dist_metrics:
            # loop through all distance metrics and calculate
            # p-value of ks-tests by applying it to the user
            # relationships
            sdist, ddist = user_grouped_dist(user_id=uid, weights=dist, profile_df=profile_df,
                                             friend_networkx=friend_networkx)

            pval = user_dist_kstest(sim_dist_vec=sdist, diff_dist_vec=ddist,
                                    fit_rayleigh=fit_rayleigh, _n=_n)

            pvals.append(pval)

        # find group whose distance metrics explained a user's existing
        # connections at the best degree.
        max_pval = max(pvals)
        max_idx = [ii for ii, pval in enumerate(pvals) if pval == max_pval][0]
        best_group = other_group[max_idx]

        if max_pval < threshold:
            # reject null hypothesis
            best_group = None
            max_pval = None

    else:
        best_group = None
        max_pval = None

    return (best_group, max_pval)