""" Group-wise distance learner
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/MM/DD
"""
import numpy as np
from numpy.random import choice
from pandas import DataFrame
import networkx as nx
from networkx import Graph
from math import floor
from datetime import datetime
from copy import deepcopy

from groupwise_distance_learning.util_functions import user_grouped_dist
from groupwise_distance_learning.util_functions import user_dist_kstest
from groupwise_distance_learning.util_functions import ldm_train_with_list
from groupwise_distance_learning.util_functions import find_fit_group
from groupwise_distance_learning.util_functions import zipf


def _init_dict_list(k):
    """ create dictionary with k items, each
        item is a empty list
    """
    res_dict = {}
    for ii in range(k):
        res_dict[ii] = []
    return res_dict


def _convert_array_to_list(x):
    """ convert data stucture's memeber to from np.array to list"""
    if isinstance(x, dict):
        for key in x.keys():
            item = x[key]
            if isinstance(item, np.ndarray):
                x[key] = item.tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return x


def all_to_list(func):
    """decorator: convert nd.array to list"""
    def func_wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        new_res = []

        for ii in res:
            new_res.append(_convert_array_to_list(ii))

        if isinstance(res, tuple):
            return tuple(new_res)
        else:
            return new_res

    return func_wrapper


def _validate_user_information(user_ids, user_profiles, user_graph):
    """ validate user-related information
    """
    uniq_user_ids = user_graph.nodes()
    num_strange_users = len([uid for uid in uniq_user_ids if not uid in user_ids])

    if len(user_ids) != user_profiles.shape[0]:
        raise ValueError("user_profiles has a different number of records than user_ids's!")

    if num_strange_users > 0:
        raise ValueError("strange users are found in user_connections!")

    if not isinstance(user_graph, Graph):
        raise ValueError("user_graph must be an instance of networkx.Graph!")


def _update_groupwise_dist(dist_metrics, fit_group, user_ids, user_profiles, user_graph,
                           min_group_size=5, random_state=None):
    """ learning gruopwise distnace metrics """
    n_feat = user_profiles.shape[1]
    # create data container
    new_dist_metrics = dist_metrics.copy()

    for gg, gg_user_ids in fit_group.items():
        # ldm() optimized distance metrics - weights
        # for selected users
        if len(gg_user_ids) > min_group_size:
            single_dist_weights = ldm_train_with_list(gg_user_ids, user_ids, user_profiles, user_graph)
            new_dist_metrics[gg] = single_dist_weights
        else:
            if not gg in new_dist_metrics:
                # intialize default distance metrics weights
                new_dist_metrics[gg] = [1] * n_feat
    return new_dist_metrics


def _update_fit_group_with_groupwise_dist(dist_matrics,
                                          fit_group, fit_pvals,
                                          user_ids, user_profiles, user_graph,
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

    fit_group = _convert_array_to_list(fit_group)
    fit_pvals = _convert_array_to_list(fit_pvals)

    # create container
    fit_group_copy = fit_group.copy()
    unfit_group = {}

    for gg, gg_user_ids in fit_group_copy.items():
        gg_dist = dist_matrics[gg]

        for ii, ii_user_id in enumerate(gg_user_ids):
            sim_dist, diff_dist = user_grouped_dist(ii_user_id, gg_dist, user_ids, user_profiles,  user_graph)
            ii_pval = user_dist_kstest(sim_dist, diff_dist)

            if ii_pval < ks_alpha:
                # remove the user from fit group, retreive [0] to ensure slice is integer
                idx = [idx for idx, uid in enumerate(fit_group[gg]) if uid == ii_user_id][0]
                # fit_group[gg].remove(idx)
                del fit_group[gg][idx]
                # fit_pvals[gg].remove(idx)
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
                         user_ids, user_profiles, user_graph, ks_alpha=0.05):
    """ return fit_group, fit_pvals, buffer_group
        redistribute member in buffer group into fit_group if fit had been found
    """
    # to keep API consistant
    # restore user_profiles to DataFrame including
    buffer_group_copy = buffer_group.copy()
    if len(buffer_group_copy) > 0:
        for ii, ii_user_id in enumerate(buffer_group_copy):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, dist_metrics,
                                                       user_ids, user_profiles, user_graph, ks_alpha,
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
                                              user_ids, user_profiles, user_graph, ks_alpha=0.05):
    """ update members in unfit_group with cross-group distance. unfit members are kept in buffer_group
    """

    unfit_group_copy = unfit_group.copy()
    for gg, gg_user_ids in unfit_group_copy.items():
        # extract cross-group distance metrics dictionary to avoid duplicate
        # tests with distance metrics associated with user's group
        other_group_keys = [group_key for group_key in dist_metrics.keys() if not group_key == gg]
        cross_group_dist_metrics = {key: dist_metrics[key] for key in other_group_keys}

        for ii, ii_user_id in enumerate(gg_user_ids):
            ii_new_group, ii_new_pval = find_fit_group(ii_user_id, cross_group_dist_metrics,
                                                       user_ids, user_profiles, user_graph, ks_alpha,
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


def _fit_score(pvals, buffer_group, C=1):
    """
    sum(pvals) / #gouped_user - C * #buffer_group/#total_user
    """
    sum_pvals = 0
    num_grouped_users = 0

    for group in pvals.keys():
        sum_pvals += sum(pvals[group])
        num_grouped_users += len(pvals[group])

    num_buffer_users = len(buffer_group)
    total_users = num_grouped_users + num_buffer_users

    score = sum_pvals / num_grouped_users - C * num_buffer_users / total_users

    return score


def _validate_input_learned_info(dist_metrics, fit_group, fit_pvals):
    """ validate input data
    """
    if not len(dist_metrics) == len(fit_group) == len(fit_pvals):
        msg = "input pacakge (dist_metrics, fit_group, fit_pvals) are not compatiable! "
        raise ValueError(msg)

    fit_group_keys = list(fit_group.keys())
    fit_pvals_keys = list(fit_pvals.keys())

    for ii, key_pairs in enumerate(zip(fit_group_keys, fit_pvals_keys)):
        gkey, pkey = key_pairs
        n_item_fit_group = len(fit_group[gkey])
        n_item_fit_pvals = len(fit_pvals[pkey])
        if n_item_fit_group != n_item_fit_pvals:
            msg = "fit_group's {} group and fit_pvals {} group has different length of items".format(gkey, pkey)
            raise ValueError(msg)


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

    # entire run's execution time
    total_time = 0

    # validate the input data is compatible
    _validate_input_learned_info(dist_metrics, fit_group, fit_pvals)

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
                            n_group=2, max_iter=200, max_nogain_streak=10,
                            min_group_size=5, ks_alpha=0.95,
                            alpha_update_freq=5, learning_rate = 0.1,
                            init="zipf", C=0.1,
                            verbose=False, is_debug=False, random_state=None):
    """ groupwise distance learning algorithm to classify users.
    it returns: ((dist_metrics, fit_group, buffer_group), _max_fit_score)

    Parameters:
    ----------
    user_ids: list of all user_id

    user_profile: matrix-like of user profiles, records should align with user_ids

    user_graph: networkx.Graph instance stores user_connections information

    n_group: integer, the number of groups to learn

    max_iter: integer, the maximum number of iteration for learning

    min_group_size: integer, the minimum number of members for a group

    ks_alpha: the initial alpha value for ks-test
        H0: distr.(conencted users) >= distr.(disconnected users)

    alpha_update_freq: integer
        update ks_alpha for every no-improving iterations of size sepcified by alpha_update_freq

    learning_rate: float [0, 1]
        change ks_alpha = ks_alpha - learing_rate * ks_alpha at each update

    init: character, {'even', 'zipf')

    verbose: boolean, optional, default value = False
       verbosity mode

    random_state: integer or numpy.RandomState, optional
       the generator used to initialize the group composition. If an integer
       is given it fixes the seed. Defaults to the global numpy random number
       generator


    """

    _validate_user_information(user_ids, user_profiles, user_connections)

    if max_iter < 0:
        msg = "Invalid number of initilizations n_group (={}) must be bigger than zero.".format(max_iter)
        raise ValueError(msg)

    if max_nogain_streak < 0:
        msg = "Invalid number of initilizations n_nogain_streak (={}) must be bigger than zero.".format(max_iter)
        raise ValueError(msg)

    # initiate containers
    dist_metrics = _init_dict_list(n_group)
    fit_group = _init_dict_list(n_group)
    fit_pvals = _init_dict_list(n_group)
    buffer_group = []

    # initiating the group's composition
    total_users = len(user_ids)

    if init == "zipf":
        group_sizes = [int(prob * total_users) for prob in zipf(n_group)]
        # margin
        margin = total_users - sum(group_sizes)
        if margin > 0:
            # append extra user cap to first group
            group_sizes[0] += margin
    else:
        # elif init == "even"
        sample_size = floor( total_users / n_group )
        group_sizes = [sample_size] * n_group
        # margin
        margin = total_users - (sample_size * n_group)
        if margin > 0:
            # append extra user cap to first group
            group_sizes[0] += margin

    # initiate fit_group, fit_pvals
    # by distributing users to different groups
    user_ids_copy = user_ids.copy()
    group_names = list(dist_metrics.keys())

    # assign users to each groups
    for gname, gsize in zip(group_names, group_sizes):
        # assign users to group
        draws = choice(user_ids_copy, gsize, replace=False)
        fit_group[gname] = draws
        # create inititial group-associated p-values
        fit_pvals[gname] = [1] * gsize

        for drawed_user_id in draws:
            user_ids_copy.remove(drawed_user_id)

    # create container to collect information to
    # track learning process
    if is_debug:
        iter_hist = []
        ks_alpha_hist = []
        fs_hist = []
        knowledge_pkgs = []
        timers = []

    # learning process
    _nogain_streak = 0
    _iterate_counter = 0
    _max_fit_score = 0
    # output package
    best_knowledge_pack = None

    for ii in range(max_iter):

        loop_start_time = datetime.now()
        iter_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                       user_ids, user_profiles, user_connections,
                                                       ks_alpha, min_group_size, verbose,
                                                       random_state)


        loop_duration = (datetime.now() - loop_start_time).total_seconds()
        dist_metrics, fit_group, fit_pvals, buffer_group = iter_res
        knowledge_pack = deepcopy(dist_metrics), deepcopy(fit_group), deepcopy(buffer_group)

        # evaluate current knowledge pack
        fit_score = _fit_score(fit_pvals, buffer_group, C=C)

        if verbose:
            msg = "-- {}th iteration's fit score: {:.4f}\n".format(_iterate_counter, fit_score)
            msg += "-- ks-alpha: {:.3f}\n".format(ks_alpha)
            msg += "-- time cost: {:.0f} seconds\n".format(loop_duration)
            msg += "-- size of buffer group: {}\n".format(len(buffer_group))
            print(msg)

        # capture the best learned knowledge
        if fit_score > _max_fit_score:
            # find a better solution
            _max_fit_score = fit_score
            best_knowledge_pack = knowledge_pack
            # reset non effective learning
            _nogain_streak = 0
        else:
            # marginal learning
            _nogain_streak += 1

        if is_debug:
            iter_hist.append(ii)
            ks_alpha_hist.append(ks_alpha)
            timers.append(loop_duration)
            fs_hist.append(fit_score)
            knowledge_pkgs.append(knowledge_pack)

        _iterate_counter += 1

        if _nogain_streak % alpha_update_freq == 0 and _nogain_streak != 0:
            # reduce ks_alpha at every 5 non-increment gain
            new_ks_alpha = ks_alpha - ks_alpha * learning_rate
            if new_ks_alpha < 0.01:
                ks_alpha = 0.01
            else:
                ks_alpha = new_ks_alpha

        if _nogain_streak >= max_nogain_streak:
            break

    if is_debug:
        track_pack = DataFrame({
            "iter_hist": iter_hist,
            "ks_alpha_hist": ks_alpha_hist,
            "timers": timers,
            "fs_hist": fs_hist,
        })
        debug_info = track_pack, knowledge_pkgs
        return best_knowledge_pack, _max_fit_score, debug_info
    else:
        return best_knowledge_pack, _max_fit_score


class GroupwiseDistLearner(object):
    """ groupwise distance learning class

    Parameters:
    ----------
    user_ids: list of all user_id

    user_profile: matrix-like of user profiles, records should align with user_ids

    user_graph: networkx.Graph instance stores user_connections information

    n_group: integer, the number of groups to learn

    max_iter: integer, the maximum number of iteration for learning

    tol: float, tolerance for incremental gain in fit score

    min_group_size: integer, the minimum number of members for a group

    ks_alpha: alpha value for ks-test
        H0: distr.(conencted users) >= distr.(disconnected users)

    init: character, {'even', 'zipf')

    verbose: boolean, optional, default value = False
       verbosity mode

    random_state: integer or numpy.RandomState, optional
       the generator used to initialize the group composition. If an integer
       is given it fixes the seed. Defaults to the global numpy random number
       generator
    """

    def __init__(self, n_group=2,
                 max_iter=200, max_nogain_streak=10,
                 min_group_size=5,
                 ks_alpha=0.95, alpha_update_freq = 5, learning_rate = 0.1,
                 C=0.1, init="zipf", verbose=False,
                 is_debug=False, random_state=None):
        self._n_group = n_group
        self._max_iter = max_iter
        self._max_nogain_streak = max_nogain_streak
        self._min_group_size = min_group_size
        self._ks_alpha = ks_alpha
        self._alpha_update_freq = alpha_update_freq
        self._learning_rate = learning_rate
        self._init = init
        self._C = C
        self._verbose = verbose
        self._is_debug = is_debug
        self._random_state = random_state
        # attributes for learned results
        self._dist_metrics = None
        self._fit_group = None
        self._buffer_group = None
        self._score = None
        self._debug_info = None

    def fit(self, user_ids, user_profiles, user_connections):

        res = groupwise_dist_learning(user_ids, user_profiles, user_connections,
                                      n_group=self._n_group, max_iter=self._max_iter,
                                      max_nogain_streak=self._max_nogain_streak,
                                      min_group_size=self._min_group_size,
                                      ks_alpha=self._ks_alpha,
                                      alpha_update_freq=self._alpha_update_freq, learning_rate=self._learning_rate,
                                      init=self._init, C=self._C, verbose=self._verbose, is_debug=self._is_debug,
                                      random_state=self._random_state)
        # unpack results
        if self._is_debug:
            knowledge_pack, best_score, debug_info = res
            dist_metrics, fit_group, buffer_group = knowledge_pack
            self._debug_info = debug_info
        else:
            knowledge_pack, best_score = res
            dist_metrics, fit_group, buffer_group = knowledge_pack

        self._score = best_score
        self._dist_metrics = dist_metrics
        self._fit_group = fit_group
        self._buffer_group = buffer_group

    def get_score(self):
        return self._score

    def get_groupwise_weights(self):
        return self._dist_metrics

    def get_user_cluster(self):
        return self._fit_group, self._buffer_group

    def get_debug_info(self):
        return self._debug_info

