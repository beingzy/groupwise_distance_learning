""" test groupwise_disetance_learner

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/03/13
"""
import unittest
from groupwise_distance_learning.tests.test_helper_func import load_sample_test_data
from groupwise_distance_learning.groupwise_distance_learner import _groupwise_dist_learning_single_run


class TestGroupWiseDistLearnerRun(unittest.TestCase):

    def setUp(self):
        pass

    def test_single_run_01(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}
        buffer_group = []

        single_run_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                             user_ids, user_profiles, user_connections,
                                                             ks_alpha=0.05, min_group_size=1, verbose=True)

        new_dist_metrics, new_fit_group, new_fit_pvals, new_buffer_group = single_run_res

        print("--- single run test ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("1nd new_dist_metrics's distance metrics: {}".format(new_dist_metrics[1]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))
        print("buffer_group: {}".format(new_buffer_group))

    def test_single_run_02(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        # definte test data
        fit_group = {0: ['a'], 1: ['d', 'e', 'b', 'c']}
        fit_pvals = {0: [1], 1: [1, 1, 1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}
        buffer_group = []

        single_run_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                             user_ids, user_profiles, user_connections,
                                                             ks_alpha=0.05, min_group_size=1, verbose=True)

        new_dist_metrics, new_fit_group, new_fit_pvals, new_buffer_group = single_run_res

        print("--- single run test ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("1nd new_dist_metrics's distance metrics: {}".format(new_dist_metrics[1]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))
        print("buffer_group: {}".format(new_buffer_group))

    def test_single_run_03(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        # definte test data
        fit_group = {0: [], 1: ['a', 'd', 'e', 'b', 'c']}
        fit_pvals = {0: [], 1: [1, 1, 1, 1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}
        buffer_group = []

        single_run_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                             user_ids, user_profiles, user_connections,
                                                             ks_alpha=0.05, min_group_size=1, verbose=True)

        new_dist_metrics, new_fit_group, new_fit_pvals, new_buffer_group = single_run_res

        print("--- single run test ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("1nd new_dist_metrics's distance metrics: {}".format(new_dist_metrics[1]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))
        print("buffer_group: {}".format(new_buffer_group))

    def test_single_run_04(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        # definte test data
        fit_group = {0: [], 1: ['a', 'd', 'e']}
        fit_pvals = {0: [], 1: [1, 1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}
        buffer_group = []

        single_run_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                             user_ids, user_profiles, user_connections,
                                                             ks_alpha=0.05, min_group_size=1, verbose=True)

        new_dist_metrics, new_fit_group, new_fit_pvals, new_buffer_group = single_run_res

        print("--- single run test ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("1nd new_dist_metrics's distance metrics: {}".format(new_dist_metrics[1]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))
        print("buffer_group: {}".format(new_buffer_group))


if __name__ == '__main__':
    unittest.main()
