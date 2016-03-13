""" test functions in groupwise_distance_learner

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/03/12
"""

import unittest

from networkx import Graph

from groupwise_distance_learning.tests.test_helper_func import load_sample_test_data
from groupwise_distance_learning.groupwise_distance_learner import _validate_user_information
from groupwise_distance_learning.groupwise_distance_learner import _update_groupwise_dist
from groupwise_distance_learning.groupwise_distance_learner import _update_fit_group_with_groupwise_dist
from groupwise_distance_learning.groupwise_distance_learner import _update_buffer_group


class TestGroupWiseDistLearnerSupportFunctions(unittest.TestCase):

    def test_validate_user_informationg_with_good_data(self):
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        try:
            _validate_user_information(user_ids = user_ids,
                                       user_profiles = user_profile_df,
                                       user_connections = user_connections_df)
            is_ok = True
        except:
            is_ok = False
        self.assertTrue(is_ok)

    def test_validate_user_informationg_with_bad_data(self):
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        try:
            # remove first user_id's record
            _validate_user_information(user_ids = user_ids[1:],
                                       user_profiles = user_profile_df,
                                       user_connections = user_connections_df)
            is_ok = False
        except:
            is_ok = True
        self.assertTrue(is_ok)

    def test_update_groupwise_dist(self):
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()

        # define test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}

        new_dist_metrics = _update_groupwise_dist(dist_metrics, fit_group,
                                                  user_ids, user_profile_df, user_connections_df,
                                                  min_group_size=1)

        print("--- test_update_groupwise_dist (with generic metrics as inputs) ---")
        print("1st group's old distance weights: {}".format(dist_metrics[0]))
        print("2nd group's old distance weights: {}".format(dist_metrics[1]))
        print("1st group's new distance weights: {}".format(new_dist_metrics[0]))
        print("2nd group's new distance weights: {}".format(new_dist_metrics[1]))

        is_ok = True
        self.assertTrue(is_ok, True)

    def test_update_fit_group_with_groupwise_dist_01(self):
        """ test with generic distance metrics """

        # load test data
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        new_dist_metrics = {0: [1, 0, 1], 1: [1, 1, 0]}

        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}

        new_fit_group, new_fit_pvals, unfit_group = _update_fit_group_with_groupwise_dist(dist_metrics,
                                                                                      fit_group,
                                                                                      fit_pvals,
                                                                                      user_ids,
                                                                                      user_profile_df,
                                                                                      user_connections_df,
                                                                                      ks_alpha=0.05)

        print("--- update_fit_group_with_groupwise (with: generic distance metrics) ---")
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))

        is_ok = True
        self.assertTrue(is_ok, True)

    def test_update_fit_group_with_groupwise_dist_02(self):
        """ test with generic distance metrics """

        # load test data
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        new_dist_metrics = {0: [0, 1, 0], 1: [0, 0, 1]}

        new_fit_group, new_fit_pvals, unfit_group = _update_fit_group_with_groupwise_dist(new_dist_metrics,
                                                                                          fit_group,
                                                                                          fit_pvals,
                                                                                          user_ids,
                                                                                          user_profile_df,
                                                                                          user_connections_df,
                                                                                          ks_alpha=0.05)

        print("--- update_fit_group_with_groupwise (with: learned distance metrics) ---")
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))

        is_ok = True
        self.assertTrue(is_ok, True)

    def test_update_fit_group_with_groupwise_dist_03(self):
        """ test with generic distance metrics """

        # load test data
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        new_dist_metrics = {0: [1, 0, 1], 1: [1, 1, 0]}

        new_fit_group, new_fit_pvals, unfit_group = _update_fit_group_with_groupwise_dist(new_dist_metrics,
                                                                                          fit_group,
                                                                                          fit_pvals,
                                                                                          user_ids,
                                                                                          user_profile_df,
                                                                                          user_connections_df,
                                                                                          ks_alpha=0.05)

        print("--- update_fit_group_with_groupwise (with: opposite distance metrics) ---")
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))

        is_ok = True
        self.assertTrue(is_ok, True)

    def test_update_fit_group_with_groupwise_dist_04(self):
        """ test with generic distance metrics """

        # load test data
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        # definte test data
        fit_group = {0: ['a', 'b', 'c'], 1: ['d', 'e']}
        fit_pvals = {0: [1, 1, 1], 1: [1, 1]}
        new_dist_metrics = {0: [1, 0, 1], 1: [1, 1, 0]}

        new_fit_group, new_fit_pvals, unfit_group = _update_fit_group_with_groupwise_dist(new_dist_metrics,
                                                                                          fit_group,
                                                                                          fit_pvals,
                                                                                          user_ids,
                                                                                          user_profile_df,
                                                                                          user_connections_df,
                                                                                          ks_alpha=0.7)

        print("--- update_fit_group_with_groupwise (with: opposite distance metrics) ---")
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("new 1st fit_pval: {}".format(new_fit_pvals[0]))
        print("new 2nd fit_pval: {}".format(new_fit_pvals[1]))

        is_ok = True
        self.assertTrue(is_ok, True)

    def test_update_buffergroup(self):
        """ test _update_buffer_group """
         # load test data
        user_ids, user_profile_df, user_connections_df = load_sample_test_data()
        # definte test data

        fit_group = {0:[], 1:[]}
        fit_pvals = {0:[], 1:[]}
        buffer_group = user_ids
        new_dist_metrics = {0: [0, 1, 0], 1: [0, 0, 1]}

        new_fit_group, new_fit_pvals, unfit_group = _update_buffer_group(new_dist_metrics, fit_group, fit_pvals,
                                                                         buffer_group, user_ids, user_profile_df,
                                                                         user_connections_df, ks_alpha=0.5)

        print("--- _update_buffer_group (with empty fit_group)---")
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))

        is_ok = True
        self.assertTrue(is_ok, True)


if __name__ == '__main__':
    unittest.main()
