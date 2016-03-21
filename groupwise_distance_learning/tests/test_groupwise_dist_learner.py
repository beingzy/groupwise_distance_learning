""" test groupwise_disetance_learner

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/03/13
"""
import unittest
from groupwise_distance_learning.tests.test_helper_func import load_sample_test_data
from groupwise_distance_learning.groupwise_distance_learner import _groupwise_dist_learning_single_run
from groupwise_distance_learning.groupwise_distance_learner import groupwise_dist_learning
from groupwise_distance_learning.groupwise_distance_learner import GroupwiseDistLearner

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

    def test_single_run_05(self):

        user_ids, user_profiles, user_connections = load_sample_test_data()

        # definte test data
        fit_group = {0: [], 1: ['a', 'd', 'e']}
        fit_pvals = {0: [], 1: [1, 1]}
        dist_metrics = {0: [1, 1, 1], 1: [1, 1, 1]}
        buffer_group = []

        try:
            single_run_res = _groupwise_dist_learning_single_run(dist_metrics, fit_group, fit_pvals, buffer_group,
                                                                 user_ids, user_profiles, user_connections,
                                                                 ks_alpha=0.05, min_group_size=1, verbose=True)
            # test internal mehtod regardin validating input data
            # it should capture the illegal input
            is_ok = False
        except:
            is_ok = True
        self.assertTrue(is_ok)


    def test_learner_01(self):

        user_ids, user_profiles, user_connections = load_sample_test_data()

        best_pack = groupwise_dist_learning(user_ids, user_profiles, user_connections, n_group=2,
                                            max_iter=20, max_nogain_streak=5, tol=0.01, min_group_size=1, ks_alpha=0.1,
                                            init='zipf', verbose=True, C=0.1)

        knowledge_pack, best_score = best_pack
        new_dist_metrics, new_fit_group, new_buffer_group = knowledge_pack

        print("--- leaner test (n_group=2) ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("1nd new_dist_metrics's distance metrics: {}".format(new_dist_metrics[1]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("new 2nd fit_group: {}".format(new_fit_group[1]))
        print("buffer_group: {}".format(new_buffer_group))

    def test_learner_02(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        best_pack = groupwise_dist_learning(user_ids, user_profiles, user_connections, n_group=1,
                                            max_iter=20, max_nogain_streak=5, tol=0.01, min_group_size=1, ks_alpha=0.1,
                                            init='zipf', verbose=True, C=0.1)

        knowledge_pack, best_score = best_pack
        new_dist_metrics, new_fit_group, new_buffer_group = knowledge_pack

        print("--- leaner test (n_group=1) ---")
        print("1st new_dist_metrics's distance metrics: {}".format(new_dist_metrics[0]))
        print("new 1st fit_group: {}".format(new_fit_group[0]))
        print("buffer_group: {}".format(new_buffer_group))

    def test_learner_class_init_even(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        gwd_learner = GroupwiseDistLearner(n_group=2, min_group_size=1, init="even", max_iter=10, verbose=True)
        gwd_learner.fit(user_ids, user_profiles, user_connections)

        print("--- learner class (n_group=2) with init='even' ---")
        print("best score: {}".format(gwd_learner.get_score()))

    def test_learner_class_init_zipf(self):
        user_ids, user_profiles, user_connections = load_sample_test_data()

        gwd_learner = GroupwiseDistLearner(n_group=2, min_group_size=1, init="zipf", max_iter=10, verbose=True)
        gwd_learner.fit(user_ids, user_profiles, user_connections)

        print("--- learner class (n_group=2) with init='zipf' ---")
        print("best score: {}".format(gwd_learner.get_score()))


if __name__ == '__main__':
    unittest.main()
