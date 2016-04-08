""" functions for developing

Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/03/10
"""
import os
import os.path
from os.path import dirname, abspath, join
import pandas as pd
from networkx import Graph, DiGraph


def get_file_parent_dir_path(level=1):
    """ return the path of the parent directory of current file """
    current_dir_path = dirname(abspath(__file__))
    path_sep = os.path.sep
    components = current_dir_path.split(path_sep)
    return path_sep.join(components[:-level])


def load_sample_test_data():
    """ load small test data """
    _root_dir = get_file_parent_dir_path(level=2)
    _data_dir = join(_root_dir, 'data', 'small_test')

    user_profile_fpath = join(_data_dir, "user_profile.csv")
    user_connections_fpath = join(_data_dir, "connections.csv")

    int_user_profile_df = pd.read_csv(user_profile_fpath, header=0, sep=',')
    user_connections_df = pd.read_csv(user_connections_fpath, header=0, sep=',')

    user_ids = int_user_profile_df.id.tolist()
    # remove id columns and cetegorical feature column
    user_profile_df = int_user_profile_df.drop(["id", "feat_3"], axis=1, inplace=False).as_matrix()
    user_connections_df = user_connections_df.as_matrix()

    user_graph = Graph()
    user_graph.add_edges_from(user_connections_df)

    return user_ids, user_profile_df, user_graph


def load_simulated_test_data():
    """ load simulationd data  with defined two groups """
    _root_dir = get_file_parent_dir_path(level=2)
    _data_dir = join(_root_dir, 'data', 'sim_two_groups')

    user_profile_fpath = join(_data_dir, "user_profiles.csv")
    user_connections_fpath = join(_data_dir, "friendships.csv")

    # prepare user profile information
    user_profile_df = pd.read_csv(user_profile_fpath, header=0, sep=",")
    # unpack data
    user_ids = user_profile_df.ID.tolist()
    user_true_groups = user_profile_df.decision_style.tolist()
    user_profile_df = user_profile_df.drop(["ID", "decision_style"], axis=1, inplace=False).as_matrix()

    user_connections_df = pd.read_csv(user_connections_fpath, header=0, sep=",")
    user_connections_df = (user_connections_df[user_connections_df.isFriend==1]
                       .drop('isFriend', axis=1, inplace=False).astype(int).as_matrix())

    user_graph = Graph()
    user_graph.add_edges_from(user_connections_df)

    return user_ids, user_profile_df, user_graph, user_true_groups




