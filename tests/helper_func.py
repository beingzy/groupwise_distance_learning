""" helper function to load data
Author: Yi Zhang <beingzy@gmail.com>
Date: 2016/02/20
"""
from os import getcwd
from os.path import join
from pandas import read_csv

def load_test_data(root_dir=None):
    """ helper function to load all dependent data for testing """

    # configure relative path
    if root_dir is None:
        root_dir = getcwd()
    data_dir = join( root_dir, 'data' )
    # load data
    user_profile_df = read_csv( join(data_dir, 'user_profile.csv'), header=0 )
    user_connection_df = read_csv( join(data_dir, 'connections.csv'), header=None )

    # structure output data
    user_ids = user_profile_df['id'].tolist()
    user_profiles = user_profile_df.drop(['id'], axis=1).as_matrix()
    user_connections = user_connection_df.as_matrix()

    return user_ids, user_profiles, user_connections
