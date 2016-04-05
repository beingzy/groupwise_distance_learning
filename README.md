#### Group Wise Distance Learning Algorithm
The algorithm clusters users with respects to their social preferrences. Their social preferrences are implicitly expressed 
by users' social connections with regards to whom they are connected with or whom they are not connected with. The 
algorithm is proposed in Yi Zhang's PhD dissertation as a means to improve the effectiveness of user recommendation 
for online social-network-based products.

#### How to use the algorithm
```python
from groupwise_distance_learning.tests.test_helper_func import load_sample_test_data
from groupwise_distance_learning.groupwise_distance_learner import GroupwiseDistLearner

# load sample data 
user_ids, user_profiles, user_connections = load_sample_test_data()

# initiate learner
gwd_learner = GroupwiseDistLearner(n_group=2, min_group_size=1)
gwd_learner.fit(user_ids, user_profiles, user_connections)

# access learned results
print(" the score of best learning metrics: {}".format(gwd_learner.get_score()) )

# extract dictionary-like distance metrics for each group
learned_dist_metrics = gwd_learner.get_groupwise_weights()
# unpack learned clusters/groups: 
# learned_fit_groups, whithin members have suitable distance metrics to decode their perferrence
# learned_buffer_groups, whithin members failed to find customized distance metrics to reprsent their perferrence
#    they will be treated with generic unweighted diteance metrics 
learned_fit_groups, learned_buffer_group = gwd_learner.get_user_cluster()
```


#### How to track the learning process
1. `GroupwiseDistLearner` class provide method to return information generated in the process of learning, if `is_debug = True` is assigned.
```python
# define learner
gwd_learner = GroupwiseDistLearner(n_group=2, max_iter=100, max_nogain_streak=10, 
                                   alpha_update_freq=2,
                                   is_debug=True, verbose=True, 
                                   C=0.5)

# load social network information and execute learning process                              
gwd_learner.fit(user_ids, user_profile_df, user_connections)
# retrive debug information                                  
track_df, knowledge_pkg = gwd_learner.get_debug_info()
```