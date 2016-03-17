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
gwd_laraner.fit(user_ids, user_profiles, user_connections)

# access learned results
print(" the score of best learning metrics: {}".format(gwd_learner.get_score()) )

# extract dictionary-like distance metrics for each group
learned_dist_metrics = gwd_learner.get_groupwise_weights()
# unpack learned clusters/groups: 
# learned_fit_groups, whithin members have suitable distance metrics to decode their perferrence
# learned_buffer_groups, whithin members failed to find customized distance metrics to reprsent their perferrence
#    they will be treated with generic unweighted diteance metrics 
learned_fit_groups, learned_buffer_group = gwd_learner.get_get_user_cluster()
```