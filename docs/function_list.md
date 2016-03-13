* user_grouped_dist(user_id, weights, profile_df, friend_networkx)
    calculate distance between a user and whose friends and distance
    between a user and whose non-friends. The grouped distance vector will
    be the outcome
    
* kstest_2samp_greater(x, y)
    calculate the test statistics and p-value for KS-test 

* user_dist_kstest(sim_dist_vec, diff_dist_vec, fit_rayligh=False, _n=100)
    calculate the goodness regarding a given weight's ability on differentiating
    friend's distances distribution vs. non-friend distance distribution of a given
    user
    
    # comment deprecate fit_rayleigh and _n 
    # need kstest_2samp_greater()
    
* users_filter_by_weights(weights, profile_df, friends_networkx, pval_threshold=0.5, ...)
    split users into two groups, "keep" (good fit group) and "mutate" (not good fit group)
    
* ldm_train_with_list(user_list, profile_df, friends, retain_type=1)
    leanring distance metrics with ldm() instance for a selection of users
    
* _init_embed_list(n)

* _init_dict_list(n)

* find_fit_group(uid, dist_metrics, profile_df, friend_networkx, ...)
    caculate user's p-value for the distance metrics of each group

* get_fit_score(fit_pvals, buffer_group, c)

