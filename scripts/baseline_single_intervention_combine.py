import pandas as pd

intervened_test = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.intervened_test.csv', comment='#')
intervened_train = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.intervened_train.csv', comment='#')
not_intervened_test = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.not_intervened_test.csv', comment='#')
not_intervened_train = pd.read_csv('../data/baseline/single_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_single_intervention.not_intervened_train.csv', comment='#')

test_data = pd.concat([intervened_test, not_intervened_test])
train_data = pd.concat([intervened_train, not_intervened_train])

test_data.to_csv('baseline.single_intervention.test.csv', mode='a', index=False)
train_data.to_csv('baseline.single_intervention.train.csv', mode='a', index=False)