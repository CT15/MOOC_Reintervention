import pandas as pd

intervened_test = pd.read_csv('../data/baseline/multiple_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_multiple_intervention.intervened_test.csv', comment='#')
intervened_train = pd.read_csv('../data/baseline/multiple_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_multiple_intervention.intervened_train.csv', comment='#')
not_intervened_test = pd.read_csv('../data/baseline/multiple_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_multiple_intervention.not_intervened_test.csv', comment='#')
not_intervened_train = pd.read_csv('../data/baseline/multiple_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_multiple_intervention.not_intervened_train.csv', comment='#')

test_data = pd.concat([intervened_test, not_intervened_test])
train_data = pd.concat([intervened_train, not_intervened_train])

test_data.to_csv('baseline.multiple_intervention.test.csv', mode='w+', index=False)
train_data.to_csv('baseline.multiple_intervention.train.csv', mode='w+', index=False)