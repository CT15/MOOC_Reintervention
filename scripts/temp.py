import pandas as pd

df = pd.read_csv('../data/baseline/multiple_intervention/post.post_order_col.post_time_elapsed_col+comment.comment_order_col.comment_time_elapsed_col.combine_post_comment.extract_multiple_intervention.csv', comment='#')
intervened = df[df.intervened == 1]
not_intervened = df[df.intervened == 0]
assert len(df) == len(intervened) + len(not_intervened)
intervened.to_csv('baseline.multiple_intervention.intervened.csv', mode='w+', index=False)
not_intervened.to_csv('baseline.multiple_intervention.not_intervened.csv', mode='w+', index=False)