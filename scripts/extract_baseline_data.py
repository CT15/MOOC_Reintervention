import pandas as pd
import re
from urlextract import URLExtract
import os
from datetime import datetime
from pathlib import Path
import argparse

VERSION = 1.0

parser = argparse.ArgumentParser(description='Concatenate posts in the same thread')
parser.add_argument('-p', action='store', 
                    type=str, required=True, help='Path to baseline data csv')
parser.add_argument('-u', action='store',
                    type=str, required=True, help='Path to user csv')
parser.add_argument('-s', action='store', choices=['single', 'multiple'],
                    type=str, required=True, help='Whether extract single intervention')


extractor = URLExtract()
def replace_url(s):
    if extractor.has_urls(s):
        urls = extractor.find_urls(s, only_unique=True)
        for url in urls:
            s = s.replace(url, "<url>")
        
    return s

def normalizeString(s):
    # Replace any .!? by a whitespace + the character --> '!' becomes ' !'.
    # \1 means the furst bracketed group --> [,!?]
    # r is to not consider \1 as a character (r to escape a backslash)
    # + means 1 or more
    s = re.sub(r"([.!?])", r" \1", s)
    # Remove any character that is NOT a sequence of lower or upper case letters. + means one or more
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    # Remove a sequence of whitespace characters
    s = re.sub(r"\s+", r" ",s).strip()
    return s

def process_text(s):
    # Assuming that none of the non-html tag contains < or > (Qn: How to be sure?)
    s = re.sub(r'<[^>]*>', '', s) # Replace HTML tags with ""
    s = re.sub('\s+', ' ', s).strip() # Replace multiple spaces with single space
    s = replace_url(s) # replace url with <url>
    s = normalizeString(s)
    s = s.lower() # change everything to lower case
    return s

def execute(path, user_path, filename, is_single_intervention):
    all_posts = pd.read_csv(path, comment='#')
    all_posts['post_text'] = all_posts['post_text'].map(lambda x: process_text(x))
    all_posts['post_text'] = all_posts['post_text'].replace([''], '<html>')

    all_users = pd.read_csv(user_path, comment='#')
    instructor_posts = set(all_users[all_users.user_title == 'Instructor'].postid)
    intervened_threads = set(all_users[all_users.user_title == 'Instructor'].threadid.unique())

    data = []
    num_of_threads = len(all_posts.thread_id.unique())
    threads_completed = 0
    for threadid in list(all_posts.thread_id.unique()):
        df = all_posts[all_posts.thread_id == threadid]
        
        text = ''

        intervened = False # useful only when is_single_intervention=True

        for postid, post_text in zip(df.id, df.post_text):
            if postid in instructor_posts and len(text) > 0: # don't include the intervention post
                text = re.sub(r"\s+", r" ", text).strip()
                row_data = [threadid, text, 1]
                data.append(row_data)

                if is_single_intervention:
                    intervened = True
                    break

            text += post_text
            text += ' '

        if not intervened:
            text = re.sub(r"\s+", r" ", text).strip()
            row_data = [threadid, text, 0]
            data.append(row_data)

        threads_completed += 1
        print(str(threads_completed) + " / " + str(num_of_threads) + "\n")

    data_df = pd.DataFrame(data, columns = ['threadid', 'text', 'intervened'])

    output_file_path = f'{os.getcwd()}/results/{filename}'
    if is_single_intervention:
        output_file_path += '.extract_single_intervention.csv'
    else:
        output_file_path += '.extract_multiple_intervention.csv'

    script_file_name = os.path.basename(__file__)
    with open(output_file_path, 'w+') as f:
        f.write(f'# Run as {script_file_name} VERSION {VERSION} on {datetime.now()}\n')
        f.write(f'# Run with command `python {script_file_name} -p {path} -u {user_path} -s {is_single_intervention}`\n')

    data_df.to_csv(output_file_path, mode='a', index=False) 

if __name__ == '__main__':
    args = vars(parser.parse_args())
    path = args['p']
    user_path = args['u']
    is_single_intervention = (args['s'] == 'single')
    filename = Path(path).stem
    
    execute(path, user_path, filename, is_single_intervention)
