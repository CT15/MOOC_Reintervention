import os

class LogWriter:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def write_log(self, text, _print=True):
        if os.path.exists(self.log_file_path):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        f = open(self.log_file_path, append_write)
        f.write(f'{text}\n')
        f.close()

        if _print:
            print(text)