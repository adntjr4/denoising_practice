import threading
import datetime, os

from .progress_msg import ProgressMsg
# from .chart import LossChart


class Logger(ProgressMsg):
    def __init__(self, max_iter:tuple, log_dir:str=None, log_lvl:str='debug', log_file_lvl:str='debug', log_include_time:bool=True):
        '''
        Args:
            session_name (str)
            max_iter (tuple) : max iteration for progress
            log_dir (str) : if None, no file out for logging
            log_lvl (str) : 'debug' < 'info' < 'warning'
            log_include_time (bool)
        '''
        self.lvl_list = ['debug', 'info', 'highlight', 'val']
        self.lvl_color = [bcolors.OKCYAN, None, bcolors.WARNING, bcolors.OKGREEN]

        assert log_lvl in self.lvl_list
        assert log_file_lvl in self.lvl_list

        # init progress message class
        ProgressMsg.__init__(self, max_iter)

        # log setting
        self.log_dir = log_dir
        self.log_lvl      = self.lvl_list.index(log_lvl)
        self.log_file_lvl = self.lvl_list.index(log_file_lvl)
        self.log_include_time = log_include_time
        
        # init logging
        if self.log_dir is not None:
            self.log_file = open(os.path.join(log_dir, 'log.log'), 'w')
            self.val_file = open(os.path.join(log_dir, 'validation.log'), 'w')

    def debug(self, txt):
        txt = str(txt)
        lvl_n = self.lvl_list.index('debug')
        if self.log_lvl <= lvl_n:
            if self.lvl_color[lvl_n] is not None:
                print('\033[K'+ self.lvl_color[lvl_n] + txt + bcolors.ENDC)
            else:
                print('\033[K'+txt)
        if self.log_file_lvl <= lvl_n:
            self.write_file(txt)

    def info(self, txt):
        txt = str(txt)
        lvl_n = self.lvl_list.index('info')
        if self.log_lvl <= lvl_n:
            if self.lvl_color[lvl_n] is not None:
                print('\033[K'+ self.lvl_color[lvl_n] + txt + bcolors.ENDC)
            else:
                print('\033[K'+txt)
        if self.log_file_lvl <= lvl_n:
            self.write_file(txt)

    def highlight(self, txt):
        txt = str(txt)
        lvl_n = self.lvl_list.index('highlight')
        if self.log_lvl <= lvl_n:
            if self.lvl_color[lvl_n] is not None:
                print('\033[K'+ self.lvl_color[lvl_n] + txt + bcolors.ENDC)
            else:
                print('\033[K'+txt)
        if self.log_file_lvl <= lvl_n:
            self.write_file(txt)

    def val(self, txt):
        txt = str(txt)
        lvl_n = self.lvl_list.index('val')
        if self.log_lvl <= lvl_n:
            if self.lvl_color[lvl_n] is not None:
                print('\033[K'+ self.lvl_color[lvl_n] + txt + bcolors.ENDC)
            else:
                print('\033[K'+txt)
        if self.log_file_lvl <= lvl_n:
            self.write_file(txt)
        if self.log_dir is not None:
            self.val_file.write(txt+'\n')
            self.val_file.flush()

    def write_file(self, txt):
        if self.log_dir is not None:
            if self.log_include_time:
                time = datetime.datetime.now().strftime('%H:%M:%S')
                txt = "[%s] "%time + txt
            self.log_file.write(txt+'\n')
            self.log_file.flush()

    def clear_screen(self):
        if os.name == 'nt': 
            os.system('cls') 
        else: 
            os.system('clear') 

# https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-python
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
