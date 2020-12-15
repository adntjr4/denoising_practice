
import time

class Timer():
    def __init__(self):
        self.data_load_time = 0.
        self.model_time = 0.

    def data_load_start(self):
        self.data_load_start_time = time.time()

    def data_load_end(self):
        self.data_load_finish_time = time.time()
        self.data_load_time = self.data_load_finish_time - self.data_load_start_time

    def model_start(self):
        self.model_start_time = time.time()

    def model_end(self):
        self.model_finish_time = time.time()
        self.model_time = self.model_finish_time - self.model_start_time

    def get_tot_time(self):
        return self.data_load_time + self.model_time
