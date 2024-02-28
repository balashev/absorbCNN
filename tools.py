import time

class Timer:
    """
    class for timing options
    """
    def __init__(self, name='', verbose=True):
        self.start = time.time()
        self.name = name
        self.verbose = verbose
        if self.name != '':
            self.name += ': '

    def restart(self):
        self.start = time.time()

    def time(self, st=None):
        s = self.start
        self.start = time.time()
        if st is not None and self.verbose:
            print(self.name + str(st) + ':', self.start - s)
        return self.start - s

    def get_time_hhmmss(self, st):
        end = time.time()
        m, s = divmod(end - self.start, 60)
        h, m = divmod(m, 60)
        time_str = "%02d:%02d:%02d" % (h, m, s)
        print(st, time_str)
        return time_str

    def sleep(self, t=0):
        time.sleep(t)



def mem(x):
    print("Memory size of numpy array in Gb:", x.size * x.itemsize / 1e9)