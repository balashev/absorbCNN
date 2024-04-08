import numpy as np
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

def add_field(a, descr, vals=None):
    """
    Return a new array that is like "a", but has additional fields.

    Arguments:
      a     -- a structured numpy array
      descr -- a numpy type description of the new fields
      vals  -- a numpy array to be added in the field. If None - nothing to add

    The contents of "a" are copied over to the appropriate fields in
    the new array, whereas the new fields are uninitialized.  The
    arguments are not modified.

    >>> sa = numpy.array([(1, 'Foo'), (2, 'Bar')], \
                         dtype=[('id', int), ('name', 'S3')])
    >>> sa.dtype.descr == numpy.dtype([('id', int), ('name', 'S3')])
    True
    >>> sb = add_field(sa, [('score', float)])
    >>> sb.dtype.descr == numpy.dtype([('id', int), ('name', 'S3'), \
                                       ('score', float)])
    True
    >>> numpy.all(sa['id'] == sb['id'])
    True
    >>> numpy.all(sa['name'] == sb['name'])
    True
    """
    if a.dtype.fields is None:
        raise ValueError("`A' must be a structured numpy array")
    b = np.empty(a.shape, dtype=a.dtype.descr + descr)
    for name in a.dtype.names:
        b[name] = a[name]
    b[descr[0][0]] = vals
    return b

def mem(x):
    print("Memory size of numpy array in Gb:", x.size * x.itemsize / 1e9)