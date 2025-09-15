import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

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

def gradient_fill(x, y, fill_color=None, ax=None, direction='down', alpha=1.0, alpha_min=0.0, **kwargs):
    """
    Plot a line with a linear alpha gradient filled beneath it.

    Parameters
    ----------
    x, y : array-like
        The data values of the line.
    fill_color : a matplotlib color specifier (string, tuple) or None
        The color for the fill. If None, the color of the line will be used.
    ax : a matplotlib Axes instance
        The axes to plot on. If None, the current pyplot axes will be used.
    Additional arguments are passed on to matplotlib's ``plot`` function.

    Returns
    -------
    line : a Line2D instance
        The line plotted.
    im : an AxesImage instance
        The transparent gradient clipped to just the area beneath the curve.
    """
    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    #alpha = line.get_alpha()
    #alpha = 1.0 if alpha else alpha

    z = np.empty((100, 1, 4), dtype=float)
    rgb = mcolors.colorConverter.to_rgb(fill_color)
    z[:,:,:3] = rgb
    if direction == 'down':
        z[:,:,-1] = np.linspace(alpha, alpha_min, 100)[:,None]
    elif direction == 'up':
        z[:,:,-1] = np.linspace(alpha_min, alpha, 100)[:, None]

    x, y = np.asarray(x), np.asarray(y)
    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax], origin='lower', zorder=zorder, interpolation='hanning')

    xy = np.column_stack([x, y])
    #xy = np.vstack([[xmin, ymax], xy, [xmax, ymax], [xmin, ymax]])
    xy = np.vstack([xy, xy[0]])
    clip_path = Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)

    #ax.autoscale(True)
    return line, im

