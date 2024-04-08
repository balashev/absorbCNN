import numpy as np
from scipy.special import wofz
from numba import jit
import os

class line():
    """
    class for calculation the line profiles using Voigt function,
    parameters:
      - l           : wavelenght, in A
      - f           : oscillator strength
      - g           : damping constant in s^-1
      - logN        : log10 of column density in cm^-2
      - b           : Doppler paremeter in km/s
      - z           : redshift
      - x           : ancillary for classmethod
    """

    def __init__(self, l=1215.6701, f=0.4164, g=6.265e8, logN=21, b=5.0, z=0.0, x=None, resolution=None):
        self.l, self.f, self.g, self.logN, self.b, self.z = l, f, g, logN, b, z
        self.c = 299792.46  # speed of light in km/s
        self.lz = self.l * (1 + self.z)
        self.ld = self.b / self.c * self.l
        self.a = self.g / 4 / np.pi / self.b / 1e5 * self.l * 1e-8
        self.tau0 = np.sqrt(np.pi) * 0.008447972556327578 * (self.l * 1e-8) * self.f * 10 ** self.logN / (self.b * 1e5)
        self.resolution = resolution

    def tau(self, x):
        return self.tau0 * self.voigt((np.asarray(x) / self.lz - 1) * self.c / self.b)

    def voigt(self, x):
        return wofz(x + 1j * self.a).real

    @classmethod
    def profile(cls, **kwargs):
        s = cls(**kwargs)
        return np.exp(-s.tau(kwargs['x']))


@jit(forceobj=True, looplift=True)
def gauss(x, s, x0=0):
    """
    gauss function
    """
    return 1 / np.sqrt(2 * np.pi) / s * np.exp(-.5 * (x - x0 / s) ** 2)


@jit(forceobj=True, looplift=True)
def errf_v2(x):
    """
    Error function
    """
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]
    t = 1 / (1 + 0.5 * np.abs(x))
    tau = t * np.exp(-x ** 2 + a[0] + t * (a[1] + t * (a[2] + t * (a[3] + t * (a[4] + t * (a[5] + t * (a[6] + t * (a[7] + t * (a[8] + t * a[9])))))))))
    if x >= 0:
        return 1 - tau
    else:
        return tau - 1

@jit(forceobj=True, looplift=True)
def convolve_res(l, f, R):
    """
    Convolve flux with instrument function specified by resolution R
    Data can be unevenly spaced.

    parameters:
        - l         : float array, shape(N)
                        wavelength array (or velocity in km/s)
        - f         : float array, shape(N)
                        flux
        - R         : float
                        resolution of the instrument function. Assumed to be constant with wavelength.
                        i.e. the width of the instrument function is linearly dependent on wavelength.
    returns:
        - fc        : float array, shape(N)
                        convolved flux
    """
    # sig = 127301 / R
    delta = 3.0

    n = len(l)
    fc = np.zeros_like(f)

    f = 1 - f

    il = 0
    for i, x in enumerate(l):
        sig = x / R / 2.355
        k = il
        while l[k] < x - delta * sig:
            k += 1
        il = k
        #s = f[il] * (1 - errf_v2((x - l[il]) / np.sqrt(2) / sig)) / 2
        s = 0
        # ie = il + 30
        while k < n - 1 and l[k + 1] < x + delta * sig:
            # s += f[k] * 1 / np.sqrt(2 * np.pi) / sig * np.exp(-.5 * ((l[k] - x) / sig) ** 2) * d[k]
            s += (f[k + 1] * gauss(l[k + 1] - x, sig) + f[k] * gauss(l[k] - x, sig)) / 2 * (l[k + 1] - l[k])
            # print(i, k , gauss(l[k] - x, sig))
            k += 1
        # input()
        #s += f[k] * (1 - errf_v2(np.abs(l[k] - x) / np.sqrt(2) / sig)) / 2
        fc[i] = s

    return 1 - fc

class H2abs():
    """
    The class to generate H2 absroption system:
    example of the usage:
        >>> h2 = H2abs()
        >>> x, f = h2.calc_profile(x=x, z=z, logN=19.8, b=5, j=6, T=100, exc='low')

        will calculate at the wavelengths specified by <x>
        the H2 asborption system with
        redshift <z>,
        total H2 column density <logN>,
        Doppler parameter <b>,
        up to <j> rotational level,
        kinetic temperature <T>,
        and type of higher rotational level excitation specified by <exc> - can be ['low', 'mid', 'high']
    """
    def __init__(self):
        self.read_data()

    def read_data(self):
        """
        read the atomic data for the H2 stored in H2_lines.dat and H2_energy_X.dat files
        """
        folder = os.path.dirname(os.path.abspath(__file__))
        print('import', folder)
        self.data = np.genfromtxt(folder + '/data/H2_lines.dat', names=True,
                                  dtype=[('band', 'U1'), ('name', 'U6'), ('vl', '<i8'), ('jl', '<i8'), ('vu', '<i8'),
                                         ('ju', '<i8'), ('lambda', '<f8'), ('f', '<f8'), ('g', '<f8')])
        self.energy = np.genfromtxt(folder + '/data/H2_energy_X.dat', skip_header=2, names=True, comments='#')
        self.energy['E'] *= 1.438777  # energy levels in K

    def mask(self, **kwargs):
        """
        filter H2 lines following keywords, e.g.:
        self.mask(band='L', vl=0, jl=0, vu=0, ju=1, lmin=912, lmax=1180)

        If a keyword is not set, then it not use these parameters to filter:
        self.mask(band='L', vl=0, jl=0, lmin=1050) will provide all transitions of Lyman band from the zero vibrational and rotational levels, which have wavelenght higher than 1050A
        """
        m = np.ones_like(self.data['lambda'], dtype=bool)
        for k, v in kwargs.items():
            if k == 'lmin':
                m *= self.data['lambda'] >= v
            elif k == 'lmax':
                m *= self.data['lambda'] <= v
            else:
                m *= (self.data[k] == v)
        return m

    def get_lines(self, **kwargs):
        """
        get a list of H2 lines using filter by following keywords, e.g.:
        self.get_lines(band='L', vl=0, jl=0, vu=0, ju=1, lmin=912, lmax=1180)

        If a keyword is not set, then it not use these parameters to filter:
        self.get_lines(band='L', vl=0, jl=0, lmin=1050) will provide all transitions of Lyman band from the zero vibrational and rotational levels, which have wavelenght higher than 1050A
        """
        return self.data[self.mask(**kwargs)]

    def get_bands(self, v):
        """
        return reference restframe wavelength for the H2 bands
        """
        d = {}
        for vi in np.arange(v):
            d[f'L{vi}-0'] = (self.get_lines(band='L', vl=0, jl=0, vu=vi, ju=1)['lambda'] +
                             self.get_lines(band='L', vl=0, jl=1, vu=vi, ju=2)['lambda'])[0] / 2
        return d

    def excitation(self, j=5, T=100, exc='low'):
        """
        simulate the FICTIVE population of the H2 levels:
        parameters:
            - j      : up to which level
            - T      : kinetic temperature for the lower transitions, in K. Typical values is 100 K
            - exc    : excitation of the higher rotational levels. Can be 'low', 'mid', 'high'. This is very rough, since the excitation depends also on the totat column density
        return:
            array of the population of the levels
        """
        Ta = {'low': 700, 'mid': 1000, 'high': 3000}
        frac = {'low': -4, 'mid': -2, 'high': -2}
        self.j = np.arange(j + 1)
        self.g = (2 * self.j + 1) * (2 * (self.j % 2) + 1)
        self.E = [self.energy['E'][(self.energy['J'] == j) * (self.energy['V'] == 0)] for j in self.j]
        z = np.sum([self.g[j] * (np.exp(-self.E[j] / T) + 10 ** frac[exc] * np.exp(-self.E[j] / Ta[exc])) for j in
                    self.j])  # excitation diagram using two temperatures for low and high rotational levels
        return np.asarray(
            [(self.g[j] / z * (np.exp(-self.E[j] / T) + 10 ** frac[exc] * np.exp(-self.E[j] / Ta[exc])))[0] for j in
             self.j])

    def calc_profile(self, x=None, z=0.0, logN=19, b=5, j=5, T=100, exc='low'):
        """
        calculate the line profile of H2 absorption system:
        parameters:
            - x         : wavelength array, at which profiles calculated, if None, use some specified
            - z         : redshift
            - logN      : log10 of the total column density in cm^-2
            - b         : Doppler parameter in km/s
            - j         : up to which rotational level
            - T         : kinetic temperature for the lower transitions, in K. Typical values is 100 K
            - exc       : excitation of the higher rotational levels. Can be 'low', 'mid', 'high'. This is very rough, since the excitation depends also on the totat column density
        return:
            x, f        : wavelength array, transmission coefficient (i.e. np.exp(-sum_i(tau_i(x))))
        """
        if x is None:
            x = np.linspace(1050, 1180, int(1e4)) * (1 + z)
        f = np.ones_like(x)
        logNi = logN + np.log10(self.excitation(j=j, T=T, exc=exc))
        for j in self.j:
            for l in self.get_lines(vl=0, jl=j, lmin=x[0] / (1 + z), lmax=x[-1] / (1 + z)):
                f *= line.profile(x=x, l=l['lambda'], f=l['f'], g=l['g'], logN=logNi[j], b=b, z=z)
        self.x, self.f = x, f
        return x, f