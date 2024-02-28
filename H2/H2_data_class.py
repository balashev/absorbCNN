import matplotlib.pyplot as plt
import numpy as np
import warnings

from ..data_class import data_structure
from ..line_profiles import H2abs

class h2_data(data_structure):
    """
    This class contains data structure that is used to DLA search.
    The individual record is one dimensional spectra region of the size <window> that also possess certain label of DLA (identification, position and column density)
    The datastructure is stored in hdf5 file given by <filename>.
    """
    def __init__(self, parent, window=64, bands=6, timing=False, filename='data_H2.hdf5'):
        """
        parameters:
            - window         :  the size of the spectral window in pixels
            - timing         :  use Timer to check the calculation time (for debug)
            - filename       :  the filename of hdf5 file where the data will be stored
        """
        super(h2_data, self).__init__(parent, timing=False, filename='data_H2.hdf5')
        self.window = window
        self.bands = bands
        self.shape = (self.window, self.bands)

    def make(self, ind=None, num=None, valid=0.3, dropout=0.7, dropout_dla=0.3, start=0):
        print('make cat')
        self.parent.cat.open()
        if num == None:
            num = self.parent.cat.cat['meta/num'][0]
        if num > self.parent.cat.cat['meta/num'][0]:
            warnings.warn(
                "The number of spectra (<num> parameter) is more than in the database! Change number <num> to correspond database size",
                UserWarning)
            num = self.parent.cat.cat['meta/num'][0]
        meta = self.parent.cat.cat['meta/qso'][:]

        if ind == None:
            self.create(dset='valid')
            self.create(dset='train')

        h2 = H2abs()

        print('Running make H2 catalog script:')
        for i in range(start, start + num):
            # print(i)

            if ind == None or ind == i:
                if self.check_bads(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID']):
                    print('bads:', i, meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])
                if (meta[i]['BI_CIV'] < 100) * (
                        not self.check_bads(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])):
                    if i * 10 % num == 0:
                        print(i, ' of ', num)
                    s = self.parent.cat[i]

                    shift = 15
                    for band, l in h2.get_bands(6).items():
                        ind = np.argmin(np.abs(10**(s['loglam']) / (1 + z) - l)) - shift
                        print(ind + x[0])
                        if ind > -x[0]:
                            f = qso[1][ind + x[0]: ind + x[-1] + 1]
                            # ax.step(qso[0][m] / (1 + z), qso[1][m], where='mid', color='k')
                            ax[k].step(x, f, where='mid', color='k')
                            if 1:
                                for lin in h2.data['lambda'][(h2.data['jl'] <= 3) * (h2.data['vl'] == 0)]:
                                    if (lin > (l - 5)) * (lin < (l + 10)):
                                        ax[k].axvline(np.log10(lin / l) / 0.0001 + shift, ls='--', lw=3, alpha=0.5,
                                                      color='tomato')
                                        if 1:
                                            ax[k].plot(np.log10(h2.x / l / (1 + z)) / 0.0001 + shift, h2.f * 40,
                                                       color='tomato')
                                        else:
                                            f = np.median(data, axis=0)

                                        ax[k].set_xlim([x[0], x[-1]])
                                        ax[k].text(0.03, 0.1, band, transform=ax[k].transAxes, color='tomato',
                                                   fontsize=20)

                                        k += 1
                                        data = np.vstack((data, f))
                                        y.append(y[-1] + 1)

                                        y = y[1:]

    def plot_spec(self, ind, add_info=True):
        """
        Make plot of the spectrum by index
        parameters:
            -  ind            :  index of the spectrum
            -  add_info       :  plot additional info (label)
        """
        self.parent.cat.open()
        s = self.parent.cat[ind]
        self.parent.cat.open()
        meta = self.parent.cat.cat['meta/qso'][ind]
        print(meta['PLATE'], meta['MJD'], meta['FIBERID'])
        z_qso = meta['Z']  # sdss.cat['meta/qso']['Z_VI'][ind]
        print(z_qso)
        i = int((np.log10(self.parent.lya * (1 + z_qso) * (1 - 3e3 / 3e5)) - s['loglam'][0]) * 1e4)
        if i > 0 and meta['BI_CIV'] < 100:
            xlims = [10 ** s['loglam'][0], 10 ** s['loglam'][i + 200]]
            if add_info:
                fig, axs = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 1, 1, 5]}, figsize=(14, 5), dpi=160)
            else:
                fig, ax = plt.subplots(figsize=(14, 5), dpi=160)
            if add_info:
                m = self.get('inds') == ind
                x = 10 ** self.get('loglams')[m]
                # print(sdss.cat['meta/{0:05d}_{1:04d}_{2:05d}/dla'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:].dtype)
                # pos = x[np.where((dla_pos[m] == 0) * dla_flags[m])[0][0]] if any(dla_flags[m]) else 0
                dla_flag, dla_pos, dla_NHI = self.get('flag')[m], self.get('pos')[m], self.get('logN')[m]
                pos = x[np.where(dla_flag == 1)[0][0]] * 10 ** (
                            -dla_pos[np.where(dla_flag == 1)[0][0]] * 0.0001) if any(dla_flag) else 0
                for l, y, mask, c, title in zip(range(3), [dla_flag, dla_pos, dla_NHI],
                                                [dla_flag > -1, dla_flag > -1, dla_NHI[dla_flag > -1] > 0],
                                                ['tomato', 'dodgerblue', 'forestgreen'],
                                                ["DLA flag", "DLA pos", "DLA N_HI"]):
                    axs[l].plot(x[mask], y[mask], 'o', c=c)
                    axs[l].text(0.02, 0.9, title, color=c, ha='left', va='top', transform=axs[l].transAxes, zorder=3)
                    axs[l].set_xlim(xlims)
                    if any(dla_flag):
                        for z in self.parent.cat.cat[
                                     'meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta['PLATE'], meta['MJD'],
                                                                               meta['FIBERID'])][:]['z_abs']:
                            axs[l].axvline(self.parent.lya * (1 + z), ls='--', color='tomato')
                axs[1].axhline(0, ls='--', color='k', lw=0.5)
                ax = axs[3]
                if any(dla_flag):
                    # meta/{0:05d}_{1:04d}_{2:05d}/ 05 05 04
                    # data/{0:05d}/{1:04d}/{2:05d}/ 05 04 05
                    for z in self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta['PLATE'], meta['MJD'],
                                                                                           meta['FIBERID'])][:][
                        'z_abs']:
                        ax.axvline(self.parent.lya * (1 + z), ls='--', color='tomato')
                        ax.axvline(self.parent.lyb * (1 + z), ls=':', color='violet')
            ax.plot(10 ** s['loglam'][:i + 200], s['flux'][:i + 200], 'k')

            m = np.nanmax(s['flux'][np.max([0, i - 50]):i + 50])
            m = np.nanquantile(s['flux'][:i + 200], 0.95)
            # print(m, s['flux'][:i+200])
            ax.set_ylim([-m * 0.1, m * 1.1])
            ax.set_xlim(xlims)
            ax.axvspan(10 ** s['loglam'][i], 10 ** s['loglam'][-1], color='w', alpha=0.5, zorder=2)
            ax.axvspan(10 ** s['loglam'][0],
                       10 ** s['loglam'][max(0, int((np.log10(911 * (1 + z_qso)) - s['loglam'][0]) * 1e4))], color='w',
                       alpha=0.5, zorder=2)
            fig.subplots_adjust(wspace=0, hspace=0)
        else:
            fig, ax = plt.subplots(figsize=(14, 5), dpi=160)
            ax.plot(10 ** s['loglam'], s['flux'], color='k')
            m = np.quantile(s['flux'], 0.99)
            ax.set_ylim([-m * 0.1, m * 1.1])
            if meta['BI_CIV'] > 100:
                ax.text(0.5, 0.5, "BAL QSO", ha='center', va='center', color='red', alpha=0.3, fontsize=100,
                        transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "OUT OF RANGE", ha='center', va='center', color='red', alpha=0.3, fontsize=100,
                        transform=ax.transAxes)

        ax.text(0.5, 0.9, f"{ind}: {meta['PLATE']} {meta['MJD']} {meta['FIBERID']}, z_qso={round(meta['Z'], 3)}",
                ha='center', va='top', transform=ax.transAxes, zorder=3)

        return fig, ax
