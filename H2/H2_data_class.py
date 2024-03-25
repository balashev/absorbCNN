import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import warnings

from ..data_class import data_structure
from ..line_profiles import H2abs, convolve_res


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
            - lines_file     :  H2_lines.dat
        """
        super(h2_data, self).__init__(parent, timing=False, filename='data_H2.hdf5')
        self.window = window
        self.bands = bands
        self.shape = (self.window, self.bands)
        self.h2 = H2abs()
        self.h2bands = self.h2.get_bands(self.bands)

    def make_mask(self, ind, z_qso=0):
        """
        Calculate spectral mask to use during the creation of the data structure
        parameters:
            -  ind           :  index of the spectra to use
            -  z_qso         :  the redshift of the quasar
            -  dlas          :  array of the dlas, to mask the data
            -  dla_windows   :  the size of the avoidance region around DLA
        """
        # print(ind)
        s = self.parent.cat[ind]
        # print(dlas)
        mask = np.ones_like(s['loglam'], dtype=bool)
        # print(i)

        mask_dla = np.zeros_like(s['loglam'], dtype=bool)

        # get position at QSO redshift and mark the region redwards:
        v_prox = 3000  # in km/s
        qso_pos = int((np.log10(self.parent.H2bands['L0-0'] * (1 + z_qso) * (1 - v_prox / 3e5)) - s['loglam'][0]) * 1e4)
        mask[max(0, qso_pos):] = False

        # masked Ly_cutoff region:
        mask[:max(0, int((np.log10(self.parent.lyc * (1 + z_qso)) - s['loglam'][0]) * 1e4))] = False

        return mask
    def make(self, ind=None, num=None, valid=0.3, dropout=0.7, dropout_h2=0.3, start=0, band='L3-0'):
        print('make cat')
        self.parent.cat.open()
        if num == None:
            num = self.parent.cat.cat['meta/num'][0]
        if num > self.parent.cat.cat['meta/num'][0]:
            warnings.warn("The number of spectra (<num> parameter) is more than in the database! Change number <num> to correspond database size", UserWarning)
            num = self.parent.cat.cat['meta/num'][0]
        meta = self.parent.cat.cat['meta/qso'][:]

        if ind == None:
            self.create(dset='valid')
            self.create(dset='train')

        delta = [l for l in self.h2.get_bands(self.bands).values()]
        delta = [int(np.log10(l/delta[0]) * 1e4) for l in delta]
        print(delta)
        print('Running make H2 catalog script:')
        for i in range(start, start + num):
            # print(i)

            if ind == None or ind == i:
                if self.check_bads(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID']):
                    print('bads:', i, meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])
                if (meta[i]['BI_CIV'] < 100) * (not self.check_bads(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])):
                    if i * 10 % num == 0:
                        print(i, ' of ', num)
                    s = self.parent.cat[i]
                    if meta[i]['H2']:
                        self.parent.cat.open()
                        sdss_name1 = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(meta[i]['PLATE'], meta[i]['FIBERID'], meta[i]['MJD'])
                        sdss_name2 = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])
                        # print(sdss_name2)
                        if sdss_name1 in self.parent.cat.cat:
                            h2 = self.parent.cat.cat['meta/{0:05d}/{1:04d}/{2:05d}/H2'.format(meta[i]['PLATE'], meta[i]['FIBERID'], meta[i]['MJD'])][:]
                        elif sdss_name2 in self.parent.cat.cat:
                            h2 = self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/H2'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])][:]
                            # dlas = sdss.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])][:]
                        else:
                            print('meta/{0:05d}/{1:05d}/{2:04d}/H2'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID']), ' vaporized in history')
                        self.parent.cat.close()
                    #print(h2)
                    if s is not None:
                        v_prox = 3000 # in km/s
                        z_min = int((np.log10(self.h2bands['L0-0'] * (1 + meta[i]['Z']) * (1 + v_prox / 3e5)) - s['loglam'][0]) * 1e4)
                        z_max = (1 + meta[i]['Z']) * (1 + v_prox / 3e5) - 1
                        z_min = (np.max([10 ** s['loglam'][0], self.parent.lyc * (1 + meta[i]['Z'])]) / self.h2bands[band] - 1)
                        #print(z_min, z_max)
                        if z_min < z_max:
                            num = int(np.trunc((np.log10(self.h2bands['L0-0'] * (1 + z_max)) - s['loglam'][0]) * 1e4 + self.window / 2)) - int(np.trunc((np.log10(self.h2bands['L0-0'] * (1 + z_min)) - s['loglam'][0]) * 1e4 - self.window / 2))
                            for band, l in self.h2bands.items():
                                #print(band, l)
                                i_max = int(np.trunc((np.log10(l * (1 + z_max)) - s['loglam'][0]) * 1e4 - self.window / 2))
                                i_min = i_max - num
                                #print(i_min, i_max, i_max - i_min)
                                stride = s['flux'].strides[0]
                                if i_max > 0:
                                    spec = as_strided(s['flux'][max(0, i_min):i_max], shape=[i_max - max(0, i_min), self.window], strides=[stride, stride])
                                    if i_min < 0:
                                        spec = np.append(np.median(specs[:-i_min, :], axis=2), spec, axis=0)
                                else:
                                    spec = np.median(specs, axis=2)
                                #print('spec:', spec.shape)
                                if band == 'L0-0':
                                    specs = spec[:, :, np.newaxis]
                                    #print(specs.shape)
                                else:
                                    specs = np.append(specs, spec[:, :, np.newaxis], axis=2)

                            i_max = int(np.trunc((np.log10(self.h2bands['L0-0'] * (1 + z_max)) - s['loglam'][0]) * 1e4))
                            #print(i_max, num)
                            reds = 10 ** s['loglam'][i_max-num:i_max] / self.h2bands['L0-0'] - 1
                            inds = np.ones(len(reds), dtype=int) * i
                            flag = np.zeros_like(reds, dtype=bool)
                            pos = np.zeros_like(reds, dtype=int)
                            logN = np.zeros_like(reds, dtype=float)

                            #print(h2['z_abs'], reds)

                            if len(h2) > 0:
                                m = (reds < (1 + h2['z_abs']) * 10 ** (self.window / 4 * 1e-4) - 1) * (reds > (1 + h2['z_abs']) * 10 ** (-self.window / 4 * 1e-4) - 1)
                                flag[m] = np.ones(np.sum(m))
                                pos[m] = np.trunc(np.log10((1 + reds[m]) / (1 + h2['z_abs'])) * 1e4)
                                logN[m] = h2['logN']
                            #print(flag, pos, logN)

                            if ind == None:
                                self.append(dset='full', specs=specs, reds=reds, inds=inds, flag=flag, pos=pos, logN=logN)

                                if np.random.random() > valid:
                                    m = np.append(np.random.choice(np.arange(len(reds))[~flag], int(sum(~flag) * (1 - dropout)), replace=False),
                                                  np.random.choice(np.arange(len(reds))[flag], int(sum(flag) * (1 - dropout_h2)), replace=False))
                                    # print(m)
                                    if len(m) > 1:
                                        self.append(dset='train', specs=specs[m], reds=reds[m], inds=inds[m], flag=flag[m], pos=pos[m], logN=logN[m])
                                else:
                                    self.append(dset='valid', specs=specs, reds=reds, inds=inds, flag=flag, pos=pos, logN=logN)
                            else:
                                return specs, reds, flag, pos, logN, inds


    def plot_data(self, ind, specs=None, z=None):
        if specs == None:
            specs, reds, flag, pos, logN, inds = self.make(ind=ind)
        if z is None:
            z = reds[flag][10]
        i = np.argmin(np.abs(reds - z))
        pos = i - pos[flag][10] + 15
        fig, ax = plt.subplots(nrows=self.bands+1, figsize=(12, 2 * self.bands + 6))
        for k in range(self.bands):
            ax[k].step(np.arange(specs.shape[1]), specs[pos, :, k], where='mid', color='k')
            if 0:
                for lin in self.h2.data['lambda'][(self.h2.data['jl'] <= 3) * (self.h2.data['vl'] == 0)]:
                    if (lin > (l - 5)) * (lin < (l + 10)):
                        ax[k].axvline(np.log10(lin / l) / 0.0001 + shift, ls='--', lw=3, alpha=0.5, color='tomato')

            #ax[k].set_xlim([x[0], x[-1]])
            #ax[k].text(0.03, 0.1, self.h2bands[k], transform=ax[k].transAxes, color='tomato', fontsize=20)

        ax[-1].imshow(specs[pos, :, :].transpose())
        return fig, ax

    def plot_preds(self, ind, fig=None):
        """
        Plot the results of the CNN search on the spectrum
        parameters:
            -  ind     :  index of the spectrum
            -  fig     :  figure to plot. If None, that it will be created using self.plot_spec(ind)
        """
        if fig == None:
            fig, ax = self.plot_spec(ind, add_info=True)
        else:
            ax = fig.get_axes()
        specs, reds, *other = self.get_spec(ind)

        if self.parent.cnn != None:
            preds = self.parent.cnn.model.predict(specs)

            x = (1 + reds) * self.h2bands['L0-0']
            fig.axes[0].plot(x, preds[0], '--k')
            fig.axes[1].plot(x, preds[1], '--k')
            fig.axes[2].plot(x, preds[2], '--k')

        return fig, ax

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
                x = (1 + self.get('reds')[m]) * self.h2bands['L0-0']
                # print(sdss.cat['meta/{0:05d}_{1:04d}_{2:05d}/dla'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:].dtype)
                # pos = x[np.where((dla_pos[m] == 0) * dla_flags[m])[0][0]] if any(dla_flags[m]) else 0
                flag, pos, logN = self.get('flag')[m], self.get('pos')[m], self.get('logN')[m]
                #pos = x[np.where(flag == 1)[0][0]] * 10 ** (-pos[np.where(flag == 1)[0][0]] * 0.0001) if any(flag) else 0
                for l, y, mask, c, title in zip(range(3), [flag, pos, logN], [flag > -1, flag > -1, logN[flag > -1] > 0],
                                                ['tomato', 'dodgerblue', 'forestgreen'], ["flag", "pos", "logN"]):
                    axs[l].plot(x[mask], y[mask], 'o', c=c)
                    axs[l].text(0.02, 0.9, title, color=c, ha='left', va='top', transform=axs[l].transAxes, zorder=3)
                    axs[l].set_xlim(xlims)
                    if any(flag):
                        for z in self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/H2'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:]['z_abs']:
                            for b in self.h2bands.values():
                                axs[l].axvline(b * (1 + z), ls='--', color='tomato')
                        axs[l].axvline(self.parent.lya * (1 + z), ls=':', color='brown')
                axs[1].axhline(0, ls='--', color='k', lw=0.5)
                ax = axs[3]
            ax.plot(10 ** s['loglam'][:i + 200], s['flux'][:i + 200], 'k')
            if any(flag):
                for z in self.parent.cat.cat[
                             'meta/{0:05d}_{1:05d}_{2:04d}/H2'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:][
                    'z_abs']:
                    for b in self.h2bands.values():
                        ax.axvline(b * (1 + z), ls='--', color='tomato')
                ax.axvline(self.parent.lya * (1 + z), ls=':', color='brown')
            m = np.nanmax(s['flux'][np.max([0, i - 50]):i + 50])
            m = np.nanquantile(s['flux'][:i + 200], 0.95)
            # print(m, s['flux'][:i+200])
            ax.set_ylim([-m * 0.1, m * 1.1])
            ax.set_xlim(xlims)
            ax.axvspan(10 ** s['loglam'][i], 10 ** s['loglam'][-1], color='w', alpha=0.5, zorder=2)
            ax.axvspan(10 ** s['loglam'][0],
                       10 ** s['loglam'][max(0, int((np.log10(911 * (1 + z_qso)) - s['loglam'][0]) * 1e4))], color='w',
                       alpha=0.5, zorder=2)
            res = self.get_abs_from_CNN(ind, plot=False, lab=self.h2bands['L0-0'])
            for r in res[::-1]:
                print('cat:', z, r)
                i = int((np.log10(self.h2bands['L0-0'] * (1 + r[2]) * (1 - 3e3 / 3e5)) - s['loglam'][0]) * 1e4) + 50
                x1, f = self.h2.calc_profile(x=10 ** s['loglam'][:i + 50], z=r[2], logN=r[5], b=5, j=6, T=100, exc='low')
                f = convolve_res(x1, f, 1800) * np.quantile(s['flux'][:i + 50], 0.95) * 1.1
                ax.plot(x1, f, lw=1.0)
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

        label = f"{ind}: {meta['PLATE']} {meta['MJD']} {meta['FIBERID']}, z_qso={round(meta['Z'], 3)}"

        ax.text(0.5, 0.9, label, ha='center', va='top', transform=ax.transAxes, zorder=3)

        return fig, ax
