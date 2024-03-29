import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import warnings
from scipy.signal import argrelextrema

from ..stats import distr1d
from ..tools import Timer
from ..data_class import data_structure
class dla_data(data_structure):
    def __init__(self, parent, window=400, timing=False, filename='data.hdf5'):
        super(dla_data, self).__init__(parent, timing=timing, filename=filename)
        self.window = window
        self.shape = (self.window, )


    def make_mask(self, ind, z_qso=0, dlas=[], dla_window=60):
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

        # get position of QSO Lya line and mask redward Lya with proximate indent:
        v_prox = 2000  # in km/s
        qso_pos = int((np.log10(self.parent.lya * (1 + z_qso) * (1 - v_prox / 3e5)) - s['loglam'][0]) * 1e4)
        mask[max(0, qso_pos):] = False

        # masked Ly_cutoff region:
        mask[:max(0, int((np.log10(self.parent.lyc * (1 + z_qso)) - s['loglam'][0]) * 1e4))] = False

        # mask dla associated pixels:
        for dla in dlas:
            dla_pos = int(1e4 * (np.log10((1 + dla['z_abs']) * self.parent.lya) - s['loglam'][0]))
            if dla_pos - dla_window >= 0:
                mask[max(0, dla_pos - int(self.window / 4)):dla_pos - dla_window] = False
            mask[dla_pos + dla_window:dla_pos + int(self.window / 4)] = False

            mask_dla[max(0, dla_pos - dla_window):dla_pos + dla_window] = True

            lyb_pos = int(1e4 * (np.log10((1 + dla['z_abs']) * self.parent.lyb - s['loglam'][0])))
            if lyb_pos + self.window >= 0:
                mask[max(0, lyb_pos - self.window):lyb_pos + self.window] = False

        mask_dla *= mask
        return mask, mask_dla

    def make(self, ind=None, num=None, valid=0.3, dropout=0.7, dropout_dla=0.3, start=0):
        """
        Make data structure
        parameters:
            - ind           :   use individual spectrum by index. If None, then run the sample
            - num           :   number of spectra to use
            - dla_window    :   the size of the spectral window.
            - valid         :   the percentage of validation sample
            - dropout       :   the percentage of the dropout in spectral windows without DLA
            - dropout_dla   :   the percentage of the dropout in spectral windows with DLA
            - start         :   number of the spectrum to start
        """
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

        print('Running make catalog script:')
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
                    specs, reds, inds, flags, pos, logN = [], [], [], [], [], []
                    # print('dla', meta[i]['dla'])
                    if meta[i]['dla']:
                        self.parent.cat.open()
                        sdss_name1 = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(meta[i]['PLATE'], meta[i]['FIBERID'], meta[i]['MJD'])
                        sdss_name2 = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])
                        # print(sdss_name2)
                        if sdss_name1 in self.parent.cat.cat:
                            dlas = self.parent.cat.cat['meta/{0:05d}/{1:04d}/{2:05d}/dla'.format(meta[i]['PLATE'], meta[i]['FIBERID'], meta[i]['MJD'])][:]
                        elif sdss_name2 in self.parent.cat.cat:
                            dlas = self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])][:]
                            # dlas = sdss.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])][:]
                        else:
                            print('meta/{0:05d}/{1:05d}/{2:04d}/dla'.format(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID']), ' vaporized in history')
                        self.parent.cat.close()
                    else:
                        dlas = []
                    # print(dlas, meta[i]['Z'])
                    if s is None:
                        mask, mask_dla = [], []
                    else:
                        mask, mask_dla = self.make_mask(i, z_qso=meta[i]['Z'], dlas=dlas, dla_window=60)
                    # print(i, mask, mask_dla)
                    # print(i, np.sum(mask), np.sum(mask_dla), np.sum(mask_dla[mask]))
                    if np.sum(mask) > 0:
                        im = np.where(np.diff(np.insert(mask, 0, 0)) != 0)[0]
                        flux = np.asarray(s['flux'][max(0, im[0] - int(self.window / 2)):min(len(s['flux']), im[-1] + int(self.window / 2))], dtype=np.float16)
                        stride = flux.strides[0]
                        specs = as_strided(np.pad(flux, (max(0, int(self.window / 2) - im[0]), np.abs(min(0, len(s['flux']) - im[-1] - int(self.window / 2)))),
                                                  'constant', constant_values=(np.quantile(flux[:50], 0.75), np.quantile(flux[-50:], 0.75))),
                                           shape=[im[-1] - im[0], self.window], strides=[stride, stride])[mask[im[0]:im[-1]]]
                        reds = 10 ** s['loglam'][mask] / self.parent.lya - 1
                        inds = np.ones(len(reds), dtype=int) * i
                        flag = mask_dla[mask]
                        pos = np.zeros_like(reds)
                        logN = np.zeros_like(reds)
                        if np.sum(flag) > 0:
                            dind = np.argmin(np.abs(np.subtract(dlas['z_abs'][:, np.newaxis], (10 ** s['loglam'][mask_dla] / self.parent.lya - 1))), axis=0)
                            pos[flag] = np.subtract(s['loglam'][mask_dla], np.log10((dlas['z_abs'][dind] + 1) * self.parent.lya)) * 1e4
                            logN[flag] = dlas['NHI'][dind]

                        if ind == None:
                            self.append(dset='full', specs=specs, reds=reds, inds=inds, flag=flag, pos=pos, logN=logN)

                            if np.random.random() > valid:
                                m = np.append(np.random.choice(np.arange(len(reds))[~flag], int(sum(~flag) * (1 - dropout)), replace=False),
                                              np.random.choice(np.arange(len(reds))[flag], int(sum(flag) * (1 - dropout_dla)), replace=False))
                                # print(m)
                                if len(m) > 1:
                                    self.append(dset='train', specs=specs[m], reds=reds[m], inds=inds[m], flag=flag[m], pos=pos[m], logN=logN[m])
                            else:
                                self.append(dset='valid', specs=specs, reds=reds, inds=inds, flag=flag, pos=pos, logN=logN)
                        else:
                            return specs, reds, flag, pos, logN, inds

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
                x = (self.get('reds')[m] + 1) * self.parent.lya
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

            x = (1 + reds) * self.parent.lya
            fig.axes[0].plot(x, preds[0], '--k')
            fig.axes[1].plot(x, preds[1], '--k')
            fig.axes[2].plot(x, preds[2], '--k')

        return fig, ax
