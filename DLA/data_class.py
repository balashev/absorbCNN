import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import h5py
import warnings
from scipy.signal import argrelextrema

from ..stats import distr1d
from ..tools import Timer
class dla_data(list):
    """
    This class contains data structure that is used to DLA search.
    The individual record is one dimensional spectra region of the size <window> that also possess certain label of DLA (identification, position and column density)
    The datastructure is stored in hdf5 file given by <filename>.
    """
    def __init__(self, parent, window=400, timing=False, filename='data.hdf5'):
        """
        parameters:
            - window         :  the size of the spectral window in pixels
            - timing         :  use Timer to check the calculation time (for debug)
            - filename       :  the filename of hdf5 file where the data will be stored
        """
        self.parent = parent
        self.window = window
        self.attrs = ['specs', 'loglams', 'dla_flag', 'dla_pos', 'dla_NHI', 'inds']
        self.dtype = {'specs': np.float32, 'loglams': np.float32, 'dla_flag': bool, 'dla_pos': int,
                      'dla_NHI': np.float32, 'inds': int, 'labels': [('dla_flag', np.bool_), ('dla_pos', np.int_), ('dla_NHI', np.single)]}
        self.timer = Timer() if timing else None
        self.filename = filename
        self.set_bads()

    def set_bads(self):
        """
        Mark bad SDSS spectra, that crush routine
        """
        self.bads = [[6190, 56210, 566], [7879, 57359, 980], [7039, 56572, 720], [7622, 56987, 660]]
        self.corr = [6190, 56210], [6190, 56210]

    def create(self, dset='full'):
        """
        Create dataset structure
        """
        self.open()
        self.data.create_group(dset)
        for attr in self.attrs + ['labels']:
            shape = (self.window, ) * (attr == 'specs') #+ (1, ) * (attr == 'labels')
            self.data.create_dataset(dset + '/' + attr, shape=(0,) + shape, dtype=self.dtype[attr], maxshape=(None,) + shape)
        print(dset, self.data[dset + '/'].keys())
        self.close()

    def append_mask(self, dset='train', mask=None, randomize=True):
        """
        Append to data structure using some mask
        """
        self.open()
        num = len(mask)
        if self.timer != None:
            self.timer.time(f'mask append')
        for attr in self.attrs:
            shape = (num, self.window) * (attr == 'specs') + (num, ) * (attr != 'specs')
            arr = np.zeros(shape, dtype=self.dtype[attr])
            if self.timer != None:
                self.timer.time(f'{attr} create')
            arr = self.data['full/' + attr][...][mask]
            #self.data['full/' + attr].read_direct(arr, source_sel=mask)
            if self.timer != None:
                self.timer.time(f'{attr} read')
            self.data[dset + '/' + attr].resize((self.data[dset + '/' + attr].shape[0] + num), axis=0)
            if self.timer != None:
                self.timer.time(f'{attr} resize')
            if attr == 'specs':
                self.data[dset + '/' + attr].write_direct(arr, dest_sel=np.s_[-num:, :])
            else:
                self.data[dset + '/' + attr].write_direct(arr, dest_sel=np.s_[-num:])
            if self.timer != None:
                self.timer.time(f'{attr} write')
        self.data[dset + '/labels'].resize((self.data[dset + '/labels'].shape[0] + num), axis=0)
        data = np.zeros((num,), dtype=self.dtype['labels'])
        for i, attr in enumerate(['dla_flag', 'dla_pos', 'dla_NHI']):
            #arr = np.zeros((num), dtype=self.dtype[attr])

            #self.data[dset + '/' + attr].read_direct(arr, source_sel=mask)
            data[attr] = self.data[dset + '/' + attr][...][-num:]# arr[:]
            #data[attr] = kwargs[attr][:, np.newaxis]
        self.data[dset + '/labels'].write_direct(data, dest_sel=np.s_[-num:])
        self.num_specs = len(self.data['full/inds'][...])
        self.close()

    def append(self, **kwargs):
        """
        Append data to data structure
        """
        self.open()
        for attr in self.attrs:
            #print(attr)
            data = np.asarray(kwargs[attr])
            self.data[kwargs['dset'] + '/' + attr].resize((self.data[kwargs['dset'] + '/' + attr].shape[0] + data.shape[0]), axis=0)
            if attr == 'specs':
                self.data[kwargs['dset'] + '/' + attr][-data.shape[0]:, :] = data
            else:
                self.data[kwargs['dset'] + '/' + attr][-data.shape[0]:] = data

        self.data[kwargs['dset'] + '/labels'].resize((self.data[kwargs['dset'] + '/labels'].shape[0] + kwargs['dla_flag'].shape[0]), axis=0)
        data = np.zeros((len(kwargs['dla_flag']),), dtype=self.dtype['labels'])
        for attr in ['dla_flag', 'dla_pos', 'dla_NHI']:
            data[attr] = kwargs[attr][:] #kwargs[attr][:, np.newaxis]
        self.data[kwargs['dset'] + '/labels'][-data.shape[0]:] = data

        self.num_specs = len(self.data['full/inds'])
        self.close()

    def new(self):
        """
        Create bew H5py file to store the data structure
        """
        try:
            os.remove(self.filename)
            print('Removed old data')
        except:
            pass
        self.data = h5py.File(self.filename, 'w')
        self.create()

    def open(self):
        """
        Open data file. It should be appropriatly closed, before next use
        """
        self.data = h5py.File(self.filename, 'r+')

    def close(self):
        """
        Close data file. It should be appropriatly closed, before next use
        """
        self.data.close()

    def get(self, attr, dset='full', batch=None, ind_batch=0):
        """
        Get the data from data structure
        parameters:
            -  attr          :  attribute to get
            -  dset          :  dataset to retrieve data
            -  batch         :  The size of the batch to get the batch of the data. If None, then all array will be retrieved
            -  ind_batch     :  Number of the batch.
        """
        self.open()
        if batch == None:
            return self.data[dset + '/' + attr][...][:]
        else:
            slices = np.s_[batch * ind_batch:min(batch * (ind_batch + 1), self.data[dset + '/' + attr].shape[0])]
            return self.data[dset + '/' + attr][slices]
        self.close()

    def load(self, attrs=None, dset='full'):
        """
        Load the data from h5py file to data structure as explicit attribute
        """
        if attrs == None:
            attrs = self.attrs
        for attr in attrs:
            setattr(self, attr, self.get(attr, dset=dset))
        self.num_specs = len(np.unique(self.inds))
        print(self.num_specs)
        self.close()

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
        v_prox = 2000 # in km/s
        qso_pos = int((np.log10(self.parent.lya * (1 + z_qso) * (1 - v_prox/3e5)) - s['loglam'][0]) * 1e4)
        mask[max(0, qso_pos):] = False

        # masked Ly_cutoff region:
        mask[:max(0, int((np.log10(self.parent.lyc * (1 + z_qso)) - s['loglam'][0]) * 1e4))] = False

        # mask dla associated pixels:
        for dla in dlas:
            dla_pos = int(1e4 * (np.log10((1 + dla['z_abs']) * self.parent.lya) - s['loglam'][0]))
            if dla_pos - dla_window >= 0:
                mask[max(0, dla_pos-int(self.window/4)):dla_pos-dla_window] = False
            mask[dla_pos + dla_window:dla_pos + int(self.window/4)] = False

            mask_dla[max(0, dla_pos-dla_window):dla_pos+dla_window] = True

            lyb_pos = int(1e4 * (np.log10((1 + dla['z_abs']) * self.parent.lyb - s['loglam'][0])))
            if lyb_pos + self.window >=0:
                mask[max(0, lyb_pos-self.window):lyb_pos+self.window] = False

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
                if (meta[i]['BI_CIV'] < 100) * (not self.check_bads(meta[i]['PLATE'], meta[i]['MJD'], meta[i]['FIBERID'])):
                    if i * 10 % num == 0:
                        print(i, ' of ', num)
                    s = self.parent.cat[i]
                    specs, loglams, inds, dla_flags, dla_pos, NHI = [], [], [], [], [], []
                    #print('dla', meta[i]['dla'])
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
                    #print(i, mask, mask_dla)
                    #print(i, np.sum(mask), np.sum(mask_dla), np.sum(mask_dla[mask]))
                    if np.sum(mask) > 0:
                        im = np.where(np.diff(np.insert(mask, 0, 0)) != 0)[0]
                        flux = np.asarray(s['flux'][max(0, im[0] - int(self.window/2)):min(len(s['flux']), im[-1] + int(self.window/2))], dtype=np.float16)
                        stride = flux.strides[0]
                        specs = as_strided(np.pad(flux, (max(0, int(self.window/2) - im[0]), np.abs(min(0, len(s['flux']) - im[-1] - int(self.window/2)))),
                                          'constant', constant_values=(np.quantile(flux[:50], 0.75), np.quantile(flux[-50:], 0.75))),
                                           shape=[im[-1] - im[0], self.window], strides=[stride, stride])[mask[im[0]:im[-1]]]
                        loglams = s['loglam'][mask]
                        inds = np.ones(len(loglams), dtype=int) * i
                        dla_flag = mask_dla[mask]
                        dla_pos = np.zeros_like(loglams)
                        dla_NHI = np.zeros_like(loglams)
                        if np.sum(dla_flag) > 0:
                            dind = np.argmin(np.abs(np.subtract(dlas['z_abs'][:, np.newaxis], (10 ** s['loglam'][mask_dla] / self.parent.lya - 1))), axis=0)
                            dla_pos[dla_flag] = np.subtract(s['loglam'][mask_dla], np.log10((dlas['z_abs'][dind] + 1) * self.parent.lya)) * 1e4
                            dla_NHI[dla_flag] = dlas['NHI'][dind]

                        if ind == None:
                            self.append(dset='full', specs=specs, loglams=loglams, inds=inds, dla_flag=dla_flag, dla_pos=dla_pos, dla_NHI=dla_NHI)

                            if np.random.random() > valid:
                                m = np.append(np.random.choice(np.arange(len(loglams))[~dla_flag], int(sum(~dla_flag) * (1 - dropout)), replace=False),
                                              np.random.choice(np.arange(len(loglams))[dla_flag], int(sum(dla_flag) * (1 - dropout_dla)), replace=False))
                                #print(m)
                                if len(m)>1:
                                    self.append(dset='train', specs=specs[m], loglams=loglams[m], inds=inds[m], dla_flag=dla_flag[m], dla_pos=dla_pos[m], dla_NHI=dla_NHI[m])
                            else:
                                self.append(dset='valid', specs=specs, loglams=loglams, inds=inds, dla_flag=dla_flag, dla_pos=dla_pos, dla_NHI=dla_NHI)
                        else:
                            return specs, loglams, dla_flag, dla_pos, dla_NHI, inds

    def make_sets(self, valid=0.3, dropout=0.0, dropout_dla=0.0, shuffle=True, batch=100):
        """
        make sets for the training and validation based on the DLA data
        """
        inds = self.get('inds')
        dla_flag = self.get('dla_flag')
        uni = np.unique(inds)
        ind = np.random.choice(uni, int(len(uni) * valid), replace=False)
        self.valid = np.zeros(self.num_specs, dtype=bool)
        self.create(dset='valid')
        for i in ind:
            m = (inds == i)
            self.valid[m] = True
            self.open()
            self.append(dset='valid', specs=self.data['full/specs'][m], loglams=self.data['full/loglams'][m], inds=self.data['full/inds'][m], dla_flag=self.data['full/dla_flag'][m], dla_pos=self.data['full/dla_pos'][m], dla_NHI=self.data['full/dla_NHI'][m])
        if self.timer:
            self.timer.time('create valid')
        #print(np.sum(dla_flag[~self.valid] == 1))
        tdi = np.arange(len(inds))[~self.valid]
        # >>> adding regions with dlas
        self.train = np.random.choice(tdi[dla_flag[~self.valid] == 1], int(np.sum(dla_flag[~self.valid] == 1) * (1 - dropout_dla)), replace=False)
        # >>> adding regions without dlas
        self.train = np.append(self.train, np.random.choice(tdi[~dla_flag[~self.valid] == 1], int(np.sum(dla_flag[~self.valid] == 0) * (1 - dropout)), replace=False))

        if shuffle:
            self.train = np.random.permutation(self.train)
        if self.timer != None:
            self.timer.time('randomize')
        self.create(dset='train')
        #print(len(self.data['full/dla_flag'][:][self.train]), np.sum(self.data['full/dla_flag'][:][self.train]))
        if self.timer != None:
            self.timer.time('create train')

        print(int(len(self.train) / batch))
        for i in range(int(len(self.train) / batch)):
            m = np.sort(self.train[i*batch:(i+1)*batch])
            print(i)
            m2 = np.random.choice(np.arange(len(m)), len(m))
            if self.timer != None:
                self.timer.time(f'randomize {i}')

            self.open()
            if self.timer != None:
                self.timer.time(f'open {i}')

            x = self.data['full/specs']

            self.append_mask(dset='train', mask=m, randomize=True) #specs=self.data['full/specs'][m,:][m2], loglams=self.data['full/loglams'][m][m2], inds=self.data['full/inds'][m][m2], dla_flag=self.data['full/dla_flag'][m][m2], dla_pos=self.data['full/dla_pos'][m][m2], dla_NHI=self.data['full/dla_NHI'][m][m2])
            if self.timer != None:
                self.timer.time(f'append {i}')

        #for attr in self.attrs:
        #    self.data.create_dataset('/'.join(dset, attr), shape=(0,) + (self.window,) * (attr == 'specs'), dtype=self.dtype[attr], maxshape=(None,) + (self.window,) * (attr == 'specs'))
        self.valid = np.where(self.valid)[0]

    def check_bads(self, plate, mjd, fiberid):
        return any([(b[0]== plate) * (b[1] == mjd) * (b[2] == fiberid) for b in self.bads]) + self.check_corr(plate, mjd, fiberid)
        #return any([(b[0]== plate) * (b[1] == mjd) for b in self.bads])

    def check_corr(self, plate, mjd, fiberid):
        return any([(b[0]== plate) * (b[1] == mjd) for b in self.corr])

    def get_inds(self, dla=False, dset='full'):
        """
        Get indixes of the spectra from the data strucutre
        """
        if dla:
            return np.unique(self.get('inds', dset=dset)[self.get('dla_flag', dset=dset) == 1])
        else:
            return np.unique(self.get('inds', dset=dset))

    def get_spec(self, ind, sdss=None):
        """
        Get all the data in data structure correspond to the certain spectrum by index
        """
        if sdss == None:
            args = np.where(self.get('inds') == ind)[0]
            inds = args[np.argsort(self.get('loglams')[args])]
            return [self.get(attr)[inds] for attr in self.attrs]
        else:
            return self.make(self.parent.cat, ind=ind, dropout=0.0, dropout_dla=0.0)

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
                dla_flag, dla_pos, dla_NHI = self.get('dla_flag')[m], self.get('dla_pos')[m], self.get('dla_NHI')[m]
                pos = x[np.where(dla_flag == 1)[0][0]] * 10 ** (-dla_pos[np.where(dla_flag == 1)[0][0]] * 0.0001) if any(dla_flag) else 0
                for l, y, mask, c, title in zip(range(3), [dla_flag, dla_pos, dla_NHI],
                                                [dla_flag > -1, dla_flag > -1, dla_NHI[dla_flag > -1] > 0],
                                                ['tomato', 'dodgerblue', 'forestgreen'],
                                                ["DLA flag", "DLA pos", "DLA N_HI"]):
                    axs[l].plot(x[mask], y[mask], 'o', c=c)
                    axs[l].text(0.02, 0.9, title, color=c, ha='left', va='top', transform=axs[l].transAxes, zorder=3)
                    axs[l].set_xlim(xlims)
                    if any(dla_flag):
                        for z in self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:]['z_abs']:
                            axs[l].axvline(self.parent.lya * (1 + z), ls='--', color='tomato')
                axs[1].axhline(0, ls='--', color='k', lw=0.5)
                ax = axs[3]
                if any(dla_flag):
                    # meta/{0:05d}_{1:04d}_{2:05d}/ 05 05 04
                    # data/{0:05d}/{1:04d}/{2:05d}/ 05 04 05
                    for z in self.parent.cat.cat['meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(meta['PLATE'], meta['MJD'], meta['FIBERID'])][:]['z_abs']:
                        ax.axvline(self.parent.lya * (1 + z), ls='--', color='tomato')
                        ax.axvline(self.parent.lyb * (1 + z), ls=':', color='violet')
            ax.plot(10 ** s['loglam'][:i + 200], s['flux'][:i + 200], 'k')

            m = np.nanmax(s['flux'][np.max([0, i - 50]):i + 50])
            m = np.nanquantile(s['flux'][:i + 200], 0.95)
            # print(m, s['flux'][:i+200])
            ax.set_ylim([-m * 0.1, m * 1.1])
            ax.set_xlim(xlims)
            ax.axvspan(10 ** s['loglam'][i], 10 ** s['loglam'][-1], color='w', alpha=0.5, zorder=2)
            ax.axvspan(10 ** s['loglam'][0], 10 ** s['loglam'][max(0, int((np.log10(911 * (1 + z_qso)) - s['loglam'][0]) * 1e4))], color='w', alpha=0.5, zorder=2)
            fig.subplots_adjust(wspace=0, hspace=0)
        else:
            fig, ax = plt.subplots(figsize=(14, 5), dpi=160)
            ax.plot(10 ** s['loglam'], s['flux'], color='k')
            m = np.quantile(s['flux'], 0.99)
            ax.set_ylim([-m * 0.1, m * 1.1])
            if meta['BI_CIV'] > 100:
                ax.text(0.5, 0.5, "BAL QSO", ha='center', va='center', color='red', alpha=0.3, fontsize=100, transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, "OUT OF RANGE", ha='center', va='center', color='red', alpha=0.3, fontsize=100, transform=ax.transAxes)

        ax.text(0.5, 0.9, f"{ind}: {meta['PLATE']} {meta['MJD']} {meta['FIBERID']}, z_qso={round(meta['Z'], 3)}", ha='center', va='top', transform=ax.transAxes, zorder=3)

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
        specs, loglams, *other = self.get_spec(ind)

        if self.parent.cnn != None:
            preds = self.parent.cnn.model.predict(specs)

            x = np.power(10, loglams)
            fig.axes[0].plot(x, preds[0], '--k')
            fig.axes[1].plot(x, preds[1], '--k')
            fig.axes[2].plot(x, preds[2], '--k')

        return fig, ax

    def get_dla_from_CNN(self, ind, plot=True):
        """
        Get the DLA catalog from the spectrum using the statistics of the CNN results.
        parameters:
            - ind       :  index of the spectrum
            - plot      :  plot intermediate results
        """
        specs, loglams, *other = self.get_spec(ind)

        dla = []
        if self.parent.cnn != None:
            preds = self.parent.cnn.model.predict(specs)

            m = (preds[0] > 0.2).flatten()
            if sum(m) > 3:
                zd = 10 ** (loglams[m] - preds[1].flatten()[m] * 1e-4) / self.parent.lya - 1
                z = distr1d(zd, bandwidth=0.7)
                z.stats()
                if plot:
                    z.plot()
                zint = [min(zd)] + list(z.x[argrelextrema(z.inter(z.x), np.less)[0]]) + [max(zd)]
                zmax = z.x[argrelextrema(z.inter(z.x), np.greater)]
                # print(zmax)
                for i in range(len(zint) - 1):
                    # print(i, zint[i], zint[i+1])
                    mz = (z.x > zint[i]) * (z.x < zint[i + 1])
                    if max([z.inter(x) for x in z.x[mz]]) > z.inter(z.point) / 3:
                        mz = (zd > zint[i]) * (zd < zint[i + 1])
                        if sum(mz) > 3:
                            # print(sum(mz))
                            dla.append([ind, np.median(preds[0][m][mz])])
                            z1 = distr1d(zd[mz])
                            z1.kde(bandwidth=0.1)
                            z1.stats()
                            if plot:
                                z1.plot()
                            N = preds[2].flatten()[m][mz]
                            N = distr1d(N)
                            N.stats()
                            if plot:
                                N.plot()
                            dla[-1].extend([z1.point] + list(z1.interval - z1.point))
                            dla[-1].extend([N.point] + list(N.interval - N.point))
                            # print(i, z1.latex(f=4), N.latex(f=2))
        else:
            warnings.warn("There is no CNN model to predict", UserWarning)
        return dla
