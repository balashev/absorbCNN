import h5py
import numpy as np
import os
from scipy.signal import argrelextrema
from sklearn.cluster import MeanShift
import warnings

from .stats import distr1d
from .tools import Timer
class data_structure(list):
    """
    This class contains data structure that is used to DLA search.
    The individual record is one dimensional spectra region of the size <window> that also possess certain label of DLA (identification, position and column density)
    The datastructure is stored in hdf5 file given by <filename>.
    """
    def __init__(self, parent, timing=False, filename='data.hdf5'):
        """
        parameters:
            - timing         :  use Timer to check the calculation time (for debug)
            - filename       :  the filename of hdf5 file where the data will be stored
        """
        self.parent = parent
        self.shape = (0, )
        self.attrs = ['specs', 'reds', 'flag', 'pos', 'logN', 'inds']
        self.dtype = {'specs': np.float32, 'reds': np.float32, 'flag': bool, 'pos': int, 'logN': np.float32,
                      'inds': int, 'labels': [('flag', np.bool_), ('pos', np.int_), ('logN', np.single)]}
        self.timer = Timer() if timing else None
        self.filename = filename
        self.set_bads()

    def set_bads(self):
        """
        Mark bad SDSS spectra, that crush routine
        """
        if 1:
            self.bads = [[6190, 56210, 566], [7879, 57359, 980], [7039, 56572, 720], [7622, 56987, 660], [7449, 56740, 541], [5731, 56363, 507]] #, [5390, 56002, 232]]
        else:
            self.bads = np.genfromtxt("sdss_mask.txt", unpack=True, names=["plate", "mjd", "fiberid", "comment"])
            #self.bads = [[x[0], x[1], x[2]] for x in d]
        self.corr = [6190, 56210], [6190, 56210]

    def create(self, dset='full'):
        """
        Create dataset structure
        """
        self.open()
        self.data.create_group(dset)
        for attr in self.attrs + ['labels']:
            shape = self.shape * (attr == 'specs') #+ (0, ) * (attr == 'labels')
            self.data.create_dataset(dset + '/' + attr, shape=(0, ) + shape, dtype=self.dtype[attr], maxshape=(None,) + shape)
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
            shape = (num, self.shape) * (attr == 'specs') + (num, ) * (attr != 'specs')
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
        for i, attr in enumerate(['flag', 'pos', 'logN']):
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

        self.data[kwargs['dset'] + '/labels'].resize((self.data[kwargs['dset'] + '/labels'].shape[0] + kwargs['flag'].shape[0]), axis=0)
        data = np.zeros((len(kwargs['flag']),), dtype=self.dtype['labels'])
        for attr in ['flag', 'pos', 'logN']:
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
        Open data file. It should be appropriately closed, before next use
        """
        self.data = h5py.File(self.filename, 'r+')

    def close(self):
        """
        Close data file. It should be appropriately closed, before next use
        """
        self.data.close()

    def get(self, attr, dset='full', inds=None, batch=None, ind_batch=0):
        """
        Get the data from data structure
        parameters:
            -  attr          :  attribute to get
            -  dset          :  dataset to retrieve data
            -  batch         :  The size of the batch to get the batch of the data. If None, then all array will be retrieved
            -  ind_batch     :  Number of the batch.
        """
        self.open()
        if batch is None:
            if inds is None:
                return self.data[dset + '/' + attr][...][:]
            else:
                return self.data[dset + '/' + attr][inds]
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

    def make_sets(self, valid=0.3, dropout=0.0, dropout_dla=0.0, shuffle=True, batch=100):
        """
        make sets for the training and validation based on the DLA data
        """
        inds = self.get('inds')
        flag = self.get('flag')
        uni = np.unique(inds)
        ind = np.random.choice(uni, int(len(uni) * valid), replace=False)
        self.valid = np.zeros(self.num_specs, dtype=bool)
        self.create(dset='valid')
        for i in ind:
            m = (inds == i)
            self.valid[m] = True
            self.open()
            self.append(dset='valid', specs=self.data['full/specs'][m], reds=self.data['full/reds'][m], inds=self.data['full/inds'][m], flag=self.data['full/flag'][m], pos=self.data['full/pos'][m], logN=self.data['full/logN'][m])
        if self.timer:
            self.timer.time('create valid')
        #print(np.sum(flag[~self.valid] == 1))
        tdi = np.arange(len(inds))[~self.valid]
        # >>> adding regions with dlas
        self.train = np.random.choice(tdi[flag[~self.valid] == 1], int(np.sum(flag[~self.valid] == 1) * (1 - dropout_dla)), replace=False)
        # >>> adding regions without dlas
        self.train = np.append(self.train, np.random.choice(tdi[~flag[~self.valid] == 1], int(np.sum(flag[~self.valid] == 0) * (1 - dropout)), replace=False))

        if shuffle:
            self.train = np.random.permutation(self.train)
        if self.timer != None:
            self.timer.time('randomize')
        self.create(dset='train')
        #print(len(self.data['full/flag'][:][self.train]), np.sum(self.data['full/flag'][:][self.train]))
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

            self.append_mask(dset='train', mask=m, randomize=True) #specs=self.data['full/specs'][m,:][m2], reds=self.data['full/reds'][m][m2], inds=self.data['full/inds'][m][m2], flag=self.data['full/flag'][m][m2], dla_pos=self.data['full/dla_pos'][m][m2], dla_NHI=self.data['full/dla_NHI'][m][m2])
            if self.timer != None:
                self.timer.time(f'append {i}')

        #for attr in self.attrs:
        #    self.data.create_dataset('/'.join(dset, attr), shape=(0,) + (self.window,) * (attr == 'specs'), dtype=self.dtype[attr], maxshape=(None,) + (self.window,) * (attr == 'specs'))
        self.valid = np.where(self.valid)[0]

    def check_bads(self, plate, mjd, fiberid):
        #print("check_bads:", (plate == self.bads["plate"]) * (mjd == self.bads["mjd"]) * (fiberid == self.bads["fiberid"]), self.check_corr(plate, mjd, fiberid))
        #return (plate == self.bads["plate"]) * (mjd == self.bads["mjd"]) * (fiberid == self.bads["fiberid"]) + self.check_corr(plate, mjd, fiberid)
        #return any([(b[0]== plate) * (b[1] == mjd) * (b[2] == fiberid) for b in self.bads]) + self.check_corr(plate, mjd, fiberid)
        #return any([(b[0]== plate) * (b[1] == mjd) for b in self.bads])
        return self.check_corr(plate, mjd, fiberid)

    def check_corr(self, plate, mjd, fiberid):
        return any([(b[0]== plate) * (b[1] == mjd) for b in self.corr])

    def get_inds(self, flag=False, dset='full'):
        """
        Get indixes of the spectra from the data structure
        """
        if flag:
            return np.unique(self.get('inds', dset=dset)[self.get('flag', dset=dset) == 1])
        else:
            return np.unique(self.get('inds', dset=dset))

    def get_spec(self, inds, sdss=None):
        """
        Get all the data in data structure correspond to the certain spectrum by index
        """
        #t = Timer('get_spec')
        if isinstance(inds, (int, float, np.int64, np.int32)):
            inds = [inds]
        if sdss == None:
            args = np.asarray([], dtype=int)
            for ind in inds:
                id = np.where(self.get('inds') == ind)[0]
                #t.time('where')
                #print(ind, id[np.argsort(self.get('reds')[id])])
                args = np.r_[args, id[np.argsort(self.get('reds')[id])]]
                #t.time('args')
            return [self.get(attr, inds=args) for attr in self.attrs]
        else:
            return self.make(self.parent.cat, ind=inds, dropout=0.0, dropout_dla=0.0)


    def get_abs_from_CNN(self, ind, reds=None, preds=None, plot=False, threshold=0.5, lab=1215.67, timer=False):
        """
        Get the DLA catalog from the spectrum using the statistics of the CNN results.
        parameters:
            - ind       :  index of the spectrum
            - plot      :  plot intermediate results
        """
        if timer:
            t = Timer('cat')

        if reds is None and preds is None:
            specs, reds, *other = self.get_spec(ind)
            #print(spec)
            if timer:
                t.time('get')
            if self.parent.cnn != None:
                preds = self.parent.cnn.model.predict(specs)
                preds[1] = preds[1] * 10
                preds[2] = preds[2] * (self.parent.N_range[1] - self.parent.N_range[0]) + self.parent.N_range[0]
            else:
                warnings.warn("There is no CNN model to predict", UserWarning)
            if timer:
                t.time('pred')
        abs = self.get_abs(preds, reds, ind=ind, plot=plot, threshold=threshold, lab=lab)
        if timer:
            t.time('stats')
        return abs

    def get_abs(self, preds, reds, ind=0, plot=False, threshold=0.5, lab=1215.67):
        print("get abs for ind:", ind)
        m = (preds[0] > 0.1).flatten()
        #print(preds[1].flatten()[m], reds[m])
        zd = (reds + 1) * 10 ** (-preds[1].flatten() * 1e-4) - 1
        abs = []
        if sum(m) > 3:
            ms = MeanShift(bandwidth=60, bin_seeding=True)
            inds = np.where(m)[0]
            X = np.array(list(zip(inds, np.zeros(len(inds)))), dtype=int)

            ms.fit(X)
            labels = ms.labels_
            #print(labels)
            n_clusters_ = len(np.unique(labels))
            zm = [np.median(zd[inds[labels == k]]) for k in range(n_clusters_)]
            #print(zm)

            #print(n_clusters_)

            l_series = []
            for k in reversed(np.argsort(zm)):
                #print(k, zm[k])
                mz = inds[labels == k]
                #print("cluster {0}: {1}".format(k, np.median(zd[mz])))
                quant = np.quantile(zd[mz], [0.159, 0.5, 0.841], weights=preds[0].flatten()[mz], method='inverted_cdf')
                #print(quant)
                #print([[(l[0] > quant[2]), (l[1] < quant[0]), (l[0] > quant[2]) + (l[1] < quant[0])] for l in l_series])
                iz = np.argmin(np.abs(quant[1] - zd))
                prob = np.mean(preds[0].flatten()[np.max([0, iz-60]):np.min([iz+60, len(preds[0])])]) #
                prob = np.median(preds[0][mz])
                #print(iz, np.median(preds[0][mz]), np.mean(preds[0].flatten()[np.max([0, iz-30]):np.min([iz+30, len(preds[0])])]), preds[0].flatten()[np.max([0, iz-30]):np.min([iz+30, len(preds[0])])])
                if prob > threshold and np.all([(l[0] > quant[2]) + (l[1] < quant[0]) for l in l_series]):
                    #print("not in lyman series")
                    abs.append([ind, prob]) #np.median(preds[0][mz])])
                    abs[-1].extend([quant[1], quant[0] - quant[1], quant[2] - quant[1]])
                    l_series.extend([[(quant[1] + 2 * (quant[0] - quant[1]) + 1) * self.parent.lyb / self.parent.lya - 1, (quant[1] + 2 * (quant[2] - quant[1]) + 1) * self.parent.lyb / self.parent.lya - 1],
                                     [(quant[1] + 2 * (quant[0] - quant[1]) + 1) * self.parent.lyc / self.parent.lya - 1, (quant[1] + 2 * (quant[2] - quant[1]) + 1) * self.parent.lyc / self.parent.lya - 1]
                                     ])
                    if len(preds) > 2:
                        N = preds[2].flatten()[mz]
                        quant = np.quantile(N, [0.159, 0.5, 0.841], weights=preds[0].flatten()[mz], method='inverted_cdf')
                        abs[-1].extend([quant[1], quant[0] - quant[1], quant[2] - quant[1]])
                #print(l_series)
                #print(abs)
        return abs


    def get_abs_pdf(self, preds, reds, ind=0, plot=False, threshold=0.8, lab=1215.67):
        m = (preds[0] > threshold).flatten()
        abs = []
        if sum(m) > 3:
            print(preds[1].flatten()[m], reds[m])
            zd = (reds + 1) *  10 ** (-preds[1].flatten() * 1e-4) - 1
            print(zd[m])
            z = distr1d(zd[m], bandwidth=0.2)
            z.stats()
            if plot:
                if not os.path.exists(self.parent.QC_folder + "/spec/"):
                    os.mkdir(self.parent.QC_folder + "/spec/")
                ax = z.plot(savefig=self.parent.QC_folder + f"/spec/{ind}_z.png")
            zint = z.x[argrelextrema(z.inter(z.x), np.greater)]
            print(zint)
            for i in range(len(zint)):
                print(i, zint[i])
                mz = (z.x > zint[i] - 0.01) * (z.x < zint[i] + 0.01)
                if max([z.inter(x) for x in z.x[mz]]) > z.inter(z.point) / 3:
                    mz = (zd > zint[i] - 0.02) * (zd < zint[i] + 0.02) * (np.abs(preds[0].flatten()) > threshold)
                    #print(sum(mz), [np.abs(a[1] - np.median(preds[0][mz])) for a in abs])
                    if sum(mz) > 3 and (len(abs) == 0 or np.min([np.abs(a[1] - np.median(preds[0][mz])) for a in abs]) > 0.05):
                        abs.append([ind, np.median(preds[0][mz])])
                        #z_point = np.sum(zd[mz] * preds[0].flatten()[mz]) / np.sum(preds[0].flatten()[mz])
                        if 1:
                            quant = np.quantile(zd[mz], [0.159, 0.5, 0.841], weights=preds[0].flatten()[mz], method='inverted_cdf')
                            abs[-1].extend([quant[1], quant[0] - quant[1], quant[2] - quant[1]])
                        else:
                            z1 = distr1d(zd[mz])
                            z1.kde(bandwidth=0.2)
                            z1.stats()
                            if plot:
                                z1.plot(savefig=self.parent.QC_folder + f"/spec/{ind}_z_{i}.png")
                            abs[-1].extend([z1.point] + list(z1.interval - z1.point))
                        # abs[-1].extend([np.median(zd[mz])] + list(z1.interval - np.median(zd[mz])))

                        if len(preds) > 2:
                            # N = preds[2].flatten()[mz] * preds[0].flatten()[mz] / np.sum(preds[0].flatten()[mz])
                            N = preds[2].flatten()[mz]
                            #print(N, preds[0].flatten()[mz])
                            if 1:
                                quant = np.quantile(N, [0.159, 0.5, 0.841], weights=preds[0].flatten()[mz], method='inverted_cdf')
                                #print(quant)
                                #print([quant[1], quant[0] - quant[1], quant[2] - quant[1]])
                                abs[-1].extend([quant[1], quant[0] - quant[1], quant[2] - quant[1]])
                            else:
                                N = distr1d(N)
                                N.stats()
                                if plot:
                                    N.plot(savefig=self.parent.QC_folder + f"/spec/{ind}_N.png")
                                abs[-1].extend([N.point] + list(N.interval - N.point))
                        #print(i, abs[-1])
        return abs