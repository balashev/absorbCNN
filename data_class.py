import h5py
import numpy as np
import os
from scipy.signal import argrelextrema
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
        self.bads = [[6190, 56210, 566], [7879, 57359, 980], [7039, 56572, 720], [7622, 56987, 660],
                     [7152, 56660, 960], [7040, 56574, 616], [7615, 56977, 114], [2848, 54453, 564],
                     [4372, 55541, 720], [2944, 54523, 444], [1930, 53347, 205], [550, 51959, 460],
                     [2430, 53815, 245], [2400, 53765, 636]]
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
        Open data file. It should be appropriatly closed, before next use
        """
        self.data = h5py.File(self.filename, 'r+')

    def close(self):
        """
        Close data file. It should be appropriatly closed, before next use
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
        return any([(b[0]== plate) * (b[1] == mjd) * (b[2] == fiberid) for b in self.bads]) + self.check_corr(plate, mjd, fiberid)
        #return any([(b[0]== plate) * (b[1] == mjd) for b in self.bads])

    def check_corr(self, plate, mjd, fiberid):
        return any([(b[0]== plate) * (b[1] == mjd) for b in self.corr])

    def get_inds(self, flag=False, dset='full'):
        """
        Get indixes of the spectra from the data strucutre
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


    def get_abs_from_CNN(self, ind, reds=None, preds=None, plot=False, threshold=0.05, lab=1215.67, timer=False):
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
            if timer:
                t.time('get')
            if self.parent.cnn != None:
                preds = self.parent.cnn.model.predict(specs)
            else:
                warnings.warn("There is no CNN model to predict", UserWarning)
            if timer:
                t.time('pred')
        abs = []
        m = (preds[0] > threshold).flatten()
        if sum(m) > 3:
            zd = (1 + reds) + 10 ** (preds[1].flatten() * 1e-4) / lab - 1
            z = distr1d(zd[m], bandwidth=0.2)
            z.stats()
            if plot:
                z.plot()
            #zint = [min(zd)] + list(z.x[argrelextrema(z.inter(z.x), np.greater)[0]]) + [max(zd)]
            zint = z.x[argrelextrema(z.inter(z.x), np.greater)]
            print(zint)
            for i in range(len(zint)):
                print(i, zint[i])
                mz = (z.x > zint[i] - 0.1) * (z.x < zint[i] + 0.1)
                if max([z.inter(x) for x in z.x[mz]]) > z.inter(z.point) / 3:
                    mz = (zd > zint[i] - 0.02) * (zd < zint[i] + 0.02) * np.abs(preds[1].flatten()) > 0.1
                    print(sum(mz))
                    if sum(mz) > 3 and (len(abs) == 0 or np.min([np.abs(a[1] - np.median(preds[0][mz])) for a in abs]) > 0.1):
                        #print(sum(mz))
                        abs.append([ind, np.median(preds[0][mz])])
                        z1 = distr1d(zd[mz])
                        z1.kde(bandwidth=0.2)
                        z1.stats()
                        if plot:
                            z1.plot()
                        #abs[-1].extend([z1.point] + list(z1.interval - z1.point))
                        abs[-1].extend([np.median(zd[mz])] + list(z1.interval - np.median(zd[mz])))

                        if len(preds) > 2:
                            N = preds[2].flatten()[mz]
                            N = distr1d(N)
                            N.stats()
                            if plot:
                                N.plot()
                            abs[-1].extend([N.point] + list(N.interval - N.point))
                        print(i, z1.latex(f=4), N.latex(f=2))
            if timer:
                t.time('stats')

        return abs