from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from .profiles import tau, convolve_res

class catalog(list):
    """
    This class is for working with the SDSS data catalog

    Create class using catalog (from offcial SDSS release) located in <filename>:
    # NOTE: This overrides the existing <filename>.hdf5 file
    >>> sdss = catalog
    >>> sdss.create(filename)

    Open already created catalog:
    >>> sdss = catalog(filename)

    Download catalog:
    >>> sdss.download

    Assess meta data (catalog of the targets):
    >>> sdss.cat['meta/qso']

    Retrieve the spectrum by index <ind>:
    >>> sdss[ind]

    Retrieve the spectrum by Plate, MJD, Fiber:
    >>> sdss.get(Plate, MJD, Fiber)
    """

    def __init__(self, parent, stored=None):
        super().__init__()
        self.parent = parent
        self.stored = stored
        if self.stored != None:
          self.open()

    def open(self, stored=None, attr='a'):
        """
        open the hdf5 file with catalog
        """
        if stored != None:
            self.stored = stored
        if attr == 'w' or os.path.isfile(self.stored):
            self.cat = h5py.File(self.stored, attr)
        else:
            print("There is not catalog file")

    def close(self):
        self.cat.close()

    def create(self, catalog_filename=None, output_filename=None):
        """
        create hdf5 file using provided catalog of the Quasars by <filename>
        The catalog will be stored in <output_filename>.hdf5 file in <meta/qso> dataset
        parameters:
            - catalog_filename        :  the filename of file, containing SDSS catalog database.
            - output_filename         :  the filename where spectral catalog will be stored. If None, then will be stored in <catalog_filename>.hdf5 file
        """
        #print(filename)
        if catalog_filename != None:
            cat = Table.read(catalog_filename)

        if output_filename is None:
            output_filename = catalog_filename.replace('.fits', '.hdf5')

        if os.path.isfile(output_filename):
            os.remove(output_filename)

        self.open(stored=output_filename, attr='w')

        attrs = ['RA', 'DEC', 'THING_ID', 'PLATE', 'MJD', 'FIBERID', 'Z', 'Z_PIPE', 'Z_PCA', 'BI_CIV']
        if 'dr12' in catalog_filename.lower():
            cat['Z'] = cat['Z_VI']

        missed = open('missed.dat', 'w')
        missed.close()
        #print(cat[attrs])
        #print(np.array(cat[attrs]).dtype)
        self.cat.create_dataset('meta/qso', data=np.array(cat[attrs]))
        print(self.cat['meta/qso'].dtype)
        self.close()

    def add_attr(self, attr, data):
        """
        Add attribute to the data structure
        """
        if attr not in self.cat['meta/qso'].dtype.fields:
            a = np.lib.recfunctions.append_fields(self.cat['meta/qso'][:], attr, data, dtypes='<f8')
            del self.cat['meta/qso']
            dset = self.cat.create_dataset('meta/qso', data=a)
        else:
            print("Attribute {0:s} is already in the dataset".format(attr))
        #self.cat['meta/qso'] = a

    def add_dla_cat(self, filename, kind='Noterdaeme'):
        """
        Adding the DLA catalog
        """
        self.open(stored=self.stored, attr='a')
        if kind == 'Noterdaeme':
            self.add_noterdaeme(filename)

        self.close()

    def add_noterdaeme(self, filename):
        """
        Adding Noterdaeme DLA catalog, organize the data
        parameters:
            - filename   :  the path to file contains Noterdaeme catalog
        """
        noter = np.genfromtxt(filename, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')
        #print(noter.dtype)
        d = np.zeros(self.cat['meta/qso'][:].shape[0])

        dt = np.dtype([('z_abs', np.float32), ('NHI', np.float32)])
        #print(self.cat['meta/qso']['FIBERID'])
        self.open(stored=self.stored, attr='a')
        for i in range(self.cat['meta/num'][0]):
            mjd, plate, fiber = self.cat['meta/qso']['MJD'][i], self.cat['meta/qso']['PLATE'][i], self.cat['meta/qso']['FIBERID'][i]
            #print(mjd, plate, fiber)
            name = 'meta/{0:05d}_{1:05d}_{2:04d}'.format(plate, mjd, fiber)
            m = (noter['MJD'] == mjd) * (noter['PLATE'] == plate) * (noter['FIBER'] == fiber)
            data = np.empty(shape=(0,), dtype=dt)
            for nind in np.where(m)[0]:
                #print(nind)
                d[i] = 1
                #print(name + '/dla')
                data = np.array([(noter[nind]['z_abs'], noter[nind]['NHI'])], dtype=dt)
                if name + '/dla' in self.cat:
                    dla = self.cat[name + '/dla'][...]
                    #print(dla)
                    #print(not any([noter[nind]['z_abs'] == d['z_abs'] for d in dla]))
                    if not any([noter[nind]['z_abs'] == d['z_abs'] for d in dla]):
                        data = np.append(dla, data)
                    #print(data)
                    del self.cat[name + '/dla']

            self.cat.create_dataset(name + '/dla', data=data, chunks=True, dtype=data.dtype)
            self.cat.flush()
                #print(self.cat[name + '/dla'][:])

        self.add_attr('dla', d)
        self.cat['meta/qso']['dla'] = d
        self.close()

    def append(self, num=None, source='web'):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is alaredy in catalog
        stored missing files in <missed.dat>
        parameters:
            - num      :  number of spectra
            - source   :  the path to file where SDSS spectra are located. If =='web' than the data will be downloaded from SDSS website (this is quite slow).
        """
        #print(source)
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso'])
        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])

        for i, q in enumerate(self.cat['meta/qso'][:num]):
            name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])
            #print(name)
            if name not in self.cat:
                print(i, name, name in self.cat)
                if source == 'web':
                    res = self.download(ra=q['RA'], dec=q['DEC'], plate=q['PLATE'], MJD=q['MJD'], fiber=q['FIBERID'])
                else:
                    sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                    #print(sdss_name, sdss_name in sdss)
                    if sdss_name in sdss:
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                    res = sdss_name in sdss
                    #try:

                if res:
                    self.cat['meta/num'][0] =  [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [1]
                else:
                    print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                    self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

    def add_dla(self, name, z_qso, debug=False):

        x, y, err = self.cat[name + '/loglam'][:], self.cat[name + '/flux'][:], 1 / np.sqrt(self.cat[name + '/ivar'][:])
        imin, imax = max(x[0], np.log10(self.parent.lyc * (1 + z_qso))), min(x[-1], np.log10(self.parent.lya * (1 + z_qso) * (1 - 2e3/3e5)))
        z_dla = 10 ** (np.random.randint(int(imin * 1e4), int(imax * 1e4)) / 1e4) / self.parent.lya - 1
        NHI = 19.5 + np.random.rand() ** 2 * 3
        t = tau(logN=NHI, b=30, z=z_dla, resolution=2000)
        if debug:
            m = (x > imin) * (x < imax)
            print(name, z_qso)
            print(imin, imax)
            print(z_dla, NHI)
            plt.plot(10 ** x[m], y[m])
        t.calctau(10 ** x)
        y = y * convolve_res(10**x, np.exp(-t.tau), t.resolution) + err * (1 - np.exp(-t.tau)) * np.random.randn(len(x)) / 2
        y[np.logical_not(np.isfinite(y))] = np.nanmedian(y)
        self.cat[name + '/flux'][...] = y

        if debug:
            plt.plot(10 ** x[m], y[m])
            plt.show()

        return z_dla, NHI

    def make_mock(self, num=None, source='web', dla_cat=None):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is alaredy in catalog
        stored missing files in <missed.dat>
        parameters:
            - num           :  number of spectra to be generated
            - source        :  filename of the catalog contained SDSS spectra
            - dla_cat       :  the path to file contains DLA catalog (in Noterdaeme catalog for now)
        """
        noter = np.genfromtxt(dla_cat, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')

        print(source)
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso'])
        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])
        d = np.zeros(num)
        mask = np.zeros(len(self.cat['meta/qso'][:]), dtype=bool)
        n = 0
        for i, q in enumerate(self.cat['meta/qso'][:]):

            m = (noter['MJD'] == q['MJD']) * (noter['PLATE'] == q['PLATE']) * (noter['FIBER'] == q['FIBERID'])
            if len(np.where(m)[0]) == 0:
                name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                #print(name)
                if name not in self.cat:
                    #print(i, name, name in self.cat, n)
                    res = False
                    sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                    if sdss_name not in sdss:
                        sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                    #print(sdss_name, sdss_name in sdss)
                    res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 100) and (self.cat['meta/qso'][i]['Z'] > 10**sdss[sdss_name + 'loglam'][:][100] / self.parent.lya / (1 - 2e3/3e5) - 1)
                    if res:
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)
                        mask[i] = True
                        z_dla, NHI = self.add_dla(name, z_qso=self.cat['meta/qso'][i]['Z'])
                        d[n] = 1
                        #print(name + '/dla')
                        data = np.array([(z_dla, NHI)], np.dtype([('z_abs', np.float32), ('NHI', np.float32)]))

                        self.cat.create_dataset(name.replace('data', 'meta') + '/dla', data=data, chunks=True, dtype=data.dtype)
                        self.cat.flush()
                        n += 1
                        print(n)
                        self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [1]
                    else:
                        #print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                        self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
            if n >= num:
                break

        print(sum(mask))
        data = self.cat['meta/qso'][...]
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=data[mask])
        self.add_attr('dla', d)
        self.cat['meta/qso']['dla'] = d

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

    def download(self, ra=None, dec=None, plate=None, MJD=None, fiber=None, kind='astroquery'):
        """
        download spectra of the catalog from the website and store it in hdf5 file in <data/> dataset
        stored missing files in <missed.dat>
        """
        name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(plate, MJD, fiber)

        # >>> direct access through SDSS website:
        if kind == 'web':
            link_boss = 'http://data.sdss.org/sas/dr14/eboss/spectro/redux/v5_10_0/spectra/lite/'
            link_sdss = 'https://data.sdss.org/sas/dr14/sdss/spectro/redux/26/spectra/lite/'

            if i % 1000 == 0:
                print(i)
            file = 'spec-{0:04d}-{1:05d}-{2:04d}.fits'.format(plate, MJD, fiber)
            folder = '{0:04d}/'.format(plate)
            if int(plate) <= 3000:
                link = link_sdss + folder + file
            else:
                link = link_boss + folder + file
            print(link)
            sp = fits.open(link)
            for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                if name + '/' + attr not in self.cat:
                    #print(name + '/' + attr)
                    self.cat.create_dataset(name + '/' + attr, data=sp[1].data[attr], dtype=sp[1].data.dtype[attr])

            sp.close()

        # >>> astroquery download:
        elif kind == 'astroquery':
            #likely slower download, but full data (is not actually needed)
            pos = SkyCoord(ra, dec, unit='deg', frame='icrs')
            xid = SDSS.query_region(pos, spectro=True)
            if xid is not None:
                sp = SDSS.get_spectra(matches=xid)
                for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                    if name + '/' + attr not in self.cat:
                        self.cat.create_dataset(name + '/' + attr, data=sp[0][1].data[attr], dtype=sp[0][1].data.dtype[attr])

            return xid is not None

    def name(self, ind):
        """
        Get the SDSS name for the spectra by index <ind>
        """
        self.open()
        q = self.cat['meta/qso'][ind]
        self.close()
        return '{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])

    def get(self, plate, mjd, fiber):
        """
        retrieve numpy recarray with the spectrum from the stored hdf5 file
        """
        self.open()
        name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(plate, mjd, fiber)
        #self.cat.visit(print)
        if name in self.cat:
            dtype = [('loglam', '<f8'), ('flux', '<f8'), ('ivar', '<f8'), ('and_mask', '<i8')]
            data = np.core.records.fromarrays([self.cat[name + '/' + attr[0]][:] for attr in dtype], dtype=dtype)
        else:
            print("There is not {0:05d}_{1:05d}_{2:04d} entry in the database".format(plate, mjd, fiber))
            data = None
        self.close()
        return data

    def __getitem__(self, i):
        self.open()
        q = self.cat['meta/qso'][i]
        self.close()
        return self.get(q['PLATE'], q['MJD'], q['FIBERID'])

