from astropy.table import Table
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astroquery.sdss import SDSS
import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.signal import medfilt


from .line_profiles import line, convolve_res, H2abs
from .tools import add_field

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

    def prepare_spectra_catalog(self, sdss_cat_file=None, dla_cat_file=None, output_filename=None):
        cat = Table.read(sdss_cat_file)
        noter = np.genfromtxt(dla_cat_file, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')
        print(cat.columns)
        dla_mask = [any((noter['MJD'] == q['MJD']) * (noter['PLATE'] == q['PLATE']) * (noter['FIBER'] == q['FIBERID'])) for q in cat[:]]

        attrs = ['RA', 'DEC', 'THING_ID', 'PLATE', 'MJD', 'FIBERID', 'Z', 'BI_CIV']
        c = cat[attrs]
        c['DLA'] = dla_mask
        c.write(output_filename, overwrite=True)

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

        attrs = ['RA', 'DEC', 'THING_ID', 'PLATE', 'MJD', 'FIBERID', 'Z', 'BI_CIV', 'DLA']
        if 'dr12' in catalog_filename.lower():
            cat['Z'] = cat['Z_VI']

        missed = open('missed.dat', 'w')
        missed.close()
        #print(cat[attrs])
        #print(np.array(cat[attrs]).dtype)
        meta = np.array(cat[attrs])
        #print("BI:", cat['BI_CIV'])
        for attr in ['SNR_DLA', 'SNR_H2']:
            meta = add_field(meta, [(attr, float)])
        self.cat.create_dataset('meta/qso', data=meta)
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

    def sdss_mask(self, meta):
        self.bads = np.genfromtxt(self.parent.folder + "/sdss_mask.txt", names=["plate", "mjd", "fiberid", "comment"])
        mask = np.ones(len(meta), dtype=bool)
        for b in self.bads:
            for ind in np.where((b["plate"] == meta["PLATE"]) * (b["mjd"] == meta["MJD"]) * (b["fiberid"] == meta["FIBERID"]))[0]:
                #print(ind, b, meta[ind])
                mask[ind] = False
        return mask

    def add_dla_cat(self, kind='Noterdaeme'):
        """
        Adding the DLA catalog
        """
        self.open(stored=self.stored, attr='a')
        print("adding DLA catalog:", kind)
        if kind == 'Noterdaeme':
            self.add_noterdaeme(self.parent.dla_cat_file)
        self.close()

    def add_noterdaeme(self, filename):
        """
        Adding Noterdaeme DLA catalog, organize the data
        parameters:
            - filename   :  the path to file contains Noterdaeme catalog
        """
        noter = np.genfromtxt(filename, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')
        print(noter.dtype)
        d = np.zeros(self.cat['meta/qso'][:].shape[0])

        dt = np.dtype([('z_abs', np.float32), ('NHI', np.float32)])
        #print(self.cat['meta/qso']['FIBERID'])
        self.open(stored=self.stored, attr='a')
        print(self.cat['meta/num'][0])
        for i in range(self.cat['meta/num'][0]):
            mjd, plate, fiber = self.cat['meta/qso']['MJD'][i], self.cat['meta/qso']['PLATE'][i], self.cat['meta/qso']['FIBERID'][i]
            #print(mjd, plate, fiber)
            #name = "meta/{0:05d}_{1:05d}_{2:04d}/DLA".format(plate, mjd, fiber)
            name = f"meta/{i}/DLA"
            m = (noter['MJD'] == mjd) * (noter['PLATE'] == plate) * (noter['FIBER'] == fiber)
            if name in self.cat:
                dla = self.cat[name][...]
                del self.cat[name]
            else:
                dla = np.empty(shape=(0,), dtype=dt)
            for nind in np.where(m)[0]:
                #print(nind)
                d[i] = 1
                if not any([noter[nind]['z_abs'] == di['z_abs'] for di in dla]):
                    dla = np.append(dla, np.array([(noter[nind]['z_abs'], noter[nind]['NHI'])], dtype=dt))
            #print(i, dla.shape, dla)

            self.cat.create_dataset(name, data=dla, chunks=True, dtype=dla.dtype)
            self.cat.flush()
                #print(self.cat[name + '/dla'][:])

        self.add_attr('dla', d)
        self.cat['meta/qso']['dla'] = d
        self.close()

    def input_spectra(self, skip=0, num=None, source='web'):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is already in catalog
        stored missing files in <missed.dat>
        parameters:
            - skip     :  number of spectra to skip
            - num      :  number of spectra to load
            - source   :  the path to file where SDSS spectra are located. If =='web' than the data will be downloaded from SDSS website (this is quite slow).
        """
        # print(source)
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso']) - skip

        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])

        meta = self.cat['meta/qso/'][...]
        mask = np.zeros(len(meta), dtype=bool)

        n = 0
        for i, q in enumerate(self.cat['meta/qso'][:]):
            q = self.cat['meta/qso'][i]
            name = f'data/{n}'
            sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
            if sdss_name not in sdss:
                sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
            #print(name)
            if source == 'web':
                res = self.download(ra=q['RA'], dec=q['DEC'], plate=q['PLATE'], MJD=q['MJD'], fiber=q['FIBERID'])
            else:
                sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                # print(sdss_name, sdss_name in sdss)
                if sdss_name not in sdss:
                    sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 50) and (self.cat['meta/qso'][i]['Z'] > 10 ** sdss[sdss_name + 'loglam'][:][100] / self.parent.lya / (1 - self.parent.v_proximate / 3e5) - 1)
                if res:
                    for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                        if attr == "flux":
                            self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:], z_mask=10**sdss[sdss_name + "loglam"][:]< (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)), dtype=sdss[sdss_name + attr].dtype)
                        else:
                            self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                    for attr, cut in zip(['SNR_DLA', 'SNR_H2'], [1230, 1150]):
                        m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                        if np.sum(m) > 0:
                            snr = np.median(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                            meta[attr][skip + i] = snr

            if res:
                self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [skip + 1]
                mask[i] = True
                n += 1
                if n % int(num / 10) == 0:
                    print(f"{n} out of {num}")

            else:
                #print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))

            if n >= num:
                break

        #print(self.cat["data"].keys())
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=meta[mask])
        self.cat['meta/num'][0] = [sum(mask)]
        print("meta len: ", len(self.cat['meta/qso'][...]), num, self.cat['meta/num'][...])
        if source != 'web':
            sdss.close()
        self.missed.close()
        self.close()

    def append(self, skip=0, num=None, source='web'):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is alaredy in catalog
        stored missing files in <missed.dat>
        parameters:
            - skip     :  number of spectra to skip
            - num      :  number of spectra to load
            - source   :  the path to file where SDSS spectra are located. If =='web' than the data will be downloaded from SDSS website (this is quite slow).
        """
        #print(source)
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso']) - skip
        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[skip])

        meta = self.cat['meta/qso/'][...]

        for i, q in enumerate(self.cat['meta/qso'][skip:num+skip]):
            name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])
            #print(name)
            if name not in self.cat:
                #print(i, name, name in self.cat)
                if source == 'web':
                    res = self.download(ra=q['RA'], dec=q['DEC'], plate=q['PLATE'], MJD=q['MJD'], fiber=q['FIBERID'])
                else:
                    sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                    #print(sdss_name, sdss_name in sdss)
                    if sdss_name in sdss.keys():
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)
                        for attr, cut in zip(['SNR_DLA', 'SNR_H2'], [1230, 1150]):
                            m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                            if np.sum(m) > 0:
                                snr = np.median(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                                meta[attr][skip+i] = snr
                    res = sdss_name in sdss.keys()
                    #try:
                if res:
                    self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [skip+1]
                else:
                    print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                    self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
            if i % int(num / 10) == 0:
                print(f"{i} out of {num}")

        self.cat['meta/qso'][...] = meta

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

    def correct_masked(self, flux, mask, z_mask=None):
        if z_mask is None:
            z_mask = np.ones_like(mask, dtype=bool)
        masked_array = np.ma.masked_array(flux[z_mask].astype(np.float64), mask=mask[z_mask] != 0)
        smooth_median_values = medfilt(masked_array.filled(np.nan), kernel_size=9)
        temp = flux[z_mask]
        temp[masked_array.mask] = smooth_median_values[masked_array.mask].astype(np.float32)
        flux[z_mask] = temp[:]
        return flux

    def add_dla(self, name, z_qso, z_dla=None, logN=None, debug=False):
        x, y, err = self.cat[name + '/loglam'][:], self.cat[name + '/flux'][:], 1 / np.sqrt(self.cat[name + '/ivar'][:])
        imin, imax = max(x[0], np.log10(self.parent.lyc * (1 + z_qso))), min(x[-1], np.log10(self.parent.lya * (1 + z_qso) * (1 - self.parent.v_proximate/3e5)))
        if z_dla is None:
            z_dla = 10 ** (np.random.randint(int(imin * 1e4), int(imax * 1e4)) / 1e4) / self.parent.lya - 1
        if logN is None:
            logN = self.parent.N_range[0] + np.random.rand() * (self.parent.N_range[1] - self.parent.N_range[0])
        t = line(logN=logN, b=30, z=z_dla, resolution=2000)
        if debug:
            m = (x > imin) * (x < imax)
            print(name, z_qso)
            print(imin, imax)
            print(z_dla, logN)
            plt.plot(10 ** x[m], y[m])
        f = convolve_res(10 ** x, np.exp(-t.tau(10 ** x)), t.resolution)
        y = y * f + err * (1 - f) * np.random.randn(len(x)) / 2
        y[np.logical_not(np.isfinite(y))] = np.nanmedian(y)
        self.cat[name + '/flux'][...] = y

        if debug:
            plt.plot(10 ** x[m], y[m])
            plt.show()

        return z_dla, logN

    def make_dla_mock(self, num=None, source='web', dla_cat=None):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is already in catalog
        stored missing files in <missed.dat>
        parameters:
            - num           :  number of spectra to be generated
            - source        :  filename of the catalog contained SDSS spectra
            - dla_cat       :  the path to file contains DLA catalog (in Noterdaeme catalog for now)
        """
        noter = np.genfromtxt(dla_cat, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12), dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')

        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso'])

        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])

        meta = self.cat['meta/qso/'][...]
        mask = np.zeros(len(meta), dtype=bool)
        
        d = np.zeros(num)
        
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
                    res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 100) and (self.cat['meta/qso'][i]['Z'] > 10**sdss[sdss_name + 'loglam'][:][100] / self.parent.lya / (1 - self.parent.v_proximate/3e5) - 1)
                    if res:
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                if attr == "flux":
                                    self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:]), dtype=sdss[sdss_name + attr].dtype)
                                else:
                                    self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                        for attr, cut in zip(['SNR_DLA', 'SNR_H2'], [1230, 1150]):
                            m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                            if np.sum(m) > 0:
                                snr = np.median(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                                meta[attr][i] = snr

                        mask[i] = True
                        z_dla, NHI = self.add_dla(name, z_qso=self.cat['meta/qso'][i]['Z'])
                        d[n] = 1
                        #print(name + '/dla')
                        data = np.array([(z_dla, NHI)], np.dtype([('z_abs', np.float32), ('NHI', np.float32)]))

                        self.cat.create_dataset(name.replace('data', 'meta') + '/DLA', data=data, chunks=True, dtype=data.dtype)
                        self.cat.flush()
                        n += 1
                        #print(n)
                        if n % int(num / 10) == 0:
                            print(f"{n} out of {num}")


                    else:
                        #print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                        self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
            if n >= num:
                break

        print(sum(mask))
        self.cat['meta/num'][0] = [sum(mask)]
        data = self.cat['meta/qso'][...]
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=data[mask])
        self.add_attr('DLA', d)
        self.cat['meta/qso']['DLA'] = d

        self.cat['meta/qso'][...] = meta[mask]

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

    def make_dla_mock_uniform(self, num=None, source='web', dla_cat=None):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is already in catalog
        stored missing files in <missed.dat>
        parameters:
            - num           :  number of spectra to be generated
            - source        :  filename of the catalog contained SDSS spectra
            - dla_cat       :  the path to file contains DLA catalog (in Noterdaeme catalog for now)
        """
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso'])

        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])

        meta = self.cat['meta/qso/'][...]

        # >>> mask spectra:
        mask = self.sdss_mask(meta)
        mask *= ((meta['DLA'] == False) *
                (meta['BI_CIV'] == 0) *
                (meta['Z'] > self.parent.z_range[0]))
                #(meta['Z'] < self.parent.z_range[1]))

        inds = []
        zs = meta['Z'][mask]
        masked_inds = np.where(mask)[0]

        # >>> iterate over the spectra to use <num> spectra
        for i in range(num):
            z_dla = np.random.uniform(self.parent.z_range[0], self.parent.z_range[1])
            z_inds = np.where(zs > z_dla * (1 + 2 * self.parent.v_proximate / 3e5))[0]
            #print(i, z_dla, len(z_inds))
            count = 1
            while True:
                ind = masked_inds[np.random.choice(z_inds)]
                #print(i, z_dla, ind)
                q = self.cat['meta/qso'][ind]
                name = f'data/{i}'
                sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                if sdss_name not in sdss:
                    sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                #print(sdss_name)
                res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 50) and (10**sdss[sdss_name + 'loglam'][:][50] < (q['Z'] + 1) * self.parent.lya * (1 - self.parent.v_proximate / 3e5)) and ((q['Z'] + 1) * self.parent.lyb < (1 + z_dla) * self.parent.lya * (1 + self.parent.v_proximate / 3e5))
                flux = sdss[sdss_name + 'flux'][:np.argmin(np.abs(10**sdss[sdss_name + 'loglam'][:] - self.parent.lya * (1 + q['Z'])))]
                #if not (np.all(np.isfinite(flux)) * np.isfinite(np.sum(flux))):
                #    print(sdss_name, flux)
                res *= np.all(np.isfinite(flux)) * np.isfinite(np.sum(flux)) * (np.sum(flux) != 0) * (np.sum(sdss[sdss_name + "and_mask"][10**sdss[sdss_name + "loglam"][:]< (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)] != 0) < 50)
                if res:
                    #print(10 ** sdss[sdss_name + "loglam"][np.where(sdss[sdss_name + "and_mask"][:] != 0)[0]])

                    #flux[sdss[sdss_name + 'and_mask']] =
                    for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                        if attr == "flux":
                            self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:], z_mask=10**sdss[sdss_name + "loglam"][:]< (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)), dtype=sdss[sdss_name + attr].dtype)
                        else:
                            self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                    for attr, cut in zip(['SNR_DLA', 'SNR_H2'], [1230, 1150]):
                        m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                        if np.sum(m) > 0:
                            snr = np.median(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                            meta[attr][i] = snr

                    inds.append(ind)
                    z_dla, NHI = self.add_dla(name, z_qso=self.cat['meta/qso'][i]['Z'], z_dla=z_dla)
                    #print(name + '/dla')
                    data = np.array([(z_dla, NHI)], np.dtype([('z_abs', np.float32), ('NHI', np.float32)]))

                    self.cat.create_dataset(name.replace('data', 'meta') + '/DLA', data=data, chunks=True, dtype=data.dtype)
                    self.cat.flush()
                    if i % int(num / 10) == 0:
                        print(f"{i} out of {num}")

                    self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [1]
                    break
                else:
                    count += 1
                if count > 10:
                    break

        print(len(inds))
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=meta[inds])
        self.add_attr('DLA', np.ones(num, dtype=bool))
        self.cat['meta/qso']['DLA'] = np.ones(num, dtype=bool)

        print(len(self.cat['meta/qso'][...]))

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

    def add_H2(self, name, z_qso, debug=False, snr_thres=2):
        H2cutoff = self.parent.H2bands['L5-0']
        x, y, err = self.cat[name + '/loglam'][:], self.cat[name + '/flux'][:], 1 / np.sqrt(self.cat[name + '/ivar'][:])
        z_max = (1 + z_qso) * (1 - self.parent.v_proximate / 3e5) - 1
        m = 10 ** x < (1 + z_max) * self.parent.H2bands['L0-0']
        z_H2, logN = None, None
        if np.sum(m) > 0:
            snr = pd.DataFrame(err[m] / y[m]).rolling(window=10).mean().fillna(method='bfill').fillna(method='ffill')
            if snr[0].gt(snr_thres).sum() > 0:
                #print(name, snr[snr[0].gt(snr_thres)].index[0])
                imin, imax = max(x[snr[snr[0].gt(snr_thres)].index[0]], np.log10(self.parent.lyc * (1 + z_qso))), min(x[-1], np.log10(H2cutoff * (1 + z_qso) * (1 + self.parent.v_proximate / 3e5)))
                z_H2 = 10 ** (imax - np.random.rand() ** 2 * (imax - imin)) / H2cutoff - 1
                logN = 19.0 + np.random.rand() ** 2 * 3
                #z_H2, logN = z_qso - 0.1, 19
                x1, f = self.H2.calc_profile(x=10**x, z=z_H2, logN=logN, b=5, j=6, T=100, exc='low')
                f = convolve_res(x1, f, 1800)
                if debug:
                    m = (x > imin) * (x < imax)
                    print(name, z_qso)
                    print(imin, imax)
                    print(z_H2, logN)
                    plt.plot(10 ** x[m], y[m])
                y = y * f + err * (1 - f) * np.random.randn(len(x)) / 2
                #y = f
                y[np.logical_not(np.isfinite(y))] = np.nanmedian(y)
                self.cat[name + '/flux'][...] = y

                if debug:
                    plt.plot(10 ** x[m], y[m])
                    plt.show()

        return z_H2, logN

    def make_H2_mock(self, num=None, source='web', dla_cat=None, snr=2):
        """
        append the spectra of the catalog from the source (website or local file) and store it in hdf5 file in <data/> dataset
        check if spectrum is alaredy in catalog
        stored missing files in <missed.dat>
        parameters:
            - num           :  number of spectra to be generated
            - source        :  filename of the catalog contained SDSS spectra
            - dla_cat       :  the path to file contains DLA catalog (in Noterdaeme catalog for now)
        """
        print(source)
        self.missed = open('missed.dat', 'a')
        self.open(stored=self.stored, attr='a')
        if source != 'web':
            sdss = h5py.File(source, 'r')

        if num == None:
            num = len(self.cat['meta/qso'])

        if 'meta/num' not in self.cat:
            self.cat.create_dataset('meta/num', data=[0])

        meta = self.cat['meta/qso/'][...]

        d = np.zeros(num)
        mask = np.zeros(len(self.cat['meta/qso'][:]), dtype=bool)
        n = 0
        self.H2 = H2abs()
        for i, q in enumerate(self.cat['meta/qso'][:]):
            name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])
            if name not in self.cat:
                res = False
                sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                if sdss_name not in sdss:
                    sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 100) and (self.cat['meta/qso'][i]['Z'] > 10 ** sdss[sdss_name + 'loglam'][:][0] / self.parent.H2bands['L5-0'] / (1 - self.parent.v_proximate / 3e5) - 1)
                if res:
                    m = ((10 ** sdss[sdss_name + 'loglam'][:] / (1 + self.cat['meta/qso'][i]['Z'])) > 1020) * ((10 ** sdss[sdss_name + 'loglam'][:] / (1 + self.cat['meta/qso'][i]['Z'])) < 1150)
                    res *= np.median(np.divide(sdss[sdss_name + 'flux'][m], 1 / np.sqrt(sdss[sdss_name + 'ivar'][m]))) > snr
                if res:
                    for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                        if name + '/' + attr not in self.cat:
                            self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                    for attr, cut in zip(['SNR_DLA', 'SNR_H2'], [1230, 1150]):
                        m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                        if np.sum(m) > 0:
                            meta[attr][i] = np.median(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))

                    z_H2, logN = self.add_H2(name, z_qso=self.cat['meta/qso'][i]['Z'])
                    if z_H2 is not None:
                        mask[i] = True
                        if 1:
                            self.add_dla(name, z_dla=z_H2, logN=20.7, z_qso=self.cat['meta/qso'][i]['Z'])
                        d[n] = 1
                        # print(name + '/dla')
                        data = np.array([(z_H2, logN)], np.dtype([('z_abs', np.float32), ('logN', np.float32)]))

                        self.cat.create_dataset(name.replace('data', 'meta') + '/H2', data=data, chunks=True,
                                                dtype=data.dtype)
                        self.cat.flush()
                        n += 1
                        if n % int(num / 10) == 0:
                            print(f"{n} out of {num}")
                        self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [1]
                else:
                    # print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
                    self.missed.write('{0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
            if n >= num:
                break

        print(sum(mask))
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=meta[mask])
        self.add_attr('H2', d)
        self.cat['meta/qso']['H2'] = d

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

    def get(self, ind=None, plate=0, mjd=0, fiber=0):
        """
        retrieve numpy recarray with the spectrum from the stored hdf5 file
        """
        self.open()
        if ind == None:
            name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(plate, mjd, fiber)
        else:
            name = f"data/{ind}"

        #self.cat.visit(print)
        if name in self.cat:
            dtype = [('loglam', '<f8'), ('flux', '<f8'), ('ivar', '<f8'), ('and_mask', '<i8')]
            data = np.core.records.fromarrays([self.cat[name + '/' + attr[0]][:] for attr in dtype], dtype=dtype)
        else:
            print(f"There is not {name} entry in the database")
            data = None
        self.close()
        return data

    def __getitem__old(self, i):
        self.open()
        q = self.cat['meta/qso'][i]
        self.close()
        return self.get(q['PLATE'], q['MJD'], q['FIBERID'])

    def __getitem__(self, i):
        return self.get(ind=i)


if __name__ == '__main__':
    cat = catalog()
    cat.prepare_spectra_catalog(sdss_cat_file="C:/science/SDSS/DR14/DR14Q_v4_4.fits", dla_cat_file="c:/science/Noterdaeme/DLAs/DLA_catalogs/Noterdaeme/DLA_DR14_v2b.dat", output_filename="c:/science/SDSS/DR14/dr14.ecsv")