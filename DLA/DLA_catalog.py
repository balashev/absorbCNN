import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

from ..line_profiles import line, convolve_res, H2abs
from ..utils import add_field
from ..catalog import catalog

class DLA_catalog(catalog):
    def __init__(self, parent, stored=None):
        super().__init__(parent, stored)
        self.parent = parent
        self.stored = stored

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
        noter = np.genfromtxt(dla_cat, skip_header=32, names=True, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12),
                              dtype='U9, <i8, <i8, <i8, U5, U4, <f8, <f8, <f8, <f8, <f8')

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
                # print(name)
                if name not in self.cat:
                    # print(i, name, name in self.cat, n)
                    res = False
                    sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                    if sdss_name not in sdss:
                        sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                    # print(sdss_name, sdss_name in sdss)
                    res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 100) and (self.cat['meta/qso'][i]['Z'] > 10 ** sdss[sdss_name + 'loglam'][:][100] / self.parent.lya / (1 - self.parent.v_proximate / 3e5) - 1)
                    if res:
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                if attr == "flux":
                                    self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:]), dtype=sdss[sdss_name + attr].dtype)
                                else:
                                    self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                        for attr, cut in zip(['SNR'], [1230]):
                            m = self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))
                            if np.sum(m) > 0:
                                snr = np.median(
                                    np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                                meta[attr][i] = snr

                        mask[i] = True
                        z_dla, logN = self.add_dla(name, z_qso=self.cat['meta/qso'][i]['Z'])
                        d[n] = 1
                        # print(name + '/dla')
                        data = np.array([(z_dla, logN)], np.dtype([('z_abs', self.parent.dt), ('logN', self.parent.dt)]))

                        self.cat.create_dataset(name.replace('data', 'meta') + '/DLA', data=data, chunks=True, dtype=data.dtype)
                        self.cat.flush()
                        n += 1
                        # print(n)
                        if n % int(num / 10) == 0:
                            print(f"{n} out of {num}")

                    else:
                        # print('missed: {0:04d} {1:05d} {2:04d} \n'.format(q['PLATE'], q['MJD'], q['FIBERID']))
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
        # (meta['Z'] < self.parent.z_range[1]))

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
                # print(i, z_dla, ind)
                q = self.cat['meta/qso'][ind]
                name = f'data/{i}'
                sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                if sdss_name not in sdss:
                    sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                # print(sdss_name)
                res = ((sdss_name in sdss) and
                       (len(sdss[sdss_name + 'loglam'][:]) > 50) and
                       (10 ** sdss[sdss_name + 'loglam'][:][50] < (q['Z'] + 1) * self.parent.lya * (1 - self.parent.v_proximate / 3e5)) and
                       ((q['Z'] + 1) * self.parent.lyb < (1 + z_dla) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)))
                flux = sdss[sdss_name + 'flux'][:np.argmin(np.abs(10 ** sdss[sdss_name + 'loglam'][:] - self.parent.lya * (1 + q['Z'])))]
                # if not (np.all(np.isfinite(flux)) * np.isfinite(np.sum(flux))):
                #    print(sdss_name, flux)
                res *= np.all(np.isfinite(flux)) * np.isfinite(np.sum(flux)) * (np.sum(flux) != 0) * (np.sum(sdss[sdss_name + "and_mask"][10 ** sdss[sdss_name + "loglam"][:] < (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)] != 0) < 50)
                if res:
                    # print(10 ** sdss[sdss_name + "loglam"][np.where(sdss[sdss_name + "and_mask"][:] != 0)[0]])

                    # flux[sdss[sdss_name + 'and_mask']] =
                    for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                        if attr == "flux":
                            self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:], z_mask=10 ** sdss[sdss_name + "loglam"][:] < (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)), dtype=sdss[sdss_name + attr].dtype)
                        else:
                            self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                    for attr, cut in zip(['SNR'], [1230]):
                        m = (self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))) * (self.cat[name + '/loglam'] > np.log10(912 * (1 + q['Z'])))
                        if np.sum(m) > 0:
                            meta[attr][ind] = np.nanmedian(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))
                            #print(i, ind, snr)

                    inds.append(ind)
                    z_dla, logN = self.add_dla(name, z_qso=self.cat['meta/qso'][i]['Z'], z_dla=z_dla)
                    # print(name + '/dla')
                    data = np.array([(z_dla, logN)], np.dtype([('z_abs', self.parent.dt), ('logN', self.parent.dt)]))

                    self.cat.create_dataset(name.replace('data', 'meta') + '/abs', data=data, chunks=True, dtype=data.dtype)
                    self.cat.flush()
                    if i % int(num / 10) == 0:
                        print(f"{i} out of {num}")

                    self.cat['meta/num'][0] = [self.cat['meta/num'][0] + 1] if 'meta/num' in self.cat else [1]
                    break
                else:
                    count += 1
                if count > 10:
                    break

        #print(len(inds))
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=meta[inds])
        self.add_attr('abs', np.ones(num, dtype=bool))
        self.cat['meta/qso']['abs'] = np.ones(num, dtype=bool)

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()

