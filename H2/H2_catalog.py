import h5py
import numpy as np
import matplotlib.pyplot as plt

from ..line_profiles import line, convolve_res, H2abs
from ..utils import add_field
from ..catalog import catalog

class H2_catalog(catalog):
    def __init__(self, parent, stored=None):
        super().__init__(parent, stored)
        self.parent = parent
        self.stored = stored

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
        # >>> mask spectra:
        mask = self.sdss_mask(meta)
        mask *= ((meta['DLA'] == False) *
                 (meta['BI_CIV'] == 0) *
                 (meta['Z'] > self.parent.z_range[0]))
        # (meta['Z'] < self.parent.z_range[1]))

        n = 0
        self.H2 = H2abs()
        inds = []
        zs = meta['Z'][mask]
        masked_inds = np.where(mask)[0]

        # >>> iterate over the spectra to use <num> spectra
        for i in range(num):
            z_H2 = np.random.uniform(self.parent.z_range[0], self.parent.z_range[1])
            z_inds = np.where(zs > z_H2 * (1 + 2 * self.parent.v_proximate / 3e5))[0]
            #print(i, z_dla, len(z_inds))
            count = 1
            while True:
                ind = masked_inds[np.random.choice(z_inds)]
                q = self.cat['meta/qso'][ind]
                #print(i, count, z_H2, ind, q['Z'])
                name = f'data/{i}'
                sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}'.format(q['PLATE'], q['MJD'], q['FIBERID'])
                if name not in self.cat:
                    res = False
                    sdss_name = 'data/{0:05d}/{1:04d}/{2:05d}/'.format(q['PLATE'], q['FIBERID'], q['MJD'])
                    if sdss_name not in sdss:
                        sdss_name = 'data/{0:05d}_{1:05d}_{2:04d}/'.format(q['PLATE'], q['MJD'], q['FIBERID'])

                    # >>> redshift conditions:
                    res = (sdss_name in sdss) and (len(sdss[sdss_name + 'loglam'][:]) > 100) and (10**sdss[sdss_name + 'loglam'][:][0] < (q['Z'] + 1) * self.parent.H2bands['L5-0'] * (1 + self.parent.v_proximate / 3e5))
                    #(self.cat['meta/qso'][i]['Z'] > 10 ** sdss[sdss_name + 'loglam'][:][0] / self.parent.H2bands['L5-0'] / (1 - self.parent.v_proximate / 3e5) - 1))
                    #print("res1", res)
                    #print((sdss_name in sdss), (len(sdss[sdss_name + 'loglam'][:]) > 100), (10**sdss[sdss_name + 'loglam'][:][0] < (q['Z'] + 1) * self.parent.H2bands['L5-0'] * (1 + self.parent.v_proximate / 3e5)))
                    # >>> good flux conditions:
                    flux = sdss[sdss_name + 'flux'][:np.argmin(np.abs(10 ** sdss[sdss_name + 'loglam'][:] - self.parent.lya * (1 + q['Z'])))]
                    res *= np.all(np.isfinite(flux)) * np.isfinite(np.sum(flux)) * (np.sum(flux) != 0) * (np.sum(sdss[sdss_name + "and_mask"][10 ** sdss[sdss_name + "loglam"][:] < (q['Z'] + 1) * self.parent.H2bands['L0-0'] * (1 + self.parent.v_proximate / 3e5)] != 0) < 50)
                    #print("res2", res)
                    # >>> snr conditions:
                    if res:
                        m = ((10 ** sdss[sdss_name + 'loglam'][:] / (1 + q['Z'])) > 1020) * ((10 ** sdss[sdss_name + 'loglam'][:] / (1 + q['Z'])) < 1150)
                        res *= np.median(np.divide(sdss[sdss_name + 'flux'][m], 1 / np.sqrt(sdss[sdss_name + 'ivar'][m]))) > snr
                    #print("res3", res)
                    if res:
                        for attr in ['loglam', 'flux', 'ivar', 'and_mask']:
                            if name + '/' + attr not in self.cat:
                                if attr == "flux":
                                    self.cat.create_dataset(name + '/' + attr, data=self.correct_masked(sdss[sdss_name + attr][:], sdss[sdss_name + "and_mask"][:], z_mask=10 ** sdss[sdss_name + "loglam"][:] < (q['Z'] + 1) * self.parent.lya * (1 + self.parent.v_proximate / 3e5)), dtype=sdss[sdss_name + attr].dtype)
                                else:
                                    self.cat.create_dataset(name + '/' + attr, data=sdss[sdss_name + attr][:], dtype=sdss[sdss_name + attr].dtype)

                        # >>> save SNR:
                        for attr, cut in zip(['SNR'], [1150]):
                            m = (self.cat[name + '/loglam'] < np.log10(cut * (1 + q['Z']))) * (self.cat[name + '/loglam'] > np.log10(912 * (1 + q['Z'])))
                            if np.sum(m) > 0:
                                meta[attr][ind] = np.nanmedian(np.divide(self.cat[name + '/flux'][m], 1 / np.sqrt(self.cat[name + '/ivar'][m])))

                        # >>> add H2 absorption system
                        z_H2, logN = self.add_H2(name, z_qso=self.cat['meta/qso'][i]['Z'], z_H2=z_H2)

                        inds.append(ind)
                        if 1:
                            self.add_dla(name, z_dla=z_H2, logN=20.7, z_qso=self.cat['meta/qso'][i]['Z'])

                        # print(name + '/dla')
                        data = np.array([(z_H2, logN)], np.dtype([('z_abs', self.parent.dt), ('logN', self.parent.dt)]))

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

        print(sum(mask))
        del self.cat['meta/qso']
        self.cat.create_dataset('meta/qso', data=meta[inds])
        self.add_attr('abs', np.ones(num, dtype=bool))
        self.cat['meta/qso']['abs'] = np.ones(num, dtype=bool)

        if source != 'web':
            sdss.close()

        self.missed.close()
        self.close()