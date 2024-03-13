import matplotlib.pyplot as plt
import numpy as np
import pickle

from .catalog import catalog

class CNN():
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.init_constants()
        self.d = None
        self.cnn = None

    def init_constants(self):
        """
        initialize constants
        """
        self.lya, self.lyb, self.lyc = 1215.67, 1025.72, 915
        self.H2bands = {'L0-0': 1108.37963, 'L1-0': 1092.461585, 'L2-0': 1077.41625, 'L3-0': 1063.16816, 'L4-0': 1049.660515, 'L5-0': 1036.84467}

    def prepare_catalog(self, action, skip=0, num=0, sdss_cat_file=None, catalog_filename=None, sdss_source=None):
        """
        prepare (create/append/load) the catalog contains the SDSS spectra
        parameters:
            - action            :   the action to do. Can be 'new' to create new catalog, ot 'load' to load already created
            - skip             :    number of spectra to skip
            - num               :   number of spectra to use
            - sdss_cat_file     :   the filename contains SDSS catalog Table. if None, use from settings
            - catalog_filename  :   the filename where the catalog will be stored (of loaded). if None, use from settings
            - sdss_source       :   the filename where raw SDSS is data located. Can be 'web' for download from SDSS website. if None, use from settings
        """

        if sdss_cat_file != None:
            self.sdss_cat_file = sdss_cat_file

        if catalog_filename != None:
            self.catalog_filename = catalog_filename

        if sdss_source != None:
            self.sdss_source = sdss_source

        if action == 'new':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            if num > 0:
                self.cat.append(num=num, skip=skip, source=self.sdss_source)
            else:
                print('the number of spectra to use is not provided')

        elif action == 'load':
            self.cat = catalog(self, stored=self.catalog_filename)

        elif action in ['add', 'update']:
            self.cat = catalog(self, stored=self.catalog_filename)
            self.cat.append(num=num, skip=skip, source=self.sdss_source)
            #self.cat.add_dla_cat(noterdaeme_file)

        elif action == 'dla_mock':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            # sdss.append(num=num_specs, source='/mnt/c/science/dr14.hdf5')
            if num > 0:
                self.cat.make_dla_mock(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
            else:
                print('the number of spectra to create is not provided')
            # sdss.make_mock(num=num_specs, source='/mnt/c/science/dr14.hdf5', dla_cat=noterdaeme_file)

        elif action == 'H2_mock':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            # sdss.append(num=num_specs, source='/mnt/c/science/dr14.hdf5')
            if num > 0:
                self.cat.make_H2_mock(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
            else:
                print('the number of spectra to create is not provided')
        self.cat.close()
        #self.catalog = catalog()
