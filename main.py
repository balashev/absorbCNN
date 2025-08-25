import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from .catalog import catalog

class CNN():
    def __init__(self, **kwargs):
        print(kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.folder = os.path.dirname(os.path.realpath(__file__))
        self.QC_folder += "/run1/"
        while os.path.exists(self.QC_folder):
            self.QC_folder = self.QC_folder[:self.QC_folder.index("run") + 3:] + f"{int(self.QC_folder[(self.QC_folder.index("run") + 3):-1]) + 1}/"
        os.mkdir(self.QC_folder)
        self.init_constants()
        self.d = None
        self.cnn = None

    def init_constants(self):
        """
        initialize constants
        """
        self.lya, self.lyb, self.lyc = 1215.67, 1025.72, 915
        self.H2bands = {'L0-0': 1108.37963, 'L1-0': 1092.461585, 'L2-0': 1077.41625, 'L3-0': 1063.16816, 'L4-0': 1049.660515, 'L5-0': 1036.84467}

    def prepare_catalog(self, action, skip=0, num=None, sdss_cat_file=None, catalog_filename=None, sdss_source=None):
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

        print(action)
        if num == None:
            num = self.num

        if sdss_cat_file != None:
            self.sdss_cat_file = sdss_cat_file

        if catalog_filename != None:
            self.catalog_filename = catalog_filename

        if sdss_source != None:
            self.sdss_source = sdss_source

        if action == 'new':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            self.cat.input_spectra(num=num, skip=skip, source=self.sdss_source)
            self.cat.add_dla_cat()

        elif action == 'load':
            self.cat = catalog(self, stored=self.catalog_filename)

        elif action in ['add', 'update']:
            self.cat = catalog(self, stored=self.catalog_filename)
            self.cat.append(num=num, skip=skip, source=self.sdss_source)
            #self.cat.add_dla_cat()

        elif action == 'dla_mock':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            # sdss.append(num=num_specs, source='/mnt/c/science/dr14.hdf5')
            #self.cat.make_dla_mock(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
            self.cat.make_dla_mock_uniform(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
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
