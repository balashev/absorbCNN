import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
import numpy as np
import os
import pickle

from ..main import CNN
from ..utils import Timer
from .H2_catalog import H2_catalog as catalog
from .H2_data_class import h2_data
from .H2_conv_model_keras import CNN_for_H2_keras
from .H2_conv_model_torch import CNN_for_H2_torch

class CNN_h2(CNN):
    def __init__(self, **kwargs):
        super(CNN_h2, self).__init__(**kwargs)

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

        elif action == 'H2_mock':
            self.cat = catalog(self)
            self.cat.create(catalog_filename=self.sdss_cat_file, output_filename=self.catalog_filename)
            # sdss.append(num=num_specs, source='/mnt/c/science/dr14.hdf5')
            if num > 0:
                self.cat.make_H2_mock(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
            else:
                print('the number of spectra to create is not provided')
        self.cat.close()

    def h2_prepare_data(self, action, num=0, window=None, bands=None, valid=0.2, dropout=0.5, dropout_dla=0.1, start=0):
        """
        Prepare appropriate CNN data structure for work with DLA. The data structure is stored locally in <self.catalog_filename>_dla_data.hdf5
        The data structure contains spectral windows (input) with appropriate labels (output) in 'training' and 'validation' samples.
        parameters:
            - action        :   the action to do. Can be 'new' to create new data structure, ot 'load' to load already created
            - num           :   number of spectra to use
            - window    :   the size of the spectral window.
            - valid         :   the percentage of validation sample
            - dropout       :   the percentage of the dropout in spectral windows without DLA
            - dropout_dla   :   the percentage of the dropout in spectral windows with DLA
            - start         :   number of the spectrum to start
        """
        if window != None:
            self.window = window

        if bands != None:
            self.bands = bands

        self.d = h2_data(self, window=self.window, bands=self.bands, timing=False, filename=self.catalog_filename.replace('.hdf5', '_data.hdf5'))

        if action == 'new':
            self.d.new()
            self.d.make()

        elif action == 'load':
            self.d.open()

    def run_cnn(self, action, epochs=0, model_filename=None, stats=False):
        if self.cnn_env == 'keras':
            self.cnn = CNN_for_H2_keras(dt=self.datatype)
            self.cnn_keras(action=action, epochs=epochs, model_filename=model_filename, stats=stats)

        elif self.cnn_env == 'torch':
            self.cnn = CNN_for_H2_torch()
            self.cnn_torch(action=action, epochs=epochs, model_filename=model_filename, stats=stats)