#from keras.utils import PyDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
from multiprocessing import Pool
import numpy as np
import os
import pickle
from scipy.signal import argrelextrema
import torch

from ..main import CNN
from ..stats import distr1d
from .DLA_catalog import DLA_catalog as catalog
from .DLA_data_class import dla_data
from .DLA_conv_model_keras import CNN_for_DLA_keras
from .DLA_conv_model_torch import CNN_for_DLA_torch

class CNN_dla(CNN):
    def __init__(self, **kwargs):
        super(CNN_dla, self).__init__(**kwargs)

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
            #self.cat.make_dla_mock(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)
            self.cat.make_dla_mock_uniform(num=num, source=self.sdss_source, dla_cat=self.dla_cat_file)

        self.cat.close()

    def dla_add_cat(self, dla_cat_file=None):
        """
        add info about DLAs from DLA catalog located at <dla_cat_file>.
        """
        if dla_cat_file != None:
            self.dla_cat_file = dla_cat_file

        self.cat.add_dla_cat(self.dla_cat_file)

    def dla_prepare_data(self, action, num=None, valid=0.2, dropout=0.7, dropout_dla=0.2, start=0):
        """
        Prepare appropriate CNN data structure for work with DLA. The data structure is stored locally in <self.catalog_filename>_dla_data.hdf5
        The data structure contains spectral windows (input) with appropriate labels (output) in 'training' and 'validation' samples.
        parameters:
            - action        :   the action to do. Can be 'new' to create new data structure, ot 'load' to load already created
            - num           :   number of spectra to use
            - valid         :   the percentage of validation sample
            - dropout       :   the percentage of the dropout in spectral windows without DLA
            - dropout_dla   :   the percentage of the dropout in spectral windows with DLA
            - start         :   number of the spectrum to start
        """
        if num == None:
            num = self.num

        self.d = dla_data(self, timing=False, filename=self.catalog_filename.replace('.hdf5', '_dla_data.hdf5'))

        if action == 'new':
            self.d.new()
            self.d.make(num=num, valid=valid, dropout=dropout, dropout_dla=dropout_dla, start=start)

        elif action == 'load':
            self.d.open()

    def run_cnn(self, action, epochs=0, model_filename=None, stats=False):

        if self.cnn_env == 'keras':
            self.cnn = self.cnn = CNN_for_DLA_keras(dt=self.datatype)
            self.cnn_keras(action=action, epochs=epochs, model_filename=model_filename, stats=stats)

        elif self.cnn_env == 'torch':
            self.cnn = CNN_for_DLA_torch()
            self.cnn_torch(action=action, epochs=epochs, model_filename=model_filename, stats=stats)

