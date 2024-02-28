import numpy as np

from ..main import CNN
from .H2_data_class import h2_data
#from .H2_conv_model import CNN_for_H2
class CNN_h2(CNN):
    def __init__(self, **kwargs):
        super(CNN_h2, self).__init__(**kwargs)

    def h2_prepare_data(self, action, num=0, window=None, bands=6, valid=0.2, dropout=0.5, dropout_dla=0.1, start=0):
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

        self.d = h2_data(self, window=self.window, bands=bands, timing=False, filename=self.catalog_filename.replace('.hdf5', '_dla_data.hdf5'))

        if action == 'new':
            self.d.new()
            self.d.make(num=num, valid=valid, dropout=dropout, dropout_dla=dropout_dla, start=start)
            # d.make_sets(valid=0.1, dropout=0.3, dropout_dla=0.8, shuffle=True, batch=30000)

        elif action == 'load':
            self.d.open()

    def plot_spec(self, ind, preds=False):
        """
        Plot SDSS spectrum regarding DLA search. This includes info from data structure for DLA search
        parameters:
            - ind        :   number of the spectrum to plot
            - preds      :   if True, plot the predicition of the CNN model
        """
        fig, ax = self.d.plot_spec(ind)
        if preds:
            fig, ax = self.d.plot_preds(ind, fig=fig)

        return fig, ax