import numpy as np

from ..main import CNN
from .H2_data_class import h2_data
from .H2_conv_model import CNN_for_H2
class CNN_h2(CNN):
    def __init__(self, **kwargs):
        super(CNN_h2, self).__init__(**kwargs)

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

        self.d = h2_data(self, window=self.window, bands=self.bands, timing=False, filename=self.catalog_filename.replace('.hdf5', '_dla_data.hdf5'), lines_file=self.lines_file, energy_file=self.energy_file)

        if action == 'new':
            self.d.new()
            self.d.make()

            #plt.show()
            #self.d.make(num=num, valid=valid, dropout=dropout, dropout_dla=dropout_dla, start=start)
            # d.make_sets(valid=0.1, dropout=0.3, dropout_dla=0.8, shuffle=True, batch=30000)

        elif action == 'load':
            self.d.open()

    def h2_cnn(self, action, epochs=0, h2_model_filename=None, stats=False):
        """
        Create and run DLA CNN model of the data_structure
        parameters:
            - action                :   the action to do. Can be 'run' to run. 'run_batch' to run using the batches (memory saving regime), and 'load' to load already created model
            - epochs                :   number of epochs to run
            - dla_model_filename    :   the filename to write/load CNN model. if None, use from settings
            - stats                 :   if True, calculate simple statistical measures on validation sample.
        """
        if h2_model_filename != None:
            self.h2_model_filename = h2_model_filename

        self.cnn = CNN_for_H2()

        self.d.open()
        if action == 'run':
            labels = np.stack((self.d.get('flag', dset='train'), self.d.get('pos', dset='train'), self.d.get('logN', dset='train')), axis=-1)
            history = self.cnn.model.fit(self.d.get('specs', dset='train'), {'ide': labels, 'red': labels, 'col': labels},
                                epochs=epochs, batch_size=700, shuffle=False)
            self.cnn.model.save(self.h2_model_filename)

        elif action == 'run_batch':
            batch = int(3e5)
            print(self.d.data['train/inds'][:].shape[0] // batch + 1)
            for i in range(int(self.d.data['train/inds'][:].shape[0] // batch + 1)):
                labels = np.stack((self.d.get('flag', dset='train', batch=batch, ind_batch=i),
                                   self.d.get('pos', dset='train', batch=batch, ind_batch=i),
                                   self.d.get('logN', dset='train', batch=batch, ind_batch=i)), axis=-1)
                print(i, np.unique(self.d.get('inds', dset='train', batch=batch, ind_batch=i)))
                history = self.cnn.model.fit(self.d.get('specs', dset='train', batch=batch, ind_batch=i),
                                    {'ide': labels, 'red': labels, 'col': labels}, epochs=epochs, batch_size=1000,
                                    shuffle=True)
        elif action == 'load':
            self.cnn.model.load_weights(self.h2_model_filename)

        if stats:
            self.cnn.h2_simple_stats()

    def h2_plot_spec(self, ind, preds=False):
        """
        Plot SDSS spectrum regarding DLA search. This includes info from data structure for DLA search
        parameters:
            - ind        :   number of the spectrum to plot
            - preds      :   if True, plot the predicition of the CNN model
        """
        fig, ax = self.d.plot_spec(ind, add_info=False)
        if preds:
            fig, ax = self.d.plot_preds(ind, fig=fig)

        return fig, ax