import matplotlib.pyplot as plt
import numpy as np
import pickle

from ..main import CNN
from .DLA_data_class import dla_data
from .DLA_conv_model import CNN_for_DLA
class CNN_dla(CNN):
    def __init__(self, **kwargs):
        super(CNN_dla, self).__init__(**kwargs)

    def dla_add_cat(self, dla_cat_file=None):
        """
        add info about DLAs from DLA catalog located at <dla_cat_file>.
        """
        if dla_cat_file != None:
            self.dla_cat_file = dla_cat_file

        self.cat.add_dla_cat(self.dla_cat_file)

    def dla_prepare_data(self, action, num=0, window=None, valid=0.2, dropout=0.5, dropout_dla=0.1, start=0):
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

        self.d = dla_data(self, window=self.window, timing=False, filename=self.catalog_filename.replace('.hdf5', '_dla_data.hdf5'))

        if action == 'new':
            self.d.new()
            self.d.make(num=num, valid=valid, dropout=dropout, dropout_dla=dropout_dla, start=start)
            # d.make_sets(valid=0.1, dropout=0.3, dropout_dla=0.8, shuffle=True, batch=30000)

        elif action == 'load':
            self.d.open()

    def dla_cnn(self, action, epochs=0, dla_model_filename=None, stats=False):
        """
        Create and run DLA CNN model of the data_structure
        parameters:
            - action                :   the action to do. Can be 'run' to run. 'run_batch' to run using the batches (memory saving regime), and 'load' to load already created model
            - epochs                :   number of epochs to run
            - dla_model_filename    :   the filename to write/load CNN model. if None, use from settings
            - stats                 :   if True, calculate simple statistical measures on validation sample.
        """
        if dla_model_filename != None:
            self.dla_model_filename = dla_model_filename

        self.cnn = CNN_for_DLA()

        self.d.open()
        if action == 'run':
            labels = np.stack((self.d.get('flag', dset='train'), self.d.get('pos', dset='train'), self.d.get('logN', dset='train')), axis=-1)
            history = self.cnn.model.fit(self.d.get('specs', dset='train'), {'ide': labels, 'red': labels, 'col': labels},
                                epochs=epochs, batch_size=700, shuffle=False)
            self.cnn.model.save(self.dla_model_filename)

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
            self.cnn.model.load_weights(self.dla_model_filename)

        if stats:
            self.cnn.dla_simple_stats()
    def dla_cnn_old(self, action, epochs=0, dla_model_filename=None, stats=False):
        """
        Create and run DLA CNN model of the data_structure
        parameters:
            - action                :   the action to do. Can be 'run' to run. 'run_batch' to run using the batches (memory saving regime), and 'load' to load already created model
            - epochs                :   number of epochs to run
            - dla_model_filename    :   the filename to write/load CNN model. if None, use from settings
            - stats                 :   if True, calculate simple statistical measures on validation sample.
        """
        if dla_model_filename != None:
            self.dla_model_filename = dla_model_filename

        self.cnn = CNN_for_DLA()

        self.d.open()
        if action == 'run':
            labels = np.stack((self.d.get('dla_flag', dset='train'), self.d.get('dla_pos', dset='train'), self.d.get('dla_NHI', dset='train')), axis=-1)
            history = self.cnn.model.fit(self.d.get('specs', dset='train'), {'ide': labels, 'red': labels, 'col': labels},
                                epochs=epochs, batch_size=700, shuffle=False)
            self.cnn.model.save(self.dla_model_filename)

        elif action == 'run_batch':
            batch = int(3e5)
            print(self.d.data['train/inds'][:].shape[0] // batch + 1)
            for i in range(int(self.d.data['train/inds'][:].shape[0] // batch + 1)):
                labels = np.stack((self.d.get('dla_flag', dset='train', batch=batch, ind_batch=i),
                                   self.d.get('dla_pos', dset='train', batch=batch, ind_batch=i),
                                   self.d.get('dla_NHI', dset='train', batch=batch, ind_batch=i)), axis=-1)
                print(i, np.unique(self.d.get('inds', dset='train', batch=batch, ind_batch=i)))
                history = self.cnn.model.fit(self.d.get('specs', dset='train', batch=batch, ind_batch=i),
                                    {'ide': labels, 'red': labels, 'col': labels}, epochs=epochs, batch_size=1000,
                                    shuffle=True)
        elif action == 'load':
            self.cnn.model.load_weights(self.dla_model_filename)

        if stats:
            self.cnn.dla_simple_stats()

    def dla_simple_stats(self):
        """
        Calculate simple statistical measure of the validation sample of the CNN DLA model. This uses only raw spectral data structure, and does not consider DLA catalogs.
        """
        self.d.open()
        labels_valid = np.stack((self.d.get('flag', dset='valid')[:], self.d.get('pos', dset='valid')[:], self.d.get('logN', dset='valid')[:]), axis=-1)
        score = self.cnn.model.evaluate(self.d.get('specs', dset='valid'), {'ide': labels_valid, 'red': labels_valid, 'col': labels_valid})
        print(f'Test loss valid any: {score[0]}')
        m = self.d.get('flag', dset='valid') == 1
        score = self.cnn.model.evaluate(self.d.get('specs', dset='valid')[m], {'ide': labels_valid[m, :], 'red': labels_valid[m, :], 'col': labels_valid[m]})
        print(f'Test loss valid DLA: {score[0]}')

    def dla_plot_spec(self, ind, preds=False):
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
    
    def dla_plot_pure_compl(self, dset='valid', NA_limit=2):

        self.cat.open()
        q = self.cat.cat['meta/qso'][...]
        print(q.dtype)
        print(q['SNR_DLA'])
        nhi, S_to_N, tp_fn, tp_fp = [], [], [], []
        for ind in self.d.get_inds(dset=dset):

            name = 'meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            real = self.cat.cat[name][...]
            nhi.append(real['logN'][0])
            S_to_N.append(q[ind]['SNR_DLA'])

            self.d.open()
            spec = self.d.get_spec(inds=[ind])[0]
            inds = self.d.get('inds', dset=dset)[:]
            labels_valid = np.stack((self.d.get('flag', dset=dset)[inds==ind], self.d.get('pos', dset=dset)[inds==ind], self.d.get('logN', dset=dset)[inds==ind]), axis=-1)
            score = self.cnn.model.evaluate(spec, {'ide': labels_valid, 'red': labels_valid, 'col': labels_valid})
            if score[-3] > score[-1]:
                tp_fn.append(1)
            elif score[-3] < score[-1]:
                tp_fn.append(-1)
            else:
                tp_fn.append(0)
            if score[-3] > score[-2]:
                tp_fp.append(1)
            elif score[-3] < score[-2]:
                tp_fp.append(-1)
            else:
                tp_fp.append(0)

        print(S_to_N)

        compl = [[[0.000001 for k in range(2)] for j in range(8)] for i in range(7)]
        for N, stn, tf in zip(nhi, S_to_N, tp_fn):
            if stn < 8 and N > 19 and N < 22.5:
                if tf == 1:
                    compl[int((N - 19) * 2)][int(stn)][0] += 1
                    compl[int((N - 19) * 2)][int(stn)][1] += 1
                elif tf == -1:
                    compl[int((N - 19) * 2)][int(stn)][1] += 1
        pure = [[[0.000001 for k in range(2)] for j in range(8)] for i in range(7)]
        for N, stn, tf in zip(nhi, S_to_N, tp_fp):
            if stn < 8 and N > 19 and N < 22.5:
                if tf == 1:
                    pure[int((N - 19) * 2)][int(stn)][0] += 1
                    pure[int((N - 19) * 2)][int(stn)][1] += 1
                elif tf == -1:
                    pure[int((N - 19) * 2)][int(stn)][1] += 1     

        compl = [[np.round(c[0] / c[1], 2) if c[1] > NA_limit else 'N/A' for c in com] for com in compl]
        pure = [[np.round(c[0] / c[1], 2) if c[1] > NA_limit else 'N/A' for c in com] for com in pure]
            
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        t = ax.table(cellText=pure, colLabels=range(8), rowLabels=[19+0.5*i for i in range(7)], loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(12)
        fig.tight_layout()
        plt.show()

        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        t = ax.table(cellText=compl, colLabels=range(8), rowLabels=[19+0.5*i for i in range(7)], loc='center')
        t.auto_set_font_size(False)
        t.set_fontsize(12)
        fig.tight_layout()
        plt.show()


    def dla_get_from_CNN(self, ind=None):
        """
        Get the DLAs from the SDSS spectrum set by <ind> using DLA CNN model
        parameters:
            - ind        :   number of the spectrum to use
        """
        if ind != None:
            self.d.get_abs_from_CNN(ind)

    def dla_make_catalog(self, action, dset='valid'):
        """
        Make the catalog of DLAs from the SDSS spectrum set by <ind> using DLA CNN model. The catalog is saved in/read from <self.catalog_filename>_dla_<dset>.pickle
        parameters:
            - action     :   the action to do. Can be 'run' to run, or 'load' for already created catalog.
            - dset       :   dataset to use. Can be 'valid' ot 'train'
        """
        if action == 'run':
            dla = []
            for ind in self.d.get_inds(dset=dset):
                res = self.d.get_abs_from_CNN(ind, plot=False)
                print(ind, res)
                if len(res) > 0:
                    # plot_preds(ind, d=d, model=model, sdss=sdss)
                    dla.extend(res)
            # print(dla)
            with open(self.catalog_filename.replace('.hdf5', f'_dla_{dset}.pickle'), 'wb') as f:
                pickle.dump(dla, f)
        elif action == 'load':
            with open(self.catalog_filename.replace('.hdf5', f'_dla_{dset}.pickle'), 'rb') as f:
                dla = pickle.load(f)

        return dla
            # print(len(dla))

    def dla_catalog_stats(self, kind=['number_count_total', 'number_count_cols', 'number_count_redshifts', 'compare_cols']):
        """
        Calculate different statistics of the false positives/negatives, results of the CNN, etc
        by comparing validation and initial DLA catalogs.
        """
        dla = self.dla_make_catalog('load', dset='valid')
        inds = [x[0] for x in dla]
        stat = {}
        for attr in ['corr', 'fp', 'fn']:
            stat[attr] = []

        self.cat.open()
        q = self.cat.cat['meta/qso'][...]

        for ind in self.d.get_inds(dset='valid')[:]:
            #print(ind, q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            name = 'meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            if name in self.cat.cat:
                real = self.cat.cat[name][...]
                #print(ind, real)
                for r in real:
                    fn = True
                    for i in np.where(ind == inds)[0]:
                        # print(dla[i])
                        if (r['z_abs'] > dla[i][2] - np.sqrt((2 * dla[i][3]) ** 2 + 0.02 ** 2)) and (r['z_abs'] < dla[i][2] + np.sqrt((2 * dla[i][4]) ** 2 + 0.02 ** 2)):
                            stat['corr'].append(dla[i] + [real['z_abs'][0], real['NHI'][0]])
                            fn = False
                        else:
                            stat['fp'].append(dla[i] + [real['z_abs'][0], real['NHI'][0]])
                            # plot_preds(ind, d=d, model=model, sdss=sdss)
                    if fn:
                        stat['fn'].append([ind, real['z_abs'][0], real['NHI'][0]])
                        # plot_preds(ind, d=d, model=model, sdss=sdss)
        if 0:
            print(stat['corr'][0])
            print(stat['fp'][0])
            print(stat['fn'][0])

        if 'number_count_total' in kind:
            print("Total number count statistics:")
            print('Ntotal  Nfp  Nfn  f_fp  f_np')
            print(len(stat['corr']), len(stat['fp']), len(stat['fn']), len(stat['fp']) / len(stat['corr']), len(stat['fn']) / len(stat['corr']))

        if 'number_count_cols' in kind:
            print("Number count statistics by column density:")
            n = np.linspace(19.5, 22.5, 7)
            print('N_l  N_r  Ntotal  Nfp  Nfn  f_fp  f_np')
            for i in range(len(n) - 1):
                cor = [s for s in stat['corr'] if (s[9] > n[i]) * (s[9] < n[i + 1])]
                pos = [s for s in stat['fp'] if (s[5] > n[i]) * (s[5] < n[i + 1])]
                neg = [s for s in stat['fn'] if (s[2] > n[i]) * (s[2] < n[i + 1])]
                if len(cor) > 0:
                    print(n[i], n[i + 1], len(cor), len(pos), len(neg), len(pos) / len(cor), len(neg) / len(cor))

        if 'number_count_redshifts' in kind:
            print("Number count statistics by redshifts:")
            z = np.linspace(2., 5, 7)
            print('z_l  z_r  Ntotal  Nfp  Nfn  f_fp  f_np')
            for i in range(len(n) - 1):
                cor = [s for s in stat['corr'] if (s[8] > z[i]) * (s[8] < z[i + 1])]
                pos = [s for s in stat['fp'] if (s[2] > z[i]) * (s[2] < z[i + 1])]
                neg = [s for s in stat['fn'] if (s[1] > z[i]) * (s[1] < z[i + 1])]
                if len(cor) > 0:
                    print(z[i], z[i + 1], len(cor), len(pos), len(neg), len(pos) / len(cor), len(neg) / len(cor))

        if 'compare_cols' in kind:
            fig, ax = plt.subplots()
            ax.plot([s[9] for s in stat['corr']], [s[5] for s in stat['corr']], '+')
            plt.show()

        return stat
