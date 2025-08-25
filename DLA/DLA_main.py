from keras.utils import PyDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
from multiprocessing import Pool
import numpy as np
import os
import pickle
from scipy.signal import argrelextrema

from ..main import CNN
from ..stats import distr1d
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

    def dla_prepare_data(self, action, num=None, window=None, valid=0.2, dropout=0.7, dropout_dla=0.2, start=0):
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
        if num == None:
            num = self.num

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
            labels = self.d.get_labels(dset="train")
            labels_validation = self.d.get_labels(dset="valid")
            history = self.cnn.model.fit(self.d.get('specs', dset='train'), {'ide': labels, 'red': labels, 'col': labels},
                                         epochs=epochs, batch_size=3000, shuffle=False,
                                         validation_data=(self.d.get('specs', dset='valid'), {'ide': labels_validation, 'red': labels_validation, 'col': labels_validation})
                                         )
            self.cnn.model.save(self.dla_model_filename)
            self.show_history(history)

        elif action == 'run_batch':
            batch = int(3e6)
            for i in range(int(self.d.data['train/inds'][:].shape[0] // batch + 1)):
                labels = self.d.get_labels(dset="train", batch=batch, ind_batch=i)
                labels_validation = self.d.get_labels(dset="valid", batch=batch, ind_batch=i)
                history = self.cnn.model.fit(self.d.get('specs', dset='train', batch=batch, ind_batch=i), {'ide': labels, 'red': labels, 'col': labels},
                                             epochs=epochs, batch_size=10000, shuffle=False,
                                             validation_data=(self.d.get('specs', dset='valid', batch=batch, ind_batch=i), {'ide': labels_validation, 'red': labels_validation, 'col': labels_validation})
                                             )
            self.cnn.model.save(self.dla_model_filename)
            self.show_history(history)

        elif action == 'load':
            self.cnn.model.load_weights(self.dla_model_filename)

        if stats:
            self.cnn.dla_simple_stats()

    def show_history(self, history, folder=None):
        if folder == None:
            folder = self.QC_folder
        fig, ax = plt.subplots(1, 4, figsize=(25, 5))
        print(history.history.keys())
        #print(history.metrics)
        val = "val_loss" in history.history.keys()
        if val:
            ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
            ax[1].plot(history.epoch, history.history["val_ide_binary_true_positives"], label="Validation tp")
            ax[2].plot(history.epoch, history.history["val_ide_binary_false_positives"], label="Validation fp")
            ax[3].plot(history.epoch, history.history["val_ide_binary_false_negatives"], label="Validation fn")
        #ax[1].set_title('acc')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[1].plot(history.epoch, history.history["ide_binary_true_positives"], label="Train tp")
        ax[2].plot(history.epoch, history.history["ide_binary_false_positives"], label="Train fp")
        ax[3].plot(history.epoch, history.history["ide_binary_false_negatives"], label="Train fn")

        ax[0].set_title('loss')
        [ax[i].legend() for i in range(4)]
        fig.savefig(folder + "history.png")
        plt.close(fig)

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

    def dla_plot_spec(self, ind, preds=False, folder=None, title=None, z=None):
        """
        Plot SDSS spectrum regarding DLA search. This includes info from data structure for DLA search
        parameters:
            - ind        :   number of the spectrum to plot
            - preds      :   if True, plot the predicition of the CNN model
        """
        fig, ax = self.d.plot_spec(ind, z=z)

        if folder != None:
            if not os.path.exists(self.QC_folder + folder):
                os.mkdir(self.QC_folder + folder)

        if preds:
            fig, ax = self.d.plot_preds(ind, fig=fig)

        if title is not None:
            ax[0].set_title(title)

        if folder != None:
            fig.savefig(self.QC_folder + folder + f"/{ind}.png")
            plt.close(fig)

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

    def dla_make_catalog(self, action, dset='valid', plot=False, threshold=0.5, lab=1215.67, batch=int(1e6)):
        """
        Make the catalog of DLAs from the SDSS spectrum set by <ind> using DLA CNN model. The catalog is saved in/read from <self.catalog_filename>_dla_<dset>.pickle
        parameters:
            - action     :   the action to do. Can be 'run' to run, or 'load' for already created catalog.
            - dset       :   dataset to use. Can be 'valid' ot 'train'
            - plot       :   plot intermediate results
            - threshold  :   the value of the probability to count systema as a detection.
            - lab        :   wavelength of the transition
            - batch      :   number of regions to batch (to reduce the memory cost)
        """
        if action == 'run':
            dla = []
            #print('cat:', self.d.get_inds(dset=dset))
            #print('cat:', len(self.d.get_inds(dset=dset)))
            for ibatch in range(int(self.d.data[dset + '/inds'][:].shape[0] // batch + 1)):
                print(ibatch, int(self.d.data[dset + '/inds'][:].shape[0] // batch + 1))
                specs = self.d.get('specs', dset=dset, batch=batch, ind_batch=ibatch)[:] #[:num]
                inds = self.d.get('inds', dset=dset, batch=batch, ind_batch=ibatch)[:] #[:num]
                allreds = self.d.get('reds', dset=dset, batch=batch, ind_batch=ibatch)[:] #[:num]
                print(specs.shape)
                print(len(np.unique(inds)))
                allpreds = self.cnn.model.predict(specs)
                allpreds[1] = allpreds[1] * 10
                allpreds[2] = allpreds[2] * (self.N_range[1] - self.N_range[0]) + self.N_range[0]
                #print(allpreds)

                for ind in np.unique(inds)[:]:
                    mind = (ind == inds)
                    preds, reds = [allpreds[0][mind], allpreds[1][mind], allpreds[2][mind]], allreds[mind]
                    abs = self.d.get_abs(preds, reds, ind=ind, threshold=threshold)
                    #print(ind, abs)
                    dla.extend(abs)

            with open(self.catalog_filename.replace('.hdf5', f'_dla_{dset}.pickle'), 'wb') as f:
                pickle.dump(dla, f)
        elif action == 'load':
            with open(self.catalog_filename.replace('.hdf5', f'_dla_{dset}.pickle'), 'rb') as f:
                dla = pickle.load(f)

        return dla

    def dla_make_catalog_loop(self, action, dset='valid'):
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

    def dla_catalog_stats(self, kind=['number_count_total', 'number_count_cols', 'number_count_redshifts', 'compare_cols', 'compare_redshifts', 'confusion_matrix'], sigma=0.02, dset="valid", folder=None):
        """
        Calculate different statistics of the false positives/negatives, results of the CNN, etc
        by comparing validation and initial DLA catalogs.
        """
        dla = self.dla_make_catalog('load', dset=dset)
        #print("dlas:", dla)
        inds = [x[0] for x in dla]
        stat = {}
        for attr in ['corr', 'fp', 'fn']:
            stat[attr] = []

        self.cat.open()
        q = self.cat.cat['meta/qso'][...]

        n = np.linspace(self.N_range[0], self.N_range[1], 6)
        z = np.linspace(self.z_range[0], self.z_range[1], 6)
        total = np.zeros([len(z) - 1, len(n) - 1], dtype=int)

        for ind in self.d.get_inds(dset=dset)[:]:
            print(ind, q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            #name = 'meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            name = f'meta/{ind}/DLA'

            if name in self.cat.cat:
                real = self.cat.cat[name][...]
                print(ind, real)
                for r in real:
                    fn = True
                    for i in np.where(ind == inds)[0]:
                        #print(dla[i])
                        if (r['z_abs'] > dla[i][2] - np.sqrt((2 * dla[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < dla[i][2] + np.sqrt((2 * dla[i][4]) ** 2 + sigma ** 2)):
                            stat['corr'].append(dla[i] + [r['z_abs'], r['NHI']])
                            fn = False
                    if fn:
                        stat['fn'].append([ind, r['z_abs'], r['NHI']])
                        # plot_preds(ind, d=d, model=model, sdss=sdss)

                    if (r['z_abs'] > z[0]) * (r['z_abs'] < z[-1]) * (r['NHI'] > n[0]) * (r['NHI'] < n[-1]):
                        total[np.searchsorted(z, r['z_abs'])-1, np.searchsorted(n, r['NHI'])-1] += 1

                for i in np.where(ind == inds)[0]:
                    print(i, dla[i][2], [r['z_abs'] for r in real])
                    print(np.any([(r['z_abs'] > dla[i][2] - np.sqrt((2 * dla[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < dla[i][2] + np.sqrt((2 * dla[i][4]) ** 2 + sigma ** 2)) for r in real]))
                    if ~np.any([(r['z_abs'] > dla[i][2] - np.sqrt((2 * dla[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < dla[i][2] + np.sqrt((2 * dla[i][4]) ** 2 + sigma ** 2)) for r in real]):
                        stat['fp'].append(dla[i])
                    # plot_preds(ind, d=d, model=model, sdss=sdss)

        print(total)
        if 0:
            print(stat['corr'][0])
            print(stat['fp'][0])
            print(stat['fn'][0])

        if 'number_count_total' in kind:
            tot = np.sum(total.flatten())
            print("Total number count statistics:")
            print('Ntotal  Ncorr  Nfp  Nfn  f_corr f_fp  f_np')
            print(tot, len(stat['corr']), len(stat['fp']), len(stat['fn']), len(stat['corr']) / tot, len(stat['fp']) / tot, len(stat['fn']) / tot)

        if 'number_count_cols' in kind:
            print("Number count statistics by column density:")
            print('logN_l  logN_r  Ntotal  Ncorr Nfp  Nfn  f_corr f_fp  f_fn')
            for i in range(len(n) - 1):
                cor = [s for s in stat['corr'] if (s[9] > n[i]) * (s[9] < n[i + 1])]
                pos = [s for s in stat['fp'] if (s[5] > n[i]) * (s[5] < n[i + 1])]
                neg = [s for s in stat['fn'] if (s[2] > n[i]) * (s[2] < n[i + 1])]
                tot = np.sum(total, axis=0)[i]
                if len(cor) > 0:
                    print(n[i], n[i + 1], tot, len(cor), len(pos), len(neg), len(cor) / tot, len(pos) / tot, len(neg) / tot)

        if 'number_count_redshifts' in kind:
            print("Number count statistics by redshifts:")
            print('z_l  z_r  Ntotal Ncorr  Nfp  Nfn  f_corr, f_fp  f_np')
            for i in range(len(z) - 1):
                cor = [s for s in stat['corr'] if (s[8] > z[i]) * (s[8] < z[i + 1])]
                pos = [s for s in stat['fp'] if (s[2] > z[i]) * (s[2] < z[i + 1])]
                neg = [s for s in stat['fn'] if (s[1] > z[i]) * (s[1] < z[i + 1])]
                tot = np.sum(total, axis=1)[i]
                if len(cor) > 0:
                    print(z[i], z[i + 1], tot, len(cor), len(pos), len(neg), len(cor) / tot, len(pos) / tot, len(neg) / tot)

        if 'compare_cols' in kind:
            fig, ax = plt.subplots(2, 2, figsize=(10, 6))
            x, sigma = np.array(self.N_range), 0.3
            for i in range(4):
                row, col = i // 2, i % 2
                m = [(s[8] > z[i]) * (s[8] < z[i + 1]) for s in stat['corr']]
                ax[row, col].plot(np.asarray([s[9] for s in stat['corr']])[m], np.asarray([s[5] for s in stat['corr']])[m], '+')
                ax[row, col].plot(x, x, '--k')
                ax[row, col].fill_between(x, x-sigma, x+sigma, color='tab:red', alpha=0.1, ls=':')
                if row == 1:
                    ax[row, col].set_xlabel(r"True $N$(HI)")
                if col == 0:
                    ax[row, col].set_ylabel(r"CNN $N$(HI)")
                ax[row, col].text(0.05, 0.95, "redshifts: {0:3.1f}..{1:3.1f}".format(z[i], z[i + 1]), transform=ax[row, col].transAxes, ha='left', va='top')
                ax[row, col].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[row, col].xaxis.set_major_locator(MultipleLocator(0.5))
                ax[row, col].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax[row, col].yaxis.set_major_locator(MultipleLocator(0.5))

            if folder == None:
                folder = self.QC_folder
            fig.savefig(folder + "compare_cols.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

            num = len(z) - 1
            fig, ax = plt.subplots(ncols=num, figsize=(16, 5))
            for i in range(num):
                col = i % num
                m = [(s[8] > z[i]) * (s[8] < z[i + 1]) for s in stat['corr']]
                data = np.asarray([s[9] for s in stat['corr']])[m] - np.asarray([s[5] for s in stat['corr']])[m]
                ax[col].hist(data, bins=30)
                ax[col].axvline(0.0, ls='--', color='k')
                if i == 2:
                    ax[col].set_xlabel(r"$\Delta N = \log N_{\rm true} - \log N_{\rm est}$")
                if col == 0:
                    ax[col].set_ylabel(r"Number of systems")
                ax[col].text(0.05, 0.95, "z bin: {0:3.1f}..{1:3.1f}".format(z[i], z[i + 1]), transform=ax[col].transAxes, ha='left', va='top')
                print(z[i], z[i + 1], np.mean(data), np.std(data))
                ax[col].text(0.05, 0.90, "mean: {0:4.2f}".format(np.mean(data)), transform=ax[col].transAxes, ha='left', va='top')
                ax[col].text(0.05, 0.85, "std: {0:4.2f}".format(np.std(data)), transform=ax[col].transAxes, ha='left', va='top')
                ax[col].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[col].xaxis.set_major_locator(MultipleLocator(0.5))
                #ax[col].yaxis.set_minor_locator(AutoMinorLocator(5))
                #ax[col].yaxis.set_major_locator(MultipleLocator(0.5))
                ax[col].set_xlim([-0.9, 0.9])
            if folder == None:
                folder = self.QC_folder
            fig.savefig(folder + "compare_cols_hist.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        if 'compare_redshifts' in kind:
            num = len(n) - 1
            fig, ax = plt.subplots(ncols=num, figsize=(16, 5))
            N = np.linspace(self.N_range[0], self.N_range[1], num+1)
            for i in range(num):
                col = i % num
                m = [(s[9] > n[i]) * (s[9] < n[i + 1]) for s in stat['corr']]
                data = np.asarray([s[8] for s in stat['corr']])[m] - np.asarray([s[2] for s in stat['corr']])[m]
                ax[col].hist(data, bins=30)
                ax[col].axvline(0.0, ls='--', color='k')
                ax[col].text(0.05, 0.95, "logN: {0:4.1f}..{1:4.1f}".format(N[i], N[i+1]), transform=ax[col].transAxes, ha='left', va='top')
                print(N[i], N[i + 1], np.mean(data), np.std(data))
                ax[col].text(0.05, 0.90, "mean: {0:5.3f}".format(np.mean(data)), transform=ax[col].transAxes, ha='left', va='top')
                ax[col].text(0.05, 0.85, "std: {0:5.3f}".format(np.std(data)), transform=ax[col].transAxes, ha='left', va='top')
                if i == 2:
                    ax[col].set_xlabel(r"$\Delta z = z_{\rm est} - z_{\rm true}$")
                if col == 0:
                    ax[col].set_ylabel(r"number of spectra")
                ax[col].set_xlim([-0.025, 0.025])
            if folder == None:
                folder = self.QC_folder
            fig.savefig(folder + "compare_redshifts.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        if 'confusion_matrix' in kind:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), )
            N, Z = np.meshgrid(n[:-1] + np.diff(n)/2, z[:-1] + np.diff(z)/2)
            cor, pos, neg = np.zeros([len(z) - 1, len(n) - 1]), np.zeros([len(z) - 1, len(n) - 1]), np.zeros([len(z) - 1, len(n) - 1])
            for i in range(len(z) - 1):
                for k in range(len(n) - 1):
                    cor[i, k] = len([s for s in stat['corr'] if (s[8] > z[i]) * (s[8] < z[i + 1]) * (s[9] > n[k]) * (s[9] < n[k + 1])])
                    pos[i, k] = len([s for s in stat['fp'] if (s[2] > z[i]) * (s[2] < z[i + 1]) * (s[5] > n[k]) * (s[5] < n[k + 1])])
                    neg[i, k] = len([s for s in stat['fn'] if (s[1] > z[i]) * (s[1] < z[i + 1]) * (s[2] > n[k]) * (s[2] < n[k + 1])])
            print(cor, pos, neg)
            for i, m, title in zip(range(3), [cor, neg, pos], ['correct', 'false negatives', 'false positives']):
                row, col = i // 2, i % 2
                ax[row, col].pcolormesh(N, Z, m / total, vmin=0, vmax=1, cmap="YlOrRd_r" if i == 0 else "YlOrRd")
                for zi in range(len(z) - 1):
                    for ni in range(len(n) - 1):
                        ax[row, col].text(N[zi, ni], Z[zi, ni], "{0:4.2f}/{1:d}".format(m[zi, ni]/total[zi, ni], int(total[zi, ni])), fontsize=18, ha='center', va='center')
                ax[row, col].set_xlabel(r"$\log N(\rm HI)$", fontsize=18)
                ax[row, col].set_ylabel(r"$z$", fontsize=18)
                ax[row, col].set_title(title, fontsize=24)
                ax[row, col].xaxis.set_tick_params(labelsize=18)
                ax[row, col].yaxis.set_tick_params(labelsize=18)
                ax[row, col].set_xticks(n[:-1] + np.diff(n)/2)
                ax[row, col].set_yticks(z[:-1] + np.diff(z)/2)
            ax[1, 1].remove()
            fig.tight_layout()
            if folder == None:
                folder = self.QC_folder
            fig.savefig(folder + "confusion_matrix.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        return stat