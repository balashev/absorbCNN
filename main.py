from copy import deepcopy
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, NullLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import torch

from .catalog import catalog
from .utils import Timer

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
        self.datatype = 'float32'
        self.dt = getattr(np, self.datatype)

    def init_constants(self):
        """
        initialize constants
        """
        self.lya, self.lyb, self.lyc = 1215.67, 1025.72, 915
        self.H2bands = {'L0-0': 1108.37963, 'L1-0': 1092.461585, 'L2-0': 1077.41625, 'L3-0': 1063.16816, 'L4-0': 1049.660515, 'L5-0': 1036.84467}


    # >>> run CNN model using keras
    def cnn_keras(self, action, epochs=0, model_filename=None, stats=False):
        """
        Create and run DLA CNN model of the data_structure
        parameters:
            - action                :   the action to do. Can be 'run' to run. 'run_batch' to run using the batches (memory saving regime), and 'load' to load already created model
            - epochs                :   number of epochs to run
            - model_filename        :   the filename to write/load CNN model. if None, use from settings
            - stats                 :   if True, calculate simple statistical measures on validation sample.
        """
        if model_filename != None:
            self.model_filename = model_filename

        self.d.open()
        if action == 'run':
            batch = int(1e4)
            for i in range(int(self.d.data['train/inds'][:].shape[0] // batch + 1)):
                labels = self.d.get_labels(dset="train", batch=batch, ind_batch=i)
                print(np.sum(np.isnan(labels)))
                labels_validation = self.d.get_labels(dset="valid", batch=batch, ind_batch=i)
                print(np.sum(np.isnan(labels_validation)))
                history = self.cnn.model.fit(self.d.get('specs', dset='train', batch=batch, ind_batch=i),
                                             {'ide': labels, 'red': labels, 'col': labels},
                                             epochs=epochs, batch_size=10000, shuffle=False,
                                             validation_data=(
                                             self.d.get('specs', dset='valid', batch=batch, ind_batch=i),
                                             {'ide': labels_validation, 'red': labels_validation,
                                              'col': labels_validation})
                                             )
            self.cnn.model.save(self.model_filename + '.keras')
            self.show_history(history)

        elif action == 'load':
            self.cnn.model.load_weights(self.model_filename + '.keras')

        if stats:
            self.cnn.simple_stats()

    # >>> run CNN model using pytorch
    def cnn_torch(self, action, epochs=0, model_filename=None, stats=False, learning_rate=1e-4, weight_decay=1e-4):
        """
        Create and run DLA CNN model of the data_structure
        parameters:
            - action                :   the action to do. Can be 'run' to run. 'run_batch' to run using the batches (memory saving regime), and 'load' to load already created model
            - epochs                :   number of epochs to run
            - model_filename    :   the filename to write/load CNN model. if None, use from settings
            - stats                 :   if True, calculate simple statistical measures on validation sample.
        """
        if model_filename != None:
            self.model_filename = model_filename

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.d.open()

        if action == 'run':
            batch = int(1e4)
            self.cnn.to(device)
            opt = torch.optim.Adam(self.cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)
            #opt = torch.optim.Adagrad(self.cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

            # initialize a dictionary to store training history
            H = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

            # measure how long training is going to take
            print("[INFO] training the network...")
            timer = Timer("CNN run")

            for epoch in range(epochs):
                self.cnn.train()  # Set model to training mode
                # initialize the total training and validation loss
                totalTrainLoss, totalValLoss = 0, 0
                # initialize the number of correct predictions in the training
                # and validation step
                trainCorrect, valCorrect = 0, 0

                trainSteps = int(self.d.data['train/inds'][:].shape[0] // batch + 1)
                print("Number of batches:", trainSteps)
                for i in range(trainSteps):
                    data, target = torch.from_numpy(self.d.get('specs', dset='train', batch=batch, ind_batch=i)[:, np.newaxis, :]).to(device), torch.from_numpy(self.d.get_labels(dset="train", batch=batch, ind_batch=i)).to(device)

                    pred = self.cnn(data)
                    # print("pred:", pred.shape, target.shape)
                    loss = self.cnn.loss(pred, target)
                    # zero out the gradients, perform the backpropagation step,
                    # and update the weights
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    # add the loss to the total training loss so far and
                    # calculate the number of correct predictions
                    totalTrainLoss += loss

                    if i % max(int((trainSteps / 10)), 1) == 0:
                        print(f'Epoch: {epoch + 1}, Batch: {i}, Loss: {loss.item():.4f}')

                # switch off autograd for evaluation
                with torch.no_grad():
                    # set the model in evaluation mode
                    self.cnn.eval()
                    # loop over the validation set
                    valSteps = int(self.d.data['valid/inds'][:].shape[0] // batch + 1)
                    for i in range(valSteps):
                        data, target = torch.from_numpy(
                            self.d.get('specs', dset='valid', batch=batch, ind_batch=i)[:, np.newaxis, :]).to(device), torch.from_numpy(self.d.get_labels(dset="valid", batch=batch, ind_batch=i)).to(device)
                        # make the predictions and calculate the validation loss
                        pred = self.cnn(data)
                        totalValLoss += self.cnn.loss(pred, target)
                        # calculate the number of correct predictions

            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # calculate the training and validation accuracy
            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["val_loss"].append(avgValLoss.cpu().detach().numpy())
            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, epochs))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(avgValLoss, valCorrect))

            # finish measuring how long training took
            timer.time("[INFO] total time taken to train the model")

            # we can now evaluate the network on the test set
            print("[INFO] evaluating network...")
            # turn off autograd for testing evaluation

            # plot the training loss and accuracy
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(H["train_loss"], label="train_loss")
            plt.plot(H["val_loss"], label="val_loss")
            plt.title("Training Loss and Accuracy on Dataset")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            # serialize the model to disk
            torch.save(self.cnn, self.model_filename + '.pytorch')

        elif action == 'load':
            self.cnn = torch.load(self.model_filename + '.pytorch', weights_only=False)
            self.cnn.to(device)

        if stats:
            self.cnn.simple_stats()


    def show_history(self, history, folder=None):
        if folder == None:
            folder = self.QC_folder
        fig, ax = plt.subplots(1, 4, figsize=(25, 5))
        print(history.history.keys())
        # print(history.metrics)
        val = "val_loss" in history.history.keys()
        if val:
            ax[0].plot(history.epoch, history.history["val_loss"], label="Validation loss")
            ax[1].plot(history.epoch, history.history["val_ide_binary_true_positives"], label="Validation tp")
            ax[2].plot(history.epoch, history.history["val_ide_binary_false_positives"], label="Validation fp")
            ax[3].plot(history.epoch, history.history["val_ide_binary_false_negatives"], label="Validation fn")
        # ax[1].set_title('acc')
        ax[0].plot(history.epoch, history.history["loss"], label="Train loss")
        ax[1].plot(history.epoch, history.history["ide_binary_true_positives"], label="Train tp")
        ax[2].plot(history.epoch, history.history["ide_binary_false_positives"], label="Train fp")
        ax[3].plot(history.epoch, history.history["ide_binary_false_negatives"], label="Train fn")

        ax[0].set_title('loss')
        [ax[i].legend() for i in range(4)]
        fig.savefig(folder + "history.png")
        plt.close(fig)

    def simple_stats(self):
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

    def plot_spec(self, ind, preds=False, folder=None, title=None, z=None):
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
            fig.savefig(self.QC_folder + folder + f"/{ind}.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        return fig, ax

    def get_from_CNN(self, ind=None):
        """
        Get the DLAs from the SDSS spectrum set by <ind> using DLA CNN model
        parameters:
            - ind        :   number of the spectrum to use
        """
        if ind != None:
            self.d.get_abs_from_CNN(ind)

    def make_catalog(self, action, name='dla', dset='valid', threshold=None, lab=1215.67, batch=int(1e4)):
        """
        Make the catalog of DLAs from the SDSS spectrum set by <ind> using DLA CNN model. The catalog is saved in/read from <self.catalog_filename>_dla_<dset>.pickle
        parameters:
            - action     :   the action to do. Can be 'run' to run, or 'load' for already created catalog.
            - dset       :   dataset to use. Can be 'valid' ot 'train'
            - threshold  :   the value of the probability to count systema as a detection.
            - lab        :   wavelength of the transition
            - batch      :   number of regions to batch (to reduce the memory cost)
        """
        if threshold is None:
            threshold = self.threshold

        if action == 'run':
            abs = []
            # print('cat:', self.d.get_inds(dset=dset))
            # print('cat:', len(self.d.get_inds(dset=dset)))
            for ibatch in range(int(self.d.data[dset + '/inds'][:].shape[0] // batch + 1)):
                print(ibatch, int(self.d.data[dset + '/inds'][:].shape[0] // batch + 1))
                specs = self.d.get('specs', dset=dset, batch=batch, ind_batch=ibatch)[:]  # [:num]
                inds = self.d.get('inds', dset=dset, batch=batch, ind_batch=ibatch)[:]  # [:num]
                allreds = self.d.get('reds', dset=dset, batch=batch, ind_batch=ibatch)[:]  # [:num]
                # print(specs.shape)
                #print(len(np.unique(inds)))
                allpreds = self.cnn.predict(specs)
                #print(allpreds)
                allpreds[1] = allpreds[1] * 10
                allpreds[2] = allpreds[2] * (self.N_range[1] - self.N_range[0]) + self.N_range[0]
                # print(allpreds)

                for ind in np.unique(inds)[:]:
                    mind = (ind == inds)
                    preds, reds = [allpreds[0][mind], allpreds[1][mind], allpreds[2][mind]], allreds[mind]
                    a = self.d.get_abs(preds, reds, ind=ind, threshold=threshold, clean=True)
                    abs.extend(a)

            with open(self.catalog_filename.replace('.hdf5', f'_{name}_{dset}.pickle'), 'wb') as f:
                pickle.dump(abs, f)
        elif action == 'load':
            with open(self.catalog_filename.replace('.hdf5', f'_{name}_{dset}.pickle'), 'rb') as f:
                abs = pickle.load(f)

        return abs

    def make_catalog_loop(self, action, dset='valid', name='dla'):
        """
        Make the catalog of DLAs from the SDSS spectrum set by <ind> using DLA CNN model. The catalog is saved in/read from <self.catalog_filename>_dla_<dset>.pickle
        parameters:
            - action     :   the action to do. Can be 'run' to run, or 'load' for already created catalog.
            - dset       :   dataset to use. Can be 'valid' ot 'train'
        """
        if action == 'run':
            abs = []
            for ind in self.d.get_inds(dset=dset):
                res = self.d.get_abs_from_CNN(ind, plot=False)
                print(ind, res)
                if len(res) > 0:
                    # plot_preds(ind, d=d, model=model, sdss=sdss)
                    abs.extend(res)
            # print(dla)

            with open(self.catalog_filename.replace('.hdf5', f'_{name}_{dset}.pickle'), 'wb') as f:
                pickle.dump(abs, f)
        elif action == 'load':
            with open(self.catalog_filename.replace('.hdf5', f'_{name}_{dset}.pickle'), 'rb') as f:
                abs = pickle.load(f)

        return abs
        # print(len(dla))

    def catalog_stats(self, conf=2, sigma=0.01, cc_thres=0.6, dset="valid", folder=None,
                          kind=['ROC', 'number_count_total', 'number_count_cols', 'number_count_redshifts',
                                'number_count_snr', 'compare_cols', 'compare_redshifts', 'confusion_matrix']):
        """
        Calculate different statistics of the false positives/negatives, results of the CNN, etc
        by comparing validation and initial absorption catalogs.
        """
        abs = self.make_catalog('load', dset=dset)
        inds = [x[0] for x in abs]

        stat = {}
        for attr in ['corr', 'fp', 'fn']:
            stat[attr] = []

        self.cat.open()
        q = self.cat.cat['meta/qso'][...]
        print(q.shape, q.dtype)
        #print(q['SNR'])
        if 1:
            print(abs)
            #print(inds)
            for ind in self.d.get_inds(dset=dset)[:]:
                print(ind, self.cat.cat[f'meta/{ind}/abs'][...])


        if folder == None:
            folder = self.QC_folder

        n = np.linspace(self.N_range[0], self.N_range[1], 6)
        z = np.linspace(self.z_range[0], self.z_range[1], 6)
        snr = np.linspace(0, 10, 6)
        total_z, total_snr = np.zeros([len(z) - 1, len(n) - 1], dtype=int), np.zeros([len(snr) - 1, len(n) - 1], dtype=int)

        thres = self.threshold

        if 'ROC' in kind:
            print("[Making ROC analysis:]")
            print("threshold  tot   corr  fp  fn ")
            fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
            th = []
            for thres in np.linspace(self.threshold, 0.99, 15):
                st, tot = deepcopy(stat), 0
                for ind in self.d.get_inds(dset=dset)[:]:
                    name = f'meta/{ind}/abs'
                    if name in self.cat.cat:
                        real = self.cat.cat[name][...]
                        #print(name, real)
                        for r in real:
                            if 'cc' not in r.dtype.names or r['cc'] > cc_thres:
                                tot += 1
                                fn = True
                                for i in np.where(ind == inds)[0]:
                                    #print(abs[i])
                                    if (abs[i][1] > thres) and (r['z_abs'] > abs[i][2] - np.sqrt((conf * abs[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < abs[i][2] + np.sqrt((conf * abs[i][4]) ** 2 + sigma ** 2)):
                                        st['corr'].append(1)  #
                                        fn = False
                                if fn:
                                    st['fn'].append(1)

                        for i in np.where(ind == inds)[0]:
                            if (abs[i][1] > thres) and ~np.any([(r['z_abs'] > abs[i][2] - np.sqrt((conf * abs[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < abs[i][2] + np.sqrt((conf * abs[i][4]) ** 2 + sigma ** 2)) and ('cc' not in r.dtype.names or r['cc'] > cc_thres) for r in real]):
                                st['fp'].append(1)

                th.append([thres, (len(st['fp']) + len(st['fn'])) / tot])
                print(thres, tot, len(st['corr']) / tot, len(st['fp']) / tot, len(st['fn']) / tot, len(st['corr']) + len(st['fn']))
                ax[0].scatter(len(st['fp']) / tot, len(st['fn']) / tot, 10, color='k')
                ax[1].scatter(thres, (len(st['fp']) + len(st['fn'])) / tot, 10, color='k')
            th = np.asarray(th).transpose()
            thres = th[0, np.argmin(th[1, :])]
            print("[Obtained best threshold value from ROC:]", thres)
            fig.savefig(folder + "ROC.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        for ind in self.d.get_inds(dset=dset)[:]:
            # name = 'meta/{0:05d}_{1:05d}_{2:04d}/dla'.format(q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
            name = f'meta/{ind}/abs'

            if name in self.cat.cat:
                real = self.cat.cat[name][...]
                # print(ind, real, q[ind]['PLATE'], q[ind]['MJD'], q[ind]['FIBERID'])
                for r in real:
                    if 'cc' not in r.dtype.names or r['cc'] > cc_thres:
                        fn = True
                        for i in np.where(ind == inds)[0]:
                            # print(dla[i])
                            if (abs[i][1] > thres) and (
                                    r['z_abs'] > abs[i][2] - np.sqrt((2 * abs[i][3]) ** 2 + sigma ** 2)) and (
                                    r['z_abs'] < abs[i][2] + np.sqrt((2 * abs[i][4]) ** 2 + sigma ** 2)):
                                stat['corr'].append(abs[i] + [r[t] for t in r.dtype.names] + [q['SNR'][ind]])
                                fn = False
                        if fn:
                            stat['fn'].append([ind] + [r[t] for t in r.dtype.names] + [q['SNR'][ind]])
                            # plot_preds(ind, d=d, model=model, sdss=sdss)

                        if (r['z_abs'] > z[0]) * (r['z_abs'] < z[-1]) * (r['logN'] > n[0]) * (r['logN'] < n[-1]):
                            total_z[np.searchsorted(z, r['z_abs']) - 1, np.searchsorted(n, r['logN']) - 1] += 1
                        if (q['SNR'][ind] > snr[0]) * (q['SNR'][ind] < snr[-1]) * (r['logN'] > n[0]) * (r['logN'] < n[-1]):
                            total_snr[np.searchsorted(snr, q['SNR'][ind]) - 1, np.searchsorted(n, r['logN']) - 1] += 1

                for i in np.where(ind == inds)[0]:
                    # print(i, abs[i][2], [r['z_abs'] for r in real])
                    # print(np.any([(r['z_abs'] > abs[i][2] - np.sqrt((2 * abs[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < abs[i][2] + np.sqrt((2 * abs[i][4]) ** 2 + sigma ** 2)) for r in real]))
                    if (abs[i][1] > thres) and ~np.any([(r['z_abs'] > abs[i][2] - np.sqrt((2 * abs[i][3]) ** 2 + sigma ** 2)) and (r['z_abs'] < abs[i][2] + np.sqrt((2 * abs[i][4]) ** 2 + sigma ** 2)) and ('cc' not in r.dtype.names or r['cc'] > cc_thres) for r in real]):
                        stat['fp'].append(abs[i] + [q['SNR'][ind]])
                    # plot_preds(ind, d=d, model=model, sdss=sdss)

        #print(total_z)
        if 1:
            print(stat['corr'][0])
            print(stat['fp'][0])
            print(stat['fn'][0])

        if 'number_count_total' in kind:
            tot = np.sum(total_z.flatten())
            print("Total number count statistics:")
            print('Ntotal  Ncorr  Nfp  Nfn  f_corr f_fp  f_np')
            print(tot, len(stat['corr']), len(stat['fp']), len(stat['fn']), len(stat['corr']) / tot,
                  len(stat['fp']) / tot, len(stat['fn']) / tot)

        if 'number_count_cols' in kind:
            print("Number count statistics by column density:")
            print('logN_l  logN_r  Ntotal  Ncorr Nfp  Nfn  f_corr f_fp  f_fn')
            for i in range(len(n) - 1):
                cor = [s for s in stat['corr'] if (s[9] > n[i]) * (s[9] < n[i + 1])]
                pos = [s for s in stat['fp'] if (s[5] > n[i]) * (s[5] < n[i + 1])]
                neg = [s for s in stat['fn'] if (s[2] > n[i]) * (s[2] < n[i + 1])]
                tot = np.sum(total_z, axis=0)[i]
                if len(cor) > 0:
                    print(n[i], n[i + 1], tot, len(cor), len(pos), len(neg), len(cor) / tot, len(pos) / tot,
                          len(neg) / tot)

        if 'number_count_redshifts' in kind:
            print("Number count statistics by redshifts:")
            print('z_l  z_r  Ntotal Ncorr  Nfp  Nfn  f_corr, f_fp  f_np')
            for i in range(len(z) - 1):
                cor = [s for s in stat['corr'] if (s[8] > z[i]) * (s[8] < z[i + 1])]
                pos = [s for s in stat['fp'] if (s[2] > z[i]) * (s[2] < z[i + 1])]
                neg = [s for s in stat['fn'] if (s[1] > z[i]) * (s[1] < z[i + 1])]
                tot = np.sum(total_z, axis=1)[i]
                if len(cor) > 0:
                    print(z[i], z[i + 1], tot, len(cor), len(pos), len(neg), len(cor) / tot, len(pos) / tot,
                          len(neg) / tot)

        if 'number_count_snr' in kind:
            print("Number count statistics by Signal to Noise ratio:")
            print('SNR_l  SNR_r  Ntotal  Ncorr Nfp  Nfn  f_corr f_fp  f_fn')
            for i in range(len(snr) - 1):
                cor = [s for s in stat['corr'] if (s[-1] > snr[i]) * (s[-1] < snr[i + 1])]
                pos = [s for s in stat['fp'] if (s[-1] > snr[i]) * (s[-1] < snr[i + 1])]
                neg = [s for s in stat['fn'] if (s[-1] > snr[i]) * (s[-1] < snr[i + 1])]
                tot = np.sum(total_snr, axis=0)[i]
                if len(cor) > 0:
                    print(n[i], n[i + 1], tot, len(cor), len(pos), len(neg), len(cor) / tot, len(pos) / tot,
                          len(neg) / tot)

        if 'compare_cols' in kind:
            fig, ax = plt.subplots(2, 2, figsize=(10, 6))
            x, sigma = np.array(self.N_range), 0.3
            for i in range(4):
                row, col = i // 2, i % 2
                m = [(s[8] > z[i]) * (s[8] < z[i + 1]) for s in stat['corr']]
                ax[row, col].plot(np.asarray([s[9] for s in stat['corr']])[m],
                                  np.asarray([s[5] for s in stat['corr']])[m], '+')
                ax[row, col].plot(x, x, '--k')
                ax[row, col].fill_between(x, x - sigma, x + sigma, color='tab:red', alpha=0.1, ls=':')
                if row == 1:
                    ax[row, col].set_xlabel(r"True $N$("+f"{self.species})")
                if col == 0:
                    ax[row, col].set_ylabel(r"CNN $N$("+f"{self.species})")
                ax[row, col].text(0.05, 0.95, "redshifts: {0:3.1f}..{1:3.1f}".format(z[i], z[i + 1]),
                                  transform=ax[row, col].transAxes, ha='left', va='top')
                ax[row, col].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[row, col].xaxis.set_major_locator(MultipleLocator(0.5))
                ax[row, col].yaxis.set_minor_locator(AutoMinorLocator(5))
                ax[row, col].yaxis.set_major_locator(MultipleLocator(0.5))

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
                ax[col].text(0.05, 0.95, "z bin: {0:3.1f}..{1:3.1f}".format(z[i], z[i + 1]),
                             transform=ax[col].transAxes, ha='left', va='top')
                #print(z[i], z[i + 1], np.mean(data), np.std(data))
                ax[col].text(0.05, 0.90, "mean: {0:4.2f}".format(np.mean(data)), transform=ax[col].transAxes, ha='left',
                             va='top')
                ax[col].text(0.05, 0.85, "std: {0:4.2f}".format(np.std(data)), transform=ax[col].transAxes, ha='left',
                             va='top')
                ax[col].xaxis.set_minor_locator(AutoMinorLocator(5))
                ax[col].xaxis.set_major_locator(MultipleLocator(0.5))
                # ax[col].yaxis.set_minor_locator(AutoMinorLocator(5))
                # ax[col].yaxis.set_major_locator(MultipleLocator(0.5))
                ax[col].set_xlim([-0.9, 0.9])

            fig.savefig(folder + "compare_cols_hist.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        if 'compare_redshifts' in kind:
            num = len(n) - 1
            fig, ax = plt.subplots(ncols=num, figsize=(16, 5))
            N = np.linspace(self.N_range[0], self.N_range[1], num + 1)
            for i in range(num):
                col = i % num
                m = [(s[9] > n[i]) * (s[9] < n[i + 1]) for s in stat['corr']]
                data = np.asarray([s[8] for s in stat['corr']])[m] - np.asarray([s[2] for s in stat['corr']])[m]
                ax[col].hist(data, bins=30)
                ax[col].axvline(0.0, ls='--', color='k')
                ax[col].text(0.05, 0.95, "logN: {0:4.1f}..{1:4.1f}".format(N[i], N[i + 1]), transform=ax[col].transAxes,
                             ha='left', va='top')
                #print(N[i], N[i + 1], np.mean(data), np.std(data))
                ax[col].text(0.05, 0.90, "mean: {0:5.3f}".format(np.mean(data)), transform=ax[col].transAxes, ha='left',
                             va='top')
                ax[col].text(0.05, 0.85, "std: {0:5.3f}".format(np.std(data)), transform=ax[col].transAxes, ha='left',
                             va='top')
                if i == 2:
                    ax[col].set_xlabel(r"$\Delta z = z_{\rm est} - z_{\rm true}$")
                if col == 0:
                    ax[col].set_ylabel(r"number of spectra")
                ax[col].set_xlim([-0.025, 0.025])

            fig.savefig(folder + "compare_redshifts.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        if 'confusion_matrix' in kind:
            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), )
            N, Z = np.meshgrid(n[:-1] + np.diff(n) / 2, z[:-1] + np.diff(z) / 2)
            cor, pos, neg = np.zeros([len(z) - 1, len(n) - 1]), np.zeros([len(z) - 1, len(n) - 1]), np.zeros([len(z) - 1, len(n) - 1])
            for i in range(len(z) - 1):
                for k in range(len(n) - 1):
                    cor[i, k] = len([s for s in stat['corr'] if (s[8] > z[i]) * (s[8] < z[i + 1]) * (s[9] > n[k]) * (s[9] < n[k + 1])])
                    pos[i, k] = len([s for s in stat['fp']   if (s[2] > z[i]) * (s[2] < z[i + 1]) * (s[5] > n[k]) * (s[5] < n[k + 1])])
                    neg[i, k] = len([s for s in stat['fn']   if (s[1] > z[i]) * (s[1] < z[i + 1]) * (s[2] > n[k]) * (s[2] < n[k + 1])])
            print(cor, pos, neg)
            for i, m, title in zip(range(3), [cor, neg, pos], ['correct', 'false negatives', 'false positives']):
                row, col = i // 2, i % 2
                ax[row, col].pcolormesh(N, Z, m / total_z, vmin=0, vmax=1, cmap="YlOrRd_r" if i == 0 else "YlOrRd")
                for zi in range(len(z) - 1):
                    for ni in range(len(n) - 1):
                        ax[row, col].text(N[zi, ni], Z[zi, ni], "{0:4.2f}/{1:d}".format(m[zi, ni] / total_z[zi, ni], int(total_z[zi, ni])), fontsize=18, ha='center', va='center')
                ax[row, col].set_xlabel(r"$\log N$("+f"{self.species})", fontsize=18)
                ax[row, col].set_ylabel(r"$z$", fontsize=18)
                ax[row, col].set_title(title, fontsize=24)
                ax[row, col].xaxis.set_tick_params(labelsize=18)
                ax[row, col].yaxis.set_tick_params(labelsize=18)
                ax[row, col].set_xticks(n[:-1] + np.diff(n) / 2)
                ax[row, col].set_yticks(z[:-1] + np.diff(z) / 2)
            ax[1, 1].remove()
            fig.tight_layout()

            fig.savefig(folder + "confusion_matrix_z_N.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

            fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10), )
            N, SNR = np.meshgrid(n[:-1] + np.diff(n) / 2, snr[:-1] + np.diff(snr) / 2)
            cor, pos, neg = np.zeros([len(snr) - 1, len(n) - 1]), np.zeros([len(snr) - 1, len(n) - 1]), np.zeros([len(snr) - 1, len(n) - 1])
            for i in range(len(snr) - 1):
                for k in range(len(n) - 1):
                    cor[i, k] = len([s for s in stat['corr'] if (s[-1] > snr[i]) * (s[-1] < snr[i + 1]) * (s[9] > n[k]) * (s[9] < n[k + 1])])
                    pos[i, k] = len([s for s in stat['fp']   if (s[-1] > snr[i]) * (s[-1] < snr[i + 1]) * (s[5] > n[k]) * (s[5] < n[k + 1])])
                    neg[i, k] = len([s for s in stat['fn']   if (s[-1] > snr[i]) * (s[-1] < snr[i + 1]) * (s[2] > n[k]) * (s[2] < n[k + 1])])
            print(cor, pos, neg)
            for i, m, title in zip(range(3), [cor, neg, pos], ['correct', 'false negatives', 'false positives']):
                row, col = i // 2, i % 2
                ax[row, col].pcolormesh(N, SNR, m / total_snr, vmin=0, vmax=1, cmap="YlOrRd_r" if i == 0 else "YlOrRd")
                for snri in range(len(snr) - 1):
                    for ni in range(len(n) - 1):
                        ax[row, col].text(N[snri, ni], SNR[snri, ni], "{0:4.2f}/{1:d}".format(m[snri, ni] / total_snr[snri, ni], int(total_snr[snri, ni])), fontsize=18, ha='center', va='center')
                ax[row, col].set_xlabel(r"$\log N$("+f"{self.species})", fontsize=18)
                ax[row, col].set_ylabel(r"$SNR$", fontsize=18)
                ax[row, col].set_title(title, fontsize=24)
                ax[row, col].xaxis.set_tick_params(labelsize=18)
                ax[row, col].yaxis.set_tick_params(labelsize=18)
                ax[row, col].set_xticks(n[:-1] + np.diff(n) / 2)
                ax[row, col].set_yticks(snr[:-1] + np.diff(snr) / 2)
            ax[1, 1].remove()
            fig.tight_layout()

            fig.savefig(folder + "confusion_matrix_N_SNR.png", bbox_inches='tight', pad_inches=0.)
            plt.close(fig)

        return stat