import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from ..conv_model import ideLoss, posLoss, colLoss

class CNN_for_DLA_torch(nn.Module):
    def __init__(self):
        super(CNN_for_DLA_torch, self).__init__()
        self.init_layers()
        self.init_loss()

    def init_layers(self, dropout=0.1, regul=0.005):
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=100, kernel_size=32, stride=3, padding='valid', padding_mode='circular')
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=dropout)
        self.pool1 = nn.MaxPool1d(kernel_size=7, stride=2)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=96, kernel_size=16, stride=1, padding='valid', padding_mode='circular')
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=dropout)
        self.pool2 = nn.MaxPool1d(kernel_size=6, stride=1)
        self.conv3 = nn.Conv1d(in_channels=96, out_channels=96, kernel_size=16, stride=1, padding='valid', padding_mode='circular')
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=dropout)
        self.pool3 = nn.MaxPool1d(kernel_size=6, stride=1)
        #self.flatten = nn.Flatten()
        self.fc = nn.Linear(96 * 19, 96)
        self.drop4 = nn.Dropout(p=dropout)
        self.fc_ide = nn.Linear(96, 1)  # Output layer for identification
        self.sigmoid = nn.Sigmoid()
        self.fc_pos = nn.Linear(96, 1)  # Output layer for position
        self.fc_col = nn.Linear(96, 1)  # Output layer for column densities

    def init_loss(self, ide_weight=1.0, pos_weight=1.0, col_weight=1.0):
        self.ide_loss = ideLoss(ide_weight)
        #self.ide_loss = nn.BCEWithLogitsLoss(weight=torch.tensor(ide_weight))
        self.pos_loss = posLoss(torch.tensor(pos_weight))
        self.col_loss = colLoss(torch.tensor(col_weight))

    def forward(self, x, debug=False):
        if debug:
            print('1:', x.shape)
            print('1:', self.drop1(self.relu1(self.conv1(x))).shape)
        x = self.pool1(self.drop1(self.relu1(self.conv1(x))))
        if debug:
            print('2:', x.shape)
            print('2:', self.drop2(self.relu2(self.conv2(x))).shape)
        x = self.pool2(self.drop2(self.relu2(self.conv2(x))))
        if debug:
            print('3:', x.shape)
            print('3:', self.drop3(self.relu3(self.conv3(x))).shape)
        x = self.pool3(self.drop3(self.relu3(self.conv3(x))))
        if debug:
            print('4:', x.shape)
        x = self.drop4(self.fc(torch.flatten(x, start_dim=1, end_dim=2)))
        if debug:
            print('5:', x.shape)
        return torch.stack([self.sigmoid(self.fc_ide(x)), self.fc_pos(x), self.fc_col(x)])

    def loss(self, outputs, targets):
        return self.ide_loss(outputs[0], targets) + self.pos_loss(outputs[1], targets) + self.col_loss(outputs[2], targets)

    def predict(self, specs):
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return self(torch.from_numpy(specs[:,np.newaxis,:]).to(device)).cpu().detach().numpy()