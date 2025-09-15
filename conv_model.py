import torch
import torch.nn as nn

class ideLoss(nn.Module):
        def __init__(self, weight=1.0):
            super(ideLoss, self).__init__()
            self.weight = weight
            self.zero = torch.tensor(0.0)
            self.eps = torch.tensor(0.00001)
            self.one = torch.tensor(1.0)

        def forward(self, y_pred, y_true):
            y_true_ide = torch.reshape(y_true[:, 0], y_pred.shape)
            return torch.nanmean(torch.multiply(torch.subtract(self.zero, self.weight), torch.add(torch.multiply(y_true_ide, torch.log(torch.add(y_pred, self.eps))), torch.multiply(torch.subtract(self.one, y_true_ide), torch.log(torch.subtract(self.one + self.eps, y_pred))))))

class posLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(posLoss, self).__init__()
        self.weight = weight
        self.eps = torch.tensor(0.000001)

    def forward(self, y_pred, y_true):
        y_true_ide = torch.reshape(y_true[:, 0], y_pred.shape)
        return torch.nanmean(torch.multiply(self.weight, torch.multiply(torch.divide(y_true_ide, torch.add(y_true_ide, self.eps)), torch.square(y_pred - torch.reshape(y_true[:, 1], y_pred.shape)))))

class colLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(colLoss, self).__init__()
        self.weight = weight
        self.eps = torch.tensor(0.000001)

    def forward(self, y_pred, y_true):
        y_true_ide = torch.reshape(y_true[:, 0], y_pred.shape)
        return torch.nanmean(torch.multiply(self.weight, torch.multiply(torch.divide(y_true_ide, torch.add(y_true_ide, self.eps)), torch.square(y_pred - torch.reshape(y_true[:, 2], y_pred.shape)))))