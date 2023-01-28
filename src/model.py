import numpy as np
import torch
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import copy
from src.group_model import EarlyStopping

class sparseModel(torch.nn.Module):
    def __init__(self, p=500, N=2, only_pos=True):
        super().__init__()
        self.p = p 
        self.N = N
        self.only_pos = only_pos
        
        self.v = torch.nn.Linear(p, 1, bias=False)
        if not self.only_pos:
            self.v2 = torch.nn.Linear(p, 1, bias=False)
            
    def forward(self, x):
        if self.only_pos:
            weights = (self.v.weight**self.N).t()
        else:
            weights = (self.v.weight**self.N - self.v2.weight**self.N).t()
        ypred = torch.matmul(x, weights)
        return ypred.squeeze(-1)
    
    def get_params(self):
        if self.only_pos:
            params = self.v.weight**self.N
        else:
            params = (self.v.weight**self.N - self.v2.weight**self.N)
        params = params.detach()
        return params

class sparseTrainer:
    def __init__(self, model, sim, lr=0.01):
        self.model = model
        self.sim = sim
        self.lr = 0.01
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.loss_criterion = torch.nn.MSELoss()
        self.monitor = []
        self.loss = []
        self.early_stopping = EarlyStopping()
        self.flag = True
        self.early_stopped_epoch = None
        self.early_stopped_model = None
        self.params_est_err = []
        
    def _one_epoch(self):
        y_pred = self.model(self.sim.X)
        loss = self.loss_criterion(y_pred, self.sim.y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        est_err = self._monitor()
        self.loss.append(loss.item())
        return loss.item(), est_err
    
    def _monitor(self):
        k = self.sim.k
        params = self.model.get_params().squeeze(0)
        est_err = self.loss_criterion(params, self.sim.w_star).item()
        
        self.params_est_err.append(est_err)
        params = params.numpy().tolist()
                
        self.monitor.append([*params[:k],max([abs(x) for x in params[k:]])])
        return est_err
    
    def train(self, epochs=500):
        for epoch in range(epochs):
            loss, est_err = self._one_epoch()
            if self.flag:
                self.early_stopping(est_err)
                if self.early_stopping.early_stop:
                    self.early_stopped_epoch = len(self.params_est_err)
                    self.early_stopped_model = copy.deepcopy(self.model)
                    self.flag = False
                    
            print(f'{epoch}/{epochs}, loss: {loss:.4f}, est error: {est_err:.4f}')  
            
    def transpose_monitor(self):
        monitor_t = {}
        monitor_t = [[one[i] for one in self.monitor] for i in range(len(self.monitor[0]))]
        return monitor_t