import torch
import numpy as np
import copy
import matplotlib.pyplot as plt

class GaussianSimulation:
    def __init__(self, m=200, p=500, seed=42, support = np.repeat([1,2,3,4],4), # np.tile
                 std = 0.5):
        self.m = m
        self.p = p
        self.k = support.shape[0]
        self.support = support
        self.seed = seed
        self.std = std
        
        np.random.seed(seed)
        X = np.random.normal(size=(m,p)) # np.random.binomial(1, 0.5, (m, p))*2 - 1
        w_star = np.hstack((support, np.zeros(self.p-self.k)))

        signal = np.matmul(X, w_star)
        noise = np.random.normal(scale=std, size=m)
        y = signal + noise
        
        self.signal, self.noise = signal, noise
        self.X = torch.tensor(X, dtype=torch.float32)
        self.w_star = torch.tensor(w_star, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        X_val = np.random.normal(size=(m//3,p)) # np.random.binomial(1, 0.5, (m, p))*2 - 1
        val_noise = np.random.normal(scale=std, size=m//3)
        y_val = np.matmul(X_val, w_star) + val_noise
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.val_noise = val_noise 

        self._least_square()
        self._snr()
        
    def _least_square(self):
        self.w_lst = torch.linalg.lstsq(self.X, self.y).solution
        self.lst_est_err = ((self.w_lst-self.w_star)**2).mean().item()
        
    def _snr(self):
        self.snr = np.sqrt((self.signal ** 2).sum()) / np.sqrt((self.noise**2).sum())
        print(f'SNR: {self.snr:.4f}')
        
class Simulation:
    def __init__(self, m=200, p=500, seed=42, support = np.repeat([1,2,3,4],4), # np.tile
                 std = 0.5):
        self.m = m
        self.p = p
        self.k = support.shape[0]
        self.support = support
        self.seed = seed
        self.std = std
        
        np.random.seed(seed)
        X = np.random.binomial(1, 0.5, (m, p))*2 - 1
        w_star = np.hstack((support, np.zeros(self.p-self.k)))
        signal = np.matmul(X, w_star)
        noise = np.random.normal(scale=std, size=m)
        y = signal + noise
        
        self.signal, self.noise = signal, noise
        self.X = torch.tensor(X, dtype=torch.float32)
        self.w_star = torch.tensor(w_star, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
        X_val = np.random.binomial(1, 0.5, (m//3, p))*2 - 1
        val_noise = np.random.normal(scale=std, size=m//3)
        y_val = np.matmul(X_val, w_star) + val_noise
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.y_val = torch.tensor(y_val, dtype=torch.float32)
        self.val_noise = val_noise 

        self._least_square()
        self._snr()

    def _least_square(self):
        self.w_lst = torch.linalg.lstsq(self.X, self.y).solution
        self.lst_est_err = ((self.w_lst-self.w_star)**2).mean().item()

    def _snr(self):
        self.snr = np.sqrt((self.signal ** 2).sum()) / np.sqrt((self.noise**2).sum())
        print(f'SNR: {self.snr:.4f}')
       
class EarlyStopping():
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(
                f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True

class groupModel(torch.nn.Module):
    def __init__(self, p=500, group_size=4, depth=2):
        super().__init__()
        self.p = p 
        self.depth = depth
        self.group_size = group_size
        num_groups = p//group_size
        self.num_groups = num_groups
        
        self.vs = torch.nn.ModuleList([torch.nn.Linear(group_size,1,bias=False) for i in range(num_groups)])
        self.u = torch.nn.Linear(num_groups, 1, bias=False)
        
    def forward(self, x):

        weights = torch.concat([self.u.weight[0, i] ** self.depth *  self.vs[i].weight for i in range(self.num_groups)], axis=1).t()
        ypred = torch.matmul(x, weights)
        return ypred.squeeze(-1)
    
    def get_params(self):
        params = torch.concat([self.u.weight[0, i] ** self.depth *  self.vs[i].weight for i in range(self.num_groups)], axis=1).squeeze(0)
        params = params.detach()
        return params

class groupTrainer:
    def __init__(self, model, sim, is_two_lr=False, varepsilon = 1e-12, tol_on_u = 5e-3, lr=0.01, is_small_train=False, is_monitor_u_diff=False):
        self.model = model
        self.sim = sim
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.u.parameters(), lr=lr)
        self.all_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.loss_criterion = torch.nn.MSELoss()
        self.monitor = {'w':[], 'u':[], 'v':[]}
        self.loss = []
        self.early_stopping = EarlyStopping()
        self.flag = True
        self.early_stopped_epoch = None
        self.early_stopped_model = None

        self.params_est_err = []
        self.dir_monitor = []
        
        self.change_epoch = 0
        self.is_two_lr = is_two_lr
        self.tol_on_u = tol_on_u
        self.is_small_train = is_small_train
        self.is_monitor_u_diff = is_monitor_u_diff

        self.num_groups = self.sim.p//self.model.group_size
        self.v_optimizers = []
        for i in range(self.num_groups):
            tmp_optim = torch.optim.SGD(self.model.vs[i].parameters(), lr=lr)
            self.v_optimizers.append(tmp_optim)
        self.varepsilon = varepsilon

    def _one_epoch(self):
        
        y_pred = self.model(self.sim.X)
        loss = self.loss_criterion(y_pred, self.sim.y)
        self.optimizer.zero_grad()
        for i in range(self.num_groups):
            self.v_optimizers[i].zero_grad()
        # for i in range(self.model.num_groups):
        #     if self.model.vs[i].weight.grad is not None:
        #         self.model.vs[i].weight.grad.data.zero_()
        # self.model.zero_grad()
        loss.backward()
        for i in range(self.num_groups):
            for g in self.v_optimizers[i].param_groups:
                g['lr'] = 1./self.model.u.weight.data.detach().clone()[0,i]**(self.model.depth*2)
            self.v_optimizers[i].step()
            # tmp_optimizer = torch.optim.SGD(self.model.vs[i].parameters(), lr=1./self.model.u.weight.data.detach().clone()[0,i]**(self.model.depth*2))
            # tmp_optimizer.step()
            self.model.vs[i].weight.data = self.model.vs[i].weight.data.clone() / (self.model.vs[i].weight.data.clone() ** 2).sum().sqrt()
        # self.model.zero_grad()
        # loss = self.loss_criterion(y_pred, self.sim.y)
        # self.optimizer.zero_grad()
        # loss.backward()
        self.optimizer.step()

        est_err = self._monitor()
        self.loss.append(loss.item())
        return loss.item(), est_err, self.model.u.weight.data.detach().clone()
    
    def _small_lr_one_epoch(self):
        y_pred = self.model(self.sim.X)
        loss = self.loss_criterion(y_pred, self.sim.y)
        self.all_optimizer.zero_grad()
        loss.backward()
        self.all_optimizer.step()
        
        for i in range(self.sim.p//self.model.group_size):
            self.model.vs[i].weight.data = self.model.vs[i].weight.data / (self.model.vs[i].weight.data ** 2).sum().sqrt()

        est_err = self._monitor()
        self.loss.append(loss.item())
        return loss.item(), est_err
    
    def _monitor(self):
        k = self.sim.k
        num = k//self.model.group_size
        params = self.model.get_params().squeeze(0)
        est_err = self.loss_criterion(params, self.sim.w_star).item()
        
        self.params_est_err.append(est_err)
        params = params.numpy().tolist()
        
        u = self.model.u.weight.detach().squeeze(0).numpy().tolist()
        v = [self.model.vs[i].weight.detach().squeeze(0).numpy().tolist() for i in range(self.model.num_groups)]
        v = [item for sublist in v for item in sublist]
        
        self.monitor['w'].append([*params[:k],max([abs(x) for x in params[k:]])])
        self.monitor['u'].append([*u[:num],max([abs(x) for x in u[num:]])])
        self.monitor['v'].append([*v[:k],max([abs(x) for x in v[k:]])])
        
        return est_err
    
    def _small_train(self, epochs):
        for epoch in range(epochs):
            loss, est_err = self._small_lr_one_epoch()
            if self.flag:
                self.early_stopping(est_err)
                if self.early_stopping.early_stop:
                    self.early_stopped_epoch = len(self.params_est_err)
                    self.early_stopped_model = copy.deepcopy(self.model)
                    self.flag = False
            print(f'{epoch}/{epochs}, loss: {loss:.4f}, est error: {est_err:.4f}')  
            
    def _optimal_train(self, epochs):
        for epoch in range(epochs):
            loss, est_err, _ = self._one_epoch()
            if self.flag:
                self.early_stopping(est_err)
                if self.early_stopping.early_stop:
                    self.early_stopped_epoch = len(self.params_est_err)
                    self.early_stopped_model = copy.deepcopy(self.model)
                    self.flag = False
            print(f'{epoch}/{epochs}, loss: {loss:.4f}, est error: {est_err:.4f}') 
            
    def _two_train(self, epochs):
        prev_u = 1
        flag = True
        for epoch in range(epochs):
            if flag:
                loss, est_err, u = self._one_epoch()
                u_diff = ((u-prev_u).abs()/np.abs(prev_u + self.varepsilon)).max()
                prev_u = u
                if self.is_monitor_u_diff:
                    print(f'{u_diff.item()}')
                if u_diff < self.tol_on_u:
                    flag = False
                    print(f'Change epoch: {epoch} with {u_diff.item()}')
                    self.change_epoch = epoch
            else:
                loss, est_err = self._small_lr_one_epoch()


            if self.flag:
                self.early_stopping(est_err)
                if self.early_stopping.early_stop:
                    self.early_stopped_epoch = len(self.params_est_err)
                    self.early_stopped_model = copy.deepcopy(self.model)
                    self.flag = False
            print(f'{epoch}/{epochs}, loss: {loss:.4f}, est error: {est_err:.4f}')  
            
    def train(self, epochs=500):
        if self.is_two_lr:
            self._two_train(epochs=epochs)
        elif self.is_small_train:
            self._small_train(epochs=epochs)
        else:
            self._optimal_train(epochs=epochs)
        self.get_dir()

    def transpose_monitor(self):
        transposed_monitor = {}
        for key, item in self.monitor.items():
            item_t = [[one[i] for one in item] for i in range(len(item[0]))]
            transposed_monitor[key] = item_t
        return transposed_monitor
    
    def get_dir(self):
        inner_product = []
        group_size = self.model.group_size
        for i in range(len(self.monitor['v'])):
            vec = np.array(self.monitor['v'][i][:-1])
            n = vec.shape[0]
            one_step = []
            for j in range(n//self.model.group_size):
                tmp_vec = vec[j*group_size : (j+1)*group_size]
                tmp_support = self.sim.support[j*group_size : (j+1)*group_size]
                res = (tmp_vec * tmp_support).sum() / (np.sqrt((tmp_vec**2).sum()) * np.sqrt((tmp_support**2).sum()))
                one_step.append(res)
            inner_product.append(one_step)
        self.dir = inner_product

class groupTrainerWithoutWN:
    def __init__(self, model, sim, is_two_lr=False, tol_on_u = 5e-3, lr=0.01, lr_v=0.0001, is_small_train=False, is_monitor_u_diff=False):
        self.model = model
        self.sim = sim
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.u.parameters(), lr=lr)
        self.v_optimizer = torch.optim.SGD(self.model.vs.parameters(), lr=lr_v)
        
        self.all_optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        self.loss_criterion = torch.nn.MSELoss()
        self.monitor = {'w':[], 'u':[], 'v':[]}
        self.loss = []
        self.early_stopping = EarlyStopping()
        self.flag = True
        self.early_stopped_epoch = None
        self.early_stopped_model = None

        self.params_est_err = []
        self.dir_monitor = []
        
        self.change_epoch = 0
        self.is_two_lr = is_two_lr
        self.tol_on_u = tol_on_u
        self.is_small_train = is_small_train
        self.is_monitor_u_diff = is_monitor_u_diff

        self.num_groups = self.sim.p//self.model.group_size
        self.v_optimizers = []
        for i in range(self.num_groups):
            tmp_optim = torch.optim.SGD(self.model.vs[i].parameters(), lr=lr)
            self.v_optimizers.append(tmp_optim)
    
    def _small_lr_one_epoch(self):
        y_pred = self.model(self.sim.X)
        loss = self.loss_criterion(y_pred, self.sim.y)
        self.all_optimizer.zero_grad()

        loss.backward()
        self.all_optimizer.step()

        est_err = self._monitor()
        self.loss.append(loss.item())
        return loss.item(), est_err
    
    def _monitor(self):
        k = self.sim.k
        num = k//self.model.group_size
        params = self.model.get_params().squeeze(0)
        est_err = self.loss_criterion(params, self.sim.w_star).item()
        
        self.params_est_err.append(est_err)
        params = params.numpy().tolist()
        
        u = self.model.u.weight.detach().squeeze(0).numpy().tolist()
        v = [self.model.vs[i].weight.detach().squeeze(0).numpy().tolist() for i in range(self.model.num_groups)]
        v = [item for sublist in v for item in sublist]
        
        self.monitor['w'].append([*params[:k],max([abs(x) for x in params[k:]])])
        self.monitor['u'].append([*u[:num],max([abs(x) for x in u[num:]])])
        self.monitor['v'].append([*v[:k],max([abs(x) for x in v[k:]])])
        
        return est_err
    
    def _small_train(self, epochs):
        for epoch in range(epochs):
            loss, est_err = self._small_lr_one_epoch()
            if self.flag:
                self.early_stopping(est_err)
                if self.early_stopping.early_stop:
                    self.early_stopped_epoch = len(self.params_est_err)
                    self.early_stopped_model = copy.deepcopy(self.model)
                    self.flag = False
            print(f'{epoch}/{epochs}, loss: {loss:.4f}, est error: {est_err:.4f}')  
        
            
    def train(self, epochs=500):
        self._small_train(epochs=epochs)
        self.get_dir()

    def transpose_monitor(self):
        transposed_monitor = {}
        for key, item in self.monitor.items():
            item_t = [[one[i] for one in item] for i in range(len(item[0]))]
            transposed_monitor[key] = item_t
        return transposed_monitor
    
    def get_dir(self):
        inner_product = []
        group_size = self.model.group_size
        for i in range(len(self.monitor['v'])):
            vec = np.array(self.monitor['v'][i][:-1])
            n = vec.shape[0]
            one_step = []
            for j in range(n//self.model.group_size):
                tmp_vec = vec[j*group_size : (j+1)*group_size]
                tmp_support = self.sim.support[j*group_size : (j+1)*group_size]
                res = (tmp_vec * tmp_support).sum() / (np.sqrt((tmp_support**2).sum())) # np.sqrt((tmp_vec**2).sum()) *
                one_step.append(res)
            inner_product.append(one_step)
        self.dir = inner_product
        
def summary_plot(trainer, n_groups = 3, group_size=2):
    fig, axes = plt.subplots(3,2)
    fig.set_size_inches(16, 16)
    axes[0,0].plot(trainer.loss)
    axes[0,0].set_title('loss')
    
    axes[0,1].plot(trainer.params_est_err)
    axes[0,1].set_title('estimation error')

    colors = ['C'+str(i) for i in range(n_groups) for j in range(group_size)] + [f'C{n_groups}']
    axes[1,0].plot(trainer.monitor['w'], label = ['group'+str(i+1) for i in range(n_groups) for j in range(group_size)] + ['non support'])
    for i, j in enumerate(axes[1,0].lines):
        j.set_color(colors[i])
    handles, labels = axes[1,0].get_legend_handles_labels()
    axes[1,0].set_title('recovered params')
    display = np.arange(0,n_groups*group_size+1,group_size)
    axes[1,0].legend([handle for i,handle in enumerate(handles) if i in display],
          [label for i,label in enumerate(labels) if i in display])#, loc=(1.04,0))

    colors = ['C'+str(i) for i in range(n_groups)] + [f'C{n_groups}']
    axes[1,1].plot(trainer.monitor['u'], label = ['group'+str(i+1) for i in range(n_groups)] + ['non support'])
    for i, j in enumerate(axes[2,0].lines):
        j.set_color(colors[i])
    axes[1,1].set_title('parameter u')
    axes[1,1].legend()

    group_labels = ['group'+str(i+1) for i in range(n_groups)]
    axes[2,0].plot(trainer.dir, label = group_labels)
    axes[2,0].set_title('inner product between v and v_star')
    axes[2,0].legend()


    colors = ['C'+str(i) for i in range(n_groups) for j in range(group_size)] + [f'C{n_groups}']
    axes[2,1].plot(trainer.monitor['v'], label = ['group'+str(i+1) for i in range(n_groups) for j in range(group_size)] + ['non support'])
    for i, j in enumerate(axes[2,1].lines):
        j.set_color(colors[i])
    handles, labels = axes[2,1].get_legend_handles_labels()
    display = np.arange(0,n_groups*group_size+1,group_size)
    axes[2,1].legend([handle for i,handle in enumerate(handles) if i in display],
          [label for i,label in enumerate(labels) if i in display], loc=(1.04,0))
    _ = axes[2,1].set_title('parameter v')
    
    if trainer.change_epoch > 0:
        for i in range(3):
            for j in range(2):
                axes[i,j].axvline(trainer.change_epoch, color='black')