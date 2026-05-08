#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import os
from torch.utils.data.dataloader import DataLoader
import pandas as pd
import torch
import numpy as np
import _pickle as pkl
import copy
import joblib
from lib.BasicTrainer_sw import Trainer
from others.attn_lstm import Attn_LSTM
from tqdm import tqdm
import time
import datetime
class NetList(torch.nn.Module):
    def __init__(self, list_of_models):
        super(NetList, self).__init__()
        self.models = torch.nn.ModuleList(list_of_models)

    def forward(self, x, idx=0):
        return self.models[idx](x)

class SGDInfluence(object):
    def __init__(self, exp, model, args, train_set, test_set, heldout_set,
                 **kwargs):

        self.args = args
        self.heldout_set = heldout_set
        self.train_set = train_set
        self.learning_rate = args.lr_init
        self.test_set = test_set

        self.max_epochs = 10

        self.batch_size = args.batch_size
        self.args.topk = 100
        if exp != None:
            self.exp = exp
        #cpu version
        self.model = model.to(args.device)
        self.device = args.device
        self.optimizer = args.optim




    def add_noise2(self, xx, yy, ratio):
        x_num = xx.shape[0]
        n_num = int(x_num * ratio)
        x_len = xx.shape[1]
        y_len = yy.shape[1]
        noiseData = np.random.normal(0.5, 0.8**2, (n_num, x_len+y_len))
        noiseData = torch.tensor(noiseData).to(self.device)
        noiseData = noiseData.float()
        noiseData_X = noiseData[:,:x_len]
        noiseData_Y = noiseData[:,-y_len:]

        new_X = torch.vstack((xx.to(self.device), noiseData_X))
        new_Y = torch.vstack((yy.to(self.device), noiseData_Y))

        b = torch.randperm(new_X.size(0))
        new_X = new_X[b,:]
        b = torch.randperm(new_Y.size(0))
        new_Y = new_Y[b,:]

        return new_X, new_Y
    def add_noise(self, xx, yy, ratio):
        x_num = xx.shape[0]
        n_num = int(x_num * ratio)
        x_len = xx.shape[1]
        y_len = yy.shape[1]
        noiseData = np.random.random((n_num, x_len+y_len))
        noiseData = torch.tensor(noiseData).to(self.device)
        noiseData = noiseData.float()
        noiseData_X = noiseData[:,:x_len]
        noiseData_Y = noiseData[:,-y_len:]

        new_X = torch.vstack((xx.to(self.device), noiseData_X))
        new_Y = torch.vstack((yy.to(self.device), noiseData_Y))

        b = torch.randperm(new_X.size(0))
        new_X = new_X[b,:]
        b = torch.randperm(new_Y.size(0))
        new_Y = new_Y[b,:]

        return new_X, new_Y



    def train(self,num_epochs_list, output_path, seed=0):
        fn = '%s/infl_sgd_at_epoch_%02d_%s.dat' % (output_path, num_epochs_list,self.args.normalizer)
        fn_final = '%s/epoch%02d_final_model.dat' % (output_path, num_epochs_list)
        if not os.path.exists(fn) and not os.path.exists(fn_final):

            # model setup
            if self.optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            elif self.optimizer == 'adam':
                optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, eps=1.0e-8,
                                             weight_decay=0, amsgrad=False)

            # init loss function, optimizer
            if self.args.loss_func == 'mae':
                loss_fun = torch.nn.L1Loss().to(self.args.device)
            elif self.args.loss_func == 'mse':
                loss_fun = torch.nn.MSELoss().to(self.args.device)
            else:
                raise ValueError


            # 生成模型
            if self.args.model in ['TimesNet', 'Autoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear',
                         'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer', 'PatchTST', 'Pyraformer',
                         'MICN', 'Crossformer', 'FiLM']:
                # exp = Exp(self.args)  # set experiments
                # exp.model_dict[args.model].Model(args).float()
                cre_Model = self.exp.model_dict[self.args.model].Model
            elif self.args.model in ['lstm-att', 'lstm']:
                cre_Model = Attn_LSTM

            # model list
            bundle_size = len(self.train_set)

            #cpu version
            list_of_models = [cre_Model(self.args) for _ in range(bundle_size)]

            # training
            # torch.manual_seed(seed)
            val_loss = []
            num_epochs = num_epochs_list
            train_loss = []
            for epoch in tqdm(range(num_epochs),total=num_epochs):
                # print(epoch)
                # training
                self.model.train()
                # np.random.seed(epoch)
                temp_loss = []
                for index, (x_train, y_true) in enumerate(self.train_set):


                    # save,一个batch，一个模型
                    # temp_state_dict = self.model.state_dict.to('cpu')
                    list_of_models[index].load_state_dict(copy.deepcopy(self.model.state_dict()))

                    # self.model.train()
                    optimizer.zero_grad()


                    ############################################
                    x_train = x_train.to(self.device).float()
                    y_true = y_true.to(self.device).float()

                    if self.args.stamp:
                        y_true = y_true[:,
                                 self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                        label = y_true[:, :, 0]  # 第2维是时间戳

                        x_enc = x_train[:, :, 0].unsqueeze(2)
                        x_mark_enc = x_train[:, :, 1:]
                        x_dec = y_true[:, :, 0].unsqueeze(2)
                        x_mark_dec = y_true[:, :, 1:]

                        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        output = output.squeeze(2)
                    else:
                        y_true = y_true[:, :self.args.window * self.args.horizon]
                        label = y_true
                        output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                    ############################################
                    loss = loss_fun(output.to(self.device), label)
                    loss.backward()
                    optimizer.step()
                    temp_loss.append(loss.item())

                self.model.eval()
                train_loss.append(np.array(temp_loss).mean())
                temp_loss.clear()

                for index, (x_train, y_true) in enumerate(self.heldout_set):
                    ############################################
                    x_train = x_train.to(self.device).float()
                    y_true = y_true.to(self.device).float()

                    if self.args.stamp:
                        y_true = y_true[:,
                                 self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                        label = y_true[:, :, 0]  # 第2维是时间戳

                        x_enc = x_train[:, :, 0].unsqueeze(2)
                        x_mark_enc = x_train[:, :, 1:]
                        x_dec = y_true[:, :, 0].unsqueeze(2)
                        x_mark_dec = y_true[:, :, 1:]

                        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        output = output.squeeze(2)
                    else:
                        y_true = y_true[:, :self.args.window * self.args.horizon]
                        label = y_true
                        output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                    ############################################
                    temp_loss.append(loss_fun(output.to(self.device), label).item())

                val_loss.append(np.array(temp_loss).mean())
                print('train loss--{}---val loss {}---at epoch {}'.format(train_loss[-1],val_loss[-1],epoch))

                # if epoch in num_epochs_list:
                # save,一个batch，一个模型，每个batch训练的模型一起保存
                fn = '%s/epoch%02d_bundled_models.dat' % (output_path, epoch)
                #cpu version
                models = NetList(list_of_models)
                torch.save(models.state_dict(), fn)

            # 保存训练集使用完之后的模型
            fn = '%s/epoch%02d_final_model.dat' % (output_path, num_epochs)
            torch.save(self.model.state_dict(), fn)
            fn = '%s/epoch%02d_final_optimizer.dat' % (output_path, num_epochs)
            torch.save(optimizer.state_dict(), fn)
            fn = '%s/epoch%02d_val_Loss.csv' % (output_path, num_epochs)
            pd.DataFrame([train_loss,val_loss]).to_csv(fn)




    def infl_sgd(self, output_path, target_epoch, seed=0):
        fn = '%s/infl_sgd_at_epoch_%02d_%s.dat' % (output_path, target_epoch,self.args.normalizer)
        if os.path.exists(fn):
            #删除多余的数据，仅保留需要的checkpoints
            for epoch in range(target_epoch - 1, -1, -1):
                try:
                    fn = '%s/epoch%02d_final_model.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass
                try:
                    fn = '%s/epoch%02d_final_optimizer.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass
                try:
                    fn = '%s/epoch%02d_bundled_models.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass

        else:
            # pass
            # model setup
            # torch.manual_seed(seed)

            # 生成模型
            if self.args.model in ['TimesNet', 'Autoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear',
                                   'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer', 'PatchTST', 'Pyraformer',
                                   'MICN', 'Crossformer', 'FiLM']:
                # exp = Exp(self.args)  # set experiments
                # exp.model_dict[args.model].Model(args).float()
                cre_Model = self.exp.model_dict[self.args.model].Model
            elif self.args.model in ['lstm-att', 'lstm']:
                cre_Model = Attn_LSTM

            model = cre_Model(self.args).to(self.device)
            #这里改为目标epoch的模型
            fn = '%s/epoch%02d_final_model.dat' % (output_path, target_epoch)
            model.load_state_dict(torch.load(fn))
            model.eval()

            # init loss function, optimizer
            if self.args.loss_func == 'mae':
                loss_fun = torch.nn.L1Loss().to(self.device)
            elif self.args.loss_func == 'mse':
                loss_fun = torch.nn.MSELoss().to(self.device)
            else:
                raise ValueError

            lr = self.learning_rate

            # gradient
            u = self.compute_gradient(model)

            ntr = len(self.train_set.dataset)
            k_len = len(self.train_set)
            # model list
            #cpu version
            list_of_models = [cre_Model(self.args) for _ in range(k_len)]
            models = NetList(list_of_models)

            # influence
            if ntr % self.args.batch_size == 0:
                infl = torch.zeros(ntr, target_epoch, requires_grad=False).to(self.device)
            else:
                ntimes = ntr - ntr % self.args.batch_size
                infl = torch.zeros(ntimes, target_epoch, requires_grad=False).to(self.device)

            for epoch in tqdm(range(target_epoch-1, -1, -1),total=target_epoch):
                # print(epoch)
                torch.cuda.empty_cache()
                fn = '%s/epoch%02d_bundled_models.dat' % (output_path, epoch)
                models.load_state_dict(torch.load(fn))

                for index, (x_train, y_true) in enumerate(self.train_set):
                    # print(index)
                    # t0 = time.time()
                    # x_train = x_train.to(self.device)
                    # y_true = y_true[:, :self.args.window * self.args.horizon]
                    # y_true = y_true.to(self.device)
                    # label = y_true[:, -self.args.window * self.args.horizon:]
                    ############################################
                    x_train = x_train.to(self.device).float()
                    y_true = y_true.to(self.device).float()

                    if self.args.stamp:
                        y_true = y_true[:,
                                 self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                        label = y_true[:, :, 0]  # 第2维是时间戳

                        x_enc = x_train[:, :, 0].unsqueeze(2)
                        x_mark_enc = x_train[:, :, 1:]
                        x_dec = y_true[:, :, 0].unsqueeze(2)
                        x_mark_dec = y_true[:, :, 1:]

                        # output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                        # output = output.squeeze(2)
                    else:
                        y_true = y_true[:, :self.args.window * self.args.horizon]
                        label = y_true
                        # output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                    ############################################


                    # 取每个batch的模型
                    m = models.models[index].to(self.device)
                    x_size = x_train.shape[0]
                    for i in range(x_size):
                        s_id = index * self.batch_size + i
                        # Forward pass
                        if self.args.stamp:
                            output = m(x_enc[i].unsqueeze(0), x_mark_enc[i].unsqueeze(0), x_dec[i].unsqueeze(0), x_mark_dec[i].unsqueeze(0))
                            output = output.squeeze(2)
                        else:
                            output, _ = m(x_train[i].unsqueeze(0), y_true[i].unsqueeze(0), teacher_forcing_ratio=0)

                        # output, _ = m(x_train[i].unsqueeze(0), y_true[i].unsqueeze(0), teacher_forcing_ratio=0.)
                        loss = loss_fun(output.to(self.device), label[i])
                        m.zero_grad()
                        loss.backward()
                        for j, param in enumerate(m.parameters()):
                            infl[s_id, epoch] = infl[s_id, epoch] + lr * (u[j].data * param.grad.data).sum() / x_size

                        # del output
                        # torch.cuda.empty_cache()
                    # update u
                    #torch.backends.cudnn.flags(enabled=True)
                    #torch.backends.cudnn.benchmark=True
                    with torch.backends.cudnn.flags(enabled=False):
                        # output, _ = m(x_train, y_true, teacher_forcing_ratio=0.)
                        if self.args.stamp:
                            output = m(x_enc, x_mark_enc, x_dec, x_mark_dec)
                            output = output.squeeze(2)
                        else:
                            output, _ = m(x_train, y_true, teacher_forcing_ratio=0)

                    loss = loss_fun(output.to(self.device), label)

                    grad_params = torch.autograd.grad(loss, m.parameters(), create_graph=True)
                    ug = 0
                    for uu, g in zip(u, grad_params):
                        ug += (uu * g).sum()
                    m.zero_grad()
                    ug.backward()
                    for j, param in enumerate(m.parameters()):
                        u[j] = u[j] - lr * param.grad.data / x_size

                    del grad_params, m, x_train, y_true, loss, ug, output, label
                    torch.cuda.empty_cache()
                    if self.args.stamp:
                        del x_enc, x_mark_enc, x_dec, x_mark_dec
                        torch.cuda.empty_cache()

                    # t1 = time.time()
                    # cost = int(round(t1-t0))
                    # print('time---{}'.format(datetime.timedelta(seconds=cost)))
                    torch.cuda.empty_cache()
                # save
                fn = '%s/infl_sgd_at_epoch_%02d_%s.dat' % (output_path, target_epoch, self.args.normalizer)
                joblib.dump(infl.cpu().numpy(), fn, compress=9)

                if epoch > 0:
                    infl[:, epoch - 1] = infl[:, epoch].clone()



            # 删除多余的数据，仅保留需要的checkpoints
            for epoch in range(target_epoch - 1, -1, -1):
                try:
                    fn = '%s/epoch%02d_final_model.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass
                try:
                    fn = '%s/epoch%02d_final_optimizer.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass
                try:
                    fn = '%s/epoch%02d_bundled_models.dat' % (output_path, epoch)
                    os.remove(fn)
                except Exception:
                    pass

    def compute_gradient(self, model):
        # 在val_dataset
        device = self.device
        n = len(self.heldout_set) * self.batch_size

        u = [torch.zeros(*param.shape, requires_grad=False).to(device) for param in model.parameters()]
        model.train()

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.args.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.args.device)
        else:
            raise ValueError

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for index, (x_train, y_true) in enumerate(self.heldout_set):
            # y_true = y_true[:, :self.args.window * self.args.horizon]
            # label = y_true[:, -self.args.window * self.args.horizon:]
            # # Forward pass
            # output, _ = model(X, y_true, teacher_forcing_ratio=0.)

            ############################################
            x_train = x_train.to(self.device).float()
            y_true = y_true.to(self.device).float()

            if self.args.stamp:
                y_true = y_true[:,
                         self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                label = y_true[:, :, 0]  # 第2维是时间戳

                x_enc = x_train[:, :, 0].unsqueeze(2)
                x_mark_enc = x_train[:, :, 1:]
                x_dec = y_true[:, :, 0].unsqueeze(2)
                x_mark_dec = y_true[:, :, 1:]

                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                output = output.squeeze(2)
            else:
                y_true = y_true[:, :self.args.window * self.args.horizon]
                label = y_true
                output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

            ############################################


            loss = loss_fun(output.to(device), label.to(device))
            # optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            for j, param in enumerate(model.parameters()):
                try:
                    u[j] += param.grad.data / n
                except (Exception):
                    u[j] += param.data / n
        return u


