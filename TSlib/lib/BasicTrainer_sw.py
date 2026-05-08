#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2022/7/1 22:05
@desc:
"""
import copy
import torch
import math
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from TSlib.lib.logger import get_logger
from TSlib.lib.metrics import All_Metrics 
import pandas as pd
from scipy.optimize import minimize


class Trainer(object):
    def __init__(self, ktype, model, loss, optimizer, train_loader, val_loader, test_loader,
                 scaler, args, lr_scheduler=None):
        super(Trainer, self).__init__()
        self.ktype = ktype
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.args = args
        self.device = args.device
        self.lr_scheduler = lr_scheduler
        self.train_per_epoch = len(train_loader)
        self.grad_dims = []
        self.confict_num = 0
        self.num_iterations = 10
        self.model = model.to(args.device)

        self.batach_atten_loss = np.zeros((1,2, self.args.batch_size,))
        if self.args.dataset == 'traffic':
            self.batach_attweight = np.zeros((1,self.args.batch_size, 576))
        else:
            self.batach_attweight = np.zeros((1, self.args.batch_size, 288))
        if val_loader != None:
            self.val_per_epoch = len(val_loader)
        self.best_path = os.path.join(self.args.log_dir, 'best_model.pth')
        self.loss_figure_path = os.path.join(self.args.log_dir, 'loss.png')
         

        self.results = pd.DataFrame(columns=['dataset', 'MAE', 'RMSE', 'MAPE', 'steps', 'bestModel'])

    def val_epoch(self, epoch, val_dataloader):
        self.model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_idx, (x_enc, y_true) in enumerate(val_dataloader):

                x_enc = x_enc.to(self.device).float()
                y_true = y_true.to(self.device).float()
 
                if self.ktype in ['sadc', 'cagrad', 'pcgrad']:
                    output, decoder_attentions = self.model(x_enc, None, y_true, None)
                else:
                    output = self.model(x_enc, None, y_true, None)
 
                
                if self.args.real_value:
                    y_true = self.scaler.inverse_transform(y_true)
                    output = self.scaler.inverse_transform(output)  # 修复：同时反归一化output保证在同一尺度

                loss = self.loss(output, y_true)
                #a whole batch of Metr_LA is filtered
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
        val_loss = total_val_loss / len(val_dataloader)
        if epoch % self.args.log_step == 0:
            # self.logger.info('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
            print('**********Val Epoch {}: average Loss: {:.6f}'.format(epoch, val_loss))
        return val_loss

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return torch.allclose(a, a.T, rtol=rtol, atol=atol)

    def cagrad_multi(self, grads, c=0.5):
        grads = grads.numpy()
        g0 = np.mean(grads, axis=1)
        g0_norm = np.linalg.norm(g0)
        phi = c * g0_norm

        K = grads.shape[1]

        # obj1 = lambda x: np.mean(x*grads, axis=0)
        # con1 = lambda x: x.sum()-1
        def obj(x):
            gw = np.mean(x * grads, axis=1)
            gw_norm = np.linalg.norm(gw)
            t1 = gw.T @ g0
            t2 = phi * gw_norm
            return t1 + t2

        def con(x):
            J = x.sum() - 1
            return J

        cons = ({'type': 'eq', 'fun': con},
                )

        bnds = tuple((0, 1) for i in range(K))

        x0 = np.zeros(K)
        res = minimize(obj, x0, constraints=cons, method='SLSQP', bounds=bnds)

        x = res.x

        gw = x * grads
        gw = np.mean(gw, axis=1)
        gw_norm = np.linalg.norm(gw)

        lmbda = phi / (gw_norm + 1e-4)
        g = g0 + lmbda * gw
        return torch.tensor(g * c)

    

    def train_epoch(self, epoch):
        # global optimizer
        self.model.train()
        total_loss = 0
        train_epoch_loss = 0

        alo_time = []
        scsq_time = []
        conf_time = []

        if self.args.optim == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                                        lr=self.args.lr_init)
        elif self.args.optim == 'adam':
            optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, self.model.parameters()),
                                         lr=self.args.lr_init, eps=1.0e-8,
                                         weight_decay=0, amsgrad=False)

        for batch_idx, (x_enc, y_true) in enumerate(self.train_loader):

            x_enc = x_enc.to(self.device).float()
            y_true = y_true.to(self.device).float()
  
            if self.ktype in ['sadc', 'cagrad', 'pcgrad']:
                output, decoder_attentions = self.model(x_enc, None, y_true, None)
            else:
                output = self.model(x_enc, None, y_true, None)

            # output = output + seq_last
            # output = output.squeeze(2)
 
            self.optimizer.zero_grad()
  
            if self.args.real_value:
                y_true = self.scaler.inverse_transform(y_true)
                output = self.scaler.inverse_transform(output)  # 修复：同时反归一化output

            if self.ktype != 'normal':
                pass

            else:
                # normal approach
                b_loss = self.loss(output, y_true)
                b_loss.backward()

                self.optimizer.step()
                total_loss += b_loss.item()

        #log information
        if epoch % self.args.log_step == 0:             
            train_epoch_loss = total_loss/self.train_per_epoch 
            print('**********Train Epoch {}: averaged Loss: {:.6f} '.format(epoch, train_epoch_loss))

        #learning rate decay
        if self.args.lr_decay:
            # self.lr_scheduler.step()
            self.lr_scheduler.step(total_loss)
        return train_epoch_loss, alo_time, scsq_time


    def train(self):
        best_model = None
        best_loss = float('inf')
        not_improved_count = 0
        train_loss_list = []
        val_loss_list = []
        start_time = time.time()
        alotimes = []
        scsqtimes = []
        for epoch in range(1, self.args.epochs + 1):
            # print(epoch)
            #epoch_time = time.time()
            train_epoch_loss, alo_time, scsq_time = self.train_epoch(epoch)
            # if epoch < 50:
            #     alotimes.extend(alo_time)
            #     scsqtimes.extend(scsq_time)
            #print(time.time()-epoch_time)
            #exit()
            if self.val_loader == None:
                val_dataloader = self.test_loader
            else:
                val_dataloader = self.val_loader
            val_epoch_loss = self.val_epoch(epoch, val_dataloader)

            #print('LR:', self.optimizer.param_groups[0]['lr'])
            train_loss_list.append(train_epoch_loss)
            val_loss_list.append(val_epoch_loss)
            if train_epoch_loss > 1e6:
                self.logger.warning('Gradient explosion detected. Ending...')
                break
            
            if best_loss - val_epoch_loss > 0.0001:
                best_loss = val_epoch_loss
                not_improved_count = 0
                best_state = True
            else:
                not_improved_count += 1
                best_state = False
            # early stop
            if self.args.early_stop:
                if not_improved_count == self.args.early_stop_patience:                     
                    print("Validation performance didn\'t improve for {} epochs. "
                          "Training stops.".format(self.args.early_stop_patience))
                    break
            # save the best state
            if best_state == True:
                '''
                if epoch % self.args.log_step == 0:
                    self.logger.info('*********************************Current best model saved!')
                '''
                best_model = copy.deepcopy(self.model.state_dict())



        finish_time = time.time()
        training_time = finish_time - start_time
        # self.logger.info("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
        print("Total training time: {:.4f}min, best loss: {:.6f}".format((training_time / 60), best_loss))
          

        #test
        self.model.load_state_dict(best_model)
        #self.val_epoch(self.args.epochs, self.test_loader)
        mae, rmse, mape = self.test(self.model, self.args, self.test_loader, self.scaler, finish_time=finish_time)

        pf = pd.DataFrame({
            'dataset': [self.args.dataset],
            'MAE': [mae],
            'RMSE': [rmse],
            'MAPE': [mape],
            'steps': [epoch],
            'bestModel': [
                './experiments/best_model_{}_{}_{}_{}.pth'.format(self.args.model, self.args.dataset, self.args.pred_len,
                                                                  finish_time)],
        })
        # self.results = self.results.append(pf, ignore_index=True)
        self.results = pd.concat([self.results, pf])

    def save_checkpoint(self):
        state = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.args
        }
        torch.save(state, self.best_path)
        # self.logger.info("Saving current best model to " + self.best_path)
        print("Saving current best model to " + self.best_path)

    @staticmethod
    def test(model, args, data_loader, scaler, path=None, finish_time=0):
        if path != None:
            check_point = torch.load(path)
            state_dict = check_point['state_dict']
            args = check_point['config']
            model.load_state_dict(state_dict)
            model.to(args.device)
        model.eval()
        y_pred_t = []
        y_true_t = []
        total_attentions = []
        total_datas = []

        with torch.no_grad():
            for batch_idx, (x_enc, y_true) in enumerate(data_loader):
                 
                x_enc = x_enc.to(args.device).float()
                y_true = y_true.to(args.device).float() 

                if args.ktype in ['sadc', 'cagrad', 'pcgrad','grand', 'vog','sgd-influence', 'd-shapely']:
                    output, decoder_attentions = model(x_enc, None, y_true, None)
                else:
                    output = model(x_enc, None, y_true, None)

                
                y_true_t.append(y_true)
                y_pred_t.append(output)

        y_pred_t = torch.cat(y_pred_t, dim=0)
        y_true_t = torch.cat(y_true_t, dim=0)
        if args.real_value:
            y_true_t = scaler.inverse_transform(y_true_t)
            y_pred_t = scaler.inverse_transform(y_pred_t)

        print_time = finish_time
         
        r_metrics = []
        for t in range(y_true.shape[1]):
            mae, rmse, mape = All_Metrics(y_pred_t[:, t], y_true_t[:, t],
                                          args.mae_thresh, args.mape_thresh)
            print("Horizon {:02d}, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
                t + 1, mae, rmse, mape * 100))
            r_metrics.append([t, mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()])


        mae, rmse, mape = All_Metrics(y_pred_t, y_true_t, args.mae_thresh, args.mape_thresh)
         
        print("Average Horizon, MAE: {:.4f}, RMSE: {:.4f}, MAPE: {:.4f}%".format(
            mae, rmse, mape * 100))
         
        return mae.cpu().item(), rmse.cpu().item(), mape.cpu().item()
 
    @staticmethod
    def _compute_sampling_threshold(global_step, k):
        """
        Computes the sampling probability for scheduled sampling using inverse sigmoid.
        :param global_step:
        :param k:
        :return:
        """
        return k / (k + math.exp(global_step / k))