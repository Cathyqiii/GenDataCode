import os
import joblib
import numpy as np
import torch
import math

class VoG(object):
    def __init__(self, model, args, train_set, test_set, heldout_set, **kwargs):

        self.args = args
        self.heldout_set = heldout_set
        self.train_set = train_set
        self.learning_rate = args.lr_init
        self.test_set = test_set
        self.batch_size = args.batch_size

        self.device = args.device
        self.optimizer = args.optim

        self.model = model.to(args.device)


    def infl_vog(self, output_path, target_epoch, seed):

        # init loss function, optimizer
        if self.args.loss_func == 'mae':
            loss_fun = torch.nn.L1Loss().to(self.device)
        elif self.args.loss_func == 'mse':
            loss_fun = torch.nn.MSELoss().to(self.device)
        else:
            raise ValueError

        if self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate, eps=1.0e-8,
                                         weight_decay=0, amsgrad=False)

        ntr = len(self.train_set.dataset)

        # influence
        if ntr % self.args.batch_size == 0:
            infl = torch.zeros(ntr, target_epoch, requires_grad=False).to(self.device)
        else:
            ntimes = ntr - ntr % self.args.batch_size
            infl = torch.zeros(ntimes, target_epoch, requires_grad=False).to(self.device)
        vog = {}

        for epoch in range(target_epoch):
            #当前参数下，所有样本的影响力
            self.model.train()
            for index, (x_train, y_true) in enumerate(self.train_set):
                x_train = x_train.to(self.device).float()
                x_train.requires_grad = True
                y_true = y_true.to(self.device).float()
                if self.args.stamp:
                    y_true = y_true[:,
                             self.args.label_len:self.args.label_len + self.args.window * self.args.horizon]  # 有label长度
                    label = y_true[:, :, 0]  # 第2维是时间戳

                    x_enc = x_train[:, :, 0].unsqueeze(2)
                    x_mark_enc = x_train[:, :, 1:]
                    x_dec = y_true[:, :, 0].unsqueeze(2)
                    x_mark_dec = y_true[:, :, 1:]

                    output , _ = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    output = output.squeeze(2)
                else:
                    y_true = y_true[:, :self.args.window * self.args.horizon]
                    label = y_true
                    output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                optimizer.zero_grad()
                b_loss = loss_fun(output.to(self.device), label)
                b_loss.backward(retain_graph=True)

                # Get gradients
                grad = x_train.grad.detach().cpu().numpy()

                optimizer.step()

                for i in range(x_train.shape[0]):
                    # 计算样本编号
                    s_id = index * self.batch_size + i

                    if s_id not in vog.keys():
                        vog[s_id] = []
                        vog[s_id].append(grad[i, :].reshape(1,-1).tolist())
                    else:
                        vog[s_id].append(grad[i, :].reshape(1,-1).tolist())


                # delete caches
                if self.args.stamp:
                    del x_train, y_true, output, label, x_enc, x_mark_enc, x_dec, x_mark_dec
                    torch.cuda.empty_cache()
                else:
                    del x_train, y_true, output, label
                    torch.cuda.empty_cache()

        #calculate vog score at each epoch
        for epoch in range(1, target_epoch):
            for index, (x_train, y_true) in enumerate(self.train_set):
                for i in range(x_train.shape[0]):
                    # 计算样本编号
                    s_id = index * self.batch_size + i
                    temp_v = np.array(vog[s_id]).squeeze()
                    temp_grad = temp_v[:epoch]
                    mean_grad = np.sum(temp_v, axis=0) / len(temp_v)
                    infl[s_id,epoch] = np.mean(
                        np.sqrt(sum([(mm - mean_grad) ** 2 for mm in temp_grad]) / len(temp_grad)))



        # save
        fn = '%s/infl_vog_at_epoch_%02d_%s.dat' % (output_path, target_epoch, self.args.normalizer)
        joblib.dump(infl.detach().cpu().numpy(), fn, compress=9)

        # delete caches
        del x_train, y_true, temp_v, temp_grad, mean_grad, infl
        torch.cuda.empty_cache()

