import os
import numpy as np
import torch
from functorch import grad
import joblib
# from others.attn_lstm import Attn_LSTM
# import torch.nn as nn
# from lib.simple_Weight import store_grad_nosadc
# from torch._functorch.deprecated import grad, vmap

class GraND(object):
    def __init__(self, model, args, train_set, test_set, heldout_set,
                 model_family='Attn_LSTM', metric='accuracy',
                 **kwargs):

        self.args = args
        self.heldout_set = heldout_set
        self.train_set = train_set
        self.learning_rate = args.lr_init
        self.test_set = test_set

        self.max_epochs = 1
        self.model_family = model_family
        self.metric = metric


        self.batch_size = args.batch_size
        self.args.topk = 100
        # if self.directory is not None:  # 文件不存在则创建
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)

        self.model = model.to(args.device)

        self.device = args.device
        self.optimizer = args.optim

        self.grad_dims = []

    def infl_grand(self, output_path, target_epoch, seed):


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
        for epoch in range(target_epoch):
            #当前参数下，所有样本的影响力
            self.model.train()
            for index, (x_train, y_true) in enumerate(self.train_set):

                x_train = x_train.to(self.device).float()
                y_true = y_true.to(self.device).float()
                if self.args.stamp:
                    y_true = y_true[:, self.args.label_len:self.args.label_len+self.args.window * self.args.horizon] #有label长度
                    label = y_true[:, :, 0] #第2维是时间戳

                    x_enc = x_train[:, :, 0].unsqueeze(2)
                    x_mark_enc = x_train[:, :, 1:]
                    x_dec = y_true[:, :, 0].unsqueeze(2)
                    x_mark_dec = y_true[:, :, 1:]

                    output ,_= self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    output = output.squeeze(2)
                else:
                    y_true = y_true[:, :self.args.window * self.args.horizon]
                    label = y_true
                    output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                self.grad_dims.clear()
                for param in self.model.parameters():
                    self.grad_dims.append(param.data.numel())
                #存储梯度
                dim = x_train.shape
                grads = torch.Tensor(1,sum(self.grad_dims))
                for i in range(dim[0]):
                    #计算样本编号
                    s_id = index * self.batch_size + i
                    optimizer.zero_grad()

                    b_item_loss = loss_fun(output[i,:].to(self.device), label[i,:])
                    b_item_loss.backward(retain_graph=True)
                    #读取样本梯度
                    cnt = 0
                    for param in self.model.parameters():
                        if param.grad is not None:
                            beg = 0 if cnt == 0 else sum(self.grad_dims[:cnt])
                            en = sum(self.grad_dims[:cnt + 1])
                            grads[0,beg: en].copy_(param.grad.data.view(-1))
                        cnt += 1

                    # 计算影响力
                    squared_norm = grads.square().sum()
                    infl[s_id, epoch] = float(squared_norm.detach().cpu().numpy() ** 0.5)

                # # delete caches
                # if self.args.stamp:
                #     del x_train, y_true, output, grads, x_enc, x_mark_enc, x_dec, x_mark_dec
                #     torch.cuda.empty_cache()
                # else:
                #     del x_train, y_true, output, grads
                #     torch.cuda.empty_cache()

            #更新模型参数

            for index, (x_train, y_true) in enumerate(self.train_set):
                # x_train = x_train.to(self.device)
                # y_true = y_true[:, :self.args.window * self.args.horizon]
                # y_true = y_true.to(self.device)
                # label = y_true[:, -self.args.window * self.args.horizon:]
                # x_train.requires_grad = True
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

                    output ,_= self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    output = output.squeeze(2)
                else:
                    y_true = y_true[:, :self.args.window * self.args.horizon]
                    label = y_true
                    output, _ = self.model(x_train, y_true, teacher_forcing_ratio=0)

                optimizer.zero_grad()
                b_loss = loss_fun(output.to(self.device), label)
                b_loss.backward(retain_graph=True)
                optimizer.step()

                # delete caches
                # if self.args.stamp:
                #     del x_train, y_true, output, grads, x_enc, x_mark_enc, x_dec, x_mark_dec
                #     torch.cuda.empty_cache()
                # else:
                #     del x_train, y_true, output, grads
                #     torch.cuda.empty_cache()

        # save
        fn = '%s/infl_grand_at_epoch_%02d_%s.dat' % (output_path, target_epoch, self.args.normalizer)
        joblib.dump(infl.cpu().numpy(), fn, compress=9)




    def run(self, epoch, K):
        # 在run的里面要重复k次，把每个样本的梯度给存下来，矩阵：每行是样本，K列是二范数，基于矩阵求均值. 求一个期望的值，一共循环k次，保证容错.
        def compute_loss_stateless_model(sample, target):
            # batch = len(sample)
            predictions = fmodel(sample, target)
            loss_fun = torch.nn.L1Loss().to(self.args.device)
            predictions = predictions[0]  # predictions : {32,192}
            loss = loss_fun(predictions, target)
            return loss

        len_train = len(self.train_set.dataset)
        value_mat = np.zeros((len_train, K))


        for num in range(K):

            # if epoch == 1:
            #
            #     with torch.no_grad():  # with条件下 必须执行 不计算梯度
            #          for index, (X, y_true) in enumerate(self.train_set):
            #             X = X.to(self.device)
            #             # X: Tensor:{32,288}
            #             outputs, _ = self.model(X, y_true)
            #             # outputs: Tensor:{32,192}

                # np.concatenate() 方法是一个用于连接多个数组的函数。该函数将多个数组按指定的轴连接起来，生成一个新的数组。连
                # 接操作可以沿着任意轴进行，即可以在水平方向、垂直方向或深度方向上连接数组。

            if epoch == -1 or epoch == 1:
                # fmodel, params, buffers = make_functional_with_buffers(self.model)
                # 使用 make_functional_with_buffers() 方法可以将一个 PyTorch 模块转换为函数式模块，从而可以获得更高的性能和更好的内存使用效率。
                fmodel = self.model
                fmodel.eval()

                # eval() 方法的作用是在评估阶段将模型设置为评估模式。该方法会递归地遍历模型的子模块，并对每个子模块调用 eval() 方法。在评估模式下，模型的 forward() 方法会被调用，但是不会计算梯度或更新权重。
                #
                # 在 PyTorch 中，可以通过调用 train() 方法将模型设置为训练模式。在训练模式下，模型的行为会恢复到训练状态，可以计算梯度并更新权重。通常，在每个 epoch 结束时，需要将模型设置为评估模式，以便在测试集上评估模型的性能。

                # def compute_loss_stateless_model(params, buffers, sample, target):
                #     batch = sample.unsqueeze(0)
                #     targets = target.unsqueeze(0)
                #
                #     predictions = fmodel(params, buffers, batch)
                #     loss = F.cross_entropy(predictions, targets)
                #     return loss


                ft_compute_grad = grad(compute_loss_stateless_model)
                # ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
                # vmap()
                # 函数会对输入的张量进行批处理，并将每个批次中的张量分别传递给函数进行操作。最终，vmap()
                # 函数将每个批次的输出张量打包成一个张量，并返回给调用方。

                print("Evaluating GRAND scores...")


                for index, (X, y_true) in enumerate(self.train_set):
                    # ft_per_sample_grads = ft_compute_sample_grad(X, y_true)
                    # X：{32，288}  y_true：{32，192} tensor
                    # train_set: {DataLoader:51}


                    ft_per_sample_grads = ft_compute_grad(X, y_true)
                    # ft_per_sample_grads : Tensor{32,288}
                    squared_norm = 0
                    for param_grad in ft_per_sample_grads:
                        squared_norm += param_grad.reshape(1,-1).square().sum()  # [8,9,10] [8,9*10]
                        # torch.normal

                    value_mat[index, num] = float(squared_norm.detach().cpu().numpy() ** 0.5)
                    # 要算每个样本的梯度，到时候放进矩阵中去求均值.
                    # [51,5]

        mat_grand = np.average(value_mat, axis=1)

        grand_scores = mat_grand

        return grand_scores



