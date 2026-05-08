# -*- coding: utf-8 -*-
"""
@author: jimapp
@time: 2022/7/15 21:55
@desc:
"""
import os
import sys
import torch

file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(file_dir)

import pandas as pd
import numpy as np
import argparse
import configparser
from datetime import datetime
import torch.nn as nn
from TSlib.others.attn_lstm import Attn_LSTM
from TSlib.others.mLSTM import mLSTM
from TSlib.lib.dataloader import get_dataloader

import time
from TSlib.lib.BasicTrainer_sw import Trainer
from TSlib.lib.dataloader import data_loader
from TSlib.lib.addnoise import add_noise2

from torch.utils.data.dataloader import DataLoader
from TSlib.exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from TSlib.exp.exp_imputation import Exp_Imputation
from TSlib.exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from TSlib.exp.exp_anomaly_detection import Exp_Anomaly_Detection
from TSlib.exp.exp_classification import Exp_Classification
import random
import pickle

today = time.time()


def unfreeze_parm(model, f_name):
    for name, p in model.named_parameters():
        # print(name, param.size())
        p.requires_grad = False
        if name in f_name:
            p.requires_grad = True


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


today = time.time()
for tt in range(3):
    # 设置随机数种子

    seed_num = 2026 + tt * 100
    setup_seed(int(seed_num))

    print(f'--------------{tt}--------------------inter----')
    for DATASET in ['etth1', 'etth2','AirQuality(bj)','AirQuality(Italian)','Traffic','FD001']:
        print(DATASET)
        if DATASET in ['Traffic']:
            continue
        Mode = 'Train'  # Train or test
        DEBUG = 'True'
        optim = 'adam'
        DEVICE = 'cuda:0'
        MODEL = 'FEDformer'  # TimesNet, DLinear, LSTM, FEDformer
        ktype = 'normal'
        noise_ratio = 0  # 0 0.3

        task_name = 'long_term_forecast'
        finish_time = 1662287958.9541638
        # config_file
        config_file = 'configs/{}/{}.conf'.format(DATASET, MODEL)  # 每个文件input_dim, output_dim不同
        config = configparser.ConfigParser()
        config.read(config_file)

        from TSlib.lib.metrics import MAE_torch


        def masked_mae_loss(scaler, mask_value):
            def loss(preds, labels):
                if scaler:
                    preds = scaler.inverse_transform(preds)
                    labels = scaler.inverse_transform(labels)
                mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
                return mae

            return loss


        # parser
        args = argparse.ArgumentParser(description='arguments')

        # basic config
        args.add_argument('--task_name', type=str, default=task_name,
                          help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
        args.add_argument('--is_training', type=int, default=1, help='status')
        args.add_argument('--model_id', type=str, default='test', help='model id')
        args.add_argument('--model', type=str, default=MODEL,
                          help='model name, options: [Autoformer, Transformer, TimesNet]')
        args.add_argument('--dataset', default=DATASET, type=str)
        args.add_argument('--data', type=str, default='m4', help='dataset type')
        args.add_argument('--mode', default=Mode, type=str)
        args.add_argument('--optim', default=optim, type=str)
        args.add_argument('--device', default=DEVICE, type=str, help='indices of GPUs')
        # args.add_argument('--debug', default=DEBUG, type=eval)
        # args.add_argument('--model', default=MODEL, type=str)
        # args.add_argument('--cuda', default=True, type=bool)

        # data
        args.add_argument('--val_ratio', default=config['data']['val_ratio'], type=float)
        args.add_argument('--test_ratio', default=config['data']['test_ratio'], type=float)
        args.add_argument('--normalizer', default=config['data']['normalizer'], type=str)
        args.add_argument('--stamp', default=config['data']['stamp'], type=bool)
        args.add_argument('--freq', type=str, default=config['data']['freq'],
                          help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
        # model
        args.add_argument('--top_k', type=int, default=config['model']['top_k'], help='for TimesBlock')
        args.add_argument('--num_kernels', type=int, default=config['model']['num_kernels'], help='for Inception')
        args.add_argument('--dec_in', default=config['model']['dec_in'], type=int)
        args.add_argument('--enc_in', default=config['model']['enc_in'], type=int)
        args.add_argument('--c_out', type=int, default=config['model']['c_out'], help='output size')
        args.add_argument('--d_model', default=config['model']['d_model'], type=int)
        args.add_argument('--n_heads', type=int, default=config['model']['n_heads'], help='num of heads')
        args.add_argument('--d_ff', type=int, default=config['model']['d_ff'], help='dimension of fcn')
        args.add_argument('--moving_avg', type=int, default=config['model']['moving_avg'],
                          help='window size of moving average')
        args.add_argument('--factor', type=int, default=config['model']['factor'], help='attn factor')
        args.add_argument('--embed', type=str, default=config['model']['embed'],
                          help='time features encoding, options:[timeF, fixed, learned]')
        args.add_argument('--dropout', type=float, default=config['model']['dropout'], help='dropout')
        args.add_argument('--timeenc', type=int, default=config['model']['timeenc'], help='dropout')

        # args.add_argument('--rnn_units', default=config['model']['rnn_units'], type=int)
        # args.add_argument('--num_layers', default=config['model']['num_layers'], type=int)
        args.add_argument('--e_layers', type=int, default=config['model']['e_layers'], help='num of encoder layers')
        args.add_argument('--d_layers', type=int, default=config['model']['d_layers'], help='num of decoder layers')

        # args.add_argument('--layer_size', default=config['model']['layer_size'], type=int)
        # args.add_argument('--res_channels', default=config['model']['res_channels'], type=int)
        # args.add_argument('--skip_channels', default=config['model']['skip_channels'], type=int)
        args.add_argument('--column_wise', default=config['model']['column_wise'], type=bool)
        args.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
        args.add_argument('--activation', type=str, default='gelu', help='activation')
        args.add_argument('--distil', action='store_false',
                          help='whether to use distilling in encoder, using this argument means not using distilling',
                          default=True)
        # train

        args.add_argument("--test_action", action='store_true')
        args.add_argument('--loss_func', default=config['train']['loss_func'], type=str)
        args.add_argument('--seed', default=config['train']['seed'], type=int)
        args.add_argument('--batch_size', default=config['train']['batch_size'], type=int)
        args.add_argument('--epochs', default=config['train']['epochs'], type=int)
        args.add_argument('--lr_init', default=config['train']['lr_init'], type=float)
        args.add_argument('--lr_decay', default=config['train']['lr_decay'], type=eval)
        args.add_argument('--lr_decay_rate', default=config['train']['lr_decay_rate'], type=float)
        args.add_argument('--lr_decay_step', default=config['train']['lr_decay_step'], type=str)
        args.add_argument('--early_stop', default=config['train']['early_stop'], type=eval)
        args.add_argument('--early_stop_patience', default=config['train']['early_stop_patience'], type=int)
        args.add_argument('--grad_norm', default=config['train']['grad_norm'], type=eval)
        args.add_argument('--max_grad_norm', default=config['train']['max_grad_norm'], type=int)
        args.add_argument('--teacher_forcing', default=config['train']['teacher_forcing'], type=eval)
        args.add_argument('--tf_decay_steps', default=config['train']['tf_decay_steps'], type=int,
                          help='teacher forcing decay steps')
        args.add_argument('--real_value', default=config['train']['real_value'], type=eval,
                          help='use real value for loss calculation')
        # test
        args.add_argument('--mae_thresh', default=config['test']['mae_thresh'], type=eval)
        args.add_argument('--mape_thresh', default=config['test']['mape_thresh'], type=float)
        # log
        # args.add_argument('--log_dir', default='./', type=str)
        args.add_argument('--log_step', default=config['log']['log_step'], type=int)
        # args.add_argument('--plot', default=config['log']['plot'], type=eval)

        # forecasting task
        args.add_argument('--seq_len', default=config['data']['seq_len'], type=int)
        args.add_argument('--pred_len', default=config['data']['pred_len'], type=int)
        # args.add_argument('--window', default=config['data']['window'], type=int)
        # args.add_argument('--interval', default=config['data']['interval'], type=int)
        # args.add_argument('--horizon', default=config['data']['horizon'], type=int)
        # args.add_argument('--label_len', type=int, default=config['data']['label_len'], help='start token length')

        # GPU
        args.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
        args.add_argument('--gpu', type=int, default=0, help='gpu')
        args.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
        args.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

        args = args.parse_args()

        ######################GPU设置#################################
        if args.device == 'cpu':
            args.device = 'cpu'
            args.use_gpu = False
        elif args.device == 'cuda:0':
            if torch.cuda.is_available():
                torch.cuda.set_device(int(args.device[5]))
                args.use_gpu = True
            else:
                args.device = 'cpu'
                args.use_gpu = False

        args.stamp = True
        args.ktype = ktype
        args.noise_ratio = noise_ratio

        if DATASET in ['etth1', 'etth2']:
            # ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'， 'OT' ]
            # 预测 'OT'
            feature = 'S'
            args.enc_in = 1
            args.dec_in = 1
            args.c_out = 1

        elif DATASET in ['AirQuality(bj)']:
            # ['TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM', 'wind_sin', 'wind_cos', 'AQI'，'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            # 预测['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
            feature = 'M'
            args.enc_in = 1
            args.dec_in = 1
            args.c_out = 1
        elif DATASET in ['AirQuality(Italian)']:
            # 使用所有保留特征  ['CO(GT)', 'PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)', 'T', 'RH', 'AH', 'CO_IAQI', 'NO2_IAQI', 'AQI']
            # 预测['PT08.S1(CO)', 'C6H6(GT)', 'PT08.S2(NMHC)', 'NOx(GT)', 'PT08.S3(NOx)', 'NO2(GT)', 'PT08.S4(NO2)','PT08.S5(O3)']
            feature = 'M'
            args.enc_in = 2
            args.dec_in = 2
            args.c_out = 2
        elif DATASET in ['Traffic']:
            feature = 'S'
            args.enc_in = 1
            args.dec_in = 1
            args.c_out = 1
        elif DATASET in ['FD001']:
            # 14个特征，预测1个RUL
            feature = 'MS'
            args.enc_in = 15
            args.dec_in = 15
            args.c_out = 1

        ##############执行任务######################################
        for hh in [16]:
            # 关键修改2：给horizon赋值（覆盖配置文件的值，保持和原逻辑一致）
            if hh == 0:
                args.horizon = 1
                args.window = 1
            else:
                args.horizon = hh
                args.window = 6

            # ##############变化######################################
            # args.seq_len = args.interval * args.lag  # 144
            # args.pred_len = args.window * args.horizon

            if os.path.exists(f'./data/{DATASET}/val_loader_{tt}.pkl') and \
                    os.path.exists(f'./data/{DATASET}/test_loader_{tt}.pkl'):
                train_loader = pickle.load(open(f'./data/{DATASET}/train_loader_{tt}.pkl', "rb"))
                val_loader = pickle.load(open(f'./data/{DATASET}/val_loader_{tt}.pkl', "rb"))
                test_loader = pickle.load(open(f'./data/{DATASET}/test_loader_{tt}.pkl', "rb"))
            else:
                train_loader, val_loader, test_loader = get_dataloader(args, feature=feature)

                if not os.path.exists(f'./data/{DATASET}'):
                    os.makedirs(f'./data/{DATASET}')

                with open(f'./data/{DATASET}/train_loader_{tt}.pkl', 'wb') as fh:
                    pickle.dump(train_loader, fh)

                with open(f'./data/{DATASET}/val_loader_{tt}.pkl', 'wb') as fh:
                    pickle.dump(val_loader, fh)
                with open(f'./data/{DATASET}/test_loader_{tt}.pkl', 'wb') as fh:
                    pickle.dump(test_loader, fh)

            # config log path
            current_time = datetime.now().strftime('%Y%m%d%H%M%S')
            current_dir = os.path.dirname(os.path.realpath(__file__))
            log_dir = os.path.join(current_dir, 'experiments', args.dataset, current_time)
            args.log_dir = log_dir

            #######################################################
            if args.task_name == 'long_term_forecast':
                Exp = Exp_Long_Term_Forecast
            elif args.task_name == 'short_term_forecast':
                Exp = Exp_Short_Term_Forecast
            elif args.task_name == 'imputation':
                Exp = Exp_Imputation
            elif args.task_name == 'anomaly_detection':
                Exp = Exp_Anomaly_Detection
            elif args.task_name == 'classification':
                Exp = Exp_Classification
            else:
                Exp = Exp_Long_Term_Forecast

            # 生成模型
            if MODEL in ['TimesNet', 'Autoformer', 'Transformer', 'Nonstationary_Transformer', 'DLinear',
                         'FEDformer', 'Informer', 'LightTS', 'Reformer', 'ETSformer', 'PatchTST', 'Pyraformer',
                         'MICN', 'Crossformer', 'FiLM']:
                exp = Exp(args)  # set experiments
                # exp.model_dict[args.model].Model(args).float()
                model = exp.model
            elif MODEL in ['lstm-att']:
                model = Attn_LSTM(args)
            elif MODEL in ['LSTM']:
                model = mLSTM(args)

            for p in model.parameters():
                if p.dim() > 1:
                    # nn.init.xavier_uniform_(p)
                    nn.init.kaiming_uniform_(p, mode='fan_in', nonlinearity='relu')
                else:
                    nn.init.uniform_(p)

            # init loss function, optimizer
            if args.loss_func == 'mask_mae':
                loss = masked_mae_loss(None, mask_value=0.0)
            elif args.loss_func == 'mae':
                loss = torch.nn.L1Loss().to(args.device)
            elif args.loss_func == 'mse':
                loss = torch.nn.MSELoss().to(args.device)
            else:
                raise ValueError

            # directory = './temp'
            quality = torch.tensor(1)
            if ktype in ['normal', 'sadc', 'cagrad', 'pcgrad']:
                # model setup
                if args.optim == 'sgd':
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_init)
                elif args.optim == 'adam':
                    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr_init, eps=1.0e-8,
                                                 weight_decay=0, amsgrad=False)

                # optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr_init, weight_decay=0)
                # learning rate decay
                lr_scheduler = None
                if args.lr_decay:
                    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode='min',  # 监控指标：'min' 表示指标越小越好（如 Loss）
                        factor=0.5,  # 缩减倍数：LR = LR * 0.5
                        patience=5,  # 容忍度：如果 5 个 Epoch 指标都没下降，就触发缩减
                        # verbose=True,  # 触发时打印消息
                        min_lr=1e-6,  # 学习率下限
                        threshold=1e-4  # 判断改进的阈值
                    )
                    print('Applying learning rate decay.')
                    # lr_decay_steps = [int(i) for i in list(args.lr_decay_step.split(','))]
                    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                    #                                                     milestones=lr_decay_steps,
                    #                                                     gamma=args.lr_decay_rate)
                # start training
                trainer = Trainer(ktype, model, loss, optimizer, train_loader, val_loader, test_loader, None,
                                  args, lr_scheduler=lr_scheduler)

                if args.mode == 'Train':
                    trainer.train()
                elif args.mode == 'test':
                    args.dataset = DATASET
                    model.load_state_dict(torch.load('./experiments/best_model_{}_{}_{}_{}_zc.pth'.format(
                        args.model,
                        args.dataset,
                        args.horizon,
                        finish_time
                    )))

                    train_loader_orig, val_loader, test_loader = get_dataloader(args,
                                                                                normalizer=args.normalizer,
                                                                                feature=feature)

                    with open(f'./data/{DATASET}/test/val_loader.pkl', 'wb') as fh:
                        pickle.dump(val_loader, fh)
                    with open(f'./data/{DATASET}/test/test_loader.pkl', 'wb') as fh:
                        pickle.dump(test_loader, fh)
                        # init model
                    args.stamp = True
                    # mae, rmse, mape = self.test(self.model, self.args, self.test_loader, self.scaler,
                    #                             finish_time=finish_time)
                    mae, rmse, mape = trainer.test(model, trainer.args, test_loader, None, finish_time=finish_time)

                    path = f'{args.dataset}_{args.model}_4_6_{finish_time}_zc1'
                    dataset = args.dataset

                    wind_pred = np.load('./results/{}/{}_pred.npy'.format(path, args.dataset))
                    wind_true = np.load('./results/{}/{}_true.npy'.format(path, args.dataset))
                    pd.DataFrame(
                        np.concatenate([wind_true[:, 2:].reshape(-1, 1), wind_pred[:, 2:].reshape(-1, 1)], axis=1),
                        columns=['true', 'pred']).to_csv(f'./results/{path}/scsq_{dataset}_true_pred_2.csv')
                    pd.DataFrame(np.concatenate([wind_true.reshape(-1, 1), wind_pred.reshape(-1, 1)], axis=1),
                                 columns=['true', 'pred']).to_csv(f'./results/{path}/scsq_{dataset}_true_pred_1.csv')
                    wind_pred[:, :1] = wind_pred[:, :1] / 2
                    pd.DataFrame(np.concatenate([wind_true.reshape(-1, 1), wind_pred.reshape(-1, 1)], axis=1),
                                 columns=['true', 'pred']).to_csv(f'./results/{path}/scsq_{dataset}_true_pred.csv')
                    # pd.DataFrame(wind_pred.reshape(-1,1)).to_csv(f'./results/{path}/scsq_{dataset}_pred.csv')

                # continue
                output_path = './ns_results/%02d' % (today)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                filename = f'{args.dataset}_{args.model}_{ktype}_{optim}_{args.early_stop_patience}_ns_{args.noise_ratio}_{args.pred_len}_{today}.csv'

                if tt == 0:
                    trainer.results.to_csv(
                        f'{output_path}/{filename}',
                        mode='a',
                        header=True
                    )
                else:
                    trainer.results.to_csv(
                        f'{output_path}/{filename}',
                        mode='a',
                        header=False
                    )
            else:
                pass

        del train_loader, val_loader, test_loader
        torch.cuda.empty_cache()