#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/8/25 17:50
@desc: load datasets
"""
import numpy as np
import torch
import pickle
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os
import sys
import random

def MinMaxScaler(data):
    """Min Max normalizer.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
    """
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    norm_data = numerator / (denominator + 1e-7)
    return norm_data

def real_data_loading(data_name, seq_len):
    """Load and preprocess real-world datasets.

    Args:
      - data_name: stock or energy
      - seq_len: sequence length

    Returns:
      - data: preprocessed data.
    """
    assert data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)','Traffic', "FD001", "FD002", "FD003", "FD004"]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(current_dir, '../../')))
    if data_name == "etth1":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh1')
        cluster_0 = np.load(
            f'{fpath}/cluster_0.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/cluster_1.npy', allow_pickle=True
        )
        cluster_2 = np.load(
            f'{fpath}/cluster_2.npy', allow_pickle=True
        )


    elif data_name == "etth2":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/ETT/output/ETTh2')
        cluster_0 = np.load(
            f'{fpath}/cluster_0.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/cluster_1.npy', allow_pickle=True
        )
        cluster_2 = np.load(
            f'{fpath}/cluster_2.npy', allow_pickle=True
        )

       

    elif data_name == "AirQuality(bj)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(bj)/output')
        cluster_0 = np.load(
            f'{fpath}/cluster_0.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/cluster_1.npy', allow_pickle=True
        )
        cluster_2 = np.load(
            f'{fpath}/cluster_2.npy', allow_pickle=True
        )
        

    elif data_name == "AirQuality(Italian)":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/AirQuality(Italian)/output')
        cluster_0 = np.load(
            f'{fpath}/cluster_0.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/cluster_1.npy', allow_pickle=True
        )
        cluster_2 = np.load(
            f'{fpath}/cluster_2.npy', allow_pickle=True
        )
       
    elif data_name == "Traffic":
        fpath = os.path.join(ROOT_DIR, 'dataProcess/Traffic/output')
        cluster_0 = np.load(
            f'{fpath}/clear.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/clouds.npy', allow_pickle=True
        )
        cluster_2 = np.load(
            f'{fpath}/drizzle.npy', allow_pickle=True
        )
        cluster_3 = np.load(
            f'{fpath}/haze.npy', allow_pickle=True
        )
        cluster_4 = np.load(
            f'{fpath}/mist.npy', allow_pickle=True
        )
        cluster_5 = np.load(
            f'{fpath}/rain.npy', allow_pickle=True
        )
        cluster_6 = np.load(
            f'{fpath}/snow.npy', allow_pickle=True
        )
        rare_data = np.concatenate([cluster_2, cluster_3, cluster_4, cluster_5, cluster_6], axis=0)
        # 合并多量类和稀少类
        ori_data = np.concatenate([cluster_0, cluster_1, rare_data], axis=0)
        # 只取最后一列
        ori_data = ori_data[:, :, -1:]
        ori_data = np.array(ori_data).astype(float)

    elif data_name in ['FD001','FD002','FD004','FD003']:
        fpath = os.path.join(ROOT_DIR, 'dataProcess/C-MAPSS/output')
        cluster_0 = np.load(
            f'{fpath}/degraded.npy', allow_pickle=True
        )
        cluster_1 = np.load(
            f'{fpath}/normal.npy', allow_pickle=True
        )
        
        all_labels = np.load(
            f'{fpath}/all_labels.npy', allow_pickle=True
        )
        all_rul_values = np.load(
            f'{fpath}/all_rul_values.npy', allow_pickle=True
        )
        
        ori_data = np.concatenate([cluster_0, cluster_1])

    #少样本方案
    if data_name in ["etth1", "etth2", 'AirQuality(bj)', 'AirQuality(Italian)']:
        orig_data = np.concatenate([cluster_0 , cluster_1,cluster_2], axis=0)
        orig_data = MinMaxScaler(orig_data)
        cluster_0 = orig_data[:cluster_0.shape[0]]
        cluster_1 = orig_data[cluster_0.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]]
        cluster_2 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]:]
        
        c0_train = int(cluster_0.shape[0]*0.1)
        c1_train = int(cluster_1.shape[0]*0.1)
        c2_train = int(cluster_2.shape[0]*0.1)
            
        c0_test = int(cluster_0.shape[0]*0.2)
        c1_test = int(cluster_1.shape[0]*0.2)
        c2_test = int(cluster_2.shape[0]*0.2)
            
        c0_test_idx = random.sample(range(c0_train*2, cluster_0.shape[0]), c0_test)
        c1_test_idx = random.sample(range(c1_train*2, cluster_1.shape[0]), c1_test)
        c2_test_idx = random.sample(range(c2_train*2, cluster_2.shape[0]), c2_test)
            
        train_data = np.concatenate([cluster_0[:c0_train] , cluster_1[:c1_train],cluster_2[:c2_train]], axis=0)
        val_data = np.concatenate([cluster_0[c0_train:c0_train*2] , cluster_1[c1_train:c1_train*2],cluster_2[c2_train:c2_train*2]], axis=0)
        test_data = np.concatenate([cluster_0[c0_test_idx] , cluster_1[c1_test_idx],cluster_2[c2_test_idx]], axis=0)
    elif data_name == "Traffic":
        orig_data = np.concatenate([cluster_0 , cluster_1,cluster_2, cluster_3, cluster_4, cluster_5, cluster_6], axis=0)
        orig_data = MinMaxScaler(orig_data)
        cluster_0 = orig_data[:cluster_0.shape[0]]
        cluster_1 = orig_data[cluster_0.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]]
        cluster_2 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]]
        cluster_3 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]]
        cluster_4 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]+cluster_4.shape[0]]
        cluster_5 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]+cluster_4.shape[0]:cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]+cluster_4.shape[0]+cluster_5.shape[0]]
        cluster_6 = orig_data[cluster_0.shape[0]+cluster_1.shape[0]+cluster_2.shape[0]+cluster_3.shape[0]+cluster_4.shape[0]+cluster_5.shape[0]:]
        
        c0_train = int(cluster_0.shape[0]*0.1)
        c1_train = int(cluster_1.shape[0]*0.1)
        c2_train = int(cluster_2.shape[0]*0.1)
        c3_train = int(cluster_3.shape[0]*0.1)
        c4_train = int(cluster_4.shape[0]*0.1)
        c5_train = int(cluster_5.shape[0]*0.1)
        c6_train = int(cluster_6.shape[0]*0.1)
        
        c0_test_idx = random.sample(range(c0_train*2, cluster_0.shape[0]), c0_train*2)
        c1_test_idx = random.sample(range(c1_train*2, cluster_1.shape[0]), c1_train*2)
        c2_test_idx = random.sample(range(c2_train*2, cluster_2.shape[0]), c2_train*2)
        c3_test_idx = random.sample(range(c3_train*2, cluster_3.shape[0]), c3_train*2)
        c4_test_idx = random.sample(range(c4_train*2, cluster_4.shape[0]), c4_train*2)
        c5_test_idx = random.sample(range(c5_train*2, cluster_5.shape[0]), c5_train*2)
        c6_test_idx = random.sample(range(c6_train*2, cluster_6.shape[0]), c6_train*2)
        
        train_data = np.concatenate([cluster_0[:c0_train] , cluster_1[:c1_train],cluster_2[:c2_train],
                                     cluster_3[:c3_train],cluster_4[:c4_train],cluster_5[:c5_train],cluster_6[:c6_train]], axis=0)
        val_data = np.concatenate([cluster_0[c0_train:c0_train*2] , cluster_1[c1_train:c1_train*2],cluster_2[c2_train:c2_train*2],
                                   cluster_3[c3_train:c3_train*2],cluster_4[c4_train:c4_train*2],cluster_5[c5_train:c5_train*2],cluster_6[c6_train:c6_train*2]], axis=0)
        test_data = np.concatenate([cluster_0[c0_test_idx] , cluster_1[c1_test_idx],cluster_2[c2_test_idx],
                                    cluster_3[c3_test_idx],cluster_4[c4_test_idx],cluster_5[c5_test_idx],cluster_6[c6_test_idx]], axis=0)
        
    elif data_name in ['FD001','FD002','FD004','FD003']:
        orig_data = np.concatenate([cluster_0 , cluster_1], axis=0)
        orig_data = MinMaxScaler(orig_data)
        cluster_0 = orig_data[:cluster_0.shape[0]]
        cluster_1 = orig_data[cluster_0.shape[0]:]
        
        c0_train = int(cluster_0.shape[0]*0.1)
        c1_train = int(cluster_1.shape[0]*0.1)
            
        c0_test = int(cluster_0.shape[0]*0.2)
        c1_test = int(cluster_1.shape[0]*0.2)
            
        c0_test_idx = random.sample(range(c0_train*2, cluster_0.shape[0]), c0_test)
        c1_test_idx = random.sample(range(c1_train*2, cluster_1.shape[0]), c1_test)
            
        train_data = np.concatenate([cluster_0[:c0_train] , cluster_1[:c1_train]], axis=0)
        val_data = np.concatenate([cluster_0[c0_train:c0_train*2] , cluster_1[c1_train:c1_train*2]], axis=0)
        test_data = np.concatenate([cluster_0[c0_test_idx] , cluster_1[c1_test_idx]], axis=0)   
            
    print(train_data.shape, val_data.shape, test_data.shape)
    
    return train_data, val_data, test_data


def load_st_dataset(dataset, feature):

    if dataset == 'electricity':
        df_raw = pd.read_csv('./data/electricity/electricity.csv')

    if dataset == 'powerLoad':
        df_raw = pd.read_csv('./data/powerLoad/NYPowerLoad.csv')

    if dataset == 'etth1':
        df_raw = pd.read_csv('./data/ETTh1/ETTh1.csv')

    if dataset == 'etth2':
        df_raw = pd.read_csv('./data/ETTh2/ETTh2.csv')

    if dataset == 'PEMS04':
        df_raw = pd.read_csv('./data/PEMS04/PEMS04.csv')

    if dataset == 'traffic':
        df_raw = pd.read_csv('./data/traffic/traffic.csv')

    if dataset == 'exchange_rate':
        df_raw = pd.read_csv('./data/exchange_rate/exchange_rate.csv')


    target = 'OT'
    tStr = 'date'

    if feature == 'S':
        df_data = df_raw[target].values
        df_dTime = df_raw[tStr]
    elif feature == 'MS' or feature == 'M':
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data].values
        df_dTime = df_raw[tStr]

    data = df_data
    dTime = df_dTime
    dTime.columns = ['date']
    print('prepare data has done!')
    return torch.tensor(data), pd.DataFrame(dTime)



if __name__ == '__main__':
    station_name = 'JSFD001'
    start_time = '20190131'
    data = load_st_dataset('powerload')
    print('prepare data has done!')