#!/usr/bin/env python
# encoding: utf-8
"""
@author: jimapp
@time: 2021/5/15 22:19
@desc: test model
"""
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

class mLSTM(torch.nn.Module):
    def __init__(self, args):
        super(mLSTM, self).__init__()
        self.enc_seq_len = args.seq_len
        self.pre_len = args.pred_len
        self.lstm_enc_in =args.enc_in
        self.lstm_out_dim = 128
        self.c_out = args.c_out
        self.dataset = args.dataset
        self.encode = nn.LSTM(self.lstm_enc_in, self.lstm_out_dim, self.c_out)
        self.decoder = nn.LSTM(self.lstm_out_dim, self.lstm_out_dim, self.c_out)

        self.fc = nn.Linear(self.lstm_out_dim, self.pre_len*self.c_out)
        self.activate = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        sample_num = x.shape[0]
        # x = x.unsqueeze(2)
        h0 = torch.randn(self.c_out, sample_num, self.lstm_out_dim).to(x.device)
        c0 = torch.randn(self.c_out, sample_num, self.lstm_out_dim).to(x.device)
        
        if self.dataset not in ['FD001', 'FD002','FD003','FD004']:
            seq_last = x[:,-1,:].unsqueeze(1)
            x = x - seq_last
            
        x = x.transpose(0,1)
        out, (hn, cn) = self.encode(x, (h0, c0))
        out, (hn, cn) = self.decoder(out, (hn, cn))

        
        if self.dataset in ['AirQuality(bj)', 'AirQuality(Italian)']:
            out = self.fc(out[-1].reshape(sample_num, -1)) #共享一个fc
            out = self.drop(out)
            out = out.reshape(sample_num, self.pre_len, self.c_out)
        elif self.dataset in ['FD001', 'FD002','FD003','FD004']:
            out = self.fc(out[-1].reshape(sample_num, -1)) #共享一个fc
            out = self.drop(out)
            out = out.reshape(sample_num, self.pre_len, self.c_out)      
            
            return out
              
        elif self.dataset in ['etth1','etth2', 'Traffic']:            
            out = self.fc(out[-1]) #共享一个fc
            out = self.drop(out)
            out.unsqueeze_(2)

        if self.dataset not in ['FD001', 'FD002', 'FD003', 'FD004']:
            out = out + seq_last

        return out