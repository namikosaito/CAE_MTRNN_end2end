#!/usr/bin/env python
# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RNN(nn.Module):
    def __init__(self, in_size=1, out_size=1, c_size=None, tau=None, variance=False):
        super(RNN, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.c_size = {'cf':5, 'cs':5} if c_size is None else c_size
        self.tau = {'tau_cf':5.0,'tau_cs':70.0} if tau is None else tau
        self.i2cf  = nn.Linear(self.in_size, self.c_size['cf'])
        self.cf2cs = nn.Linear(self.c_size['cf'], self.c_size['cs'])
        self.cf2cf = nn.Linear(self.c_size['cf'], self.c_size['cf'])
        self.cs2cf = nn.Linear(self.c_size['cs'], self.c_size['cf'])
        self.cs2cs = nn.Linear(self.c_size['cs'], self.c_size['cs'])
        self.cf2o  = nn.Linear(self.c_size['cf'], self.out_size)


    def initialize_c_state(self, rand=False, cf=False, cs=False):
        self.init_cf = torch.zeros(self.c_size["cf"])
        self.init_cs = torch.zeros(self.c_size["cs"])
        if rand:
            self.init_cf = torch.randn(self.c_size["cf"])
            self.init_cs = torch.randn(self.c_size["cs"])
        if cf:
            self.init_cf = nn.Parameter(self.init_cf)
        if cs:
            self.init_cs = nn.Parameter(self.init_cs)


    def forward(self, x, ts=999):
        ### fast context
        if ts==0:
            self.cf_state = F.tanh(self.init_cf)
            self.cs_state = F.tanh(self.init_cs)
            self.cf_inter = (1.0-1.0/self.tau['cf']) * self.init_cf + (1.0/self.tau['cf']) \
                            * (self.i2cf(x) + self.cf2cf(self.cf_state) + self.cs2cf(self.cs_state))
        else:
            self.cf_inter = (1.0-1.0/self.tau['cf']) * self.cf_inter + (1.0/self.tau['cf']) \
                            * (self.i2cf(x) + self.cf2cf(self.cf_state) + self.cs2cf(self.cs_state))
        self.cf_state = F.tanh(self.cf_inter)

        ### slow context
        if ts==0:
            self.cs_inter = (1.0-1.0/self.tau['cs']) * self.init_cs + (1.0/self.tau['cs']) \
                            * (self.cf2cs(self.cf_state) + self.cs2cs(self.cs_state))
        else:
            self.cs_inter = (1.0-1.0/self.tau['cs']) * self.cs_inter + (1.0/self.tau['cs']) \
                            * (self.cf2cs(self.cf_state) + self.cs2cs(self.cs_state))
        self.cs_state = F.tanh(self.cs_inter)

        ### output
        y = F.tanh(self.cf2o(self.cf_state))

        return y, self.cf_state, self.cs_state, self.cf_inter, self.cs_inter


