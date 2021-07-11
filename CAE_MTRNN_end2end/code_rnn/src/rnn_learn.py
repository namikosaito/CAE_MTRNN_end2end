#!/usr/bin/env python
# coding:utf-8

import os, sys, six, shutil
import _pickle as pickle
from tqdm import tqdm
import numpy as np
from utils.logger import Logger
import torch
import torch.nn.functional as F
from torch.autograd import Variable


ignore_value = -999


def train_partical(model, criterion, in_data, out_data, input_param, sp, n_name):
    loss = [0]*(len(n_name))
    for ts in range(in_data.shape[1]):
        mask = np.where(out_data[:,ts,0] != ignore_value)[0]
        x_ndarray = np.array(in_data[:,ts])
        t_ndarray = np.array(out_data[:,ts])
        x_ndarray = torch.tensor(x_ndarray)
        t_ndarray = torch.tensor(t_ndarray)
        if ts != 0:
            prev_out = y.data
            num = 0
            sp_ = sp[n_name[num]]
            for i in range(x_ndarray.shape[1]):
                if i < sp_:
                    ip = input_param[n_name[num]]
                    x_ndarray[:,i] = ip * x_ndarray[:,i] + (1.0-ip) * prev_out[:,i]
                else:
                    num += 1
                    ip = input_param[n_name[num]]
                    x_ndarray[:,i] = ip * x_ndarray[:,i] + (1.0-ip) * prev_out[:,i]
                    sp_  += sp[n_name[num]]
        x = Variable(x_ndarray)
        t = Variable(t_ndarray)
        y, cf, cs, cf_inter, cs_inter = model.forward(x, ts)

        ### loss
        if len(mask)!=0:
            d1 = 0
            for i,name in enumerate(n_name):
                d2 = d1 + sp[name]
                loss[i] += criterion(y[mask,d1:d2], t[mask,d1:d2])
                d1 = d2
    return loss


def train(params, rnn, lqr=False, lqrstep=5):
    ###----------------- setup -----------------###
    logger = Logger(params["outdir"], name=["loss"]+params["name_node"], 
                    loc=[1]*(len(params["name_node"])+1))
    if os.path.isdir(os.path.join(params["outdir"], "code")):
        shutil.rmtree(os.path.join(params["outdir"], "code"))
    shutil.copytree("../code_rnn", os.path.join(params["outdir"], "code"))
    with open(os.path.join(params["outdir"], "code", "nn_params.pickle"), mode='wb') as f:
        pickle.dump(params, f)

    ### dataset
    with open(params["train"], "rb") as f:
        dataset_train = pickle.load(f)
    with open(params["test"], "rb") as f:
        dataset_test = pickle.load(f)
    teach_in = dataset_train[:,:-1,:]
    teach_out = dataset_train[:,1:,:]
    test_in = dataset_test[:,:-1,:]
    test_out = dataset_test[:,1:,:]
    #print(teach_in, teach_in.shape)

    ### model, loss, optimizer
    _, steps, insize = teach_in.shape
    model = rnn(insize, insize, params["c_size"], params["tau"])
    model.initialize_c_state(rand=False, cf=True, cs=False)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    ### load
    if params["resume"]:
        checkpoint = torch.load(params["resume"])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    ###----------------- learn -----------------###
    pbar = tqdm(six.moves.range(1,params["epoch"]+1),total=params["epoch"],desc="epoch",ascii=True)
    for epoch in pbar:
        ### train
        model.train()
        optimizer.zero_grad()
        loss = train_partical(model, criterion, teach_in, teach_out, 
               params["input_param"], params["split_node"], params["name_node"])
        #print(sum(loss), type(torch.Tensor(loss)))
        sum(loss).backward()#torch.Tensor(sum(loss)).backward()
        optimizer.step()

        if (epoch%params["print_iter"])==0 or epoch==params["epoch"]:
            info_train = "train: {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum(loss).data/steps)
            for i,name in enumerate(params["name_node"]):
                info_train += " {}: {:.2e}".format(name, loss[i].data/steps)
            logger(info_train)

            ### test
            model.eval()
            loss = train_partical(model, criterion, test_in, test_out, 
                   params["input_param_test"], params["split_node"], params["name_node"])
            info_test = "test:  {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum(loss).data/steps)
            for i,name in enumerate(params["name_node"]):
                info_test += " {}: {:.2e}".format(name, loss[i].data/steps)
            logger(info_test)

        if (epoch%params["snap_iter"])==0 or epoch==params["epoch"]:
            logger.save_model(epoch, model, optimizer)

    print (model.init_cf)
    print (model.init_cs)

