#!/usr/bin/env python
# coding:utf-8

import os, sys, six, shutil, time
import pickle
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from utils.logger  import Logger
from utils.trainer import dataClass
import torch


def train(train_path, test_path, params, ae):

    ###----------------- setup -----------------###

    ### logger
    logger = Logger(params["outdir"], name=["loss"], loc=[1], log=True)
    if os.path.isdir(os.path.join(params["outdir"], "code")):
        shutil.rmtree(os.path.join(params["outdir"], "code"))
    shutil.copytree("../code_ae", os.path.join(params["outdir"], "code"))
    with open(os.path.join(params["outdir"], "code", "nn_params.pickle"), mode='wb') as f:
        pickle.dump(params, f)

    ### dataset
    traindata = dataClass(train_path, params["size"], params["dsize"], params["batch"], distort=True, test=False)
    testdata  = dataClass(test_path, params["size"], params["dsize"], params["batch"], distort=True, test=True)

    ### model, loss, optimizer
    model = ae().cuda(params["gpu"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(model.parameters()))

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
        sum_loss = 0.0
        traindata.minibatch_reset(rand=False)
        while traindata.loop:
            optimizer.zero_grad()
            traindata.minibatch_next()
            x_in, x_out = traindata()
            x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in))).cuda(params["gpu"])
            x_out = torch.autograd.Variable(torch.tensor(np.asarray(x_out))).cuda(params["gpu"])
            y = model(x_in)
            loss = criterion(y, x_out)
            sum_loss += float(loss.data) * len(x_in.data)
            loss.backward()
            optimizer.step()

        if (epoch%params["print_iter"])==0 or epoch==params["epoch"]:
            info_train = "train: {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum_loss/len(traindata))
            logger(info_train)

            ### test
            model.eval()
            sum_loss_test = 0.0
            testdata.minibatch_reset()

            while testdata.loop:
                testdata.minibatch_next()
                x_in, x_out = testdata()
                x_in = torch.autograd.Variable(torch.tensor(np.asarray(x_in))).cuda(params["gpu"])
                x_out = torch.autograd.Variable(torch.tensor(np.asarray(x_out))).cuda(params["gpu"])
                y = model(x_in)
                loss = criterion(y, x_out)
                sum_loss_test += float(loss.data) * len(x_in.data)

            info_test = "test:  {}/{} loss: {:.2e}".format(epoch, params["epoch"], sum_loss_test/len(testdata))
            logger(info_test)

        if (epoch%params["snap_iter"])==0 or epoch==params["epoch"]:
            logger.save_model(epoch, model, optimizer)
