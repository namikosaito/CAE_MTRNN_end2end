#!/usr/bin/env python
# coding:utf-8

import os, sys
import getpass
import numpy as np
import _pickle as pickle
from model.rnn import RNN as rnn
from src.rnn_learn import train
from src.rnn_test  import test

args = sys.argv
mode = args[1]

### dataset
data_path = "../results_ae/test_torobo_ae/snap/00020_mid"
train_path = os.path.join(data_path, "train.pickle")
test_path = os.path.join(data_path, "test.pickle")

if mode=="train":
    angle_dim = 8
    image_dim = 30
    log = "../results_rnn/test_torobo_rnn"
    resume  = ""
    if resume:
        log  = resume.rstrip(".tar")

    ### name
    name_nodes  = ["mot", "img"]
    split_nodes = {"mot":angle_dim, "img":image_dim}
    input_param = {"mot":1.0, "img":1.0}
    input_param_test = {"mot":1.0, "img":1.0}

    ### param
    c_size = {"cf":30, "cs":5}
    tau = {"io":2.0, "cf":5.0, "cs":30.0}
    nn_params = {"tau":tau, 
                 "name_node":name_nodes, 
                 "input_param":input_param, 
                 "input_param_test":input_param_test, 
                 "gpu":-1, 
                 "epoch":500, 
                 "print_iter":10, 
                 "snap_iter":50, 
                 "c_size":c_size, 
                 "outdir":log, 
                 "split_node":split_nodes,
                 "train":train_path,
                 "test":test_path, 
                 "resume":resume}
    ### train
    train(nn_params, rnn)


elif mode=="test":
    ### path
    resume = "../results_rnn/test_torobo_rnn/snap/00500.tar"
    with open(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/nn_params.pickle"), "rb") as f:
        nn_params = pickle.load(f)
    nn_params["resume"] = resume
    nn_params["input_param_test"] = {"mot":0.0, "img":1.0}
    test(nn_params, rnn, pca_anim=False)


