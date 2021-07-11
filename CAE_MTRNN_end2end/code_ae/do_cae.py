#!/usr/bin/env python
# coding:utf-8

import os, sys, glob, shutil
import pickle
from model.cae import CAE as ae
from src.cae_learn import train
from src.cae_eval  import reconstract, extract

args = sys.argv
mode = args[1]
dirpath = "../dataset/torobo_data"


### dataset
train_path = []
test_path = []
for dir in glob.glob(os.path.join(dirpath,"*")):
    if "train_" in dir:
        train_path.append(os.path.join(dir,"imglist.dat"))
    elif "test_" in dir:
        test_path.append(os.path.join(dir,"imglist.dat"))

if mode=="train":
    log = "../results_ae/test_torobo_ae"
    resume = ""
    if resume:
        log  = resume.rstrip(".tar")
    nn_params = {"gpu":0,
                 "batch":124,
                 "size":128,
                 "dsize":10,
                 "epoch":20,
                 "print_iter":2,
                 "snap_iter": 5,
                 "outdir":log,
                 "train":train_path,
                 "test":test_path,
                 "resume":resume}
    train(train_path, test_path, nn_params, ae)


elif mode=="test":
    resume = "../results_ae/test_torobo_ae/snap/00020.tar"
    if os.path.isfile("./model/test_cae.py"):
        os.remove("./model/test_cae.py")
    shutil.copyfile(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/model/cae.py"), "./model/test_cae.py")
    from model.test_cae import CAE as ae
    with open(os.path.join(resume.rstrip(resume.split("/")[-1]), "../code/nn_params.pickle"), "rb") as f:
        nn_params = pickle.load(f)
    nn_params["resume"] = resume
    nn_params["batch"] = 12
    nn_params["gpu"] = 0

    files = glob.glob(os.path.join(resume.replace(".tar", "_mid"), "*.dat")) # fileが残っていたら削除
    for file in files:
        os.remove(file)

    mot_path = []
    for path in test_path:
        mot_path.append(path.replace(path.split("/")[-1], "motion/motion.txt"))
    extract(test_path, nn_params, ae, mot_paths=mot_path)
    reconstract(test_path, nn_params, ae)

    mot_path = []
    for path in train_path:
        mot_path.append(path.replace(path.split("/")[-1], "motion/motion.txt"))
    extract(train_path, nn_params, ae, mot_paths=mot_path)
    #reconstract(train_path, nn_params, ae)
