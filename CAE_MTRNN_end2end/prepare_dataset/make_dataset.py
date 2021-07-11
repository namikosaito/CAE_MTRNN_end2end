#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import copy
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data import normalize


ignore_value = -999


def pre_process(path, val_range=[-1.0,1.0], plot=False):

    ### fileを読み込み
    files = glob.glob(os.path.join(path, "*.dat"))
    files.sort()
    data = []
    train_bool = []
    test_bool = []
    len_vals = 0
    #length = []
    print(files)
    for file in files:
        with open(file, "r") as fr:
            lines = fr.readlines()
        angles = []
        for i, line in enumerate(lines):
            vals = []
            for j, val in enumerate(line.rstrip("\n").split(" ")):
                if (j<7) or (j>12): #poseをmask
                    vals.append(float(val))
            len_vals = len(vals)
            angles.append(vals)
        data.append(angles)
        #length.append(len(angles))
        if "train" in file:
            train_bool.append(True)
            test_bool.append(False)
        elif "test" in file:
            train_bool.append(False)
            test_bool.append(True)
    print(train_bool)
    ### 正規化範囲を計算
    len_seqs = []
    vals = {}
    for i in range(len_vals):
        vals[i] = []
    for i in range(len(data)):
        for j in range(len(data[i])):
            len_seqs.append(len(data[i]))
            for k in range(len(data[i][j])):
                vals[k].append(data[i][j][k])
    range_min_max = {}
    for i in range(len_vals):
        range_min_max[i] = [min(vals[i]), max(vals[i])]

    ### 正規化(各要素ごと), 範囲を保存(生成用)
    for i in range(len(data)):
        for j in range(len(data[i])):
            for k in range(len(data[i][j])):
                if range_min_max[k][0]==range_min_max[k][1]:
                    data[i][j][k] = (val_range[0] + val_range[1]) / 2.
                else:
                    if (data[i][j][k] < range_min_max[k][0]) or (data[i][j][k] > range_min_max[k][1]):
                        print (i,j,k, data[i][j][k], range_min_max[k])
                    data[i][j][k] = normalize(data[i][j][k], range_min_max[k], val_range)

    ### numpyに変換して保存
    min_max_vals = []
    for i in range(len_vals):
        min_max_vals.append(range_min_max[i])
    np.save(os.path.join(path, "min_max_vals.npy"), np.asarray(min_max_vals))
    np.save(os.path.join(path, "val_range.npy"), np.asarray(val_range))

    ### numpyに変換
    for i,d in enumerate(data):
        for j in range(max(len_seqs)-len(d)):
            d.append([ignore_value]*len_vals) ### データ長を揃える(ignore valueで埋める)
        d = np.asarray(d)
        data[i] = d
    data = np.asarray(data, dtype=np.float32)

    ### 可視化(前処理後)
    if plot:
        plt.figure(figsize=(18, 5))
        for i, d in enumerate(data):
            mask = np.where(d[:,0] != ignore_value)[0]
            plt.plot(d[mask,:8])
            plt.grid()
            plt.savefig(os.path.join(path,"seq{}_angles.png".format(i)), bbox_inches='tight', pad_inches=0)
            plt.clf()
            #break
        for i, d in enumerate(data):
            mask = np.where(d[:,0] != ignore_value)[0]
            plt.plot(d[mask,8:])
            plt.grid()
            plt.savefig(os.path.join(path,"seq{}_features.png".format(i)), bbox_inches='tight', pad_inches=0)
            plt.clf()
            #break

    ### RNN用に保存
    print ("="*20)
    print ("train data shape : {}".format(data[train_bool].shape))
    print ("test data shape  : {}".format(data[test_bool].shape))
    with open(os.path.join(path, "train.pickle"), 'wb') as wb:
        pickle.dump(data[train_bool], wb)
    with open(os.path.join(path, "test.pickle"), 'wb') as wb:
        pickle.dump(data[test_bool], wb)


if __name__ == "__main__":
    path = "../results_ae/test_torobo_ae/snap/00020_mid"
    pre_process(path, plot=True)


