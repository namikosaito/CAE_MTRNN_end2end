#!/usr/bin/env python
# coding:utf-8

import os, sys
import _pickle as pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.vis_pca import vis_pca
import torch
from torch.autograd import Variable


ignore_value = -999.


def train_partical(model, in_data, out_data, input_param, sp, n_name):
    t_seq, x_seq, y_seq, cf_seq, cs_seq = [], [], [], [], []
    for ts in range(in_data.shape[1]):
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

        ### 可視化用に出力を保持
        if ts==0:
            t_seq   = t.data[:,None,:]
            x_seq   = x.data[:,None,:]
            y_seq   = y.data[:,None,:]
            cf_seq  = cf.data[:,None,:]
            cs_seq  = cs.data[:,None,:]
        else:
            t_seq  = np.concatenate([t_seq, t.data[:,None,:]], axis=1)
            x_seq  = np.concatenate([x_seq, x.data[:,None,:]], axis=1)
            y_seq  = np.concatenate([y_seq, y.data[:,None,:]], axis=1)
            cf_seq = np.concatenate([cf_seq, cf.data[:,None,:]], axis=1)
            cs_seq = np.concatenate([cs_seq, cs.data[:,None,:]], axis=1)
    return t_seq, x_seq, y_seq, cf_seq, cs_seq



def test(params, rnn, pca_anim=False):

    ###----------------- setup -----------------###

    dim = params["split_node"]["mot"]

    outdir_seq = params["resume"].replace(".tar", "_seq")
    if not os.path.isdir(outdir_seq):
        os.makedirs(outdir_seq)
    outdir_pca = params["resume"].replace(".tar", "_pca")
    if not os.path.isdir(outdir_pca):
        os.makedirs(outdir_pca)

    ### dataset
    with open(params["train"], "rb") as f:
        dataset_train = pickle.load(f)
    with open(params["test"], "rb") as f:
        dataset_test = pickle.load(f)
    teach_in = dataset_train[:,:-1,:]
    teach_out = dataset_train[:,1:,:]
    test_in = dataset_test[:,:-1,:]
    test_out = dataset_test[:,1:,:]

    ### model, load
    N, steps, insize = teach_in.shape
    tN, _, _ = test_in.shape
    model = rnn(insize, insize, params["c_size"], params["tau"])
    model.initialize_c_state(rand=False, cf=True, cs=False)
    checkpoint = torch.load(params["resume"])
    model.load_state_dict(checkpoint['model_state_dict'])
    print (model.init_cf)
    print (model.init_cs)


    ###----------------- test (seq) -----------------###

    ### plot (seq)
    colors = ["g", "b", "r", "k", "y", "c", "m", "g"]
    def func(data, linestyles, title="", xlabel="", ylabel="", path="fig.jpg", lim=[-1.2, 1.2]):
        fig = plt.figure()
        figs = []
        for i in range(len(data[0][0])):
            fig_n = len(data[0][0])*100 + 10 + (i+1)
            figs.append(fig.add_subplot(fig_n))
        for seq, linestyle in zip(data, linestyles):
            for j in range(seq.shape[-1]):
                val = seq[:,j]
                figs[j].plot(val, linestyle=linestyle, color=colors[j])
                figs[j].grid(True)
                figs[j].set_ylim(lim)
        plt.savefig(path)
        plt.clf()

    def plot_seq(_y_seq, _t_seq, name=""):
        for i, (seq1, seq2) in enumerate(zip(_y_seq, _t_seq)):
            mask = np.where(seq2[:,0] != ignore_value)[0]
            func([seq1[mask,:dim],seq2[mask,:dim]], ["solid","dashed"], \
                 path=os.path.join(outdir_seq, "{}_angle_{}.png".format(name,i+1)))

    model.eval()
    t_seq, x_seq, y_seq, cf_seq, cs_seq = train_partical(model, teach_in, teach_out, 
                           params["input_param_test"], params["split_node"], params["name_node"])
    plot_seq(y_seq, t_seq, name="train")
    tt_seq, tx_seq, ty_seq, tcf_seq, tcs_seq = train_partical(model, test_in, test_out, 
                           params["input_param_test"], params["split_node"], params["name_node"])
    plot_seq(ty_seq, tt_seq, name="test")


    ###----------------- test (pca) -----------------###

    for cs_cf in ["cs", "cf"]:
        if cs_cf == "cs":
            h_seq  = cs_seq  
            th_seq = tcs_seq 
        else:
            h_seq  = cf_seq  
            th_seq = tcf_seq 

        len_seqs = []
        for i, (h, t) in enumerate(zip(h_seq, t_seq)):
            mask = np.where(t[:,0] != ignore_value)[0]
            h = h[mask]
            len_seqs.append(len(h))
            if i==0:
                h_seq_all = h
            else:
                h_seq_all = np.concatenate([h_seq_all, h], axis=0)
        for i, (h, t) in enumerate(zip(th_seq, tt_seq)):
            mask = np.where(t[:,0] != ignore_value)[0]
            h = h[mask]
            len_seqs.append(len(h))
            h_seq_all = np.concatenate([h_seq_all, h], axis=0)
        if not (np.nan in h_seq_all):
            try:
                pca_h = vis_pca(os.path.join(outdir_pca, "pca_h"), h_seq_all, component=3)

                start_id = 0
                for n in range(N+tN):
                    ids = range(start_id, start_id+len_seqs[n]) # 可視化する時はここを調整
                    color_ids = [0] * len_seqs[n] # 可視化する時はここを調整
                    mark_ids  = [0] * len_seqs[n] # 可視化する時はここを調整
                    pca_h.plot2d(ids, mark_ids, color_ids, figname=cs_cf+"_seq{}".format(n+1)+".png")
                    pca_h.plot3d(ids, mark_ids, color_ids, figname=cs_cf+"_seq{}".format(n+1)+".png")
                    start_id = start_id+len_seqs[n]
            except:
                print("there is Nan")


