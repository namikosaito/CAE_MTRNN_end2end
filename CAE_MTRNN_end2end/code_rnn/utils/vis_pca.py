#!/usr/bin/env python
# coding:utf-8

import os
import numpy as np
import itertools
from sklearn.decomposition import PCA
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['animation.ffmpeg_args'] = '-report'
matplotlib.rcParams['animation.bitrate'] = 2000

class vis_pca():
    def __init__(self, outdir, dataset, component):
        self.colors = plt.get_cmap("tab10")
        self.markers = ["o","x","s"]
        self.component = component

        ### directory
        self.outdir = outdir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)

        self.comb_3d = list(itertools.combinations(range(self.component),3)) # directory (3d axis)
        self.dirnames_3d = []
        for c in self.comb_3d:
            dirname = os.path.join(self.outdir, "pc{}_pc{}_pc{}".format(c[0]+1,c[1]+1,c[2]+1))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            self.dirnames_3d.append(dirname)

        self.comb_2d = list(itertools.combinations(range(self.component),2)) # directory (2d axis)
        self.dirnames_2d = []
        for c in self.comb_2d:
            dirname = os.path.join(self.outdir, "pc{}_pc{}".format(c[0]+1,c[1]+1))
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            self.dirnames_2d.append(dirname)

        ### setup
        self.set_pc(dataset)


    def cmap(self, id):
        c_list = ["b","g","r","c","m","y","k","w","orangered","#FF4500"]
        while True:
            if id < len(c_list):
                return c_list[id]
            id = id -len(c_list)

    def set_pc(self, dataset):
        pca = PCA(n_components=self.component)
        pca_p = pca.fit(dataset)
        self.pca_v = pca.transform(dataset)
        self.ratio = pca_p.explained_variance_ratio_

    def plot3d(self, ids, color_ids, mark_ids, figname="pca.png"):
        for num, c in enumerate(self.comb_3d):
            ax = Axes3D(plt.figure(figsize=(6,6)))
            for i, j, k in zip(ids, color_ids, mark_ids):
                ax.scatter(self.pca_v[i,c[0]], self.pca_v[i,c[1]], self.pca_v[i,c[2]], \
                           c=self.colors(j), marker=self.markers[k])
            ax.set_xlabel("pc{} ({:.3g})".format((c[0]+1), self.ratio[0]))
            ax.set_ylabel("pc{} ({:.3g})".format((c[1]+1), self.ratio[1]))
            ax.set_zlabel("pc{} ({:.3g})".format((c[2]+1), self.ratio[2]))
            plt.savefig(os.path.join(self.dirnames_3d[num], figname))
            plt.close()

    def plot3d_ave(self, len_seqs, N, tN, figname="pca.png"):
        for num, c in enumerate(self.comb_3d):
            ax = Axes3D(plt.figure(figsize=(6,6)))
            start_id = 0
            for n in range(N):
                ids = range(start_id, start_id+len_seqs[n])
                pca_v1 = self.pca_v[ids,c[0]].mean()
                pca_v2 = self.pca_v[ids,c[1]].mean()
                pca_v3 = self.pca_v[ids,c[2]].mean()
                ax.scatter(pca_v1, pca_v2, pca_v3, c=self.colors(n), marker=self.markers[0], \
                           label="train{}".format(n+1))
                start_id = start_id+len_seqs[n]
            for tn in range(tN):
                ids = range(start_id, start_id+len_seqs[n+tn+1])
                pca_v1 = self.pca_v[ids,c[0]].mean()
                pca_v2 = self.pca_v[ids,c[1]].mean()
                pca_v3 = self.pca_v[ids,c[2]].mean()
                ax.scatter(pca_v1, pca_v2, pca_v3, c=self.colors(n), marker=self.markers[1], \
                           label="test{}".format(n+tn+2))
                start_id = start_id+len_seqs[n+tn+1]
            ax.set_xlabel("pc{} ({:.3g})".format((c[0]+1), self.ratio[0]))
            ax.set_ylabel("pc{} ({:.3g})".format((c[1]+1), self.ratio[1]))
            ax.set_zlabel("pc{} ({:.3g})".format((c[2]+1), self.ratio[2]))
            ax.legend()
            plt.savefig(os.path.join(self.dirnames_3d[num], figname))
            plt.close()

    def plot2d(self, ids, color_ids, mark_ids, figname="pca.png"):
        for num, c in enumerate(self.comb_2d):
            plt.figure(figsize=(6,6))
            for i, j, k in zip(ids, color_ids, mark_ids):
                plt.scatter(self.pca_v[i,c[0]], self.pca_v[i,c[1]], \
                           c=self.colors(j), marker=self.markers[k])
            plt.xlabel("pc{} ({:.3g})".format((c[0]+1), self.ratio[0]))
            plt.ylabel("pc{} ({:.3g})".format((c[1]+1), self.ratio[1]))
            plt.savefig(os.path.join(self.dirnames_2d[num], figname), bbox_inches="tight",pad_inches=0)
            plt.close()

    def plot2d_ave(self, len_seqs, N, tN, figname="pca.png"):
        for num, c in enumerate(self.comb_2d):
            plt.figure(figsize=(6,6))
            start_id = 0
            for n in range(N):
                ids = range(start_id, start_id+len_seqs[n])
                pca_v1 = self.pca_v[ids,c[0]].mean()
                pca_v2 = self.pca_v[ids,c[1]].mean()
                plt.scatter(pca_v1, pca_v2, c=self.colors(n), marker=self.markers[0], \
                            label="train{}".format(n+1))
                start_id = start_id+len_seqs[n]
            for tn in range(tN):
                ids = range(start_id, start_id+len_seqs[n+tn+1])
                pca_v1 = self.pca_v[ids,c[0]].mean()
                pca_v2 = self.pca_v[ids,c[1]].mean()
                plt.scatter(pca_v1, pca_v2, c=self.colors(n), marker=self.markers[1], \
                            label="test{}".format(n+tn+2))
                start_id = start_id+len_seqs[n+tn+1]
            plt.xlabel("pc{} ({:.3g})".format((c[0]+1), self.ratio[0]))
            plt.ylabel("pc{} ({:.3g})".format((c[1]+1), self.ratio[1]))
            plt.legend()
            plt.savefig(os.path.join(self.dirnames_2d[num], figname))
            plt.close()

    def plt_anim(self, ids, color_ids, mark_ids, figname="pc123.png"):
        fig = plt.figure()
        ax = Axes3D(fig)
        mark = "o"
        def init_pca123():
            for num,c in enumerate(self.comb_3d):
                for i, j, k in zip(ids, color_ids, mark_ids):
                    ax.scatter(self.pca_v[i,c[0]], self.pca_v[i,c[1]], self.pca_v[i,c[2]], \
                               c=self.colors(j), marker=self.markers[k])
            ax.set_xlabel("pc1: {:.5f}".format(self.ratio[0]))
            ax.set_ylabel("pc2: {:.5f}".format(self.ratio[1]))
            ax.set_zlabel("pc3: {:.5f}".format(self.ratio[2]))
            return fig,
        def animate(i):
            ax.view_init(elev=10., azim=i)
            return fig,
        ### full quality: frames=360, fps=30
        anim = animation.FuncAnimation(fig, animate, init_func=init_pca123,
                                       frames=90, interval=20, blit=True)
        anim.save(os.path.join(self.outdir,figname), fps=5)


