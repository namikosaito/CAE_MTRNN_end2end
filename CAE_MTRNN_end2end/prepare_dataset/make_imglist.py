#!/usr/bin/env python
# coding:utf-8

import os, sys
import glob


def make_list(dirpath):
    for dir in glob.glob(os.path.join(dirpath,"*")):
        files = glob.glob(os.path.join(dir,"motion/img/*"))
        print os.path.join(dir,"img/*")
        files.sort()
        with open(os.path.join(dir,"imglist.dat"), "w") as f:
            for file in files:
                path = file[len(dir)+1:]
                f.write("{}\n".format(path))


if __name__ == "__main__":
    dirpath = "../dataset/torobo_data"
    make_list(dirpath)

