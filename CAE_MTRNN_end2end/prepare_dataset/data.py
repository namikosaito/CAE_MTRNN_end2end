#!/usr/bin/python
#-*-coding:utf-8-*-
import sys,os,re

def normalize(data, indataRange, outdataRange):
    """
    return normalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data


def denormalize(data, indataRange, outdataRange):
    """
    上記のinとoutのrangeを入れ替えればよい
    return denormalized data
    it need two list (indataRange[x1,x2] and outdataRange[y1,y2])
    """
    if indataRange[0]!=indataRange[1]:
        data = (data - indataRange[0]) / (indataRange[1] - indataRange[0])
        data = data * (outdataRange[1] - outdataRange[0]) + outdataRange[0]
    else:
        data = (outdataRange[0] + outdataRange[1]) / 2.
    return data

def as_mat(x):
    if x.ndim == 1:
        return x.reshape(-1, len(x))
    else:
        return x

def sortFile(files):
    convert = lambda text: int(text) if text.isdigit() else text 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    files.sort( key=alphanum_key ) 
    return files


