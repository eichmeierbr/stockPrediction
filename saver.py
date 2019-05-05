#!/usr/bin/python

import pickle as cPickle

layers=3
version=3
netStyle='ann'
memory=25
lr=0.0001
funcs=['tanh','tanh','tanh']
depths=[75,75,200]
file_name=netStyle+'%iLayer_%iV%i.pck' %(layers,memory,version)
ann=(netStyle,memory,lr,funcs,depths)
with open(file_name, 'wb') as fp:
    cPickle.dump(ann, fp)
