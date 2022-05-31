'''
    Dataset for ShapeNetPart segmentation
'''

import os
import os.path
import json
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from random import sample
import math

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class PartNormalDataset():
    def __init__(self, filename, root = '../data', classification = False, normalize=True, return_cls_label = False):
        self.root = root
        self.cat = {}
        
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        
        self.cat = {
          # 'class_name' : 'class_folder',
          'Single' : 'single',
        }

        #print(self.cat)
            
        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = [filename]            
                
            #print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0]) 
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))
                #print(os.path.join(dir_point, token + '.txt'))
        
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))
            
         
        self.classes = dict(zip(self.cat, range(len(self.cat))))  
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Single': [0, 1]}

        for cat in sorted(self.seg_classes.keys()):
            print(cat, self.seg_classes[cat])
        
        self.cache = {} # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000
        
    def __getitem__(self, index):
        if index in self.cache:
            point_set, normal, seg, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            point_set = data[:,0:3]
            if self.normalize:
                point_set = pc_normalize(point_set)
            normal = data[:,3:6]
            seg = data[:,-1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, normal, seg, cls)

        if self.classification:
            return point_set, normal, cls
        else:
            if self.return_cls_label:
                return point_set, normal, seg, cls
            else:
                return point_set, normal, seg
        
    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    d = PartNormalDataset(filename='M01_0313_a.txt')
    print(len(d))

    i = 0
    ps, normal, seg = d[i]
    print(d.datapath[i])
    print(np.max(seg), np.min(seg))
    print(ps.shape, seg.shape, normal.shape)
    print(ps)
    print(normal)