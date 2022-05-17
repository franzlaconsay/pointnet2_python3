'''
    Dataset for ShapeNetPart segmentation
'''

import os
import os.path
import json
import numpy as np
import sys
from random import sample
import math

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class Pheno4dNormalDataset():
    def __init__(self, root='../data/pheno4d_mt', classification=False, return_cls_label=False, npoints=10000, normalize=True, category='maize', indices = [1,2,3,4,5,6,7], train_samples=1.0):
        self.npoints = npoints
        self.root = root
        self.cat = {}
        
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label

        if category == 'maize':
          self.cat = {'Maize' : 'Maize'} # 'class_name' : 'class_folder'
          models = ['M0'+str(i) for i in indices]
        elif category == 'tomato':
          self.cat = {'Tomato' : 'Tomato'}
          models = ['T0'+str(i) for i in indices]
        else:
          print('Unknown category: %s. Exiting..' % category)
          exit(-1)

        #print(self.cat)
            
        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-11])
            fns = [fn for fn in fns if ((fn[0:-11] in models))]
            nsample = math.ceil(len(fns) * train_samples)
            fns = sample(fns, nsample)
            
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
        #self.seg_classes = {'Maize': [0, 1], 'Tomato': [2, 3]}
        if category == 'maize':
          self.seg_classes = {'Maize': [0, 1]}
        elif category == 'tomato':
          self.seg_classes = {'Tomato': [0, 1]}

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
                
        
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        #resample
        point_set = point_set[choice, :]
        seg = seg[choice]
        normal = normal[choice,:]
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
    d = Pheno4dNormalDataset(category='tomato', indices=[2,3], samples=0.5)
    print(len(d))

    i = 0
    ps, normal, seg = d[i]
    print(d.datapath[i])
    print(np.max(seg), np.min(seg))
    print(ps.shape, seg.shape, normal.shape)
    print(ps)
    print(normal)
