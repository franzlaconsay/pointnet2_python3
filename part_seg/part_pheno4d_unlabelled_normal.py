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
    def __init__(self, root = '../data/pheno4d_unlabelled', npoints = 2500, classification = False, split='train', normalize=True, return_cls_label = False, train_sample = 1, category='maize_tomato'):
        self.npoints = npoints
        self.root = root
        self.cat = {}
        
        self.classification = classification
        self.normalize = normalize
        self.return_cls_label = return_cls_label
        
        self.cat = {
          # 'class_name' : 'class_folder',
          'Maize' : 'Maize',
          'Tomato' : 'Tomato'
        }
        maize_train = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07']
        maize_test = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07']
        tomato_train = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07']
        tomato_test = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07']

        if category == 'maize':
          self.cat = {'Maize' : 'Maize'}
          maize_train = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07']
          maize_test = ['M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07']
          tomato_train = []
          tomato_test = []
          self.root = '../data/pheno4d_unlabelled'

        if category == 'tomato':
          self.cat = {'Tomato' : 'Tomato'}
          maize_train = []
          maize_test = []
          tomato_train = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07']
          tomato_test = ['T01', 'T02', 'T03', 'T04', 'T05', 'T06', 'T07']
          self.root = '../data/pheno4d_unlabelled'

        #print(self.cat)
            
        self.meta = {}
        for item in self.cat:
            #print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            #print(fns[0][0:-11])
            if split=='trainval':
                fns = [fn for fn in fns if ((fn[0:-9] in maize_train) or (fn[0:-9] in tomato_train))]
                samples = math.ceil(len(fns) * train_sample)
                fns = sample(fns, samples)
            elif split=='test':
                fns = [fn for fn in fns if ((fn[0:-9] in maize_test) or (fn[0:-9] in tomato_test))]
            else:
                print('Unknown split: %s. Exiting..'%(split))
                exit(-1)
                
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
        self.seg_classes = {'Maize': [0, 1], 'Tomato': [2, 3]}

        if category == 'maize':
          self.seg_classes = {'Maize': [0, 1]}

        if category == 'tomato':
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
    d = PartNormalDataset(split='test', npoints=10000, train_sample=1.0, category='tomato')
    print(len(d))

    i = 0
    ps, normal, seg = d[i]
    print(d.datapath[i])
    print(np.max(seg), np.min(seg))
    print(ps.shape, seg.shape, normal.shape)
    print(ps)
    print(normal)
