import argparse
import math
from datetime import datetime
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import provider
import tf_util
import part_pheno4d_all_normal

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt', help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--category', default='maize_tomato')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
CATEGORY = FLAGS.category

MODEL_PATH = FLAGS.model_path
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 2

TEST_DATASET = part_pheno4d_all_normal.PartNormalDataset(npoints=NUM_POINT, classification=False, split='test', category=CATEGORY)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            batch = tf.Variable(0)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl)
            saver = tf.train.Saver()
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'merged': merged,
               'step': batch}

        eval_one_epoch(sess, ops, test_writer)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg = dataset[idxs[i+start_idx]]
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
    return batch_data, batch_label      
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1)/BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]

    seg_classes = TEST_DATASET.seg_classes
    shape_ious = {cat:[] for cat in seg_classes.keys()}
    seg_label_to_cat = {} # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    batch_data = np.zeros((BATCH_SIZE, NUM_POINT, 6))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT)).astype(np.int32)
    for batch_idx in range(int(num_batches)):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label

        # ---------------------------------------------------------------------
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        # ---------------------------------------------------------------------
    
        # Select valid data
        cur_pred_val = pred_val[0:cur_batch_size]
        #print(len(cur_pred_val[0]))
        #print(cur_pred_val[0])
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            #print(cat)
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]
            #print(cur_pred_val.shape)

        for i in range(cur_batch_size):
          extract_output(cur_batch_data[i], cur_pred_val[i], i, batch_idx)

        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        # if cur_batch_size==BATCH_SIZE:
        #     loss_sum += loss_val
        loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            #print(segp.shape)
            segl = cur_batch_label[i,:] 
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l-seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            shape_ious[cat].append(np.mean(part_ious))

    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(list(shape_ious[cat]))
    mean_shape_ious = np.mean(list(shape_ious.values()))
    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(list(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))))
    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (np.mean(list(all_shape_ious))))
         
    EPOCH_CNT += 1
    return total_correct/float(total_seen)

def print_specs():
  print('BATCH_SIZE: ' + str(BATCH_SIZE))
  print('NUM_POINT: ' + str(NUM_POINT))
  print('LOG_DIR: ' + str(LOG_DIR))
  print('CATEGORY: ' + CATEGORY)
  print('TEST_DATASET: ' + str(len(TEST_DATASET)))

def extract_output(points, labels, index, batch_idx):
  out_data = points
  out_labels = (labels).reshape(-1,1) # convert to np array with 1 column
  out = np.hstack((out_data, out_labels))

  lines = []
  for i in range(len(out)):
    lines.append(' '.join(map(str, out[i])))

  filepath = TEST_DATASET.datapath[index+BATCH_SIZE*batch_idx][1]
  filename = os.path.basename(filepath)
  output_folder = os.path.join(LOG_DIR, 'output')
  if not os.path.exists(output_folder): os.makedirs(output_folder)
  save_file = os.path.join(output_folder, filename)
  print(save_file)

  output = []
  for i in range(len(out)):
    nums = lines[i].split()
    for j in range(len(nums)):
      nums[j] = "{0:.6f}".format(float(nums[j]))
    output.append(' '.join(nums))

  #print(lines)

  content = '\n'.join(output)
  with open(save_file, 'w') as fid:
      fid.write(content)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    print_specs()
    evaluate()
    print_specs()
    LOG_FOUT.close()
