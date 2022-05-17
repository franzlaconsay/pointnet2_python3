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
from pheno4d_dataset_split import Pheno4dDataset
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet2_part_seg', help='Model name [default: model]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=10000, help='Point Number [default: 10000]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
parser.add_argument('--train_sample', type=float, default=1.0)
parser.add_argument('--category', default='maize')
parser.add_argument('--random_state', type=int, default=42)
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
TRAIN_SAMPLE = FLAGS.train_sample
CATEGORY = FLAGS.category
RANDOM_STATE = FLAGS.random_state

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

NUM_CLASSES = 2

pheno4d_dataset = Pheno4dDataset()
TRAIN_DATASET, TEST_DATASET = pheno4d_dataset.pheno4d_train_test_split(npoints=NUM_POINT, category=CATEGORY, train_sample=TRAIN_SAMPLE, random_state=RANDOM_STATE)

epochs_log = []

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            print("--- Get model and loss")
            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_POINT)
            tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
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
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT, 6))
    batch_label = np.zeros((bsize, NUM_POINT), dtype=np.int32)
    for i in range(bsize):
        ps,normal,seg = dataset[idxs[i+start_idx]]
        batch_label[i] = np.zeros((len(ps)), dtype=np.int32)
        batch_data[i,:,0:3] = ps
        batch_data[i,:,3:6] = normal
        batch_label[i,:] = seg
    return batch_data, batch_label

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    train_idxs = np.arange(0, len(TRAIN_DATASET))
    np.random.shuffle(train_idxs)
    num_batches = len(TRAIN_DATASET)/BATCH_SIZE
    
    log_string(str(datetime.now()))

    total_correct = 0
    total_seen = 0
    loss_sum = 0
    for batch_idx in range(int(num_batches)):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        batch_data, batch_label = get_batch(TRAIN_DATASET, train_idxs, start_idx, end_idx)
        # Augment batched point clouds by rotation and jittering
        #aug_data = batch_data
        #aug_data = provider.random_scale_point_cloud(batch_data)
        batch_data[:,:,0:3] = provider.jitter_point_cloud(batch_data[:,:,0:3])
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == batch_label)
        total_correct += correct
        total_seen += (BATCH_SIZE*NUM_POINT)
        loss_sum += loss_val

        if (batch_idx+1)%10 == 0:
            log_string(' -- %03d / %03d --' % (batch_idx+1, num_batches))
            log_string('mean loss: %f' % (loss_sum / 10))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0
        

        
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

    # add
    total_part_ious = [0.0 for _ in range(NUM_CLASSES)]
    total_part_recalls = [0.0 for _ in range(NUM_CLASSES)]
    total_part_precisions = [0.0 for _ in range(NUM_CLASSES)]
    total_matrix = np.zeros((2,2))

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
        # Constrain pred to the groundtruth classes (selected by seg_classes[cat])
        cur_pred_val_logits = cur_pred_val
        cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
        for i in range(cur_batch_size):
            cat = seg_label_to_cat[cur_batch_label[i,0]]
            logits = cur_pred_val_logits[i,:,:]
            cur_pred_val[i,:] = np.argmax(logits[:,seg_classes[cat]], 1) + seg_classes[cat][0]

        correct = np.sum(cur_pred_val == cur_batch_label)
        total_correct += correct
        total_seen += (cur_batch_size*NUM_POINT)
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val
        #loss_sum += loss_val
        for l in range(NUM_CLASSES):
            total_seen_class[l] += np.sum(cur_batch_label==l)
            total_correct_class[l] += (np.sum((cur_pred_val==l) & (cur_batch_label==l)))

        for i in range(cur_batch_size):
            segp = cur_pred_val[i,:]
            segl = cur_batch_label[i,:] 
            cat = seg_label_to_cat[segl[0]]
            part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
            part_recalls = [0.0 for _ in range(len(seg_classes[cat]))]
            part_precisions = [0.0 for _ in range(len(seg_classes[cat]))]
            for l in seg_classes[cat]:
                if (np.sum(segl==l) == 0) and (np.sum(segp==l) == 0): # part is not present, no prediction as well
                    part_ious[l-seg_classes[cat][0]] = 1.0
                else:
                    part_ious[l-seg_classes[cat][0]] = np.sum((segl==l) & (segp==l)) / float(np.sum((segl==l) | (segp==l)))
            
            matrix = confusion_matrix(segl, segp)
            TP = matrix[1][1]
            TN = matrix[0][0]
            FP = matrix[0][1]
            FN = matrix[1][0]
            part_recalls = np.diag(matrix) / np.sum(matrix, axis = 1)
            part_precisions = np.diag(matrix) / np.sum(matrix, axis = 0)
            total_matrix += matrix

            total_part_ious = np.sum([total_part_ious, part_ious],axis=0)
            total_part_recalls = np.sum([total_part_recalls, part_recalls],axis=0)
            total_part_precisions = np.sum([total_part_precisions, part_precisions],axis=0)
            shape_ious[cat].append(np.mean(part_ious))

    epoch_stats = {} # for json output
    
    all_shape_ious = []
    for cat in shape_ious.keys():
        for iou in shape_ious[cat]:
            all_shape_ious.append(iou)
        shape_ious[cat] = np.mean(list(shape_ious[cat]))
    mean_shape_ious = np.mean(list(shape_ious.values()))
    mean_loss = loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)
    accuracy = total_correct / float(total_seen)
    avg_class_accuracy = np.mean(list(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float)))

    log_string('eval mean loss: %f' % (mean_loss))
    log_string('eval accuracy: %f' % (accuracy))
    log_string('eval avg class acc: %f' % (avg_class_accuracy))

    for cat in sorted(shape_ious.keys()):
        log_string('eval mIoU of %s:\t %f'%(cat, shape_ious[cat]))

    mean_miou_all = np.mean(list(all_shape_ious))
    log_string('eval mean mIoU: %f' % (mean_shape_ious))
    log_string('eval mean mIoU (all shapes): %f' % (mean_miou_all))

    # additional metrics
    total_part_ious /= float(len(TEST_DATASET))
    total_part_recalls /= float(len(TEST_DATASET))
    total_part_precisions /= float(len(TEST_DATASET))

    stem_iou = total_part_ious[0]
    leaf_iou = total_part_ious[1]
    stem_recall = total_part_recalls[0]
    leaf_recall = total_part_recalls[1]
    stem_precision = total_part_precisions[0]
    leaf_precision = total_part_precisions[1]
    log_string('stem_iou: %f' % stem_iou)
    log_string('leaf_iou: %f' % leaf_iou)
    log_string('stem_recall: %f' % stem_recall)
    log_string('leaf_recall: %f' % leaf_recall)
    log_string('stem_precision: %f' % stem_precision)
    log_string('leaf_precision: %f' % leaf_precision)

    total_matrix /= float(len(TEST_DATASET))
    log_string('confusion matrix: \n%s' % str(total_matrix))
    TN = total_matrix[0][0]
    FP = total_matrix[0][1]
    FN = total_matrix[1][0]
    TP = total_matrix[1][1]
    log_string('TN: %f' % TN)
    log_string('FP: %f' % FP)
    log_string('FN: %f' % FN)
    log_string('TP: %f' % TP)

    epoch_stats['accuracy'] = accuracy
    epoch_stats['miou'] = mean_shape_ious
    epoch_stats['stem_iou'] = stem_iou
    epoch_stats['leaf_iou'] = leaf_iou
    epoch_stats['stem_recall'] = stem_recall
    epoch_stats['leaf_recall'] = leaf_recall
    epoch_stats['stem_precision'] = stem_precision
    epoch_stats['leaf_precision'] = leaf_precision
    epoch_stats['tn'] = TN
    epoch_stats['fp'] = FP
    epoch_stats['fn'] = FN
    epoch_stats['tp'] = TP

    epochs_log.append(epoch_stats.copy())

    EPOCH_CNT += 1
    return total_correct/float(total_seen)

def get_log_specs():
  log = (
    'CATEGORY: ' + CATEGORY +
    '\nTRAIN_SAMPLE: ' + str(TRAIN_SAMPLE) +
    '\nRANDOM_STATE: ' + str(RANDOM_STATE) +
    '\nNUM_POINT: ' + str(NUM_POINT) +
    '\nBATCH_SIZE: ' + str(BATCH_SIZE) +
    '\nMAX_EPOCH: ' + str(MAX_EPOCH) +
    '\nLOG_DIR: ' + str(LOG_DIR) +
    '\n' +
    pheno4d_dataset.get_log_stats()
  )
  return log

if __name__ == "__main__":
    start_time = datetime.now()
    log_string('pid: %s'%(str(os.getpid())))
    print(get_log_specs())
    train()
    end_time = datetime.now()
    log_string(get_log_specs())
    log_string(str(start_time))
    log_string(str(end_time))
    log_string(str(end_time-start_time))
    LOG_FOUT.close()

    # output to json
    data = {}
    data['epochs'] = epochs_log
    json_data = json.dumps(data)
    print(json_data)
    LOG_JSON = open(os.path.join(LOG_DIR, 'epochs.json'), 'w')
    LOG_JSON.write(json_data)
    LOG_JSON.flush()
    LOG_JSON.close()