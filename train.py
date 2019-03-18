'''
    Single-GPU training.
    Will use H5 dataset in default. If using normal, will shift to the normal dataset.
'''
import argparse
import math
from datetime import datetime
#  import h5py
import numpy as np
import tensorflow as tf
from scipy import stats
import socket
import importlib
import os
import sys
from shutil import copy2
import logging
import multiprocessing as mp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from utils import provider, tf_util
import dataset
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet', help='Model name [default: pointnet_cls_ssg]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=400, help='Point Number [default: 1024]')
parser.add_argument('--num_channels', type=int, default=140, help='Channel Number [default: 140]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 20]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate [default: 0.0001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=100000, help='Decay step for lr decay [default: 100000]')
parser.add_argument('--decay_rate', type=float, default=0.9, help='Decay rate for lr decay [default: 0.9]')
parser.add_argument('--normal', action='store_true', help='Whether to use normal information')
parser.add_argument('--loss', type=str, default='l2_loss', help='The loss type used in training [options: l1_loss, l2_loss (default), corre_loss]')
parser.add_argument('--dataset', type=str, default='coordinate', help='The dataset used to train the model [default: coordinate]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_CHANNELS = FLAGS.num_channels
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
LOSS_TYPE = FLAGS.loss
DATASET = FLAGS.dataset

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, FLAGS.model+'.py')

time = datetime.now()
log_path = os.path.join(FLAGS.log_dir, time.strftime("%Y%m%d_%H%M"))
LOG_DIR = os.path.join(ROOT_DIR, log_path)
if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)
copy2(MODEL_FILE, LOG_DIR)
copy2("train.py", LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

HOSTNAME = socket.gethostname()

if LOSS_TYPE == "cross_entropy":
    TRAIN_DATASET = dataset.AtomwiseDataset(
        split='training', shuffle=True, batch_size=BATCH_SIZE,
        label_type="onehot")
    TEST_DATASET = dataset.AtomwiseDataset(
        split='validation', batch_size=BATCH_SIZE,
        label_type="onehot")
elif DATASET == "coordinate":
    TRAIN_DATASET = dataset.CoordinateDataset(
        split='training', shuffle=True, batch_size=BATCH_SIZE)
    TEST_DATASET = dataset.CoordinateDataset(
        split='validation', batch_size=BATCH_SIZE)
else:
    TRAIN_DATASET = dataset.AtomwiseDataset(
        split='training', shuffle=True, batch_size=BATCH_SIZE)
    TEST_DATASET = dataset.AtomwiseDataset(
        split='validation', batch_size=BATCH_SIZE)

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
            if LOSS_TYPE == "cross_entropy":
                pointclouds_pl, labels_pl = MODEL.placeholder_inputs(
                    BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS,
                    TRAIN_DATASET.NUM_CHANNEL, label_type="onehot") 
            else:
                pointclouds_pl, labels_pl = MODEL.placeholder_inputs(
                    BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS,
                    TRAIN_DATASET.NUM_CHANNEL)
            if DATASET == "coordinate":
                mask_pl = tf.placeholder(tf.bool, shape=(BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS, 1, 768))

            is_training_pl = tf.placeholder(tf.bool, shape=())
            tf.summary.histogram("labels", labels_pl)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            if LOSS_TYPE == "cross_entropy":
                pred = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay, label_type="onehot")
            elif DATASET == "coordinate":
                pred = MODEL.get_model(pointclouds_pl, is_training_pl, mask_pl, bn_decay=bn_decay)
            else:
                pred = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            tf.summary.histogram("predictions", pred)
            loss = MODEL.get_loss(pred, labels_pl, type=LOSS_TYPE)
            # losses = tf.get_collection('losses')
            # total_loss = tf.add_n(losses, name='total_loss')
            tf.summary.scalar('loss', loss)
            # for l in losses + [total_loss]:
            #     tf.summary.scalar(l.op.name, l)

            # correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            # accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            # tf.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
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
        sess.run(init, {is_training_pl: True})

        ops = {
            'pointclouds_pl': pointclouds_pl,
            'labels_pl': labels_pl,
            'mask_pl': mask_pl,
            'is_training_pl': is_training_pl,
            'pred': pred,
            'loss': loss,
            'train_op': train_op,
            'merged': merged,
            'step': batch,
        }

        # best_acc = -1

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)


def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS, TRAIN_DATASET.NUM_CHANNEL), dtype=np.float32)
    if LOSS_TYPE == "cross_entropy":
        cur_batch_label = np.zeros((BATCH_SIZE, 10), dtype=np.int8)
    else:
        cur_batch_label = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    if DATASET == "coordinate":
        cur_batch_mask = np.full((BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS, 1, 768), True)

    loss_sum = 0
    batch_idx = 0
    while TRAIN_DATASET.has_next_batch():
        if DATASET == "coordinate":
            batch_data, batch_label, batch_mask = TRAIN_DATASET.next_batch()
        else:
            batch_data, batch_label = TRAIN_DATASET.next_batch()
        bsize = batch_data.shape[0]
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize, ...] = batch_label
        if DATASET == "coordinate":
            for i in range(bsize):
                cur_batch_mask[i, :, 0, :] = np.expand_dims(batch_mask[i], 0).transpose()
        feed_dict = {
            ops['pointclouds_pl']: cur_batch_data,
            ops['labels_pl']: cur_batch_label,
            ops['mask_pl']: cur_batch_mask,
            ops['is_training_pl']: is_training
        }
        _, summary, step, loss_val, _ = sess.run(
            [ops['train_op'],
            ops['merged'],
            ops['step'],
            ops['loss'],
            ops['pred']], feed_dict=feed_dict
        )
        # print("batch: {}, pred: {}".format(batch_idx, pred))
        train_writer.add_summary(summary, step)
        loss_sum += loss_val
        # print("pred: {}".format(pred))
        # print("batch loss: {}".format(loss_val))
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            loss_sum = 0
        batch_idx += 1

    TRAIN_DATASET.reset()
        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

    all_labels = []
    all_preds = []
    num_decoys = 0

    # Make sure batch data is of same size
    cur_batch_data = np.zeros((BATCH_SIZE,TEST_DATASET.MAX_ATOMS,TEST_DATASET.NUM_CHANNEL), dtype=np.float32)
    if LOSS_TYPE == "cross_entropy":
        cur_batch_label = np.zeros((BATCH_SIZE, 10), dtype=np.int8)
    else:
        cur_batch_label = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
    if DATASET == "coordinate":
        cur_batch_mask = np.full((BATCH_SIZE, TRAIN_DATASET.MAX_ATOMS), True)

    loss_sum = 0
    batch_idx = 0
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d EVALUATION ----'%(EPOCH_CNT))
    
    while TEST_DATASET.has_next_batch():
        if DATASET == "coordinate":
            batch_data, batch_label, batch_mask = TEST_DATASET.next_batch()
        else:
            batch_data, batch_label = TEST_DATASET.next_batch()
        bsize = batch_data.shape[0]
        cur_batch_label[0:bsize] = batch_label
        all_labels += cur_batch_label.squeeze().tolist()
        num_decoys += bsize
        # for the last batch in the epoch, the bsize:end are from last batch
        cur_batch_data[0:bsize, ...] = batch_data
        cur_batch_label[0:bsize, ...] = batch_label
        if DATASET == "coordinate":
            cur_batch_mask[0:bsize, ...] = batch_mask

        feed_dict = {ops['pointclouds_pl']: cur_batch_data,
                     ops['labels_pl']: cur_batch_label,
                     ops['mask_pl']: cur_batch_mask,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        # print("validation labels: {}".format(cur_batch_label))
        # print("validation pred values: {}".format(pred_val))

        all_preds += pred_val.squeeze().tolist()
        test_writer.add_summary(summary, step)
        
        loss_sum += loss_val
        batch_idx += 1

    # pearson correlation loss
    # pearson = stats.pearsonr(all_preds, all_labels)[0]

    # l2 loss
    # x_y = np.array(all_labels) - np.array(all_preds)
    # eval_loss = np.sum(x_y ** 2) * 0.5 / num_decoys

    # log_string("eval l2_loss: %f" % eval_loss)

    # need to be changed to a evaluation function showing correlation
    #     for i in range(0, bsize):
    #         l = batch_label[i]
    #         total_seen_class[l] += 1
    #         total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(batch_idx)))
    # log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    # log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))

    EPOCH_CNT += 1

    TEST_DATASET.reset()
    return loss_sum / float(batch_idx)


if __name__ == "__main__":
    mp.freeze_support()
    logging.basicConfig(level=logging.DEBUG)
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
