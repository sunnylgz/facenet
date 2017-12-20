"""Validate a face recognizer on private dataset and the ROC curve
is calculated and plotted. Both the model metagraph and the model parameters need to exist
in the same directory, and the metagraph should have the extension '.meta'.
"""
# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
import math
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

def main(args):

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Get the paths for the corresponding images
            paths, actual_issame = get_paths(os.path.expanduser(args.data_dir), args.test_pairs, args.nrof_images)
            print("paths length: %d, actual_issame length: %d" % (len(paths), len(actual_issame)))
            n_same = actual_issame.count(True)#np.sum(actual_issame)
            n_diff = actual_issame.count(False)#np.sum(np.logical_not(actual_issame))
            print("num of same is %d, num of diff is %d" % (n_same, n_diff))

            # Load the model
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            #image_size = images_placeholder.get_shape()[1]  # For some reason this doesn't work for frozen graphs
            image_size = args.image_size
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Runnning forward pass on test images')
            batch_size = args.batch_size
            nrof_images = len(paths)
            nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches):
                start_index = i*batch_size
                end_index = min((i+1)*batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            evaluate(emb_array, actual_issame, nrof_folds=args.nrof_folds)


def evaluate(embeddings, actual_issame, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds)

    print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-2, nrof_folds=nrof_folds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-1, nrof_folds=nrof_folds)
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)

    return tpr, fpr, accuracy, val, val_std, far


def get_paths(data_dir, test_pairs, total_nrof_images = 0):
    def select(array, num):
        length = len(array)
        if num >= length:
            return array
        i_shuffle = np.random.permutation(length)[0:num]
        return array[i_shuffle]

    class IndexClass():
        "Stores the class/index for an image"
        def __init__(self, cls, idx):
            self.cls = cls
            self.idx = idx

        def __str__(self):
            return "class: %d, index: %d" % (self.cls, self.idx)

    path_list = []
    issame_list = []
    if os.path.isfile(test_pairs):
        with open(test_pairs, "r") as f:
            for line in f:
                path0,path1,is_same = line.split(',')
                path_list += (path0, path1)
                issame_list.append(bool(int(is_same)))
        print("Load information from ", test_pairs)
        print("The 1st line is:")
        print(path_list[0], path_list[1], issame_list[0])
        return path_list, issame_list

    test_set = facenet.get_dataset(data_dir)

    nrof_classes = len(test_set)
    path_list_i = []
    for i in range(nrof_classes):
        nrof_images = len(test_set[i])
        for j in range(nrof_images):
            path0 = IndexClass(i,j)#test_set[i].image_paths[j]
            for k in range(j+1, nrof_images):
                path1 = IndexClass(i,k)#test_set[i].image_paths[k]
                path_list_i += (path0, path1)
                issame_list.append(True)
            for k in range(i+1, nrof_classes):
                for l in range(len(test_set[k].image_paths)):
                    path1 = IndexClass(k,l)
                    path_list_i += (path0, path1)
                    issame_list.append(False)

    n_same = issame_list.count(True)
    n_diff = issame_list.count(False)
    print("original num of same is %d, num of diff is %d" % (n_same, n_diff))
    path_array = np.reshape(np.array(path_list_i), (-1,2))
    issame_array = np.array(issame_list)
    index_same = np.reshape(np.where(issame_array), (-1))
    index_diff = np.reshape(np.where(issame_array==False), (-1,))
    # assure the numb of same & diff is equal
    if n_same > n_diff:
        n_same = n_diff
    else:
        n_diff = n_same

    # assure total num of images is not exceeded
    if total_nrof_images > 0 and n_same + n_diff > total_nrof_images:
        n_same = n_diff = total_nrof_images//2

    # select and shuffle dataset
    index_same = select(index_same, n_same)
    index_diff = select(index_diff, n_diff)
    # maybe current numpy has bugs, the np.concatenate() will return error
    # when concatenate index_same and index_diff
    index_final = list(index_same) + list(index_diff)
    np.random.shuffle(index_final)
    path_list_i = np.reshape(path_array[index_final], (-1,)).tolist()
    issame_list = issame_array[index_final].tolist()

    for path_i in path_list_i:
        path_list.append(test_set[path_i.cls].image_paths[path_i.idx])

    with open(test_pairs, "w") as f:
        for i in range(len(issame_list)):
            f.write(path_list[2*i] + ',' + path_list[2*i+1] + ',' + str(int(issame_list[i]))+'\n')
        print("Dump information to ", test_pairs)

    return path_list, issame_list

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned face patches.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch in the test set.', default=100)
    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file', default='checkpoint/20171219-004017/20171219-004017_49975.pb')
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--test_pairs', type=str,
        help='The file containing the pairs to use for validation. Will generate one if the specified file not exist', default='data/test_pairs.txt')
    parser.add_argument('--nrof_images', type=int,
        help='Number of images used for testing.', default=0)
    parser.add_argument('--nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
