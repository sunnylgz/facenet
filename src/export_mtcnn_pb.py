#! /usr/bin/python3
"""convert mtcnn models from .npy to tensorflow pb files
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

import sys
import os
import argparse
import align.detect_face
#import facenet
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.tools.freeze_graph import freeze_graph as freeze_graph

ckpt_name = "./checkpoint/model.chkp"
pbtxt_name = "mtcnn.pbtxt"
def main(args):
  with tf.Session() as sess:
      pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
      saver = tf.train.Saver(max_to_keep = 0, keep_checkpoint_every_n_hours = 2)

      output_node_names = "pnet/conv4-2/BiasAdd,pnet/prob1,rnet/conv5-2/conv5-2,rnet/prob1,onet/conv6-2/conv6-2,onet/conv6-3/conv6-3,onet/prob1"

      output_graph_def = graph_util.convert_variables_to_constants(  
          sess,   
          sess.graph_def,#input_graph_def,   
          output_node_names.split(",") # We split on comma for convenience  
      )
      saver.save(sess, ckpt_name)

      '''
      with tf.gfile.GFile(args.out, "wb") as f:  
          f.write(output_graph_def.SerializeToString())
          print("save to", args.out)
      print("%d ops in the final graph." % len(output_graph_def.node))
      '''
      tf.train.write_graph(sess.graph_def, "./", pbtxt_name)
      freeze_graph(pbtxt_name, None, False, ckpt_name, output_node_names, None, None, args.out, True, "")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-o', '--out', type=str, default = 'frozen_mtcnn.pb', help='save output to disk')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
