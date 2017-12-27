#! /usr/bin/python3
"""find faces from input image based on mtcnn and locate the locations and landmarks
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
from PIL import Image, ImageDraw
import tensorflow as tf
from scipy import misc
import numpy as np
import facenet
import datetime,time
import align.detect_face

def drawRectange(draw, rect, width = 1, outline = None, fill = None):
  if width > 1:
    for x in range(-width//2, width//2):
      box = (rect[0] - x, rect[1] - x, rect[2] + x, rect[3] + x)
      draw.rectangle(box, outline=outline, fill=fill)
  else:
    draw.rectangle(rect, outline=outline, fill=fill)

def drawPoint(draw, xy, width = 1, fill = None):
  if width > 1:
    xs = xy[0::2]
    ys = xy[1::2]

    for i in range(len(xs)):
      box = (xs[i] - width//2, ys[i] - width//2, xs[i] + width//2, ys[i] + width//2)
      draw.ellipse(box, fill=fill)
  else:
    draw.point(xy, fill=fill)

def main(args):
  print('Creating networks and loading parameters')
  
  with tf.Graph().as_default():
      #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
      sess = tf.Session()
      with sess.as_default():
          pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
  factor = 0.709 # scale factor

  img = misc.imread(os.path.expanduser(args.image), mode='RGB')

  start_t = time.time()
  start_c = time.clock()

  face_locations, points = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

  end_t = time.time()
  end_c = time.clock()

  total_cpu_time = end_c - start_c
  total_real_time = end_t - start_t
  print("The inference time cost: %.2fs" % (total_real_time))

  print("I found {} face(s) in this photograph.".format(len(face_locations)))

  pil_image = Image.fromarray(img)
  draw = ImageDraw.Draw(pil_image)
  p_shape = [0,5,1,6,2,7,3,8,4,9]
  i = 0
  for face_location in face_locations:

      # Print the location of each face in this image
      left, top, right, bottom = face_location[0:4]
      landmarks = points[p_shape,i]
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

      drawRectange(draw, (left, top, right, bottom), width = 4, outline='green')
      draw.text((left+4,top+4), "%.2f" % (face_location[4]), fill='green')
      drawPoint(draw, (landmarks), width = 3, fill='green')

      if args.dump:
        face_image = img[int(top):int(bottom), int(left):int(right)]
        pil_image = Image.fromarray(face_image)
        pil_image.save("%d.jpg"%i)
      i += 1
  if args.out:
    pil_image.save(args.out)
  else:
    pil_image.show()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image', type=str, help='Images to load')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--dump', action="store_true",
                        default=False,
                        help='dump face cropped results')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

