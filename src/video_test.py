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
import cv2
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

MIN_INPUT_SIZE = 160
def faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor):
  #print(img.shape)
  h=img.shape[0]
  w=img.shape[1]
  minl=np.amin([h, w])
  print("original image is %dx%d" % (w, h))

  scale = 1
  if minl > MIN_INPUT_SIZE:
    scale = minl // MIN_INPUT_SIZE
    hs=int(np.ceil(h/scale))
    ws=int(np.ceil(w/scale))
    #im_data = imresample(img, (hs, ws))
    im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    print("scaled image is %dx%d" % (ws, hs))
  else:
    im_data = img

  face_locations, points = align.detect_face.detect_face(im_data, minsize, pnet, rnet, onet, threshold, factor)
  #for face_location in face_locations:
  #  face_location[0:4] = face_location[0:4] * scale

  return face_locations, points, scale

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
  total_cpu_time = 0
  total_real_time = 0

  video = args.video or 0 # if args.video == '': open camera
  videoCapture = cv2.VideoCapture(args.video)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
          int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print("fps = {}, size = {}".format(fps, size))

  if not videoCapture.isOpened():
    sys.exit("open video failed")

  if args.out:
    videoWriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'H264'), fps, size)
  count = 0
  while True:
    success, frame = videoCapture.read()
    if not success:
      break

    start_t = time.time()
    start_c = time.clock()
    
    img = frame#misc.imread(os.path.expanduser(args.image), mode='RGB')
    face_locations, points, scale = faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor)

    end_t = time.time()
    end_c = time.clock()

    total_cpu_time += end_c - start_c
    total_real_time += end_t - start_t

    #print("I found {} face(s) in this photograph.".format(len(face_locations)))

    pil_image = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_image)
    p_shape = [0,5,1,6,2,7,3,8,4,9]
    i = 0
    for face_location in face_locations:

        # Print the location of each face in this image
        left, top, right, bottom = face_location[0:4] * scale
        landmarks = points[p_shape,i] * scale
        #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        drawRectange(draw, (left, top, right, bottom), width = 4, outline='green')
        draw.text((left+4,top+4), "%.2f" % (face_location[4]), fill='green')
        drawPoint(draw, (landmarks), width = 3, fill='green')

        i += 1
    if args.out:
      videoWriter.write(np.array(pil_image))
    else:
      cv2.imshow("Oto Video", np.array(pil_image))

    count = count + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  print("The inference time cost: %.2fs, fps: %.2f" % (total_real_time, count/total_real_time))
  videoCapture.release()
  cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video', type=str, default = '', help='Video to load')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

