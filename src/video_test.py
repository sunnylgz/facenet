#! /usr/bin/env python3
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
__debug = True
margin = 44 # Margin for the crop around the bounding box (height, width) in pixels.
image_size = 160 # Image size (height, width) of cropped face in pixels.

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
  #print("original image is %dx%d" % (w, h))

  scale = 1
  if minl > MIN_INPUT_SIZE:
    scale = minl // MIN_INPUT_SIZE
    hs=int(np.ceil(h/scale))
    ws=int(np.ceil(w/scale))
    #im_data = imresample(img, (hs, ws))
    im_data = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    #print("scaled image is %dx%d" % (ws, hs))
  else:
    im_data = img

  face_locations, points = align.detect_face.detect_face(im_data, minsize, pnet, rnet, onet, threshold, factor)
  #for face_location in face_locations:
  #  face_location[0:4] = face_location[0:4] * scale

  return face_locations, points, scale

def create_facenet(facenet_model):
  if __debug:
    start_t = time.time()
    start_c = time.clock()

  with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
      # Load the model
      facenet.load_model(facenet_model)

      if __debug:
        end_t = time.time()
        end_c = time.clock()

        elapsed_real_time = end_t - start_t
        elapsed_user_time = end_c - start_c
        print("load face model cost (real/user): %.2fs/%.2fs" % (elapsed_real_time, elapsed_user_time))
        start_t,start_c = end_t,end_c

      # Get input and output tensors
      images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
      embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
      phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

      # Run forward pass to calculate embeddings
      emb_fun = lambda images : sess.run(embeddings, feed_dict={ images_placeholder: images, phase_train_placeholder:False })

  return emb_fun

def crop_face(img, bounding_boxes, scale):
  img_size = np.asarray(img.shape)[0:2]

  prewhitened_faces = []
  for bounding_box in bounding_boxes:
    det = np.squeeze(bounding_box[0:4]) * scale
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    prewhitened_face = facenet.prewhiten(aligned)
    prewhitened_faces.append(prewhitened_face)

  return prewhitened_faces

def main(args):
  print('Creating networks and loading parameters')
  
  with tf.Graph().as_default():
      #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
      sess = tf.Session()
      with sess.as_default():
          pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

  emb_fun = create_facenet(args.model)
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

    if __debug:
      start_t = time.time()
      start_c = time.clock()
    
    img = frame#misc.imread(os.path.expanduser(args.image), mode='RGB')
    face_locations, points, scale = faster_face_detect(img, minsize, pnet, rnet, onet, threshold, factor)
    faces = crop_face(img, face_locations, scale)
    face_embs = emb_fun(faces)

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

    if __debug:
      end_t = time.time()
      end_c = time.clock()

      total_cpu_time += end_c - start_c
      total_real_time += end_t - start_t

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  if __debug:
    print("The inference time cost: %.2fs, fps: %.2f" % (total_real_time, count/total_real_time))
  videoCapture.release()
  cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video', type=str, default = '', help='Video to load')
    parser.add_argument('--model', type=str, default = './checkpoint', help='Facenet model, either to be a checkpoint folder or pb file')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

