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
import pickle
import numpy as np
import cv2
import facenet
import datetime,time
import align.detect_face

__debug = True
margin = 44 # Margin for the crop around the bounding box (height, width) in pixels.
image_size = 160 # Image size (height, width) of cropped face in pixels.
cnt_skip_frames = 29

# MEDIANFLOW has the lowest time cost
# GOTURN has issue under opencv3.4.0
# 
def create_tracker(tracker_type = 'KCF'):
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
  tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
  #tracker_type = tracker_types[2]

  if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
  else:
    if tracker_type == 'BOOSTING':
      tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
      tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
      tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
      tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
      tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
      tracker = cv2.TrackerGOTURN_create()

  return tracker

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
def image_down_scale(img):
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
    img_down = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_AREA)
    #print("scaled image is %dx%d" % (ws, hs))
  else:
    img_down = img

  return img_down, scale

def create_facenet(sess, facenet_model):
  if __debug:
    start_t = time.time()
    start_c = time.clock()

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

def create_mtcnn(sess, model):
    facenet.load_model(model)

    pnet_fun = lambda img : sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'), feed_dict={'pnet/input:0':img})
    rnet_fun = lambda img : sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'), feed_dict={'rnet/input:0':img})
    onet_fun = lambda img : sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'), feed_dict={'onet/input:0':img})
    return pnet_fun, rnet_fun, onet_fun

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
          pnet, rnet, onet = create_mtcnn(sess, "./frozen_mtcnn.pb")


          if args.model:
            emb_fun = create_facenet(sess, args.model)
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.9 ]  # three steps's threshold
  factor = 0.709 # scale factor
  total_cpu_time = 0
  total_real_time = 0

  video = args.video or 0 # if args.video == '': open camera
  videoCapture = cv2.VideoCapture(video)
  #videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  #videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
          int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
  print("fps = {}, size = {}".format(fps, size))

  if not videoCapture.isOpened():
    sys.exit("open video failed")

  if args.out:
    videoWriter = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'H264'), fps, size)
  count = 0
  if __debug:
    start_t = time.time()
    start_c = time.clock()

  trackers = []
  tracker_cnt = 0
  while True:
    success, frame = videoCapture.read()
    if not success:
      break


    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)
    img_down,scale = image_down_scale(frame) #misc.imread(os.path.expanduser(args.image), mode='RGB')
    if count % (cnt_skip_frames+1) == 0:
      face_locations, points = align.detect_face.detect_face(img_down, minsize, pnet, rnet, onet, threshold, factor)
      if args.model:
        faces = crop_face(frame, face_locations, scale)
        if len(faces):
          face_embs = emb_fun(faces)

          if args.classifier_filename:
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
              (model, class_names) = pickle.load(infile)

            predictions = model.predict_proba(face_embs)
            best_class_indices = np.argmax(predictions, axis=1)
            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

      #print("I found {} face(s) in this photograph.".format(len(face_locations)))

      p_shape = [0,5,1,6,2,7,3,8,4,9]
      i = 0
      trackers.clear()
      for face_location in face_locations:

          # Print the location of each face in this image
          left, top, right, bottom = face_location[0:4] * scale
          landmarks = points[p_shape,i] * scale
          #print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

          drawRectange(draw, (left, top, right, bottom), width = 4, outline='green')
          if args.classifier_filename:
            class_name = class_names[best_class_indices[i]] if best_class_probabilities[i] > 0.9 else 'unknow'
            #print('%4d  %s: %.3f' % (i, class_name, best_class_probabilities[i]))
            draw.text((left+4,top+4), "%s %.2f" % (class_name,best_class_probabilities[i]), fill='green')
          else:
            draw.text((left+4,top+4), "%.2f" % (face_location[4]), fill='green')
          drawPoint(draw, (landmarks), width = 3, fill='green')

          left, top, right, bottom = face_location[0:4]
          bbox = (left, top, right-left+1, bottom-top+1)
          trackers.append(create_tracker(args.tracker))
          trackers[i].init(img_down, bbox)
          i += 1
      #tracker_cnt = i
    else:
      i = 0
      for tracker in trackers:
        ok, bbox = tracker.update(img_down)
        if ok:
          left, top, right, bottom = (bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1])
          left, top, right, bottom = (left*scale, top*scale, right*scale, bottom*scale)
          drawRectange(draw, (left, top, right, bottom), width = 4, outline='green')

          if args.classifier_filename:
            class_name = class_names[best_class_indices[i]] if best_class_probabilities[i] > 0.9 else 'unknow'
            draw.text((left+4,top+4), "%s %.2f" % (class_name,best_class_probabilities[i]), fill='green')

        i += 1

    if args.out:
      videoWriter.write(np.array(pil_image))
    else:
      cv2.imshow("Oto Video", np.array(pil_image))

    count = count + 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  if __debug:
    end_t = time.time()
    end_c = time.clock()

    total_cpu_time += end_c - start_c
    total_real_time += end_t - start_t
    print("The inference time cost: %.2fs, fps: %.2f" % (total_real_time, count/total_real_time))
  videoCapture.release()
  cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--video', type=str, default = '', help='Video to load')
    parser.add_argument('--model', type=str, default = '', help='Facenet model, either to be a checkpoint folder or pb file')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    parser.add_argument('--tracker', type=str, default = 'KCF', help='Tracker type, \'BOOSTING\', \'MIL\',\'KCF\', \'TLD\', \'MEDIANFLOW\', \'GOTURN\'')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--classifier_filename',
        type=str, default = '',
        help='Classifier model file name as a pickle (.pkl) file. ' +
        'For training this is the output and for classification this is an input.')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

