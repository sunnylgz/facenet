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

def main(args):
  print('Creating networks and loading parameters')
  
  with tf.Graph().as_default():
      #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
      sess = tf.Session()
      with sess.as_default():
          pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  minsize = 20 # minimum size of face
  threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
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
  i = 0
  for face_location in face_locations:

      # Print the location of each face in this image
      left, top, right, bottom = face_location[0:4]
      print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

      draw.rectangle((left, top, right, bottom), outline='red')
      draw.text((left,top), "%.2f" % (face_location[4]), fill='red')

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

