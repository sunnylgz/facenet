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
import cv2

def main(args):
  videoCapture = cv2.VideoCapture(args.video)
  fps = videoCapture.get(cv2.CAP_PROP_FPS)
  size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),   
        int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

  success, frame = videoCapture.read()
  while success:
    cv2.imshow("Oto Video", frame)
    cv2.waitKey(1000//int(fps))
    success, frame = videoCapture.read()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('video', type=str, help='Video to load')
    parser.add_argument('-o', '--out', type=str, default = '', help='save output to disk')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


