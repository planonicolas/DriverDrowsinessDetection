#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2019 Iván de Paz Centeno
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

import cv2
from mtcnn import MTCNN
import time

detector = MTCNN()

image = cv2.cvtColor(cv2.imread("try.jpg"), cv2.COLOR_BGR2RGB)
start_time = time.time()
result = detector.detect_faces(image)
print("--- %s seconds ---" % (time.time() - start_time))

# Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
image_mod = image
i=0
for r in result:

    b = r['box']
    keypoints = r['keypoints']
    cv2.rectangle(image_mod,
              (b[0], b[1]),
              (b[0]+b[2], b[1] + b[3]),
              (0,155,255),
              2)

    cv2.circle(image_mod,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image_mod,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image_mod,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image_mod,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image_mod,(keypoints['mouth_right']), 2, (0,155,255), 2)

    cv2.imwrite("try_"+str(i)+".jpg", cv2.cvtColor(image_mod, cv2.COLOR_RGB2BGR))
    i+=1
    image_mod = image

print(result)
