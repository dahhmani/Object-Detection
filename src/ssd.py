"""
MIT License

Copyright (c) 2020 Mahmoud Dahmani

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2 as cv
from time import time

from model import DetectionModel

class MobileNet_SSD(DetectionModel):
	def __init__(self, pretrainedModelsPath):
		self.inputResolution = (300, 300)
		configPath = f'{pretrainedModelsPath}/MobileNet-SSD/MobileNetSSD_deploy.prototxt.txt'
		weightsPath = f'{pretrainedModelsPath}/MobileNet-SSD/MobileNetSSD_deploy.caffemodel'
		self.labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
	                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
	                    'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
	                    'sofa', 'train', 'tvmonitor'] # COCO class labels (20 classes)
		np.random.seed(1)
		self.classColors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
		
		self._loadModel(configPath, weightsPath) 

	def predict(self, frame, minimumConfidence=0.3):
		frameHeight, frameWidth = frame.shape[:2]
		preprocessedFrame = cv.dnn.blobFromImage(frame, 0.007843, self.inputResolution, 127.5)
		self.net.setInput(preprocessedFrame)

		start = time()
		outputs = self.net.forward()
		end = time()
		print(f'Forward pass of a single frame took {end-start:.3f} s')

		return self._decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence)

	def _loadModel(self, configPath, weightsPath):
		self.net = cv.dnn.readNetFromCaffe(configPath, weightsPath) # pre-trained on COCO
		self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

	@staticmethod
	def _decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence):
		classIDs, confidences, boxes = [], [], []
		for i in np.arange(outputs.shape[2]):
			confidence = outputs[0, 0, i, 2]
			if confidence > minimumConfidence:
				classID = int(outputs[0, 0, i, 1]) 
				box = outputs[0, 0, i, 3:7] * [frameWidth, frameHeight, frameWidth, frameHeight]
				startX, startY, endX, endY = box.astype('int')
				width = endX - startX
				height = endY - startY

				classIDs.append(classID)
				confidences.append(confidence)
				boxes.append([startX, startY, width, height])

		indices = np.arange(len(boxes))

		return (indices, classIDs, confidences, boxes)
