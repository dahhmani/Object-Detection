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

class YOLOv4(DetectionModel):
	def __init__(self, pretrainedModelsPath):
		inputResolution = (512, 512)
		configPath = f'{pretrainedModelsPath}/YOLO/yolov4.cfg'
		weightsPath = f'{pretrainedModelsPath}/YOLO/yolov4.weights'
		labelsPath = f'{pretrainedModelsPath}/YOLO/coco.names'
		self.labels = open(labelsPath).read().strip().split('\n') # COCO class labels (80 classes)
		np.random.seed(1)
		self.classColors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
		
		self._loadModel(configPath, weightsPath)
		self.net.setInputParams(size=inputResolution, scale=1.0/255, swapRB=True)

	def predict(self, frame, minimumConfidence=0.5, minimumOverlap=0.3):
		start = time()
		classIDs, confidences, boxes = self.net.detect(frame, minimumConfidence, minimumOverlap)
		end = time()
		print(f'Forward pass of a single frame took {end-start:.3f} s')
		indices = np.arange(len(boxes))

		return (indices, classIDs.reshape(-1), confidences.reshape(-1), boxes)

	def _loadModel(self, configPath, weightsPath):
		self.net = cv.dnn_DetectionModel(configPath, weightsPath) # pre-trained on COCO
		self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
		
class YOLOv3(DetectionModel):
	def __init__(self, pretrainedModelsPath):
		self.inputResolution = (416, 416)
		configPath = f'{pretrainedModelsPath}/YOLO/yolov3.cfg'
		weightsPath = f'{pretrainedModelsPath}/YOLO/yolov3.weights'
		labelsPath = f'{pretrainedModelsPath}/YOLO/coco.names'
		self.labels = open(labelsPath).read().strip().split('\n') # COCO class labels (80 classes)
		np.random.seed(1)
		self.classColors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype=np.uint8)
		
		self._loadModel(configPath, weightsPath) 

	def predict(self, frame, minimumConfidence=0.5, minimumOverlap=0.3):
		frameHeight, frameWidth = frame.shape[:2]
		preprocessedFrame = cv.dnn.blobFromImage(frame, 1/255, self.inputResolution, swapRB=True, crop=False)
		self.net.setInput(preprocessedFrame)

		start = time()
		outputs = self.net.forward(self.outputLayers)
		end = time()
		print(f'Forward pass of a single frame took {end-start:.3f} s')

		return self._decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence, minimumOverlap)

	def _loadModel(self, configPath, weightsPath):
		self.net = cv.dnn.readNetFromDarknet(configPath, weightsPath) # pre-trained on COCO
		self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
		layers = self.net.getLayerNames()
		self.outputLayers = [layers[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

	@staticmethod
	def _decodeOutputs(outputs, frameWidth, frameHeight, minimumConfidence, minimumOverlap):
		classIDs, confidences, boxes = [], [], []
		for output in outputs:
			for detection in output:
				scores = detection[5:] # class probabilities
				classID = np.argmax(scores)
				confidence = scores[classID]

				# filter out weak predictions 
				if confidence > minimumConfidence:
					# map box parameters from YOLO representation to opencv representation
					box = detection[0:4] * [frameWidth, frameHeight, frameWidth, frameHeight]
					x_center, y_center, width, height = box
					x = x_center - width/2
					y = y_center - height/2
				
					classIDs.append(classID)
					confidences.append(float(confidence))
					boxes.append([int(x), int(y), int(width), int(height)])

		# apply non-maximum suppression
		indices = cv.dnn.NMSBoxes(boxes, confidences, minimumConfidence, minimumOverlap).reshape(-1)

		return (indices, classIDs, confidences, boxes)
