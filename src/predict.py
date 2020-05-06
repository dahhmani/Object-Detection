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

import cv2 as cv
import glob

from yolo import YOLOv3
from ssd import MobileNet_SSD

def main():
	pretrainedModelsPath = '../models'
	
	detector = YOLOv3(pretrainedModelsPath) # specify detection model
	predict_images(detector)
	# predict_video(detector)

def predict_images(detector):
	inputPath = '../data/input/frames/*'
	outputPath = '../data/output/frames'
	dataset = sorted(glob.glob(inputPath))

	for image in dataset:
		frame = cv.imread(image)
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)
		detector.logResults(frame, detections, outputPath, image[len(inputPath)-1:-4])

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		cv.imwrite(f'{outputPath}/{image[len(inputPath)-1:]}', frame)
		if cv.waitKey(1) >= 0:
			break

def predict_video(detector):
	video = 'testVideo.mp4'
	inputPath = '../data/input/video'
	outputPath = '../data/output/video'
	inputVideo = cv.VideoCapture(f'{inputPath}/{video}')
	outputVideo = cv.VideoWriter(f'{outputPath}/{video}', cv.VideoWriter_fourcc(*'XVID'), 30, (int(inputVideo.get(3)),int(inputVideo.get(4))))
	
	frameIndex = 0
	while True:
		frameIndex += 1
		read, frame = inputVideo.read()
		if not read:
			break
		detections = detector.predict(frame)
		detector.drawBoxes(frame, detections)
		detector.logResults(frame, detections, outputPath, f'{video[:-4]}_{frameIndex}')

		cv.namedWindow('Live', cv.WINDOW_NORMAL)
		cv.imshow('Live', frame)
		outputVideo.write(frame)
		if cv.waitKey(1) >= 0:
			break

if __name__ == '__main__':
    main()
