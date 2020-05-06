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

class DetectionModel:
    def drawBoxes(self, frame, detections):
        indices, classIDs, confidences, boxes = detections
        if len(indices) > 0:
            for i in indices:
                x, y, w, h = boxes[i]
                caption = f'{self.labels[classIDs[i]]}: {confidences[i]:.3f}'
                classColor = self.classColors[classIDs[i]].tolist()

                cv.rectangle(frame, (x, y), (x+w, y+h), classColor, 2)
                cv.putText(frame, caption, (x, y-5), cv.LINE_AA, 0.5, classColor, 2)

    def logResults(self, frame, detections, outputPath, frameName):
        indices, classIDs, confidences, boxes = detections
        frameHeight, frameWidth = frame.shape[:2]
        
        with open(f'{outputPath}/{frameName}.txt', 'w') as file:
            if len(indices) > 0:
                for i in indices.reshape(-1):
                    x, y, w, h = boxes[i]
                    left, top, right, bottom = x, y, x+w, y+h
                    if left < 0:
                        left = 0
                    if right >= frameWidth:
                        right = frameWidth - 1
                    if top < 0:
                        top = 0
                    if bottom >= frameHeight:
                        bottom = frameHeight - 1
                    file.write(f"{self.labels[classIDs[i]].replace(' ','')} {confidences[i]} {left} {top} {right} {bottom}\n")
