# Object Detection
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
This project implements single-stage object detection algorithms for self-driving cars.
Sample evaluation results are found [here](https://drive.google.com/drive/folders/1-xO9VBnPSBuyKxqMTnUafN0mK_VStSwW?usp=sharing).
<p align="center">
  <img src="https://github.com/dahhmani/Object-Detection/blob/master/data/output/frames/testImage.jpg?raw=true">
</p>

## Requirements
* Ubuntu / MacOS (Operating System)
* Python >= 3.8 (Programming Language)
* pip (Package Manager)

## Setup
Run the following commands in a new terminal:
```
git clone --recursive https://github.com/dahhmani/Object-Detection.git
cd <path to repository>/scripts/
source setup.sh
sh download_pretrained_models.sh
```

## Run
Run the following commands in a new terminal:

### Inference
```
cd <path to repository>/src/
source ../scripts/activate_venv.sh
python predict.py
```

### Evaluation
```
cd <path to repository>/src/mAP/
Follow the instructions in README.md
```
