import json

import cv2
import cv2 as cv
from os.path import join, abspath, dirname

import numpy as np

import jigsaw_puzzle_utils

def nothing(x):
    pass


class Jigsaw:
    def __init__(self, contour):
        self.contour = contour
        self.raw_img = cv.imread(input_path)

    def