import cv2
import cv2 as cv
from os.path import join, abspath, dirname

import numpy as np

import jigsaw_puzzle_utils

def nothing(x):
    pass


class EdgeFinder:
    def __init__(self, input_path):
        self.input_path = input_path,
        self.img = cv.imread(input_path)


    def generate_edge_highlights(self, output_path):
        """given an image of puzzle pieces generate another image of edge pieces highlight"""

        self.run_threshold()

        cv.imshow("image", self.img)
        cv2.setMouseCallback("image", self.select_bg_color_from_image)
        cv.waitKey(0)  # Wait for a keystroke in the window
        cv2.destroyAllWindows()


    def run_threshold(self):
        cv.imshow("image", self.img)
        cv2.createTrackbar('HMin', 'image', 0, 179, nothing)
        cv2.createTrackbar('HMax', 'image', 0, 179, nothing)
        cv2.createTrackbar('SMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('SMax', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMin', 'image', 0, 255, nothing)
        cv2.createTrackbar('VMax', 'image', 0, 255, nothing)
        cv2.setMouseCallback("image", self.set_threshold)


        while True:
            # Get current positions of all trackbars
            hMin = cv2.getTrackbarPos('HMin', 'image')
            hMax = cv2.getTrackbarPos('HMax', 'image')
            sMin = cv2.getTrackbarPos('SMin', 'image')
            sMax = cv2.getTrackbarPos('SMax', 'image')
            vMin = cv2.getTrackbarPos('VMin', 'image')
            vMax = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([hMin, sMin, vMin])
            upper = np.array([hMax, sMax, vMax])

            # Convert to HSV format and color threshold
            hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(self.img, self.img, mask=mask_inv)

            cv2.imshow('threshold_map', result)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    def set_threshold(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at coordinates: ({x}, {y})")
            if self.img.shape[0] < y or self.img.shape[1] < x:
                print('skipped')
                return
            img_hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            selected_hsv = img_hsv[y, x, :]
            cv2.setTrackbarPos('HMin', 'image', max(int(selected_hsv[0]) - 10, 0))
            cv2.setTrackbarPos('HMax', 'image', min(int(selected_hsv[0]) + 10, 179))
            cv2.setTrackbarPos('SMin', 'image', max(int(selected_hsv[1]) - 10, 0))
            cv2.setTrackbarPos('SMax', 'image', min(int(selected_hsv[1]) + 10, 255))
            cv2.setTrackbarPos('VMin', 'image', max(int(selected_hsv[2]) - 10, 0))
            cv2.setTrackbarPos('VMax', 'image', min(int(selected_hsv[2]) + 10, 255))




if __name__ == '__main__':
    DATA_DIR = join(dirname(dirname(abspath(jigsaw_puzzle_utils.__file__))), "tests", 'data')
    input_path = join(DATA_DIR, 'sample_pieces.png')
    output_path = join(DATA_DIR, 'edge_highlights.png')
    ef = EdgeFinder(input_path)
    ef.generate_edge_highlights(output_path)