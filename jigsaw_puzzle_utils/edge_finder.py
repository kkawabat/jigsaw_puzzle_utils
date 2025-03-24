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
        self.raw_img = cv.imread(input_path)
        self.img_hsv = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2HSV)
        self.temp_img = None


    def generate_edge_highlights(self, output_path):
        """given an image of puzzle pieces generate another image of edge pieces highlight"""

        threshold_img = self.run_threshold()

        cv2.imshow('threshold_img', threshold_img)
        contours = self.get_contours(threshold_img)


        cv2.drawContours(self.raw_img, contours, -1, (0, 255, 0), 2)
        cv2.imshow('Image with Contours', self.raw_img)


        cv.waitKey(0)  # Wait for a keystroke in the window
        cv2.destroyAllWindows()


    def run_threshold(self):
        """press `a` to add another mask, `r` to reset, `q` to finish"""
        self.temp_img = self.quantize_color(self.raw_img)
        cur_img = self.temp_img
        cv.imshow("image", cur_img)
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
            mask = cv2.inRange(cur_img, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(cur_img, cur_img, mask=mask_inv)

            cv2.imshow('threshold_map', result)
            cv2.moveWindow("threshold_map", 1000, 0)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyWindow('image')
                cv2.destroyWindow('threshold_map')

                threshold_img_grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                threshold_img_grey[threshold_img_grey != 0] = 255

                kernel = np.ones((5, 5), np.uint8)
                mask_cleaned = cv2.morphologyEx(threshold_img_grey, cv2.MORPH_CLOSE, kernel, iterations=2)
                return mask_cleaned
            elif cv2.waitKey(10) & 0xFF == ord('a'):
                cv.imshow("image", result)
                cur_img = result
                print('saved')
            elif cv2.waitKey(10) & 0xFF == ord('r'):
                cv.imshow("image", self.temp_img)
                cur_img = self.temp_img

    def set_threshold(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at coordinates: ({x}, {y})")
            if self.raw_img.shape[0] < y or self.raw_img.shape[1] < x:
                print('skipped')
                return

            selected_hsv = self.temp_img[y, x, :]
            cv2.setTrackbarPos('HMin', 'image', max(int(selected_hsv[0]) - 10, 0))
            cv2.setTrackbarPos('HMax', 'image', min(int(selected_hsv[0]) + 10, 179))
            cv2.setTrackbarPos('SMin', 'image', max(int(selected_hsv[1]) - 10, 0))
            cv2.setTrackbarPos('SMax', 'image', min(int(selected_hsv[1]) + 10, 255))
            cv2.setTrackbarPos('VMin', 'image', max(int(selected_hsv[2]) - 10, 0))
            cv2.setTrackbarPos('VMax', 'image', min(int(selected_hsv[2]) + 10, 255))

    def quantize_color(self, img):
        image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_bins = 10  # Example: Reduce hue to 10 levels
        saturation_bins = 5  # Example: Reduce saturation to 5 levels
        value_bins = 5  # Example: Reduce value to 5 levels

        quantized_hue = image_hsv[:, :, 0] // (180 // hue_bins) * (180 // hue_bins)
        quantized_saturation = image_hsv[:, :, 1] // (256 // saturation_bins) * (256 // saturation_bins)
        quantized_value = image_hsv[:, :, 2] // (256 // value_bins) * (256 // value_bins)
        quantized_hsv = np.stack([quantized_hue, quantized_saturation, quantized_value], axis=-1).astype(np.uint8)
        return quantized_hsv


    def get_contours(self, threshold_img):
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


if __name__ == '__main__':
    DATA_DIR = join(dirname(dirname(abspath(jigsaw_puzzle_utils.__file__))), "tests", 'data')
    input_path = join(DATA_DIR, 'sample_pieces.jpg')
    output_path = join(DATA_DIR, 'edge_highlights.jpg')
    ef = EdgeFinder(input_path)
    ef.generate_edge_highlights(output_path)