import json

import cv2
import cv2 as cv
from os.path import join, abspath, dirname

import numpy as np

import jigsaw_puzzle_utils
from jigsaw_puzzle_utils.jigsaw import Jigsaw


def nothing(x):
    pass


class EdgeFinder:
    def __init__(self, input_path):
        self.input_path = input_path
        self.raw_img = None
        self.img_hsv = None
        self.quantized_img = None
        self.threshold_img = None
        self.contours = None
        self.labeled_pieces_img = None
        self.contour_img = None

    def run(self):
        """given an image of puzzle pieces generate another image of edge pieces highlight"""
        self.raw_img = cv.imread(self.input_path)
        self.img_hsv = cv2.cvtColor(self.raw_img, cv2.COLOR_BGR2HSV)
        self.threshold_img = self.run_threshold()
        self.contours = self.get_contours(self.threshold_img)
        self.labeled_pieces_img = self.label_pieces(self.threshold_img.shape[1], self.threshold_img.shape[0], self.contours)


    def run_threshold(self ):
        """press `a` to add another mask, `r` to reset, `q` to finish"""
        self.quantized_img = self.quantize_color(self.raw_img)
        cur_img = self.quantized_img

        cv2.namedWindow('image', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('threshold_map', cv2.WINDOW_KEEPRATIO)
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
            h_min = cv2.getTrackbarPos('HMin', 'image')
            h_max = cv2.getTrackbarPos('HMax', 'image')
            s_min = cv2.getTrackbarPos('SMin', 'image')
            s_max = cv2.getTrackbarPos('SMax', 'image')
            v_min = cv2.getTrackbarPos('VMin', 'image')
            v_max = cv2.getTrackbarPos('VMax', 'image')

            # Set minimum and maximum HSV values to display
            lower = np.array([h_min, s_min, v_min])
            upper = np.array([h_max, s_max, v_max])

            # Convert to HSV format and color threshold
            mask = cv2.inRange(cur_img, lower, upper)
            mask_inv = cv2.bitwise_not(mask)
            result = cv2.bitwise_and(cur_img, cur_img, mask=mask_inv)

            cv2.imshow('threshold_map', result)
            cv2.moveWindow("threshold_map", 1000, 0)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                cv2.destroyWindow('image')
                cv2.destroyWindow('threshold_map')

                threshold_img_grey = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
                threshold_img_grey[threshold_img_grey != 0] = 255

                kernel = np.ones((5, 5), np.uint8)
                mask_cleaned = cv2.morphologyEx(threshold_img_grey, cv2.MORPH_CLOSE, kernel, iterations=2)
                return mask_cleaned
            elif key == ord('a'):
                cv.imshow("image", result)
                cur_img = result
                print('saved')
            elif key == ord('r'):
                cv.imshow("image", self.quantized_img)
                cur_img = self.quantized_img
                print('reset')

    def set_threshold(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked at coordinates: ({x}, {y})")
            if self.quantized_img.shape[0] < y or self.quantized_img.shape[1] < x:
                print('skipped')
                return

            selected_hsv = self.quantized_img[y, x, :]
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
        quantized_hsv = cv2.medianBlur(quantized_hsv, 3)
        return quantized_hsv


    def get_contours(self, threshold_img):
        kernel = np.ones((3, 3), np.uint8)
        mask_cleaned = cv2.morphologyEx(threshold_img, cv2.MORPH_OPEN, kernel, iterations=1)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)

        edges = cv2.Canny(mask_cleaned, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.namedWindow('contour_image', cv2.WINDOW_KEEPRATIO)

        cv2.createTrackbar('h_min', 'contour_image', 0, threshold_img.shape[0], nothing)
        cv2.createTrackbar('h_max', 'contour_image', 0, threshold_img.shape[0], nothing)
        cv2.createTrackbar('w_min', 'contour_image', 0, threshold_img.shape[1], nothing)
        cv2.createTrackbar('w_max', 'contour_image', 0, threshold_img.shape[1], nothing)

        while True:
            h_min = cv2.getTrackbarPos('h_min', 'contour_image')
            h_max = cv2.getTrackbarPos('h_max', 'contour_image')
            w_min = cv2.getTrackbarPos('w_min', 'contour_image')
            w_max = cv2.getTrackbarPos('w_max', 'contour_image')

            filtered_contour = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Keep only large enough objects
                if w_max > w > w_min and h_max > h > h_min and len(contour) > 30:
                    filtered_contour.append(contour)

            contour_img = self.raw_img.copy()
            cv2.drawContours(contour_img, filtered_contour, -1, (0, 255, 0), 2)
            cv2.imshow('contour_image', contour_img)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                cv2.destroyWindow('contour_image')
                return filtered_contour

    def refine_contours(self):
        for cnt in self.contours:
            x, y, w, h = cv2.boundingRect(cnt)
            x_min = max(int(x - w * .1), 0)
            y_min = max(int(y - h * .1), 0)
            x_max = min(int(x_min + w * 1.2), self.raw_img.shape[1])
            y_max = min(int(y_min + h * 1.2), self.raw_img.shape[0])
            cropped_img = self.raw_img[y_min:y_max, x_min:x_max].copy()

            kernel = np.ones((3, 3), np.uint8)
            mask_cleaned = cv2.morphologyEx(cropped_img, cv2.MORPH_OPEN, kernel, iterations=1)
            dilated = cv2.dilate(mask_cleaned, kernel, iterations=1)
            gradient = cv2.morphologyEx(dilated, cv2.MORPH_GRADIENT, kernel)
            gradient_hsv = cv2.cvtColor(gradient, cv2.COLOR_BGR2HSV)

            edges = cv.Canny(gradient_hsv[:, :, 2], 50, 150)
            contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            filtered_contours = self.custom_contour_filter(contours, w, h)
            v2 = cropped_img * 0
            cv2.drawContours(v2, contours, -1, (0, 255, 0), 1)

            v3 = cropped_img * 0
            kernel = np.ones((5, 5), np.uint8)
            closed = cv2.morphologyEx(v2, cv2.MORPH_CLOSE, kernel)
            # cv2.drawContours(v3, approx_contour, -1, (0, 255, 0), 1)

            img_edge = cropped_img.copy()
            cv2.drawContours(img_edge, filtered_contours, -1, (0, 255, 0), 2)

            cv2.imshow('cropped_img', cropped_img)
            cv2.imshow('gradient_v', gradient_hsv[:,:,2])
            cv2.imshow('edges', edges)
            cv2.imshow('img_edge', img_edge)
            cv2.imshow('v2', v2)
            cv2.imshow('v3', closed)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        pass

    def label_pieces(self, height, width, contours):
        org_blank_img = np.zeros((width, height, 3), dtype=np.uint8)

        blank_img = org_blank_img.copy()
        cv2.drawContours(blank_img, [np.array(c) for c in contours[0::2]], -1, (0, 255, 0), 2)
        img = cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB)

        for i, c in enumerate(contours):
            # for some reason there are two set of contours that overlap each other so we skip one of them
            if i % 2 == 0:
                continue
            x = int(np.mean([cc[0][0] for cc in c]))
            y = int(np.mean([cc[0][1] for cc in c]))
            cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)

        return img

    def custom_contour_filter(self, contours, w, h):
        """
        filter rule from heuristics
        - filter contours with too little points
        - filter contours if the size of the contour is way smaller than the original contour
        - filter contours if the ends of the contour is further than half the dimension (it's probably not a closed loop)
        """
        filtered_contours = []
        for cnt in contours:
            x, y, _w, _h = cv2.boundingRect(cnt)
            if len(cnt) < 20:
                continue

            if _w * _h < (w * h) * .5:
                continue

            if np.linalg.norm(cnt[0] - cnt[-1]) > max(w, h) * .5:
                continue
            filtered_contours.append(cnt)
        return filtered_contours

    def process_jigsaws(self):
        # for c in self.contours:
        #     j = Jigsaw(c)
        #     j.analyze()
        #     j.plot()
        self.contours[0]
        j = Jigsaw(self.contours[0])
        j.analyze()
        j.plot()

        self.contours[821]
        j = Jigsaw(self.contours[821])
        j.analyze()
        j.plot()

    def save(self, output_path):
        cv2.imwrite(join(output_path, 'origin.jpg'), self.raw_img)
        cv2.imwrite(join(output_path, 'quantized.jpg'), self.quantized_img)
        cv2.imwrite(join(output_path, 'threshold.jpg'), self.threshold_img)
        cv2.imwrite(join(output_path, 'labeled_pieces.jpg'), self.labeled_pieces_img)

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        with open(join(output_path, 'contours.json'), 'w') as ofile:
            json.dump({"contours": self.contours,
                       "height": self.threshold_img.shape[1],
                       "width": self.threshold_img.shape[0]}, ofile, cls=NumpyEncoder)
        pass

    @classmethod
    def load(cls, input_path):
        ef = cls(input_path)
        ef.raw_img = cv2.imread(join(input_path, 'origin.jpg'))
        ef.quantized_img = cv2.imread(join(input_path, 'quantized.jpg'))
        ef.threshold_img = cv2.imread(join(input_path, 'threshold.jpg'))
        ef.labeled_pieces_img = cv2.imread(join(input_path, 'labeled_pieces.jpg'))

        with open(join(input_path, 'contours.json'), 'r') as ifile:
            data_dict = json.load(ifile)
        ef.contours = [np.array(c) for c in data_dict['contours']]
        return ef
