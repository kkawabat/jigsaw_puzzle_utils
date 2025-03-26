import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.signal import find_peaks


class Jigsaw:
    def __init__(self, contour):
        self.contour = contour
        self.x = contour[:, 0, 0]
        self.y = contour[:, 0, 1]
        self.x_mean = None
        self.y_mean = None

    def analyze(self):
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)

        self.dist = np.sqrt((self.x - self.x_mean) ** 2 + (self.y - self.y_mean) ** 2)
        self.angle = np.arctan2((self.y - self.y_mean), (self.x - self.x_mean))

        peaks, _ = find_peaks(self.dist)
        self.x_peaks = self.x[peaks]
        self.y_peaks = self.y[peaks]


    def plot(self):
        colors = cm.rainbow(np.linspace(0, 1, len(self.angle)))
        plt.subplot(1, 2, 1)
        plt.scatter(self.angle, self.dist, c=colors[:,0:3])
        plt.subplot(1, 2, 2)
        plt.scatter(self.x, self.y, c=colors[:,0:3])
        plt.scatter(self.x_mean, self.y_mean)
        plt.scatter(self.x_peaks, self.y_peaks, marker='x', color='black')
        plt.gca().set_aspect('equal')
        plt.show()
