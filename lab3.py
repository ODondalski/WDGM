from lab2 import *
import matplotlib
from matplotlib.pyplot import imshow
import numpy as np
from typing import Any


class GrayScaleTransform(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    " metoda zwracajaca obraz w skali szarosci jako obiekt klasy BaseImage "

    def to_gray(self) -> BaseImage:
        R, G, B = self.get_layers()

        avR = R * 0.299
        avG = G * 0.587
        avB = B * 0.114

        self.pixels = avR + avG + avB
        self.color_model = ColorModel.gray
        return self
    
    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        if alpha_beta[0] > 1 and alpha_beta[1] < 1 and alpha_beta[0] + alpha_beta[1] == 2:
            self.to_gray()
            L0, L1, L2 = self.data, self.data, self.data
            L0 = np.where(L0 * alpha_beta[0] > 255, 255, L0 * alpha_beta[0])
            L2 = np.where(L2 * alpha_beta[1] > 255, 255, L2 * alpha_beta[1])
            self.data = np.dstack((L0, L1, L2)).astype('i')
            self.color_model = ColorModel.sepia
            return self
        elif w is not None and alpha_beta[0] is None and alpha_beta[1] is None:
            self.fromRgbToGray()
            L0, L1, L2 = self.pixels, self.pixels, self.pixels
            L0 = np.where(L0 + 2 * w > 255, 255, L0 + 2 * w)
            L1 = np.where(L1 + w > 255, 255, L1 + w)
            self.pixels = np.dstack((L0, L1, L2)).astype('i')
            self.color_model = ColorModel.sepia
            return self
