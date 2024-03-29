"""
lena_gray = cv2.imread('../data/lena.jpg', cv2.IMREAD_GRAYSCALE)
_, thresh_otsu = cv2.threshold(

    lena_gray,

    thresh=0,

    maxval=255,

    type=cv2.THRESH_BINARY + cv2.THRESH_OTSU

)

plt.imshow(thresh_otsu, cmap='gray')
th_adaptive = cv2.adaptiveThreshold(

    lena_gray,

    maxValue=255,

    adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,

    thresholdType=cv2.THRESH_BINARY,

    blockSize=13,

    C=8

)

plt.imshow(th_adaptive, cmap='gray')
canny_edges = cv2.Canny(lena_gray,16,40,3)
lines_edges = cv2.Canny(lines_thresh, 20, 50, 3)

plt.imshow(lines_edges, cmap='gray')
lines = cv2.HoughLinesP(

    lines_edges,

    2,

    np.pi / 180,

    30

)

result_lines_img = cv2.cvtColor(lines_img, cv2.COLOR_GRAY2RGB)

for line in lines:

  x0, y0, x1, y1 = line[0]

  cv2.line(result_lines_img, (x0, y0), (x1, y1), (0, 255, 0), 5)

plt.imshow(result_lines_img)

okręgi 

checkers_img = cv2.imread('checkers.png')

checkers_gray = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2GRAY)

checkers_color = cv2.cvtColor(checkers_img, cv2.COLOR_BGR2RGB)

circles = cv2.HoughCircles(

    checkers_gray,

    method=cv2.HOUGH_GRADIENT,

    dp=2,

    minDist=60,

    minRadius=20,

    maxRadius=100

)

len(circles[0])

24

for (x, y, r) in circles.astype(int)[0]:

  cv2.circle(checkers_color, (x, y), r, (0, 255, 0), 4)

plt.imshow(checkers_color)
--------------------------------------------------
lake_color = cv2.imread('swiecajty.jpg', cv2.IMREAD_COLOR)

lake_gray = cv2.cvtColor(lake_color, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(

    clipLimit=2.0,

    tileGridSize=(4, 4)

)

equalized_lake_gray = clahe.apply(lake_gray)

plt.subplot(221)

plt.imshow(lake_gray, cmap='gray')

plt.subplot(222)

plt.hist(lake_gray.ravel(), bins=256, range=(0, 256), color='gray')

plt.subplot(223)

plt.imshow(equalized_lake_gray, cmap='gray')

plt.subplot(224)

plt.hist(equalized_lake_gray.ravel(), bins=256, range=(0, 256), color='gray')

plt.show()
--------------------------------------------
lake_rgb = cv2.cvtColor(lake_color, cv2.COLOR_BGR2RGB)
lake_lab = cv2.cvtColor(lake_color, cv2.COLOR_BGR2LAB)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
lake_lab[..., 0] = clahe.apply(lake_lab[..., 0])
lake_color_equalized = cv2.cvtColor(lake_lab, cv2.COLOR_LAB2RGB)
plt.subplot(221)
plt.imshow(lake_rgb)

plt.subplot(222)
plt.hist(lake_rgb[..., 0].ravel(), bins=256, range=(0, 256), color='b')
plt.hist(lake_rgb[..., 1].ravel(), bins=256, range=(0, 256), color='g')
plt.hist(lake_rgb[..., 2].ravel(), bins=256, range=(0, 256), color='r')

plt.subplot(223)
plt.imshow(lake_color_equalized)

plt.subplot(224)
plt.hist(lake_color_equalized[..., 0].ravel(), bins=256, range=(0, 256), color='b')
plt.hist(lake_color_equalized[..., 1].ravel(), bins=256, range=(0, 256), color='g')
plt.hist(lake_color_equalized[..., 2].ravel(), bins=256, range=(0, 256), color='r')

plt.show()
"""
from typing import Any
import matplotlib
from matplotlib.image import imread
from matplotlib.pyplot import imshow
from matplotlib.image import imsave
import numpy as np
import cv2
from enum import Enum


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4
    sepia = 5


class BaseImage:
    data: np.ndarray
    color_model: ColorModel

    def __init__(self, data: Any, color_model: ColorModel) -> None:
        self.data = imread(data)
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        imsave(path, self.data)

    def show_img(self) -> None:
        imshow(self.data)
        matplotlib.pyplot.show()

    def get_layer(self, layer_id: int) -> 'np.ndarray':
        return self.data[:, :, layer_id]

    def get_layers(self) -> []:
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

    def from_rgb_to_hsv(self) -> 'BaseImage':
        R, G, B = self.get_layers() / 255
        M = np.max([R, G, B], axis=0)
        m = np.min([R, G, B], axis=0)
        V = M / 255
        S = np.where(M > 0, 1 - (m / M), 0)
        H = np.where(G >= B,
                     np.cos((R - (G / 2.0) - (B / 2.0)) /
                            np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                    - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                            ) ** (-1),
                     360 - np.cos((R - (G / 2.0) - (B / 2.0)) /
                                  np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                          - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                                  ) ** (-1),
                     )
        self.data = np.dstack((H, S, V))
        self.color_model = ColorModel.hsv
        return self

    def from_hsv_to_rgb(self) -> 'BaseImage':
        H, S, V = self.get_layers()
        M = 255 * V
        m = M * (1 - S)
        z = (M - m) * (1 - np.absolute(((H / 60) % 2) - 1))
        R = np.where(H < 60, M,
                     np.where(H < 120, z + m,
                              np.where(H < 240, m,
                                       np.where(H < 300, z + m, M))))

        G = np.where(H < 60, z + m,
                     np.where(H < 240, M, m))

        B = np.where(H < 120, m,
                     np.where(H < 240, z + m,
                              np.where(H < 300, M, z + m)))
        self.data = np.dstack((R, G, B))
        self.color_model = ColorModel.rgb
        return self

    def from_rgb_to_hsi(self) -> 'BaseImage':
        R, G, B = self.get_layers() / 255
        M = np.max([R, G, B], axis=0)
        m = np.min([R, G, B], axis=0)
        I = (R + G + B) / 3
        S = np.where(M > 0, 1 - (m / M), 0)
        H = np.where(G >= B,
                     np.cos((R - (G / 2.0) - (B / 2.0)) /
                            np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                    - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                            ) ** (-1),
                     360 - (np.cos((R - (G / 2.0) - (B / 2.0)) /
                                   np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                           - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                                   )) ** (-1),
                     )
        self.data = np.dstack((H, S, I))
        self.color_model = ColorModel.hsi
        return self

    def from_hsi_to_rgb(self) -> 'BaseImage':
        for layer in self.data:
            for pixel in layer:
                H, S, I = pixel[0], pixel[1], pixel[2]
                S = S * 0.5
                if 0 <= H <= 120:
                    pixel[2] = I * (1 - S)
                    pixel[0] = I * (1 + (S * np.cos(np.radians(H)) / np.cos(np.radians(60) - np.radians(H))))
                    pixel[1] = I * 3 - (pixel[0] + pixel[2])
                elif 120 < H <= 240:
                    H = H - 120
                    pixel[0] = I * (1 - S)
                    pixel[1] = I * (1 + (S * np.cos(np.radians(H)) / np.cos(np.radians(60) - np.radians(H))))
                    pixel[2] = 3 * I - (pixel[0] + pixel[1])
                elif 0 < H <= 360:
                    H = H - 240
                    pixel[1] = I * (1 - S)
                    pixel[2] = I * (1 + (S * np.cos(np.radians(H)) / np.cos(np.radians(60) - np.radians(H))))
                    pixel[0] = I * 3 - (pixel[1] + pixel[2])

        self.color_model = ColorModel.rgb
        return self

    def from_rgb_to_hsl(self) -> 'BaseImage':
        R, G, B = self.get_layers() / 255
        M = np.max([R, G, B], axis=0)
        m = np.min([R, G, B], axis=0)
        d = (M - m) / 255
        L = ((M + m) / 2) / 255
        S = np.where(L > 0, (1 * d) / (1 - np.absolute((2 * L) - 1)), 0)
        H = np.where(G >= B,
                     np.cos((R - (G / 2.0) - (B / 2.0)) /
                            np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                    - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                            ) ** (-1),
                     360 - np.cos((R - (G / 2.0) - (B / 2.0)) /
                                  np.sqrt(np.power(R, 2.0) + np.power(G, 2.0), np.power(B, 2.0)
                                          - np.multiply(R, G) - np.multiply(R, B) - np.multiply(G, B))
                                  ) ** (-1),
                     )
        self.data = np.dstack((H, S, L))
        self.color_model = ColorModel.hsl
        return self

    def from_hsl_to_rgb(self) -> 'BaseImage':
        H, S, L = self.get_layers()
        d = S * (1 - np.absolute((2 * L) - 1))
        m = 255 * (L - (d / 2))
        x = d * (1 - np.absolute(((H / 60) % 2) - 1))

        R = np.where(H < 60, (255 * d) + m,
                     np.where(H < 120, (255 * x) + m,
                              np.where(H < 240, m,
                                       np.where(H < 300, (255 * x) + m, (255 * d) + m))))

        G = np.where(H < 60, (255 * x) + m,
                     np.where(H < 180, (255 * d) + m,
                              np.where(H < 240, (255 * x) + m, m)))

        B = np.where(H < 120, m,
                     np.where(H < 180, (255 * x) + m,
                              np.where(H < 300, (255 * d) + m, (255 * x) + m)))

        self.data = np.dstack((R, G, B))
        self.color_model = ColorModel.rgb
        return self

    def to_rgb(self) -> 'BaseImage':
        if self.color_model == ColorModel.hsv:
            return self.from_hsv_to_rgb()
        elif self.color_model == ColorModel.hsi:
            return self.from_hsi_to_rgb()
        elif self.color_model == ColorModel.hsl:
            return self.from_hsl_to_rgb()


class GrayScaleTransform(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def to_gray(self) -> BaseImage:
        R, G, B = self.get_layers()
        avR = np.multiply(R, 0.299)
        avG = np.multiply(G, 0.587)
        avB = np.multiply(B, 0.114)
        self.data = np.round((avR + avG + avB)).astype('i')
        self.color_model = ColorModel.gray
        return self

    def to_sepia(self, alpha_beta: tuple = (None, None), w: int = None) -> BaseImage:
        if w is None and alpha_beta is not None and \
                (alpha_beta[0] > 1 and alpha_beta[1] < 1 and alpha_beta[0] + alpha_beta[1] == 2):
            self.to_gray()
            L0, L1, L2 = self.data, self.data, self.data
            L0 = np.where(L0 * alpha_beta[0] > 255, 255, L0 * alpha_beta[0])
            L2 = np.where(L2 * alpha_beta[1] > 255, 255, L2 * alpha_beta[1])
            self.data = np.dstack((L0, L1, L2)).astype('i')
            self.color_model = ColorModel.sepia
            return self
        elif w is not None and alpha_beta[0] is None and alpha_beta[1] is None:
            self.to_gray()
            L0, L1, L2 = self.data, self.data, self.data
            L0 = np.where(L0 + 2 * w > 255, 255, L0 + 2 * w)
            L1 = np.where(L1 + w > 255, 255, L1 + w)
            self.data = np.dstack((L0, L1, L2)).astype('i')
            self.color_model = ColorModel.sepia
            return self


class Histogram:
    values: np.ndarray

    def __init__(self, values: np.ndarray) -> None:
        if values.ndim == 2:
            self.values = np.histogram(values, bins=256, range=(0, 255))[0]
        else:
            self.values = values

    def plot(self) -> None:
        if self.values.ndim == 1:
            matplotlib.pyplot.figure()
            matplotlib.pyplot.title("Gray Scale Histogram")
            matplotlib.pyplot.xlabel("Gray value")
            matplotlib.pyplot.ylabel("Pixels")
            matplotlib.pyplot.xlim([0, 255])
            bin_edges = np.linspace(0, 254.9, 256)
            matplotlib.pyplot.plot(bin_edges, self.values, color="gray")
            matplotlib.pyplot.show()
        else:
            matplotlib.pyplot.figure(figsize=(14, 8))
            bin_edges = np.linspace(0, 254.9, 256)
            matplotlib.pyplot.subplot(131)
            matplotlib.pyplot.title("red layer")
            matplotlib.pyplot.xlim([0, 255])
            matplotlib.pyplot.ylabel("Pixels")
            matplotlib.pyplot.xlabel("Red value")
            matplotlib.pyplot.plot(bin_edges, self.values[:, :, 0].flatten(), color="red")
            matplotlib.pyplot.subplot(132)
            matplotlib.pyplot.title("green layer")
            matplotlib.pyplot.xlabel("Green value")
            matplotlib.pyplot.xlim([0, 255])
            matplotlib.pyplot.plot(bin_edges, self.values[:, :, 1].flatten(), color="green")
            matplotlib.pyplot.subplot(133)
            matplotlib.pyplot.title("blue layer")
            matplotlib.pyplot.xlabel("Blue value")
            matplotlib.pyplot.xlim([0, 255])
            matplotlib.pyplot.plot(bin_edges, self.values[:, :, 2].flatten(), color="blue")
            matplotlib.pyplot.show()

    def to_cumulative(self) -> 'Histogram':
        self.values = np.cumsum(self.values)
        return self


class ImageDiffMethod(Enum):
    mse = 0
    rmse = 1


class Image(GrayScaleTransform):
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)


class ImageComparison(BaseImage):
    def __init__(self, data: Any, color_model: ColorModel) -> None:
        super().__init__(data, color_model)

    def histogram(self) -> Histogram:
        return Histogram(self.data)

    def compare_to(self, other: Image, method: ImageDiffMethod) -> float:
        grayImage1 = GrayScaleTransform(self.data, color_model=ColorModel.rgb).to_gray()
        grayImage2 = GrayScaleTransform(other.data, color_model=ColorModel.rgb).to_gray()
        grayImage1HistogramValues = Histogram(grayImage1.data).values
        grayImage2HistogramValues = Histogram(grayImage2.data).values
        n = len(grayImage1HistogramValues)
        sumHistogram = 0
        for x in range(n):
            sumHistogram = sumHistogram + ((grayImage1HistogramValues[x] - grayImage2HistogramValues[x]) ** 2)
        sumHistogram = np.sum(sumHistogram) * 1 / n

        if method == ImageDiffMethod.rmse:
            sumHistogram = np.sqrt(sumHistogram)

        return sumHistogram


# alpha1 = 1.1
# alpha2 = 1.5
# alpha3 = 1.9

# beta1 = 0.9
# beta2 = 0.5
# beta3 = 0.1

# w1 = 20
# w2 = 30
# w3 = 40
