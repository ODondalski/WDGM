from matplotlib.image import imread, imsave
from matplotlib.pyplot import imshow
from enum import Enum
import numpy as np


class ColorModel(Enum):
    rgb = 0
    hsv = 1
    hsi = 2
    hsl = 3
    gray = 4  # obraz 2d


class BaseImage:
    data: np.ndarray  # tensor przechowujacy piksele obrazu
    color_model: ColorModel  # atrybut przechowujacy biezacy model barw obrazu

    def __init__(self, path: str, color_model: ColorModel) -> None:
        self.data = imread(path)
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        imsave(path, self.data)

    def show_img(self) -> None:
        imshow(self.data)

    def get_layer(self, layer_id: int) -> 'BaseImage':
        return self.data[:, :, layer_id]

    def get_layers(self) -> []:
        return np.squeeze(np.dsplit(self.data, self.data.shape[-1]))

    def to_hsv(self) -> 'BaseImage':
        for i in self.data:
            for col in i:
                red, green, blue = col[0], col[1], col[2]

                M = max(red, green, blue)
                m = min(red, green, blue)

                V = M / 255
                if (M > 0):
                    S = 1 - (m / M)
                else:
                    S = 0

                zm = (red ** 2 + green ** 2 + blue ** 2) - (red * green - red * blue - green * blue)

                if green >= blue:
                    H = np.cos((red - green / 2 - blue / 2) / np.sqrt(zm)) ** (-1)
                else:
                    H = 360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(zm)) ** (-1)

                col[0] = H * 255
                col[1] = S * 255
                col[2] = V * 255

        self.color_model = ColorModel.hsv
        return self

    def to_hsi(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsi
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        for i in self.data:
            for col in i:
                red, green, blue = col[0], col[1], col[2]

                M = max(red, green, blue)
                m = min(red, green, blue)
                I = (red + green + blue) / 3
                if M > 0:
                    S = 1 - (m / M)
                else:
                    S = 0

                zm = (red ** 2 + green ** 2 + blue ** 2) - (red * green - red * blue - green * blue)

                if green >= blue:
                    H = np.cos((red - green / 2 - blue / 2) / np.sqrt(zm)) ** (-1)
                else:
                    H = 360 - np.cos((red - green / 2 - blue / 2) / np.sqrt(zm)) ** (-1)

                col[0] = H * 150
                col[1] = S * 150
                col[2] = I * 150

        self.color_model = ColorModel.hsv
        return self

    def to_hsl(self) -> 'BaseImage':
        red, green, blue = self.get_layers()
        M = np.max([red, green, blue], axis=0)
        m = np.min([red, green.blue], axis=0)
        d = (M - m) / 255
        L = ((M + m) / 2) / 255
        if L > 0:
            S = d / (1 - np.absolute((2 * L) - 1))
        else:
            S = 0
        if green >= blue:
            H = np.cos(red - green / 2 - blue / 2) / (np.sqrt(red ** 2 + green ** 2 +
                                                              blue ** 2 - red * green -
                                                              red * blue - green * blue)) ** (-1)
        self.color_model = ColorModel.hsl
        return self

    def to_rgb(self) -> 'BaseImage':
        H, S, V = self.get_layers()
        M = 255 * V
        m = M * (1 - S)
        z = (M - m) * (1 - np.absolute(((H / 60) % 2) - 1))

        red = np.where(H < 60, M,
                       np.where(H < 120, z + m,
                                np.where(H < 240, m,
                                         np.where(H < 300, z + m, M))))

        green = np.where(H < 60, z + m,
                         np.where(H < 240, M, m))

        blue = np.where(H < 120, m,
                        np.where(H < 240, z + m,
                                 np.where(H < 300, M, z + m)))
        self.color_model = ColorModel.rgb
        return self
