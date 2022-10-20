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

    def __init__(self, path: str) -> None:
        """
        inicjalizator wczytujacy obraz do atrybutu data na podstawie sciezki
        """
        self.data = imread(path)
        self.color_model = color_model

    def save_img(self, path: str) -> None:
        """
        metoda zapisujaca obraz znajdujacy sie w atrybucie data do pliku
        """
        imsave('image.jpg', self.data)

    def show_img(self) -> None:
        """
        metoda wyswietlajaca obraz znajdujacy sie w atrybucie data
        """
        imshow(self.data)

    def get_layer(self, layer_id: int) -> 'BaseImage':
        """
        metoda zwracajaca warstwe o wskazanym indeksie
        """
        return self.data[:, :, layer_id]

    def to_hsv(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsv
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        for i in self.data:
            for col in i:
                red, green, blue = col[0], col[1], col[2]

                M = max(red, green, blue)
                m = min(red, green, blue)

                V = M / 255
                if(M > 0):
                    S = 1 - (m / M)
                else:
                    S = 0

                zm = (red ** 2 + green ** 2 + blue ** 2) - (red * green - red * blue - green * blue)

                if green >= blue:
                    H = np.cos((red - green/2-blue/2) / np.sqrt(zm)) ** (-1)
                else:
                    H = 360 - np.cos((red - green/2-blue/2) / np.sqrt(zm)) ** (-1)

                col[0] = H * 255
                col[1] = S * 255
                col[2] = V * 255

        self.colorModel = ColorModel.hsv
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
                if(M > 0):
                    S = 1 - (m / M)
                else:
                    S = 0

                zm = (red ** 2 + green ** 2 + blue ** 2) - (red * green - red * blue - green * blue)

                if green >= blue:
                    H = np.cos((red - green/2-blue/2) / np.sqrt(zm)) ** (-1)
                else:
                    H = 360 - np.cos((red - green/2-blue/2) / np.sqrt(zm)) ** (-1)

                col[0] = H * 150
                col[1] = S * 150
                col[2] = I * 150

        self.colorModel = ColorModel.hsv
        return self

    def to_hsl(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu hsl
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        pass

    def to_rgb(self) -> 'BaseImage':
        """
        metoda dokonujaca konwersji obrazu w atrybucie data do modelu rgb
        metoda zwraca nowy obiekt klasy image zawierajacy obraz w docelowym modelu barw
        """
        pass
