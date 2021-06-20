import cv2 as cv
import random


class PlaceLogo:
    def __init__(self, logo_path, standard_logo_size=200, standard_logo_offset=(150, 150)):
        self.standard_logo_size = standard_logo_size
        self.standard_logo_offset = standard_logo_offset
        self.logo = cv.imread(logo_path, cv.IMREAD_UNCHANGED)

    def __call__(self, img):
        logo_size = 150 + random.randint(-50, 50)
        offset = (150 + random.randint(-50, 50), 150 + random.randint(-50, 50))
        logo = cv.resize(self.logo, (logo_size, logo_size))
        img = cv.cvtColor(img, cv.COLOR_RGB2RGBA)
        img[offset[0]:(logo_size + offset[0]), offset[1]:(logo_size + offset[1])] = cv.addWeighted(
            img[offset[0]:(logo_size + offset[0]), offset[1]:(logo_size + offset[1])], 1,
            logo, 0.3 + random.uniform(0, 0.3), 0)
        return img


class Resize:
    def __init__(self, target_width):
        self.target_width = target_width

    def __call__(self, img):
        return cv.resize(img, (self.target_width, int(720 / 1280 * self.target_width)))


class Gray:
    def __call__(self, img):
        return cv.cvtColor(img, cv.COLOR_RGBA2GRAY)
