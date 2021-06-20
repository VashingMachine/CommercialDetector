import os
import cv2 as cv
from torch.utils.data import Dataset
import random
from torchvision import transforms
from transforms import PlaceLogo, Gray, Resize

LOGO_SIZE = 150
SCALE = 300


def get_datasets(path, split_ratio=0.8):
    commercials = [os.path.join(path, 'frames/commercial', file) for file in
                   os.listdir(os.path.join(path, 'frames/commercial'))]
    movies = [os.path.join(path, 'frames/movie', file) for file in
              os.listdir(os.path.join(path, 'frames/movie'))]

    random.shuffle(commercials)
    random.shuffle(movies)

    commercials_split_point = int(len(commercials) * split_ratio)
    movies_split_point = int(len(movies) * split_ratio)
    train_commercials = commercials[:commercials_split_point]
    test_commercials = commercials[commercials_split_point:]
    train_movies = movies[:movies_split_point]
    test_movies = movies[movies_split_point:]
    return ImageDataset(path, train_commercials, train_movies), ImageDataset(path, test_commercials, test_movies)


class TestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.frames = [os.path.join(path, file) for file in os.listdir(path)]

        self.tran = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = cv.imread(self.frames[idx])
        img = cv.resize(cv.cvtColor(img, cv.COLOR_RGBA2GRAY), (SCALE, int(720 / 1280 * SCALE)))
        return self.tran(img), 0


class ImageDataset(Dataset):
    def __init__(self, path, commercials, movies):
        self.path = path
        self.commercials = commercials
        self.movies = movies
        self.frames = [*self.commercials, *self.movies]

        self.commercial_transform = transforms.Compose([
            Resize(SCALE),
            Gray(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

        self.movie_transform = transforms.Compose([
            PlaceLogo(os.path.join(self.path, 'logo.png'), 300, (225, 150)),
            Resize(SCALE),
            Gray(),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5)
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        img = cv.imread(self.frames[idx])
        if idx < len(self.commercials):
            return self.commercial_transform(img), 0
        else:
            return self.movie_transform(img), 1


if __name__ == "__main__":
    train, test = get_datasets('data/preset')
    print(len(train))
    print(len(test))

    cv.imshow("image", train[len(train) - 1][0])
    cv.waitKey(0)
