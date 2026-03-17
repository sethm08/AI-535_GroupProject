# Seth Mackovjak
# OSU AI 535 - Group Project
# Desc: This is to provide the tools to create datasets relative
#       to the group project.

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import tensor, Tensor, from_numpy
from torchvision.transforms.functional import pil_to_tensor
from torchvision.ops import box_convert
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import kagglehub

# 1) download
# 2) structure (input/output as a Dataset)
# 3) dataset split (train/test)
# 4) augmentation

# Make a directory to store the data:
data_path = Path('./data')
cache_path = data_path / '.cache'
cache_path.mkdir(exist_ok=True)


def get_img_files(dir: str | Path) -> list:
    """
    Desc:   Compile the image data into a dataset.
    """

    path = Path(dir)

    # Compile a list of image files:
    img_extensions = ('*.jpg', '*.jpeg', '*.tiff', '*.tif', '*.png')
    files_list = []
    for extension in img_extensions:
        try:
            files_list.extend([file for file in path.rglob(extension)])
        except Exception as e:
            print('Error finding files:', e)

    return files_list


def image_import(path: str | Path) -> Tensor:
    """
    Desc:   Import the image and extract the blast value.
    Args:
        Path {str | Path}: Path to the image file.
    Return:
        Tensor
    """

    path = str(path)
    extension = path.rsplit('.', maxsplit=1)[1]

    if extension in ('jpg', 'jpeg', 'png'):
        return read_image(path=path)
    else:
        return pil_to_tensor(Image.open(path))


class idb2_dataset(Dataset):
    def __init__(self, img_dir: str | Path = './data/ALL_IDB/ALL_IDB2',
                 transform=None, train: bool = True,
                 split_ratio: float = 0.8):

        self.transform = transform

        def label_value(path: str | Path) -> int:
            """
            Desc:   Extract the blast value from the file path.
            Args:
                path {str | Path}: Path to the image file.
            Return:
                int: Blast value 0 = non-blast cell, 1 = blast cell.
            """

            path = str(path)
            blast_value = path.rsplit('.', maxsplit=1)[0]
            blast_value = int(blast_value.rsplit('_', maxsplit=1)[1])

            return blast_value

        # Compile a list of the image files & labels:
        img_files = get_img_files(dir=img_dir)
        labels = [label_value(f) for f in img_files]

        # Split the train and test files so that the labels are balanced:
        train_files, test_files = train_test_split(
            img_files,
            test_size=1.0 - split_ratio,
            random_state=5,
            stratify=labels
        )

        # Set the appropriate train/test file paths and labels.
        self.img_files = train_files if train else test_files
        self.labels = [label_value(f) for f in self.img_files]

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx) -> tuple:
        image = image_import(self.img_files[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class bcicd_dataset(Dataset):
    def __init__(self, transform=None, train: bool = True,
                 split_ratio: float = 0.8, dir: str | Path = cache_path):

        self.transform = transform

        # Download the dataset from Kaggle:
        path = Path(
            kagglehub.dataset_download(
                "sumithsingh/blood-cell-images-for-cancer-detection",
                output_dir=str(dir)
                )
        )

        # File prefix to cell type dictionary:
        prefix_dict = {
            'BA': {'name': 'basophil', 'id': 0},
            'ERB': {'name': 'erythroblast', 'id': 1},
            'MO': {'name': 'monocyte', 'id': 2},
            'MYO': {'name': 'myeloblast', 'id': 3},
            'NGS': {'name': 'seg_neutrophil', 'id': 4},
        }

        def label_value(path: str | Path) -> int:
            """
            Desc:   Convert the file prefix to a label value.
            Args:
                path {str | Path}: Filepath.
            Return:
                int
            """

            filename = str(path).rsplit('\\', maxsplit=1)[1]
            prefix = filename.split('_', maxsplit=1)[0]

            return prefix_dict[prefix]['id']

        img_files = get_img_files(dir=path)
        labels = [label_value(f) for f in img_files]

        # Split the train and test files so that the labels are balanced:
        train_files, test_files = train_test_split(
            img_files,
            test_size=1.0 - split_ratio,
            random_state=5,
            stratify=labels
        )

        # Set the appropriate train/test file paths and labels.
        self.img_files = train_files if train else test_files
        self.labels = [label_value(f) for f in self.img_files]

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        image = image_import(self.img_files[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class idb1_dataset(Dataset):
    def __init__(self,
                 dir_path: str | Path = './data/ALL_IDB/ALL_IDB1',
                 transform=None, train: bool = True,
                 split_ratio: float = 0.8):

        self.transform = transform

        self.dir_path = Path(dir_path)
        print("Searching", self.dir_path)
        img_files = list(self.dir_path.rglob('*.jpg'))
        centroid_files = list(self.dir_path.rglob('*.xyc'))
        img_files.sort()
        centroid_files.sort()
        print("Found", len(img_files), "images and", len(centroid_files), "centroid files.")

        if len(img_files) != len(centroid_files):
            raise AttributeError('Unequal number of image and centroid files.')
        
        count_list = [c.shape[0] for c in ]

        # Split the train and test files so that the labels are balanced:
        train_files, test_files = train_test_split(
            img_files,
            test_size=1.0 - split_ratio,
            random_state=5,
            stratify=labels
        )

        self.paired_paths = list(zip(img_files, centroid_files))

    def __len__(self) -> int:
        return len(self.paired_paths)

    def __getitem__(self, idx) -> tuple[Tensor, Tensor]:
        img_path, centroid_file = self.paired_paths[idx]
        centroids = np.loadtxt(centroid_file, dtype=np.int16, delimiter='\t')
        radius = 128
        bboxes = tensor([[x, y, radius, radius] for x, y in centroids])
        bboxes = box_convert(boxes=bboxes, in_fmt='cxcywh', out_fmt='xyxy')
        image = image_import(img_path)

        return image, bboxes


def category_balance_plot(train_dataset: Dataset, test_dataset: Dataset):
    train_label_count = Counter(train_dataset.labels)
    test_label_count = Counter(test_dataset.labels)

    categories = sorted(set(train_label_count) | set(test_label_count))

    train_percent = [
        100 * train_label_count.get(cat, 0) / len(train_dataset)
        for cat in categories]
    test_percent = [
        100 * test_label_count.get(cat, 0) / len(test_dataset)
        for cat in categories]

    label_x = np.arange(len(categories))
    width = 0.4

    plt.figure(figsize=(8,5))

    plt.bar(label_x - width/2, train_percent, width, label="Train")
    plt.bar(label_x + width/2, test_percent, width, label="Test")

    plt.xticks(label_x, categories)
    plt.ylabel("Percentage of Samples (%)")
    plt.xlabel("Category")
    plt.title("Train vs Test Label Distribution")
    plt.legend()

    plt.tight_layout()
    plt.show()
