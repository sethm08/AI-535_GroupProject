# Seth Mackovjak
# OSU AI 535 - Group Project
# Desc: This is to provide the tools to create datasets relative
#       to the group project.

from pathlib import Path
from skimage.io import imread
from sklearn.model_selection import train_test_split
from torch import tensor, Tensor
from torchvision.io import read_image
from torch.utils.data import Dataset

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


def idb2_image_import(path: str | Path) -> tuple[Tensor, int]:
    """
    Desc:   Import the image and extract the blast value.
    Args:
        Path {str | Path}: Path to the image file.
    Return:
        tuple(tensor, int)
    """
    path = str(path)
    blast_value, extension = path.rsplit('.', maxsplit=1)
    blast_value = int(blast_value.rsplit('_', maxsplit=1)[1])

    if extension in ('jpg', 'jpeg', 'png'):
        img_tensor = read_image(path=path)
    else:
        img_tensor = tensor(imread(path))

    return img_tensor, blast_value


class idb2_dataset(Dataset):
    def __init__(self, img_dir: str | Path = './data/ALL_IDB/ALL_IDB2',
                 transform=None, train: bool = True,
                 split_ratio: float = 0.8):

        self.transform = transform

        # Compile a list of the image files & labels:
        img_files = get_img_files(dir=img_dir)
        labels = [idb2_image_import(f)[1] for f in img_files]

        # Split the train and test files so that the labels are balanced:
        train_files, test_files = train_test_split(
            img_files,
            test_size=1.0 - split_ratio,
            random_state=5,
            stratify=labels
        )

        self.img_files = train_files if train else test_files

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx) -> tuple:
        image, label = idb2_image_import(self.img_files[idx])
        if self.transform:
            image = self.transform(image)
        return image, label
