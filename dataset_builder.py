# Seth Mackovjak
# OSU AI 535 - Group Project
# Desc: This is to provide the tools to create datasets relative
#       to the group project.

import os
from pathlib import Path
from typing import Tuple, Dict, Any, List
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split
import cv2
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
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


def image_import(path: str | Path) -> torch.Tensor:
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
        return TF.pil_to_tensor(Image.open(path))


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
                 transform=None,
                 train: bool = True,
                 split_ratio: float = 0.8,
                 mask_radius: int = 64,
                 image_size: tuple[int, int] = (256, 256),
                 augment: bool = True,
                 image_only_transform=None):

        self.transform = transform
        self.mask_radius = mask_radius
        self.image_size = image_size
        self.augment = augment and train
        self.image_only_transform = image_only_transform

        self.dir_path = Path(dir_path)
        print("Searching", self.dir_path)

        img_files = sorted(self.dir_path.rglob('*.jpg'))
        centroid_files = sorted(self.dir_path.rglob('*.xyc'))

        print("Found", len(img_files), "images and",
              len(centroid_files), "centroid files.")

        if len(img_files) != len(centroid_files):
            raise AttributeError('Unequal number of image and centroid files.')

        paired_paths = list(zip(img_files, centroid_files))

        count_list = np.array([
            self._load_centroids(c_file).shape[0]
            for _, c_file in paired_paths
            ], dtype=np.int32)

        # Split the train and test files so that the labels are balanced:
        stratify_labels = self._build_stratify_labels(count_list)
        train_files, test_files = train_test_split(
            paired_paths,
            test_size=1.0 - split_ratio,
            random_state=5,
            stratify=stratify_labels
        )

        self.paired_paths = train_files if train else test_files

    def __len__(self) -> int:
        """ Return the size of the dataset."""
        return len(self.paired_paths)

    def __getitem__(self, idx) -> tuple[torch.Tensor, dict]:
        img_path, centroid_file = self.paired_paths[idx]

        image = image_import(img_path)
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        centroids = self._load_centroids(centroid_file)

        instance_masks = self._mask_generator(image, centroids)

        if image.shape[0] in (1, 3):   # CHW
            _, H, W = image.shape
        else:
            raise ValueError('Expected a CHW image tensor, got',
                             tuple(image.shape))

        if instance_masks.shape[0] > 0:
            semantic_mask = torch.amax(instance_masks, dim=0, keepdim=True)
        else:
            semantic_mask = torch.zeros((1, H, W), dtype=torch.int16)

        image, semantic_mask, instance_masks = self._joint_transform(
            image, semantic_mask, instance_masks
        )

        boxes = self._boxes_from_masks(instance_masks)
        centroids_t = self._centroids_from_masks(instance_masks)
        labels = torch.ones((boxes.shape[0],), dtype=torch.long)

        if self.image_only_transform is not None:
            image = self.image_only_transform(image)

        target = {
            'semantic_mask': semantic_mask.float(),
            'instance_masks': instance_masks.float(),
            'boxes': boxes.float(),
            'centroids': centroids_t.float(),
            'labels': labels
        }

        return image, target

    def _centroids_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """
        Recompute centroids from transformed instance masks.
        masks: (N,H,W)
        returns: (N,2) in x,y order
        """
        centroids = []

        for mask in masks:
            ys, xs = torch.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            cx = xs.float().mean().item()
            cy = ys.float().mean().item()
            centroids.append([cx, cy])

        if len(centroids) == 0:
            return torch.empty((0, 2), dtype=torch.int16)

        return torch.tensor(centroids, dtype=torch.int16)

    def _joint_transform(self,
                         image: torch.Tensor,
                         semantic_mask: torch.Tensor,
                         instance_masks: torch.Tensor,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply identical geometric transforms to image and masks.

        image: (C,H,W)
        semantic_mask: (1,H,W)
        instance_masks: (N,H,W)

        Returns transformed image, semantic_mask, instance_masks
        """
        # Resize first
        image = TF.resize(
            image,
            self.image_size,
            interpolation=InterpolationMode.BILINEAR,
            antialias=True
        )

        semantic_mask = TF.resize(
            semantic_mask,
            self.image_size,
            interpolation=InterpolationMode.NEAREST
        )

        if instance_masks.shape[0] > 0:
            # Resize as a stack of masks
            instance_masks = TF.resize(
                instance_masks,
                self.image_size,
                interpolation=InterpolationMode.NEAREST
            )
        else:
            Ht, Wt = self.image_size
            instance_masks = torch.empty((0, Ht, Wt), dtype=torch.float32)

        # Random flips (only in training mode if augment=True)
        if self.augment:
            if torch.rand(1).item() < 0.5:
                image = TF.hflip(image)
                semantic_mask = TF.hflip(semantic_mask)
                if instance_masks.shape[0] > 0:
                    instance_masks = TF.hflip(instance_masks)

            if torch.rand(1).item() < 0.5:
                image = TF.vflip(image)
                semantic_mask = TF.vflip(semantic_mask)
                if instance_masks.shape[0] > 0:
                    instance_masks = TF.vflip(instance_masks)

        # Ensure masks remain binary after resize/transform
        semantic_mask = (semantic_mask > 0.5).float()
        if instance_masks.shape[0] > 0:
            instance_masks = (instance_masks > 0.5).float()

        return image, semantic_mask, instance_masks

    def _build_stratify_labels(self, count_list: np.ndarray) -> np.ndarray:
        bins = []
        for c in count_list:
            if c == 0:
                bins.append('0')
            elif c == 1:
                bins.append('1')
            elif c == 2:
                bins.append('2')
            elif c <= 4:
                bins.append('3-4')
            else:
                bins.append('5+')

        return np.array(bins)

    def _load_centroids(self, file_path: str | Path) -> np.ndarray:
        try:
            if Path(file_path).stat().st_size == 0:
                return np.empty((0, 2), dtype=np.int16)

            centroids = np.loadtxt(file_path, dtype=np.int16, delimiter='\t')

            if centroids.ndim == 1:
                centroids = centroids.reshape(1, -1)

            if centroids.shape[1] < 2:
                return np.empty((0, 2), dtype=np.int16)

            return centroids

        except (OSError, ValueError, UserWarning):
            return np.empty((0, 2), dtype=np.int16)

    def _mask_generator(self,
                        image: torch.Tensor,
                        centroids: np.ndarray) -> torch.Tensor:
        """
        Desc: Generate masks for the provided image and centroids.
        """

        if image.ndim != 3:
            raise ValueError('Expected image with 3 dims, got shape',
                             tuple(image.shape))

        if image.shape[0] in (1, 3):    # Channel x Height x Width
            _, H, W = image.shape
        else:                           # Height x Width x Channel
            H, W, _ = image.shape

        if centroids.shape[0] == 0:
            return torch.empty((0, H, W), dtype=torch.int16)

        # Build disks around each centroid:
        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )

        r2 = float(self.mask_radius ** 2)

        masks = []
        for row in centroids:
            cx, cy = float(row[0]), float(row[1])

            # Pass on invalid centroids:
            if cx < 0 or cy < 0 or cx >= W or cy >= H:
                continue

            disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r2
            masks.append(disk.float())

        if len(masks) == 0:
            return torch.empty((0, H, W), dtype=torch.int16)

        return torch.stack(masks, dim=0)

    def _boxes_from_masks(self, masks: torch.Tensor) -> torch.Tensor:

        if masks.ndim != 3:
            raise ValueError('Expected masks shape (N,H,W), got',
                             tuple(masks.shape))

        boxes = []
        for mask in masks:
            ys, xs = torch.where(mask > 0)

            if len(xs) == 0 or len(ys) == 0:
                continue

            x1 = xs.min().item()
            y1 = ys.min().item()
            x2 = xs.max().item()
            y2 = ys.max().item()

            boxes.append([x1, y1, x2, y2])

        if len(boxes) == 0:
            return torch.empty((0, 4), dtype=torch.int16)

        return torch.tensor(boxes, dtype=torch.int16)


class IDB1PatchTrainDataset(Dataset):
    def __init__(
        self,
        dir_path: str | Path = "./data/ALL_IDB/ALL_IDB1",
        split_ratio: float = 0.8,
        random_state: int = 5,
        patch_size: int = 512,
        mask_radius: int = 12,
        image_only_transform=None,
        include_background_patches: bool = False,
        background_per_image: int = 0,
    ):
        self.dir_path = Path(dir_path)
        self.patch_size = patch_size
        self.mask_radius = mask_radius
        self.image_only_transform = image_only_transform
        self.include_background_patches = include_background_patches
        self.background_per_image = background_per_image

        img_files = sorted(self.dir_path.rglob("*.jpg"))
        centroid_files = sorted(self.dir_path.rglob("*.xyc"))

        if len(img_files) != len(centroid_files):
            raise AttributeError("Unequal number of image and centroid files.")

        paired_paths = list(zip(img_files, centroid_files))

        count_list = np.array([
            self._load_centroids(c_file).shape[0]
            for _, c_file in paired_paths
        ], dtype=np.int32)

        stratify_labels = self._build_stratify_labels(count_list)

        train_files, test_files = train_test_split(
            paired_paths,
            test_size=1.0 - split_ratio,
            random_state=random_state,
            stratify=stratify_labels,
        )

        self.train_files = train_files

        # Expand into per-patch samples
        self.samples = []
        rng = np.random.default_rng(random_state)

        for img_path, centroid_file in self.train_files:
            centroids = self._load_centroids(centroid_file)

            # one patch per centroid
            for i in range(centroids.shape[0]):
                self.samples.append({
                    "img_path": img_path,
                    "centroid_file": centroid_file,
                    "centroid_idx": i,
                    "mode": "positive",
                })

            # optional background patches
            if self.include_background_patches and self.background_per_image > 0:
                for _ in range(self.background_per_image):
                    self.samples.append({
                        "img_path": img_path,
                        "centroid_file": centroid_file,
                        "centroid_idx": None,
                        "mode": "background",
                        "seed": int(rng.integers(0, 1_000_000)),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]
        img_path = sample["img_path"]
        centroid_file = sample["centroid_file"]

        image = image_import(img_path)  # expected CHW torch.Tensor
        centroids = self._load_centroids(centroid_file)

        instance_masks_full = self._mask_generator(image, centroids)   # (N,H,W)
        semantic_mask_full = self._semantic_from_instances(instance_masks_full, image)

        if sample["mode"] == "positive":
            cx, cy = centroids[sample["centroid_idx"], :2]
        else:
            cx, cy = self._sample_background_center(
                image=image,
                centroids=centroids,
                seed=sample["seed"]
            )

        patch_img, patch_sem, patch_inst, x0, y0 = self._crop_patch(
            image=image,
            semantic_mask=semantic_mask_full,
            instance_masks=instance_masks_full,
            cx=float(cx),
            cy=float(cy),
            patch_size=self.patch_size,
        )

        # recompute patch-local boxes / centroids from cropped masks
        patch_boxes = self._boxes_from_masks(patch_inst)
        patch_centroids = self._centroids_from_masks(patch_inst)

        labels = torch.ones((patch_boxes.shape[0],), dtype=torch.long)

        patch_img = patch_img.float()
        if patch_img.max() > 1.0:
            patch_img = patch_img / 255.0

        if self.image_only_transform is not None:
            patch_img = self.image_only_transform(patch_img)

        target = {
            "semantic_mask": patch_sem,        # (1,Hp,Wp)
            "instance_masks": patch_inst,      # (N,Hp,Wp)
            "boxes": patch_boxes,              # (N,4)
            "centroids": patch_centroids,      # (N,2)
            "labels": labels,
            "patch_origin": torch.tensor([x0, y0], dtype=torch.float32),
            "source_image": str(img_path),
        }
        return patch_img, target

    def _load_centroids(self, file_path: str | Path) -> np.ndarray:
        try:
            if Path(file_path).stat().st_size == 0:
                return np.empty((0, 2), dtype=np.int16)

            centroids = np.loadtxt(file_path, dtype=np.int16, delimiter="\t")

            if centroids.ndim == 1:
                centroids = centroids.reshape(1, -1)

            if centroids.shape[1] < 2:
                return np.empty((0, 2), dtype=np.int16)

            return centroids
        except (OSError, ValueError, UserWarning):
            return np.empty((0, 2), dtype=np.int16)

    def _mask_generator(self, image: torch.Tensor, centroids: np.ndarray) -> torch.Tensor:
        _, H, W = image.shape
        if centroids.shape[0] == 0:
            return torch.empty((0, H, W), dtype=torch.float32)

        yy, xx = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing="ij",
        )
        r2 = float(self.mask_radius ** 2)
        masks = []

        for row in centroids:
            cx = float(row[0])
            cy = float(row[1])

            if cx < 0 or cy < 0 or cx >= W or cy >= H:
                continue

            disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r2
            masks.append(disk.float())

        if len(masks) == 0:
            return torch.empty((0, H, W), dtype=torch.float32)

        return torch.stack(masks, dim=0)

    def _semantic_from_instances(self, instance_masks: torch.Tensor,
                                 image: torch.Tensor) -> torch.Tensor:
        _, H, W = image.shape
        if instance_masks.shape[0] == 0:
            return torch.zeros((1, H, W), dtype=torch.float32)
        return torch.amax(instance_masks, dim=0, keepdim=True)

    def _crop_patch(
        self,
        image: torch.Tensor,
        semantic_mask: torch.Tensor,
        instance_masks: torch.Tensor,
        cx: float,
        cy: float,
        patch_size: int,
    ):
        _, H, W = image.shape
        half = patch_size // 2

        x0 = max(0, int(round(cx)) - half)
        y0 = max(0, int(round(cy)) - half)
        x1 = min(W, x0 + patch_size)
        y1 = min(H, y0 + patch_size)

        # shift back if near edges
        x0 = max(0, x1 - patch_size)
        y0 = max(0, y1 - patch_size)

        patch_img = image[:, y0:y1, x0:x1]
        patch_sem = semantic_mask[:, y0:y1, x0:x1]

        if instance_masks.shape[0] > 0:
            patch_inst = instance_masks[:, y0:y1, x0:x1]
            # keep only masks that have some positive pixels in this patch
            keep = patch_inst.flatten(1).sum(dim=1) > 0
            patch_inst = patch_inst[keep]
        else:
            patch_inst = torch.empty((0, y1 - y0, x1 - x0), dtype=torch.float32)

        return patch_img, patch_sem, patch_inst, x0, y0

    def _boxes_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        boxes = []
        for mask in masks:
            ys, xs = torch.where(mask > 0)
            if len(xs) == 0:
                continue
            boxes.append([xs.min().item(), ys.min().item(), xs.max().item(), ys.max().item()])
        if len(boxes) == 0:
            return torch.empty((0, 4), dtype=torch.float32)
        return torch.tensor(boxes, dtype=torch.float32)

    def _centroids_from_masks(self, masks: torch.Tensor) -> torch.Tensor:
        cents = []
        for mask in masks:
            ys, xs = torch.where(mask > 0)
            if len(xs) == 0:
                continue
            cents.append([xs.float().mean().item(), ys.float().mean().item()])
        if len(cents) == 0:
            return torch.empty((0, 2), dtype=torch.float32)
        return torch.tensor(cents, dtype=torch.float32)

    def _sample_background_center(self, image: torch.Tensor, centroids: np.ndarray, seed: int):
        _, H, W = image.shape
        rng = np.random.default_rng(seed)

        for _ in range(100):
            cx = rng.integers(self.patch_size // 2, max(self.patch_size // 2 + 1, W - self.patch_size // 2))
            cy = rng.integers(self.patch_size // 2, max(self.patch_size // 2 + 1, H - self.patch_size // 2))

            if centroids.shape[0] == 0:
                return cx, cy

            d2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
            if np.all(d2 > (2 * self.patch_size // 5) ** 2):
                return cx, cy

        # fallback
        return W // 2, H // 2

    def _build_stratify_labels(self, count_list: np.ndarray) -> np.ndarray:
        bins = []
        for c in count_list:
            if c == 0:
                bins.append("0")
            elif c == 1:
                bins.append("1")
            elif c == 2:
                bins.append("2")
            elif c <= 4:
                bins.append("3-4")
            else:
                bins.append("5+")
        return np.array(bins)


class IDB1FullImageTestDataset(Dataset):
    def __init__(
        self,
        dir_path: str | Path = "./data/ALL_IDB/ALL_IDB1",
        split_ratio: float = 0.8,
        random_state: int = 5,
        mask_radius: int = 12,
        image_only_transform=None,
    ):
        self.dir_path = Path(dir_path)
        self.mask_radius = mask_radius
        self.image_only_transform = image_only_transform

        img_files = sorted(self.dir_path.rglob("*.jpg"))
        centroid_files = sorted(self.dir_path.rglob("*.xyc"))

        if len(img_files) != len(centroid_files):
            raise AttributeError("Unequal number of image and centroid files.")

        paired_paths = list(zip(img_files, centroid_files))

        count_list = np.array([
            self._load_centroids(c_file).shape[0]
            for _, c_file in paired_paths
        ], dtype=np.int32)

        stratify_labels = self._build_stratify_labels(count_list)

        train_files, test_files = train_test_split(
            paired_paths,
            test_size=1.0 - split_ratio,
            random_state=random_state,
            stratify=stratify_labels,
        )

        self.test_files = test_files

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, idx):
        img_path, centroid_file = self.test_files[idx]

        image = image_import(img_path)  # CHW
        centroids = self._load_centroids(centroid_file)
        instance_masks = self._mask_generator(image, centroids)
        semantic_mask = self._semantic_from_instances(instance_masks, image)
        boxes = self._boxes_from_masks(instance_masks)
        centroids_t = self._centroids_from_masks(instance_masks)
        labels = torch.ones((boxes.shape[0],), dtype=torch.long)

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        if self.image_only_transform is not None:
            image = self.image_only_transform(image)

        target = {
            "semantic_mask": semantic_mask,
            "instance_masks": instance_masks,
            "boxes": boxes,
            "centroids": centroids_t,
            "labels": labels,
            "image_path": str(img_path),
        }
        return image, target

    # reuse same helper methods
    _load_centroids = IDB1PatchTrainDataset._load_centroids
    _mask_generator = IDB1PatchTrainDataset._mask_generator
    _semantic_from_instances = IDB1PatchTrainDataset._semantic_from_instances
    _boxes_from_masks = IDB1PatchTrainDataset._boxes_from_masks
    _centroids_from_masks = IDB1PatchTrainDataset._centroids_from_masks
    _build_stratify_labels = IDB1PatchTrainDataset._build_stratify_labels


def full_image_collate_fn(batch):
    assert len(batch) == 1, "Use batch_size=1 for full-resolution test inference."
    return batch[0]


def predicted_mask_to_boxes(pred_mask: torch.Tensor, min_area: int = 20):
    """
    pred_mask: (1,H,W) or (H,W), binary
    returns Tensor (N,4) in xyxy
    """
    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)

    mask_np = pred_mask.detach().cpu().numpy().astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, connectivity=8)

    boxes = []
    for i in range(1, num_labels):   # skip background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        boxes.append([x, y, x + w, y + h])

    if len(boxes) == 0:
        return torch.empty((0, 4), dtype=torch.float32)

    return torch.tensor(boxes, dtype=torch.float32)


def save_box_visualization(
    image: torch.Tensor,                 # (C,H,W), assumed float [0,1]
    true_boxes: torch.Tensor,            # (N,4)
    pred_boxes: torch.Tensor,            # (M,4)
    save_path: str,
):
    img = image.detach().cpu().clone()
    if img.max() <= 1.0:
        img = (img * 255.0).clamp(0, 255)

    img = img.byte().permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    # green = true
    for box in true_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    # red = predicted
    for box in pred_boxes:
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

    pil_img.save(save_path)


def patch_collate_fn(batch):
    images = []
    semantic_masks = []
    instance_masks = []
    boxes = []
    centroids = []
    labels = []
    patch_origins = []
    source_images = []

    for image, target in batch:
        images.append(image)
        semantic_masks.append(target["semantic_mask"])
        instance_masks.append(target["instance_masks"])
        boxes.append(target["boxes"])
        centroids.append(target["centroids"])
        labels.append(target["labels"])
        patch_origins.append(target["patch_origin"])
        source_images.append(target["source_image"])

    return torch.stack(images), {
        "semantic_mask": torch.stack(semantic_masks),
        "instance_masks": instance_masks,
        "boxes": boxes,
        "centroids": centroids,
        "labels": labels,
        "patch_origin": patch_origins,
        "source_image": source_images,
    }


def idb1_collate_fn(batch):
    images = []
    semantic_masks = []
    instance_masks = []
    boxes = []
    centroids = []
    labels = []

    for image, target in batch:
        images.append(image)
        semantic_masks.append(target["semantic_mask"])
        instance_masks.append(target["instance_masks"])
        boxes.append(target["boxes"])
        centroids.append(target["centroids"])
        labels.append(target["labels"])

    images = torch.stack(images, dim=0)               # (B, C, H, W)
    semantic_masks = torch.stack(semantic_masks, dim=0)  # (B, 1, H, W)

    batched_target = {
        "semantic_mask": semantic_masks,
        "instance_masks": instance_masks,   # list length B
        "boxes": boxes,                     # list length B
        "centroids": centroids,             # list length B
        "labels": labels,                   # list length B
    }

    return images, batched_target


def crop_boxes_from_image(image: torch.Tensor,
                          boxes: torch.Tensor,
                          output_size: int = 224):
    """
    Desc:   Crops the image based on boxes so that it can be fed to another classifer.
    """
    crops = []

    C, W, H = image.shape
    for box in boxes:
        x1, y1, x2, y2 = box.round().long()

        x1 = torch.clamp(x1, 0, W - 1)
        x2 = torch.clamp(x2, 0, W)
        y1 = torch.clamp(y1, 0, H - 1)
        y2 = torch.clamp(y2, 0, H)

        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[:, y1:y2, x1:x2]

        # Resize:
        crop = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=(output_size, output_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        crops.append(crop)

    return crops


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
