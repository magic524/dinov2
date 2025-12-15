# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
from typing import Callable, Optional
from enum import Enum

import numpy as np
from PIL import Image

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")


class _Split(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @property
    def value_str(self) -> str:
        return self.value


class CustomImageFolder(ExtendedVisionDataset):
    """
    自定义图像文件夹数据集，支持标准的ImageNet格式：
    root/
        train/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img1.jpg
        val/
            class1/
            class2/
    """
    
    Split = _Split
    
    def __init__(
        self,
        *,
        split: Split,
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        self._split = split
        
        self._entries = None
        self._class_ids = None
        self._class_names = None

    @property
    def split(self):
        return self._split
    
    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries-{self._split.value.upper()}.npy"

    @property
    def _class_ids_path(self) -> str:
        return f"class-ids-{self._split.value.upper()}.npy"

    @property
    def _class_names_path(self) -> str:
        return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self):
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self):
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self):
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return class_names[class_index]

    def get_image_data(self, index: int):
        entries = self._get_entries()
        class_ids = self._get_class_ids()
        
        image_relpath = entries[index].decode('utf-8')
        class_id = class_ids[index]
        
        image_full_path = os.path.join(self.root, image_relpath)
        image = Image.open(image_full_path).convert("RGB")
        
        return image, class_id

    def get_target(self, index: int) -> int:
        class_ids = self._get_class_ids()
        return int(class_ids[index])

    def get_targets(self):
        class_ids = self._get_class_ids()
        return class_ids.tolist()

    def get_class_id(self, index: int) -> str:
        class_ids = self._get_class_ids()
        return self.find_class_id(class_ids[index])

    def get_class_name(self, index: int) -> str:
        class_id = self.get_class_id(index)
        return class_id

    def __len__(self) -> int:
        entries = self._get_entries()
        assert entries is not None
        return len(entries)

    def __getitem__(self, index: int):
        image, target = self.get_image_data(index)
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        
        return image, target

    def dump_extra(self):
        """生成并保存元数据文件"""
        split_dir = os.path.join(self.root, self._split.value)
        
        if not os.path.exists(split_dir):
            logger.warning(f"Split directory does not exist: {split_dir}")
            return
        
        # 获取所有类别
        classes = sorted([d for d in os.listdir(split_dir) 
                         if os.path.isdir(os.path.join(split_dir, d))])
        
        entries = []
        class_ids = []
        
        logger.info(f"Processing {self._split.value} split...")
        logger.info(f"Found classes: {classes}")
        
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(split_dir, class_name)
            
            # 获取该类别的所有图片
            image_files = sorted([f for f in os.listdir(class_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            
            logger.info(f"Class {class_name}: {len(image_files)} images")
            
            for img_file in image_files:
                rel_path = os.path.join(self._split.value, class_name, img_file)
                entries.append(rel_path)
                class_ids.append(class_idx)
        
        # 转换为numpy数组
        entries_array = np.array(entries, dtype='S')
        class_ids_array = np.array(class_ids, dtype=np.int32)
        class_names_array = np.array(classes, dtype='S')
        
        # 保存
        logger.info(f"Saving {len(entries)} entries to {self._entries_path}")
        self._save_extra(entries_array, self._entries_path)
        self._save_extra(class_ids_array, self._class_ids_path)
        self._save_extra(class_names_array, self._class_names_path)
        
        logger.info(f"Successfully created metadata for {self._split.value} split")
        logger.info(f"Total images: {len(entries)}")
        logger.info(f"Classes: {classes}")
