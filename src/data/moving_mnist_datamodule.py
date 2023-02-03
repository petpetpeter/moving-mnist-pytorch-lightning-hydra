from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from components.moving_mnist_dataset import MovingMNistDataset
from torchvision.datasets import MNIST
from torchvision.transforms import transforms



import numpy as np
import cv2
import matplotlib.pyplot as plt

class MonvingMnistDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        train_val_test_split: Tuple[int, int, int] = (55_000, 5_000, 10_000),
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor()]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 10

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        #download with wget
        #
        #result = subprocess.run(['wget', 'http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy'], stdout=subprocess.PIPE)
        return 1

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        data_shape = (10,1,64,64) # (seq_len, Channel, height, width)
        # load and split datasets only if not loaded already
        data_numpy = np.load('/root/dia_ws/moving-mnist-pytorch-lightning-hydra/data/mnist_test_seq.npy')
        data_numpy = np.swapaxes(data_numpy, 0, 1)
        #print(f"shape of data_numpy: {data_numpy.shape}")
        #cp_data_numpy = data_numpy.copy()
        #cp_data_numpy = np.swapaxes(cp_data_numpy, 0, 1)
        #cp_data_1 = cp_data_numpy[0]
        # for img in cp_data_1:
        #     cv2.imshow('real', img)
        #     cv2.waitKey(0)
        # (seq_len,num_seq,height,width)
        # swap to (num_seq,seq_len,height,width)
        #data_numpy = np.swapaxes(data_numpy,0,1)
        #data_numpy = data_numpy/255
        
        #random split to train, val, test 0.7, 0.15, 0.15
        len_train = int(data_numpy.shape[0]*0.7)
        len_val = int(data_numpy.shape[0]*0.15)
        len_test = data_numpy.shape[0] - len_train - len_val
        self.data_train = MovingMNistDataset(data_numpy[:len_train])
        self.data_val = MovingMNistDataset(data_numpy[len_train:len_train+len_val])
        self.data_test = MovingMNistDataset(data_numpy[len_train+len_val:])
        

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    mmd = MonvingMnistDataModule()
    mmd.prepare_data()
    mmd.setup()
    x_frames,y_frames = next(iter(mmd.train_dataloader()))
    print(f"shape of x_frames: {x_frames.shape}")
    print(f"shape of y_frames: {y_frames.shape}")
    first_batch_x = x_frames[1]
    first_batch_y = y_frames[1]

    for img in first_batch_x:
        cv2.imshow('image',img.numpy())
        cv2.waitKey(0)
    for img in first_batch_y:
        cv2.imshow('image2',img.numpy())
        cv2.waitKey(0)





