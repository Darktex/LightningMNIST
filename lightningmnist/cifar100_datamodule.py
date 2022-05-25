# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pathlib import Path

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from torchvision.datasets import CIFAR100


class Cifar100DataModule(LightningDataModule):
    def __init__(self, job_dir, batch_size=32):
        super().__init__()
        self.save_hyperparameters()

    @property
    def transform(self):
        return T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
                ),
            ]
        )

    def prepare_data(self) -> None:
        CIFAR100(Path(self.hparams.job_dir) / "./data", download=True, train=True)
        CIFAR100(Path(self.hparams.job_dir) / "./data", download=True, train=False)

    def train_dataloader(self):
        train_dataset = CIFAR100(
            Path(self.hparams.job_dir) / "./data",
            train=True,
            download=False,
            transform=self.transform,
        )
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=self.hparams.batch_size
        )

    def test_dataloader(self):
        test_dataset = CIFAR100(
            Path(self.hparams.job_dir) / "./data",
            train=False,
            download=False,
            transform=self.transform,
        )
        return torch.utils.data.DataLoader(
            test_dataset, batch_size=self.hparams.batch_size
        )
