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
"""Simple MNIST image classifier example with LightningModule and LightningDataModule.
To run: python train.py --trainer.max_epochs=50
"""
import os

from lightningmnist.cifar100_datamodule import Cifar100DataModule
from lightningmnist.classifier import ImageClassifier
from pytorch_lightning.plugins.environments import ClusterEnvironment
from pytorch_lightning.utilities.cli import LightningCLI


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument("--job-dir", default=".")
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("job_dir", "data.job_dir")
        parser.link_arguments("job_dir", "trainer.default_root_dir")


def cli_main():
    if 'CLUSTER_SPEC' in os.environ:
        print(f"CLUSTER_SPEC: {os.environ['CLUSTER_SPEC']}")

    if 'RANK' in os.environ:
        print(f"RANK: {os.environ['RANK']}")

    if 'LOCAL_RANK' in os.environ:
        print(f"LOCAL_RANK: {os.environ['LOCAL_RANK']}")

    if 'WORLD_SIZE' in os.environ:
        print(f"WORLD_SIZE: {os.environ['WORLD_SIZE']}")

    if 'MASTER_ADDR' in os.environ:
        print(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")

    if 'MASTER_PORT' in os.environ:
        print(f"MASTER_PORT: {os.environ['MASTER_PORT']}")

    # The LightningCLI removes all the boilerplate associated with arguments parsing. This is purely optional.
    cli = MyLightningCLI(
        ImageClassifier,
        Cifar100DataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)


if __name__ == "__main__":
    cli_main()
