from typing import Tuple, Optional, List, Dict
import torch.nn as nn
import torch, os
import numpy as np
from tqdm import tqdm
import pandas as pd
from DomainAdaptationTrainer.CST_Utils.dalib.modules.grl import WarmStartGradientReverseLayer


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    return torch.log(torch.clamp(x + offset, max=1.))


class ImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, bottleneck: Optional[nn.Module] = None,
                 bottleneck_dim: Optional[int] = -1, head: Optional[nn.Module] = None, finetune=True):
        super(ImageClassifier, self).__init__()
        self.backbone = nn.Sequential(backbone,nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten())
        bottleneck = nn.Sequential(
            
            nn.Linear(backbone.out_features, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self.num_classes = num_classes
        if bottleneck is None:
            self.bottleneck = nn.Sequential(
            )
            self._features_dim = backbone.out_features
        else:
            self.bottleneck = bottleneck
            assert bottleneck_dim > 0
            self._features_dim = bottleneck_dim

        if head is None:
            self.head = nn.Linear(self._features_dim, num_classes)
        else:
            self.head = head
        self.finetune = finetune

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        f = self.backbone(x)
        f1 = self.bottleneck(f)
        predictions = self.head(f1)
        return predictions, f

    def get_parameters(self, base_lr=1.0) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr": 0.1 * base_lr if self.finetune else 1.0 * base_lr},
            {"params": self.bottleneck.parameters(), "lr": 1.0 * base_lr},
            {"params": self.head.parameters(), "lr": 1.0 * base_lr},
        ]

        return params

def construct_data_list_file(dir_name):
    label = np.load(
        os.path.join(dir_name, 'label.npy')
    ).reshape(-1)

    domain = np.load(
        os.path.join(dir_name, 'domain.npy')
    ).reshape(-1)

    assert len(label) == len(domain)

    fnames = [f'image_{i}.npy' for i in range(len(label))]
    for fname in tqdm(fnames):
        try:
            assert os.path.exists(
                os.path.join(dir_name, fname)
            )
        except:
            print(f'{os.path.join(dir_name, fname)} does not exist!')
            raise

    df = pd.DataFrame()
    df['path'] = fnames
    df['label'] = label
    df['domain'] = domain
    df.to_csv( os.path.join(dir_name, 'image_list.csv'), index=False)


def re_organize(dir_name, csv_fname):
    df = pd.read_csv(
        os.path.join(dir_name, csv_fname)
    )
    label = df.label.values
    domain = df.domain.values
    fnames = df.path.values
    image_list = [
        "image_list/Art.txt",
        "image_list/Clipart.txt",
        "image_list/Product.txt",
        "image_list/Real_World.txt",
    ]
    for d_id in range(4):
        d_paths = fnames[domain.__eq__(d_id)]
        d_label = label[domain.__eq__(d_id)]
        lines = [f"{d_paths[idx]} {d_label[idx]}\n" for idx in range(len(d_paths))]
        with open(
                os.path.join(dir_name, image_list[d_id]),
                'w'
        ) as fw:
            fw.writelines(lines)
