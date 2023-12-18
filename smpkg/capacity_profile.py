from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import torch
from dataclasses import dataclass
from neomodel import db
from datetime import datetime
from typing import Callable, Any, Optional, Literal


@dataclass
class ProfileFeature:
    r""" 单个特征信息 """
    feature_cls: str # 特征类型
    feature_value: Optional[str] = None # 特征值
    feature_time: Optional[datetime] = None # 特征时间


@dataclass
class ProfileCapacity:
    r""" 单个能力信息 """
    capacity_name: str # 能力类型名称
    capacity_level: Optional[str] = None # 能力等级


@dataclass
class Profile:
    r""" 人员能力画像 """
    static_features: list[ProfileFeature] # 静态特征
    dynamic_features: list[ProfileFeature] # 动态特征
    capacities: list[ProfileCapacity] # 能力


@dataclass
class DatasetPadding:
    static_feature_pad_len: int
    dynamic_feature_pad_len: int
    capa_pad_len: int

    capa_cls_pad_id: int              = 0
    capa_level_pad_id: int            = 0
    static_feature_cls_pad_id: int    = 0
    dynamic_feature_cls_pad_id: int   = 0
    dynamic_feature_value_pad_id: int = 0
    dynamic_feature_pad_interval: float = 1e6 # use a large interval let weight be 0

    capa_pad_strategy: Literal['all_capa', 'fix_len'] = 'all_capa'


class FeatureCapacityDataset(Dataset):
    r"""
    basic class for 
    """
    def __init__(self, 
        profiles: list[Profile], 
        static_feature_embedder: Callable[[ProfileFeature], int],
        dynamic_feature_embedder: Callable[[ProfileFeature], tuple[int, int, float]],
        target_embedder: Callable[[ProfileCapacity], tuple[int, int]],
        pad_setting: Optional[DatasetPadding] = None
    ) -> None:
        super().__init__()
        self.data = profiles
        self.static_feature_embedder = static_feature_embedder
        self.dynamic_feature_embedder = dynamic_feature_embedder
        self.target_embedder = target_embedder
        self.pad_setting = pad_setting

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[ list[int], list[int], list[int], list[int], list[int], list[int] ]:
        r"""
        
        Returns:
            0: list of static feature cls
            1: list of dynamic feature cls
            2: list of dynamic feature value
            3: list of dynamic feature time
            4: list of target capacity_clses
            5. list of target capacity_levels
            6. mask of static features
            7. mask of dynamic features
            8. mask of cpacity queries
        """
        perdata: Profile = self.data[idx]

        static_feature_cls = [ self.static_feature_embedder(feature.feature_cls) for feature in perdata.static_features ]
        static_feature_attn_mask = [1] * len(static_feature_cls)

        dynamic_features = [ self.dynamic_feature_embedder(feature) for feature in perdata.dynamic_features ]
        dynamic_feature_cls = [ feature[0] for feature in dynamic_features ]
        dynamic_feature_value = [ feature[1] for feature in dynamic_features ]
        dynamic_feature_time = [ feature[2] for feature in dynamic_features ]
        dynamic_feature_attn_mask = [1] * len(dynamic_feature_cls)

        capas = [ self.target_embedder(capacity) for capacity in perdata.capacities ]
        capacity_cls = [ capacity[0] for capacity in capas ]
        capacity_levels = [ capacity[1] for capacity in capas ]
        capa_mask = [1] * len(capacity_cls)

        if self.pad_setting:
            static_feature_cls = static_feature_cls[:self.pad_setting.static_feature_pad_len]
            static_feature_attn_mask = static_feature_attn_mask[:self.pad_setting.static_feature_pad_len]
            static_pad_len = max(0, self.pad_setting.static_feature_pad_len - len(static_feature_cls))
            static_feature_cls += [self.pad_setting.static_feature_cls_pad_id] * static_pad_len
            static_feature_attn_mask += [0] * static_pad_len

            dynamic_feature_cls = dynamic_feature_cls[:self.pad_setting.dynamic_feature_pad_len]
            dynamic_feature_value = dynamic_feature_value[:self.pad_setting.dynamic_feature_pad_len]
            dynamic_feature_time = dynamic_feature_time[:self.pad_setting.dynamic_feature_pad_len]
            dynamic_feature_attn_mask = dynamic_feature_attn_mask[:self.pad_setting.dynamic_feature_pad_len]
            dynamic_pad_len = max(0, self.pad_setting.dynamic_feature_pad_len - len(dynamic_feature_cls))
            dynamic_feature_cls += [self.pad_setting.dynamic_feature_cls_pad_id] * dynamic_pad_len
            dynamic_feature_value += [self.pad_setting.dynamic_feature_value_pad_id] * dynamic_pad_len
            dynamic_feature_time += [self.pad_setting.dynamic_feature_pad_interval] * dynamic_pad_len
            dynamic_feature_attn_mask += [0] * dynamic_pad_len
            
            if self.pad_setting.capa_pad_strategy == 'all_capa':
                # this means use all capacities, and use level_pad_id show the capacity of this person is not exist
                # this case capa_pad_len should equal to number of all might capacities
                # !advised, in this case model can figure if person has capacity
                # len(self.target_embedder) should be the num of capacities
                num_capacities = len(self.target_embedder)
                new_capacity_levels = [self.pad_setting.capa_level_pad_id] * num_capacities
                for i, cls_id in enumerate(capacity_cls):
                    new_capacity_levels[cls_id] = capacity_levels[i]
                capacity_levels = new_capacity_levels
                capacity_cls = list(range(num_capacities))
                capa_mask = [1] * num_capacities

            elif self.pad_setting.capa_pad_strategy == 'fix_len':
                # this case pad capacity by fix length use capa_cls_pad_id and capa_level_pad_id
                # !not advised, in this case model can't figure person who don't have capacity
                capacity_cls = capacity_cls[:self.pad_setting.capa_pad_len]
                capacity_levels = capacity_levels[:self.pad_setting.capa_pad_len]
                capa_pad_len = max(0, self.pad_setting.capa_pad_len - len(capacity_cls))
                capa_mask += [0] * capa_pad_len
                capacity_cls += [self.pad_setting.capa_cls_pad_id] * capa_pad_len
                capacity_levels += [self.pad_setting.capa_level_pad_id] * capa_pad_len
            else:
                raise ValueError(f'unkown capa_pad_strategy: {self.pad_setting.capa_pad_strategy}')

        return (
            torch.tensor(static_feature_cls, dtype=torch.long),
            torch.tensor(dynamic_feature_cls, dtype=torch.long), 
            torch.tensor(dynamic_feature_value, dtype=torch.long), 
            torch.tensor(dynamic_feature_time, dtype=torch.float), 
            torch.tensor(capacity_cls, dtype=torch.long), 
            torch.tensor(capacity_levels, dtype=torch.long), 
            torch.tensor(static_feature_attn_mask, dtype=torch.bool),
            torch.tensor(dynamic_feature_attn_mask, dtype=torch.bool),
            torch.tensor(capa_mask, dtype=torch.bool)
        )
        
class DataPrefetcher:

    def __init__(self, 
        dataset: FeatureCapacityDataset,
        batch_size: int,
        num_workers: int = 1,
        shuffle: bool = True,
        device: torch.device = torch.device('cpu')
    ) -> None:
        self.d_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.stream = torch.cuda.Stream(device)
        self.device = torch.device(device)
        self.loader = None
    
    def slash(self):
        self.loader = iter( self.d_loader ) 
        self.preload()

    def preload(self):
        try:
            (
                self.static_feature_cls,
                self.dynamic_feature_cls, 
                self.dynamic_feature_value, 
                self.dynamic_feature_time, 
                self.capacity_cls, 
                self.capacity_levels, 
                self.static_feature_attn_mask,
                self.dynamic_feature_attn_mask,
                self.capa_mask 
            ) = next(self.loader)
        except StopIteration:
            (
                self.static_feature_cls,
                self.dynamic_feature_cls, 
                self.dynamic_feature_value, 
                self.dynamic_feature_time, 
                self.capacity_cls, 
                self.capacity_levels, 
                self.static_feature_attn_mask,
                self.dynamic_feature_attn_mask,
                self.capa_mask 
            ) = (None, None, None, None, None, None, None, None, None)
            return
        if self.device.type == 'cuda':
            # using cuda.stream to speed up
            with torch.cuda.stream(self.stream):
                (
                    self.static_feature_cls,
                    self.dynamic_feature_cls, 
                    self.dynamic_feature_value, 
                    self.dynamic_feature_time, 
                    self.capacity_cls, 
                    self.capacity_levels, 
                    self.static_feature_attn_mask,
                    self.dynamic_feature_attn_mask,
                    self.capa_mask 
                ) = (
                    self.static_feature_cls.cuda(non_blocking=True),
                    self.dynamic_feature_cls.cuda(non_blocking=True),
                    self.dynamic_feature_value.cuda(non_blocking=True),
                    self.dynamic_feature_time.cuda(non_blocking=True),
                    self.capacity_cls.cuda(non_blocking=True),
                    self.capacity_levels.cuda(non_blocking=True),
                    self.static_feature_attn_mask.cuda(non_blocking=True),
                    self.dynamic_feature_attn_mask.cuda(non_blocking=True),
                    self.capa_mask.cuda(non_blocking=True),
                )
            
    def next(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        if self.device.type == 'cuda':
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
        static_feature_cls = self.static_feature_cls
        dynamic_feature_cls = self.dynamic_feature_cls
        dynamic_feature_value = self.dynamic_feature_value
        dynamic_feature_time = self.dynamic_feature_time
        capacity_cls = self.capacity_cls
        capacity_levels = self.capacity_levels
        static_feature_attn_mask = self.static_feature_attn_mask
        dynamic_feature_attn_mask = self.dynamic_feature_attn_mask
        capa_mask = self.capa_mask
        self.preload()
        return (
            static_feature_cls,
            dynamic_feature_cls, 
            dynamic_feature_value, 
            dynamic_feature_time, 
            capacity_cls, 
            capacity_levels, 
            static_feature_attn_mask,
            dynamic_feature_attn_mask,
            capa_mask 
        )

