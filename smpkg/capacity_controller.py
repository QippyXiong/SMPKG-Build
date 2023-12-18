from dataclasses import dataclass, asdict
from os import PathLike
from torch.utils.data import DataLoader
from typing import Any, overload, Union, Optional, Callable, TypeVar
from datetime import datetime, date
from torch.optim import SGD
from torch import nn
from transformers.optimization import get_linear_schedule_with_warmup
import torch, gc, numpy
from sklearn.metrics import classification_report
from pathlib import Path

from smpkg.attention_evaluator_model import VanillaCapacityEvaluatorConfig, VanillaCapacityEvaluator, AttentionEvaluatorModelConfig
from smpkg.capacity_profile import Profile, ProfileFeature, FeatureCapacityDataset, ProfileCapacity, DatasetPadding, DataPrefetcher
from smpkg.logger import Logger, logger
import json5
from functools import wraps


@dataclass
class CapacityGenerationSettings:
    capacity_names: list[str] # 能力名称，无默认初始化空值
    capacity_levels: list[str] # 能力可选等级，建议从低到高，最低为不包含此能力
    static_feature_classes: list[str] # 静态特征类型
    dynamic_feature_classes: list[str] # 动态特征类型
    dynamic_feature_values: list[str] # 动态特征的可能值
    # init_null_values: bool = False # 是否初始化空值

    def __post_init__(self) -> None:
        self.capacity_names2id = {name: id for id, name in enumerate(self.capacity_names)}
        self.capacity_levels2id = {level: id for id, level in enumerate(self.capacity_levels)}
        self.static_feature_classes2id = {cls: id for id, cls in enumerate(self.static_feature_classes)}
        self.dynamic_feature_classes2id = {cls: id for id, cls in enumerate(self.dynamic_feature_classes)}
        self.dynamic_feature_values2id = {value: id for id, value in enumerate(self.dynamic_feature_values)}

        # initialize null values

# class DateTimeEncoder(json5.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, datetime):
#             return obj.strftime('%Y-%m-%d %H:%M:%S')
#         elif isinstance(obj, date):
#             return obj.strftime('%Y-%m-%d')
#         return json5.JSONEncoder.default(self, obj)

method_T = TypeVar('method_T', bound=Callable)

def check_model(func: method_T) -> method_T:
    @wraps(func)
    def wrapper(self, *args, **kargs):
        if self.model is None:
            raise ValueError(f'model is not initialized for function call { func.__name__ }')
        else:
            return func(self, *args, **kargs)
    return wrapper

class CapacityGenerationController:
    r"""
    执行模型的创建、训练、预测、保存和加载。
    """
    model: Optional[VanillaCapacityEvaluator]
    @overload
    def __init__(
        self,
        settings: CapacityGenerationSettings,
        model_config: VanillaCapacityEvaluatorConfig,
        logger: Optional[Logger] = None
    ):
        ...
    @overload
    def __init__(
        self,
        settings: CapacityGenerationSettings,
        model: Optional[VanillaCapacityEvaluator] = None,
        logger: Optional[Logger] = None
    ):
        ...

    def __init__(self,
        settings: CapacityGenerationSettings,
        model_or_config: Union[VanillaCapacityEvaluatorConfig, VanillaCapacityEvaluator] = None,
        logger: Optional[Logger] = None
    ) -> None:
        self.settings = settings
        self.logger = logger

        if isinstance(model_or_config, VanillaCapacityEvaluatorConfig):
            self.init_model(model_or_config)
        elif isinstance(model_or_config, VanillaCapacityEvaluator):
            self.model = model_or_config
        elif model_or_config is None:
            self.model = None
        else:
            raise TypeError('model_or_config must be VanillaCapacityEvaluatorConfig or VanillaCapacityEvaluator')
        
        # create embedders
        self.static_feature_embedder = self.FeatureEmbedder(self.settings.static_feature_classes2id)
        self.dynamic_feature_embedder = self.FeatureEmbedder(
            self.settings.dynamic_feature_classes2id,
            self.settings.dynamic_feature_values2id
        )
        self.capacity_embedder = self.CapacityEmbedder(
            self.settings.capacity_names2id,
            self.settings.capacity_levels2id
        )
    
    def init_model(
        self,
        config: VanillaCapacityEvaluatorConfig
    ) -> None:
        r""" 初始化模型，如果模型以创建会摧毁原先的模型根据config重新创建 """
        config.num_capacity_cls = len(self.settings.capacity_names)
        config.num_dynamic_feature_cls = len(self.settings.dynamic_feature_classes)
        config.num_static_feature_cls = len(self.settings.static_feature_classes)
        config.num_dynamic_feature_values = len(self.settings.dynamic_feature_values)
        config.num_labels = len(self.settings.capacity_levels)
        self.model = VanillaCapacityEvaluator(config)
    
    def release_model(self) -> None:
        r""" 释放模型，如果模型没有创建什么都不会发生 """
        if self.model is None:
            return
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    @property
    def default_config(self):
        r"""
        """
        config = VanillaCapacityEvaluatorConfig(
            query_size=32,
            query_hidden_size=32,
            num_query_mapper_layers=2,
            feature_size=32,
            feature_hidden_size=32,
            num_feature_mapper_layers=2,
            embed_size=32,
            num_heads=4,
            cls_hidden_size=32,
            num_cls_layers=2,
            num_labels=len(self.settings.capacity_levels),
            num_transformer_blks=0,
            
            # training params
            num_epochs=5,
            batch_size=24,
            lr=0.001,
            weight_decay=0.1,
            dropout=0.1,
            warming_up_proportion=0.1,

            num_capacity_cls=len(self.settings.capacity_names),
            num_dynamic_feature_cls=len(self.settings.dynamic_feature_classes),
            num_static_feature_cls=len(self.settings.static_feature_classes),
            num_dynamic_feature_values=len(self.settings.dynamic_feature_values),
            time_decay_func="norm",
            includeing_days=200.0,
            static_feature_num=50,
            dynamic_feature_num=50,
            capacity_num=20,
        )
        return config
    
    def save(self, path: PathLike) -> None:
        r"""
        保存模型和配置文件

        Args:
            path: 保存路径
        """
        self.save_model(path)
        settings_path = Path(path).joinpath('settings.json')
        with open(settings_path, 'w', encoding='UTF-8') as fp:
            json5.dump(asdict(self.settings), fp, quote_keys=True, trailing_commas=False)

    @check_model
    def save_model(self, path: PathLike) -> None:
        self.model.save(path)

    def load_model(self, path: PathLike) -> None:
        self.release_model()
        self.model = VanillaCapacityEvaluator.load(path)

    @classmethod
    def load(cls, path: PathLike):
        settings_path = Path(path).joinpath('settings.json')
        with open(settings_path, 'r', encoding='UTF-8') as fp:
            settings = CapacityGenerationSettings(**json5.load(fp))
        model = VanillaCapacityEvaluator.load(str(path))
        controller = cls(settings, model)
        return controller
    
    class FeatureEmbedder:
        r""" 用于将特征转换为张量 """
        def __init__(self, feature_cls2id: dict, feature_value2id: Optional[dict] = None, compute_time: Optional[datetime] = None):
            self.feature_cls2id = feature_cls2id
            self.feature_value2id = feature_value2id
            self.compute_time = compute_time
        
        def __call__(self, feature: ProfileFeature) -> Union[int, tuple[int, int, int]]:
            feature_cls = self.feature_cls2id.get(feature.feature_cls, None)
            if feature_cls is None:
                raise ValueError(f'invalid feature class: {feature.feature_cls}')
            elif feature.feature_value is None and feature.feature_time is None:
                return feature_cls
            feature_value = self.feature_value2id.get(feature.feature_value, None)
            if feature_value is None:
                raise ValueError(f'invalid feature value: {feature.feature_value}')
            compute_time = self.compute_time if self.compute_time is not None else datetime.now()
            feature_time = (compute_time - feature.feature_time).days
            return feature_cls, feature_value, feature_time
            
    class CapacityEmbedder:
        r""" 用于将能力信息转换为张量 """
        def __init__(self, capacity_name2id: dict, capacity_level2id = dict) -> None:
            self.capacity_name2id = capacity_name2id
            self.capacity_level2id = capacity_level2id
        
        def __call__(self, target: ProfileCapacity) -> Any:
            cls_id = self.capacity_name2id.get(target.capacity_name, None)
            if cls_id is None:
                raise ValueError(f'unkown capacity name: {target}')
            if target.capacity_level is None:
                level_id = 0
            else:
                level_id = self.capacity_level2id.get(target.capacity_level, None)
            if level_id is None:
                raise ValueError(f'unkown capacity level: {target}')
            return cls_id, level_id
        
        def __len__(self) -> int:
            return len(self.capacity_name2id)
        
        @property
        def num_levels(self) -> int:
            return len(self.capacity_level2id)

    @check_model
    def train_model(self,
        profiles: list[Profile],
        begin_time: datetime,
        device: torch.device,
        num_workers: int
    ) -> None:
        r"""
        训练人员能力生成模型

        Args:
            profiles: 人员画像列表
            begin_time: 训练数据参考开始时间，比如训练数据是采集到2023年11月12号，那么begin_time应该是2023年11月12号
            device: 训练设备，'cpu' or 'cuda'
        """
        
        # create dataset
        previous_dynamic_time = self.get_rel_time()
        self.set_rel_time(begin_time)
        padding_setting = DatasetPadding(
            self.model.config.static_feature_num,
            self.model.config.dynamic_feature_num,
            self.model.config.num_capacity_cls,
            capa_pad_strategy='all_capa'
        )
        train_dataset = FeatureCapacityDataset(
            profiles,
            self.static_feature_embedder,
            self.dynamic_feature_embedder,
            self.capacity_embedder,
            padding_setting
        )
        pre_fetcher = DataPrefetcher(
            train_dataset, 
            self.model.config.batch_size,
            num_workers,
            True,
            device
        )

        # optimizer & scheduler
        optimizer = SGD(list(self.model.parameters()), lr=self.model.config.lr, weight_decay=self.model.config.weight_decay)
        total_training_step = self.model.config.num_epochs * len(train_dataset) 
        warm_up_steps = self.model.config.num_epochs * len(train_dataset) * self.model.config.warming_up_proportion
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_training_step)
        loss = nn.CrossEntropyLoss()
        model = self.model
        model.to(device)
        model.train()
        step = 1
        for epoch in range(1, self.model.config.num_epochs+1):
            pre_fetcher.slash()
            (
                static_feature_cls,
                dynamic_feature_cls, 
                dynamic_feature_value, 
                dynamic_feature_time, 
                capacity_cls, 
                capacity_levels, 
                static_feature_attn_mask,
                dynamic_feature_attn_mask,
                capa_mask 
            ) = pre_fetcher.next()
            while static_feature_cls is not None:
                pred = model.forward(
                    capacity_cls,
                    static_feature_cls,
                    dynamic_feature_cls,
                    dynamic_feature_value,
                    dynamic_feature_time,
                    static_feature_attn_mask,
                    dynamic_feature_attn_mask
                )
                optimizer.zero_grad()
                # capacity_levels = capacity_levels.masked_fill_(~capa_mask, padding_setting.capa_level_pad_id)
                l = loss.forward(pred.permute(0, 2, 1), capacity_levels)
                l.backward()
                optimizer.step()
                scheduler.step()
                (
                    static_feature_cls,
                    dynamic_feature_cls, 
                    dynamic_feature_value, 
                    dynamic_feature_time, 
                    capacity_cls, 
                    capacity_levels, 
                    static_feature_attn_mask,
                    dynamic_feature_attn_mask,
                    capa_mask 
                ) = pre_fetcher.next()
                # callback
                self.logger.log_train(epoch, step, l.item())
                step += 1
        
        # set back compute time
        self.set_rel_time(previous_dynamic_time)

    def set_rel_time(self, rel_time: datetime) -> None:
        r"""
        设置动态特征计算时间
        """
        self.dynamic_feature_embedder.compute_time = rel_time

    def get_rel_time(self) -> datetime:
        r"""
        获取动态特征计算时间
        """
        if self.dynamic_feature_embedder.compute_time is None:
            return datetime.now()
        else:
            return self.dynamic_feature_embedder.compute_time
    
    def move_model(self, device: torch.device) -> None:
        r"""
        move model to device
        """
        self.model.to(device)

    @torch.no_grad()
    @check_model
    def predicate(self,
        profiles: list[Profile],
        rel_time: Optional[datetime] = None,
        validation: bool = False,
        report_text: bool = False
    ) -> Union[list[list[ProfileCapacity]], Union[dict, str]]:
        r"""
        batched predicate
        Args:
            profiles: 要预测的人员画像列表，每个画像中的能力等级信息将被忽略，预测对应能力类型信息的等级
        
        Returns:
            if validation is True return classification report ( str for report_text is True else dict )
            else return predicate capacity profiles
        """
        self.model.eval()
        # dataset pad setting
        if len(profiles) == 1:
            pad_setting = None
        else:
            max_static_feature_len = max([len(profile.static_features) for profile in profiles])
            max_dynamic_feature_len = max([len(profile.dynamic_features) for profile in profiles])
            max_capacity_len = max([len(profile.capacities) for profile in profiles])
            if rel_time is None:
                rel_time = datetime.now()
            pad_setting = DatasetPadding(
                max_static_feature_len,
                max_dynamic_feature_len,
                max_capacity_len,
                capa_pad_strategy='all_capa'
            )
        
        previous_time = self.get_rel_time()
        self.set_rel_time(rel_time)

        pred_dataset = FeatureCapacityDataset( 
            profiles, 
            self.static_feature_embedder, 
            self.dynamic_feature_embedder, 
            self.capacity_embedder,
            pad_setting
        )

        loader = DataLoader(pred_dataset, batch_size= self.model.config.batch_size if validation else len(pred_dataset), shuffle=False)
        device = self.model.device
        pred_levels = []
        label_levels = []

        for (
            static_feature_cls,
            dynamic_feature_cls, 
            dynamic_feature_value, 
            dynamic_feature_time, 
            capacity_cls, 
            capacity_levels, 
            static_feature_attn_mask,
            dynamic_feature_attn_mask,
            capa_mask 
        ) in loader:

            (
                static_feature_cls,
                dynamic_feature_cls, 
                dynamic_feature_value, 
                dynamic_feature_time, 
                capacity_cls, 
                capacity_levels, 
                static_feature_attn_mask,
                dynamic_feature_attn_mask,
                capa_mask 
            ) = (
                static_feature_cls.to(device),
                dynamic_feature_cls.to(device),
                dynamic_feature_value.to(device), 
                dynamic_feature_time.to(device), 
                capacity_cls.to(device), 
                capacity_levels.to(device), 
                static_feature_attn_mask.to(device),
                dynamic_feature_attn_mask.to(device),
                capa_mask.to(device)
            )

            pred = self.model.forward(
                capacity_cls,
                static_feature_cls,
                dynamic_feature_cls,
                dynamic_feature_value,
                dynamic_feature_time,
                static_feature_attn_mask,
                dynamic_feature_attn_mask
            )
            pred_levels.append(torch.argmax(pred, dim=-1).detach().cpu().numpy())
            if validation:
                label_levels.append(capacity_levels.detach().cpu().numpy())
        
        self.set_rel_time(previous_time)
        pred_levels = numpy.concatenate(pred_levels, axis=0)
        if validation:
            r"""
            two schema:
            1. compute classification of each query result
            2. set label name be capacity_name + '_' + capacity_level

            implementation schema 2
            set previous level idx to be capacity_idx * num_capacity_levels + previous_level_idx
            """
            label_levels = numpy.concatenate(label_levels, axis=0) # shape (num_profiles, num_capacities)
            add_levels = numpy.arange(len(self.settings.capacity_names)) * len(self.settings.capacity_levels) # shape (num_capacities)
            label_levels += add_levels
            pred_levels += add_levels
            label_names = [f'{name}_{level}' for name in self.settings.capacity_names for level in self.settings.capacity_levels]
            return classification_report(
                label_levels.reshape(-1), 
                pred_levels.reshape(-1), 
                labels=numpy.arange(len(label_names)), 
                target_names=label_names, 
                output_dict=not report_text, 
                zero_division=0
            )
        # else:
        pred_levels = pred_levels.tolist()
        pred_capacities = [
            [
                ProfileCapacity(
                    capacity_name=capa_name,
                    capacity_level=self.settings.capacity_levels[pred_level[i]]
                ) for i, capa_name in enumerate(self.settings.capacity_names)
            ]
            for pred_level in pred_levels
        ]
        return pred_capacities
