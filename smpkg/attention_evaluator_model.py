import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional, Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers.optimization import get_linear_schedule_with_warmup


__all__ = [
    'AttentionEvaluatorModelConfig',
    'AttentionEvaluatorModel',
    'VanillaCapacityEvaluatorConfig',
    'VanillaCapacityEvaluator'
]


@dataclass
class AttentionEvaluatorModelConfig:
    r""" config of AttentionEvaluatorModel """
    # hyper parameters
    query_feature_size: int
    embed_size: int

    num_query_mapper_layers: int
    num_feature_mapper_layers: int
    num_value_mapper_layers: int
    num_cls_layers: int
    num_labels: int

    num_transformer_blks: int
    num_heads: int
    
    # training params:
    num_epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    dropout: float
    warming_up_proportion: float


class EvaluatorMapperBlock(nn.Module):

    def __init__(self, embed_size: int, dropout: float = 0., activate_block: Optional[nn.Module] = None) -> None:
        super().__init__()
        if activate_block is None:
            activate_block = nn.ReLU()
        
        self.block = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            activate_block,
            nn.Dropout(dropout),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.block(x) + x
            

class AttentionEvaluatorModel(nn.Module):
    config_cls = AttentionEvaluatorModelConfig

    def __init__(self, config: AttentionEvaluatorModelConfig):
        super().__init__()

        self.config: AttentionEvaluatorModelConfig = config

        # initial mappers
        self.query_mapper = nn.Sequential()
        for _ in range(config.num_query_mapper_layers):
            block = EvaluatorMapperBlock(config.query_feature_size, config.dropout)
            for param in block.parameters():
                if param.dim() > 1:
                    nn.init.xavier_normal_(param)
                else:
                    nn.init.zeros_(param)
            self.query_mapper.append(block)
        
        self.feature_mapper = nn.Sequential()
        for _ in range(config.num_feature_mapper_layers):
            self.feature_mapper.append(EvaluatorMapperBlock(config.query_feature_size, config.dropout))

        self.value_mapper = nn.Sequential()
        for _ in range(config.num_value_mapper_layers):
            self.value_mapper.append(EvaluatorMapperBlock(config.embed_size, config.dropout))
        
        self.cls = nn.Sequential()
        for _ in range(config.num_cls_layers):
            self.cls.append(EvaluatorMapperBlock(config.embed_size, config.dropout))
        self.cls.append(nn.Linear(config.embed_size, config.num_labels))

        # initial transformers layer
        if config.num_transformer_blks:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.embed_size,
                nhead=config.num_heads,
                batch_first=True,
                dropout=config.dropout
            )
            self.transformers = nn.TransformerEncoder(
                encoder_layer,
                num_layers=config.num_transformer_blks
            )
        else:
            self.transformers = None

    def forward(
        self,
        query: Tensor,
        features: Tensor,
        values: Tensor,
        attention_mask: Optional[Tensor] = None,
        feature_time_weights: Optional[Tensor] = None
    ) -> Tensor:
        r"""
        Args:
            query: shape(batch_size, num_queries, query_size)
            features: shape(batch_size, num_features, feature_size)
            values: shape(batch_size, num_features, value_size)
            attention_mask: shape(batch_size, num_features)
            feature_time_weights: shape(batch_size, num_features)
        Returns:
            shape(batch_size, num_queries, num_labels)
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(features)
        if feature_time_weights is None:
            feature_time_weights = torch.ones_like(features)
        feature_time_weights = feature_time_weights.unsqueeze(1).repeat(1, query.shape[1], 1)
        # shape (batch_size, num_queries, num_features)

        input_query = self.query_mapper(query)
        input_features = self.feature_mapper(features)

        input_features = input_features.unsqueeze(1).repeat(1, input_query.shape[1], 1, 1)
        input_query = input_query.unsqueeze(2).repeat(1, 1, input_features.shape[2], 1)
        # shape (batch_size, num_queries, num_features, embed_size)

        attn_output_weights = nn.functional.sigmoid((input_features * input_query).sum(-1))
        # shape (batch_size, num_queries, num_features)
        if self.transformers:
            transformer_mask = attention_mask.unsqueeze(1).repeat(self.config.num_heads, values.shape[1], 1)
            values = self.transformers(values, ~transformer_mask)
            # shape(batchsize*num_heads, num_features, num_features)
        input_values = self.value_mapper(values)

        attention_mask = attention_mask.unsqueeze(1).repeat(1, query.shape[1], 1)
        # shape (batch_size, num_queries, num_features)
        input_values = input_values.unsqueeze(1).repeat(1, query.shape[1], 1, 1)
        # shape (batch_size, num_queries, num_features, value_size)

        weights = (attn_output_weights * feature_time_weights * attention_mask.float()).unsqueeze(-1)
        cross_pooling_result = input_values.permute(0, 1, 3, 2) @ weights
        cross_pooling_result = cross_pooling_result.squeeze(-1)
        # shape (batch_size, num_queries, embed_size)

        cls_r = self.cls(cross_pooling_result.squeeze(-1))
        # shape (batch_size, num_queries, num_labels)
        return cls_r

    def save(self, save_path: str):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_path = Path(save_path).joinpath('pytorch_model.bin')
        config_path = Path(save_path).joinpath('config.json')
        torch.save(self.state_dict(), model_path)
        fp = open(config_path, "w", encoding='UTF-8')
        json.dump(asdict(self.config), fp)
        fp.close()

    @classmethod
    def load(cls, save_path: str, *args):
        r"""
        Return:
            type is cls who use this method
        """
        if not os.path.exists(save_path):
            raise FileNotFoundError(f'{save_path} not found')
        model_path = Path(save_path).joinpath('pytorch_model.bin')
        config_path = Path(save_path).joinpath('config.json')

        fp = open(config_path, "r", encoding='UTF-8')
        config = json.load(fp)
        fp.close()
        config = cls.config_cls(**config)
        model = cls(config, *args)
        model.load_state_dict(torch.load(model_path))
        return model
    
    @property
    def device(self):
        return next(self.parameters()).device


@dataclass
class VanillaCapacityEvaluatorConfig(AttentionEvaluatorModelConfig):
    r""" config of VanillaCapacityEvaluator """
    num_capacity_cls: int
    num_dynamic_feature_cls: int
    num_static_feature_cls: int
    num_dynamic_feature_values: int
    # num_scale: float = 0.001
    time_decay_func: Literal['exp', 'norm', 'linear', 'fit_exp', 'fit_linear', 'fit_norm']
    includeing_days: float  # 主要考虑的包含在内的天数

    # training params
    static_feature_num: int
    dynamic_feature_num: int
    capacity_num: int
    alpha: float  # for dot product regular loss
    beta: float  # for l2 norm regular loss

    @property
    def k(self):
        return 1/self.includeing_days

    @property
    def sigma(self):
        return self.includeing_days / 3

    @property
    def e_axis(self):
        return self.includeing_days / 4

    @property
    def b(self):
        return self.includeing_days

    @property
    def norm_scale(self):
        return 1 / ((2 * 3.1415926) ** 0.5 * self.sigma)


class VanillaCapacityEvaluator(AttentionEvaluatorModel):
    config_cls = VanillaCapacityEvaluatorConfig
    config: VanillaCapacityEvaluatorConfig
    r"""
    implement capacity evaluator by nn.Embedding
    """
    def __init__(self, config: VanillaCapacityEvaluatorConfig):
        super().__init__(config)
        self.config = config
        self.capacity_embed = nn.Embedding(config.num_capacity_cls, config.query_feature_size)
        self.static_feature_cls_embed = nn.Embedding(config.num_static_feature_cls, config.query_feature_size)
        self.static_feature_values_embed = nn.Embedding(config.num_static_feature_cls, config.embed_size)
        self.dynamic_feature_cls_embed = nn.Embedding(config.num_dynamic_feature_cls, config.query_feature_size)
        self.dynamic_feature_values_emebed = nn.Embedding(config.num_dynamic_feature_values, config.embed_size)
        # default including days = 3652
        
        self.feature_static_code = nn.Parameter(torch.zeros((config.query_feature_size,), dtype=torch.float), requires_grad=True)
        self.feature_dynamic_code = nn.Parameter(torch.zeros((config.query_feature_size,), dtype=torch.float), requires_grad=True)
        self.value_static_code = nn.Parameter(torch.zeros((config.embed_size,), dtype=torch.float), requires_grad=True)
        self.value_dynamic_code = nn.Parameter(torch.zeros((config.embed_size,), dtype=torch.float), requires_grad=True)

        for code in [self.feature_static_code, self.feature_dynamic_code, self.value_static_code, self.value_dynamic_code]:
            nn.init.normal_(code, 0., 0.01)

        # capacity cls for tanh
        nn.init.xavier_normal_(self.capacity_embed.weight)

        time_param_data = 0.0
        if config.time_decay_func == 'fit_linear':
            time_param_data = 1 / config.includeing_days
        elif config.time_decay_func == 'fit_exp':
            time_param_data = self.config.includeing_days / 4
        elif config.time_decay_func == 'fit_norm':
            time_param_data = self.config.includeing_days / 3
        self.time_param = nn.Parameter(torch.tensor( time_param_data , dtype=torch.float), requires_grad=True)

    def time_decay(self, feature_cls_time: Tensor) -> Tensor:
        r"""
        Args:
            feature_cls_time: shape(batch_size, num_features)
        Return:
            shape(batch_size, num_features)
        """
        if self.config.time_decay_func == 'exp':
            return torch.exp(-feature_cls_time  / self.config.e_axis)
        elif self.config.time_decay_func == 'norm':
            return torch.exp(- feature_cls_time ** 2 / (2 * self.config.sigma ** 2))
        elif self.config.time_decay_func == 'linear':
            return torch.max(1 - self.config.k * feature_cls_time, torch.zeros_like(feature_cls_time))
        elif self.config.time_decay_func == 'fit_exp':
            return torch.exp(- feature_cls_time  / self.time_param)
        elif self.config.time_decay_func == 'fit_linear':
            return torch.max(1 - self.time_param * feature_cls_time, torch.zeros_like(feature_cls_time))
        elif self.config.time_decay_func == 'fit_norm':
            return torch.exp(- feature_cls_time ** 2 / (2 * self.time_param ** 2))
        else:
            raise ValueError(f'not support time_decay_func {self.config.time_decay_func}')

    # noinspection PyMethodOverriding
    def forward(
        self,
        capacities: Tensor, 
        static_features: Tensor,
        dynamic_features: Tensor,
        dynamic_feature_values: Tensor,
        dynamic_feature_intervals: Tensor,
        static_attention_masks: Optional[Tensor] = None,
        dynamic_attention_masks: Optional[Tensor] = None,
        return_embeds: bool = False
    ) -> Tensor:
        r"""
        Args:
            capacities: input capacity cls as queries
            static_features: input static feature cls as keys and values
            dynamic_features: input dynamic feature cls as keys
            dynamic_feature_values: input dynamic feature values as values
            dynamic_feature_intervals: input dynamic feature intervals as time weights
            static_attention_masks: mask for static features
            dynamic_attention_masks: mask for dynamic features

        Shapes:
            capacities: shape(batch_size, num_capacities)(dtype:int)
            static_features: shape(batch_size, num_static_features)(dtype:int)
            dynamic_features: shape(batch_size, num_dynamic_features)(dtype:int)
            dynamic_feature_values: shape(batch_size, num_dynamic_features)(dtype:int)
            dynamic_feature_intervals: shape(batch_size, num_dynamic_features)(dtype:float),
            static_attention_masks: shape(batch_size, num_static_features)(dtype:bool)
            dynamic_attention_masks: shape(batch_size, num_dynamic_features)(dtype:bool)

        Return:
            Tensor, shape(batch_size, num_capacities, num_labels)
        """
        if static_attention_masks is None:
            static_attention_masks = torch.ones_like(static_features)
        if dynamic_attention_masks is None:
            dynamic_attention_masks = torch.ones_like(dynamic_features)
        input_queries = self.capacity_embed(capacities)

        static_time_weights = torch.ones_like(static_features)
        dynamic_time_weights = self.time_decay(dynamic_feature_intervals)
        time_weights = torch.concat([static_time_weights, dynamic_time_weights], dim=1)

        # combine two features into one
        static_key_input = self.static_feature_cls_embed(static_features)
        static_key_input = static_key_input + self.feature_static_code
        static_value_input = self.static_feature_values_embed(static_features)
        static_value_input = static_value_input + self.value_static_code
        dynamic_key_input = self.dynamic_feature_cls_embed(dynamic_features)
        dynamic_key_input = dynamic_key_input + self.feature_dynamic_code
        dynamic_value_input = self.dynamic_feature_values_emebed(dynamic_feature_values)
        dynamic_value_input = dynamic_value_input + self.value_dynamic_code

        input_keys = torch.cat([static_key_input, dynamic_key_input], dim=1)
        input_values = torch.cat([static_value_input, dynamic_value_input], dim=1)

        attention_masks = torch.cat([static_attention_masks, dynamic_attention_masks], dim=1)

        pred_scores = super().forward(
            input_queries,
            input_keys,
            input_values,
            attention_masks, 
            time_weights
        )  # shape (batch_size, num_capacities, num_labels)

        if not return_embeds:
            return pred_scores

        return pred_scores, input_queries, static_key_input, dynamic_key_input
        
    def train_model(
        self,
        dataset,
        step_call_back: Callable[[int, int, float], None],
        device: torch.device,
        num_workers: int = 1,
        ignore_capa_level_id: int = -1
    ) -> None:
        r"""
        Args:
            dataset: get item of (capacities, static_features, dynamic_features, target)
            step_call_back: (epoch, step, loss)->None
            device: device to train
            num_workers: num workers of dataloader
            ignore_capa_level_id: ignore capa level id
        """
        train_iter = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers
        )
        optimizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        total_training_step = self.config.num_epochs * len(dataset) 
        warm_up_steps = self.config.num_epochs * len(dataset) * self.config.warming_up_proportion
        scheduler = get_linear_schedule_with_warmup(optimizer, warm_up_steps, total_training_step)

        loss = nn.CrossEntropyLoss(ignore_index=ignore_capa_level_id)
        step = 0
        for epoch in range(1, self.config.num_epochs+1):
            self.train()
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
            ) in train_iter:

                (
                    capacity_cls,
                    static_feature_cls,
                    dynamic_feature_cls,
                    dynamic_feature_value,
                    dynamic_feature_time,
                    static_feature_attn_mask,
                    dynamic_feature_attn_mask,
                ) = (
                    capacity_cls.to(device),
                    static_feature_cls.to(device),
                    dynamic_feature_cls.to(device),
                    dynamic_feature_value.to(device),
                    dynamic_feature_time.to(device),
                    static_feature_attn_mask.to(device),
                    dynamic_feature_attn_mask.to(device),
                )

                output = self.forward(
                    capacity_cls,
                    static_feature_cls,
                    dynamic_feature_cls,
                    dynamic_feature_value,
                    dynamic_feature_time,
                    static_feature_attn_mask,
                    dynamic_feature_attn_mask,
                )
                capacity_levels.masked_fill_(capa_mask, ignore_capa_level_id)
                loss_value = loss.forward(output.permute(0, 2, 1), capacity_levels)
                optimizer.zero_grad()
                loss_value.backward()
                optimizer.step()
                scheduler.step()
                step += 1
                if step_call_back:
                    step_call_back(epoch, step, loss_value.item())


class ContrastiveCrossEntrophyRegularLoss(nn.CrossEntropyLoss):
    r"""
    带嵌入正则的batch内对比学习损失函数。
    """
    def __init__(self, alpha: float, beta: float, ignore_index: int = -1) -> None:
        super().__init__(ignore_index=ignore_index)
        self.alpha = alpha
        self.beta = beta
    
    def forward(
        self, 
        pred: Tensor, 
        target: Tensor, 
        reprs: Optional[list[Tensor]] = None, 
        embedes: Optional[list[Tensor]] = None
    ) -> Tensor:
        r"""
        Args:
            pred: 模型输出的预测对各个能力的结果。
            target: 人员画像的各个不同能力的实际能力等级。
            reprs: 模型特征变换层的输出。
            embeds: 模型嵌入层的输出。
        Shapes:
            pred: shape(batch_size, num_capacities, num_labels)
            target: shape(batch_size, num_capacities)
            reprs: list of shape(batch_size, num_cls, embed_size)
            embeds: list of shape(batch_size, num_cls, embed_size)
        """
        loss = super().forward(pred, target)

        if reprs is None:
            reprs = []
        if embedes is None:
            embedes = []

        # compute dot product regular loss
        for repr in reprs:
            num_cls = repr.shape[1]
            embedings_left = repr.unsqueeze(1).repeat(1, num_cls, 1, 1)
            embedings_right = repr.unsqueeze(2).repeat(1, 1, num_cls, 1)
            regular_loss1 = torch.linalg.vecdot(embedings_left, embedings_right).sum(-1).sum(-1).mean()
            loss = loss + self.alpha * regular_loss1
        # compute l2 norm regular loss
        for embed in embedes:
            num_cls = embed.shape[1]
            embedings_left = embed.unsqueeze(1).repeat(1, num_cls, 1, 1)
            embedings_right = embed.unsqueeze(2).repeat(1, 1, num_cls, 1)
            regular_loss2 = - torch.linalg.norm(embedings_left - embedings_right, dim=-1).mean()
            loss = loss + self.beta * regular_loss2

        return loss
