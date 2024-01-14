import os
from copy import copy
from pathlib import Path

import torch
import json5
import uvicorn

from kgdatabase.maintenance_personnel import MaintenanceWorker, Capacity, MaintenanceRecord
from smpkg.attention_evaluator_model import VanillaCapacityEvaluatorConfig
from smpkg.capacity_controller import CapacityGenerationController, CapacityGenerationSettings, ILogger
from smpkg.database_utils import build_database_dataset, connect_to_neo4j, collect_attrib_values_from_db


class Logger(ILogger):
    def __init__(self, max_epoch: int):
        self.clear(max_epoch)
    
    def clear(self, max_epoch: int):
        self.max_epoch = max_epoch
        self.current_epoch = 1
        self.current_step = 1
        self.total_loss = 0.

    def log_train(self, epoch: int, step: int, loss_value: float):
        if epoch > self.current_epoch:
            num_steps = step - self.current_step
            print("Epoch: %4d, Step: %4d, Loss: %f" % (self.current_epoch, step, self.total_loss / num_steps))
            self.current_epoch = epoch
            self.total_loss = loss_value
            self.current_step = step
        else:
            self.total_loss += loss_value


def main():

    connect_to_neo4j("10.181.8.236:7687", "neo4j", "neo4j_pass")

    # collect statics
    dynamic_cls = collect_attrib_values_from_db(MaintenanceRecord, 'malfunction')
    dynamic_values = collect_attrib_values_from_db(MaintenanceWorker, 'MaintenancePerformance', 'performance')
    capa_cls = collect_attrib_values_from_db(Capacity, 'name')
    capa_levels = collect_attrib_values_from_db(Capacity, 'CapacityRate', 'level')
    capa_levels = [ '无能力' ] + capa_levels
    # capa_levels = ['无能力', '初级', '中级', '高级', '专家', '资深']
    times = collect_attrib_values_from_db(MaintenanceRecord, 'complish_time')
    # maxtime for training and validation
    max_time = max(times)

    print("collect statics done")

    settings = CapacityGenerationSettings(
        capacity_names = capa_cls,  # 能力名称，无默认初始化空值
        capacity_levels = capa_levels,  # 能力可选等级，建议从低到高，最低为不包含此能力
        static_feature_classes = ['None'],  # 静态特征类型
        dynamic_feature_classes = dynamic_cls,  # 动态特征类型
        dynamic_feature_values = dynamic_values  # 动态特征的可能值
    )

    profiles = build_database_dataset(
        MaintenanceWorker,
        'CapacityRate',
        Capacity,
        [ ('MaintenancePerformance', False) ],
        [ 'complish_time'],
        ['malfunction'],
        ['performance'],
        'name',
        'level'
    )

    configs: list[VanillaCapacityEvaluatorConfig] = []
    config_dir = Path('configs')
    with open(config_dir.joinpath(f'model_config.json'), 'r', encoding='UTF-8') as fp:
        load_json = json5.load(fp)
        base_config = load_json['base_config']
        test_cases = load_json['test_cases']

        for test_case in test_cases:
            temp_config: dict = copy(base_config)
            temp_config.update(test_case)
            configs.append(VanillaCapacityEvaluatorConfig(**temp_config))

    logger = Logger(configs[0].num_epochs)
    controller = CapacityGenerationController(settings, configs[0], logger)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    best_f1 = 0.
    best_idx = 0
    for i, config in enumerate(configs):
        controller.init_model(config)
        controller.logger.clear(config.num_epochs)
        controller.train_model(profiles, max_time, device, 1)
        controller.save(path=f'models/capacity_evaluator{i}')
        report = controller.predicate(profiles, max_time, False, True, False)
        report['final_time_param'] = controller.model.time_param.item()
        f1_score = report['weighted avg']['f1-score']
        if f1_score > best_f1:
            best_idx = i
            best_f1 = f1_score
        with open(f'reports/report_case{i}.json', 'w', encoding='UTF-8') as fp:
            json5.dump(report, fp, indent=4, quote_keys=True, trailing_commas=False)

    print(f'best config for config{ best_idx } f1-score: {best_f1}')

    # random 10 samples to see prediction
    con = CapacityGenerationController.load(f'models/capacity_evaluator{best_idx}')
    import numpy as np
    indices = np.random.choice(len(profiles), 10)
    pred_profiles = [ profiles[i] for i in indices ]
    pre_capas = con.predicate(pred_profiles, max_time)
    print('pre_capas', *pre_capas, sep='\n')
    print('tru_capas', *[ profiles[i].capacities for i in indices ], sep='\n')
    # uvicorn.run("server:app", port=5200, log_level="info", host="0.0.0.0")
    return

if __name__ == "__main__":
    main()
