import json5
from copy import copy
from pathlib import Path

import uvicorn

from database.maintenance_personnel import MaintenanceWorker, Capacity, MaintenanceRecord
from smpkg.attention_evaluator_model import VanillaCapacityEvaluatorConfig
from smpkg.capacity_controller import CapacityGenerationController, CapacityGenerationSettings
from smpkg.database_utils import build_database_dataset, connect_to_neo4j, collect_attrib_values_from_db


def main():
    config_dir = Path('configs')

    connect_to_neo4j("10.181.8.236:7687", "neo4j", "neo4j_pass")

    # collect statics
    dynamic_cls = collect_attrib_values_from_db(MaintenanceRecord, 'malfunction')
    dynamic_values = collect_attrib_values_from_db(MaintenanceWorker, 'MaintenancePerformance', 'performance')
    capa_cls = collect_attrib_values_from_db(Capacity, 'name')
    # capa_levels = collect_attrib_values_from_db(Capacity, 'CapacityRate', 'level')
    # capa_levels = [ '无能力' ] + capa_levels
    capa_levels = ['无能力', '初级', '中级', '高级', '专家', '资深']
    times = collect_attrib_values_from_db(MaintenanceRecord, 'complish_time')
    # maxtime for training and validation
    max_time = max(times)

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

    best_f1 = 0.
    best_idx = 0
    configs = []
    for i in range(9):
        with open(config_dir.joinpath(f'default_cpagen_params{i}.json'), 'r', encoding='UTF-8') as fp:
            config_dict = json5.load(fp)
            configs.append(VanillaCapacityEvaluatorConfig(**config_dict))

    controller = CapacityGenerationController(settings, configs[0])
    for i, config in enumerate(configs[8:]):
        controller.init_model(config)
        controller.train_model(profiles, max_time, 'cuda:0', 1)
        controller.save(path=f'models/capacity_evaluator_config{i}')
        report = controller.predicate(profiles, max_time, True, False)
        report['final_time_param'] = controller.model.time_param.item()
        f1_score = report['weighted avg']['f1-score']
        if f1_score > best_f1:
            best_idx = i
            best_f1 = f1_score
        with open(f'reports/report{i}.json', 'w', encoding='UTF-8') as fp:
            json5.dump(report, fp, indent=4, quote_keys=True, trailing_commas=False)
        print(f'best config for config{ best_idx } f1-score: {best_f1}')
    return
    # uvicorn.run("server:app", port=5200, log_level="info", host="0.0.0.0")


if __name__ == '__main__':
    main()
