import os
from copy import copy
from pathlib import Path

import torch
import json5
import uvicorn

from kgdatabase.maintenance_personnel import MaintenanceWorker, Capacity, MaintenanceRecord
from smpkg.attention_evaluator_model import VanillaCapacityEvaluatorConfig
from smpkg.capacity_controller import CapacityGenerationController, CapacityGenerationSettings, ILogger
from smpkg.database_utils import load_excel_file_to_graph, build_database_dataset, connect_to_neo4j, collect_attrib_values_from_db
from server import init_local_capa_controller, init_local_llm_controller
from smpkg.prompt import extract_json

class Logger(ILogger):
    def __init__(self, max_epoch: int):
        self.clear(max_epoch)
    
    def clear(self, max_epoch: int):
        self.max_epoch = max_epoch
        self.current_epoch = 1
        self.current_step = 1
        self.total_loss = 0.

    def log_train(self, epoch: int, step: int, loss_value: float):
        r""" 打印训练结果，每个epoch打印一次平均loss """
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
    load_excel_file_to_graph("data/维保人员数据.xlsx")  # load data to neo4j

    # collect statics
    dynamic_cls = collect_attrib_values_from_db(MaintenanceRecord, 'malfunction')
    dynamic_values = collect_attrib_values_from_db(MaintenanceWorker, 'MaintenancePerformance', 'performance')
    capa_cls = collect_attrib_values_from_db(Capacity, 'name')
    capa_levels = collect_attrib_values_from_db(Capacity, 'CapacityRate', 'level')
    # 数据库中不存放无能力关系，这里手动添加上
    capa_levels = [ '无能力' ] + capa_levels
    # 收集结果为 capa_levels = ['无能力', '初级', '中级', '高级', '专家', '资深']，次序可能变化
    times = collect_attrib_values_from_db(MaintenanceRecord, 'complish_time')
    # 取收集结果中时间最大的项作为训练用的相对时间
    max_time = max(times)

    print("collect statics done")

    settings = CapacityGenerationSettings(
        capacity_names = capa_cls,  # 能力名称，无默认初始化空值
        capacity_levels = capa_levels,  # 能力可选等级，建议从低到高，最低为不包含此能力
        static_feature_classes = ['None'],  # 静态特征类型，这里没有用到
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

    # 加载configs中的model_config中包含的各个配置，从这些配置中挑选最好的模型
    # 这会需要很长时间，可以跳过此步（76-112），直接加载我们已经训练好的模型 models/capacity_evaluator0
    # configs: list[VanillaCapacityEvaluatorConfig] = []
    # config_dir = Path('configs')
    # with open(config_dir.joinpath(f'model_config.json'), 'r', encoding='UTF-8') as fp:
    #     load_json = json5.load(fp)
    #     base_config = load_json['base_config']
    #     test_cases = load_json['test_cases']

    #     for test_case in test_cases:
    #         temp_config: dict = copy(base_config)
    #         temp_config.update(test_case)
    #         configs.append(VanillaCapacityEvaluatorConfig(**temp_config))

    # logger = Logger(configs[0].num_epochs)
    # controller = CapacityGenerationController(settings, configs[0], logger)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # os.makedirs('models', exist_ok=True)
    # os.makedirs('reports', exist_ok=True)
    
    # best_f1 = 0.
    # best_idx = 0
    # for i, config in enumerate(configs[:1]):
    #     controller.init_model(config)
    #     controller.logger.clear(config.num_epochs)
    #     controller.train_model(profiles, max_time, device, 1)
    #     controller.save(path=f'models/capacity_evaluator{i}')  # 保存模型到指定路径
    #     report = controller.predicate(profiles, max_time, False, True, False)  # 返回模型验证报告
    #     report['final_time_param'] = controller.model.time_param.item()  # 记录模型最终的时间参数
    #     f1_score = report['weighted avg']['f1-score']  # 取f1分数作为考量模型效果的指标
    #     if f1_score > best_f1:
    #         best_idx = i
    #         best_f1 = f1_score
    #     with open(f'reports/report_case{i}.json', 'w', encoding='UTF-8') as fp:
    #         json5.dump(report, fp, indent=4, quote_keys=True, trailing_commas=False)

    # print(f'best config for config{ best_idx } f1-score: {best_f1}')

    # 加载模型，并随机采样10个样本查看实际预测的效果，这里加载的是我们训练好的模型 models/capacity_evaluator0
    con = CapacityGenerationController.load(f'models/capacity_evaluator0')
    import numpy as np
    indices = np.random.choice(len(profiles), 10)
    pred_profiles = [ profiles[i] for i in indices ]  # 随机采样10个样本
    pre_capas = con.predicate(pred_profiles, max_time)
    # 查看预测结果与真实结果是否有误
    print('pre_capas', *pre_capas, sep='\n')
    print('tru_capas', *[ profiles[i].capacities for i in indices ], sep='\n')

    # 初始化两个控制器，分别用于信息处理和能力生成，用于webserver
    # 加载信息处理模型需要 14~22 GB GPU memory 运行功能，注意选择合适的device
    llm_con = init_local_llm_controller("models/Qwen-14B-Chat-Int4", device=torch.device('cuda:1'))
    llm_con.load()
    # 加载能力生成即当前模型， 少于 4 GB GPU memory
    con.move_model(torch.device('cuda:0'))
    init_local_capa_controller(con)
    # 启动webserver，可以查看server.py源代码或者访问 'http://localhost:5200/docs' 来了解
    uvicorn.run("server:app", port=5200, log_level="info", host="0.0.0.0")
    return


if __name__ == "__main__":
    main()
