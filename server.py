from typing import Union, Optional
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel
from functools import reduce
from neomodel import db, StructuredNode, RelationshipManager, StructuredRel
from dataclasses import dataclass
import sys

import torch

from kgdatabase.maintenance_personnel import MaintenanceWorker, Capacity, MaintenanceRecord, MaintenancePerformance, \
    CapacityRate
from smpkg.llm import LocalLLMController
from smpkg.capacity_controller import CapacityGenerationController
from smpkg.database_utils import (EntityQueryByAtt, RelQueryByEnt, getRelEnt, CreateEnt,
                                  handle_time_key, CreateRel, UpdateEnt, UpdateRel, DeleteEnt, DeleteRel,
                                  RelQueryByEntsAttr, get_persons_profiles, parse_record_to_dict)
from smpkg.prompt import InformationExtractionArrayPrompt, PromptDateFormatConstraint, PromptSelectedValueConstraint


app = FastAPI()
local_llm_controller: Optional[LocalLLMController] = None
local_capa_controller: Optional[CapacityGenerationController] = None


def init_local_llm_controller(model_dir: str, device: torch.device) -> LocalLLMController:
    global local_llm_controller
    local_llm_controller = LocalLLMController(model_dir, device)
    return local_llm_controller


def init_local_capa_controller(controller: CapacityGenerationController):
    global local_capa_controller
    local_capa_controller = controller


database_module_name = 'database'
database_modules = [sys.modules[key] for key in sys.modules if
                    key.startswith(database_module_name + '.') and key != database_module_name]
database_dics = [{k: v for k, v in module.__dict__.items() if not k.startswith('_')} for module in database_modules]

kg_mapping: dict[str, type[StructuredNode]] = dict()
[kg_mapping.update(
    {key: value for key, value in dic.items() if issubclass(value, StructuredNode) and value != StructuredNode})
 for dic in database_dics]

kg_rel_mapping: dict[str, type[StructuredRel]] = dict()
[kg_rel_mapping.update(
    {key: value for key, value in dic.items() if issubclass(value, StructuredRel) and value != StructuredRel})
 for dic in database_dics]


@app.get("/")
def read_root():
    return "Hello"


@app.get("/items/{item_id}")
def read_item(item_id: str, q: Union[str, None] = None):
    print('fuck you')
    return {"item_id": item_id, "q": q}


class SearchData(BaseModel):
    properties: dict
    relation: str


@app.post("/search/entity/{ent_type}")
def read_entity(ent_type: str, data: SearchData):
    # return {"None":None, "[]":[]}
    # return {'ok': True, 'msg': 'success', 'data': ent_type}
    # print(ent_type)
    if len(data.properties) == 0:
        return {'ok': False, 'msg': 'properties is null', 'data': None}
    try:
        if data.relation == "None":
            ret_arr = EntityQueryByAtt(kg_mapping[ent_type], attr=data.properties)
            if isinstance(ret_arr, str):
                return {'ok': False, 'msg': ret_arr, 'data': None}
            if any(ret_arr):
                return {'ok': True, 'msg': 'success', 'data': ret_arr}
            else:
                return {'ok': False, 'msg': f'{ent_type} not exsists', 'data': None}
        if data.relation == "All":
            ret_arr = RelQueryByEnt(kg_mapping[ent_type], attr=data.properties, rel_type=None)
        else:
            ret_arr = RelQueryByEnt(kg_mapping[ent_type], attr=data.properties, rel_type=data.relation)
        if isinstance(ret_arr, str):
            return {'ok': False, 'msg': ret_arr, 'data': None}
        if any(ret_arr):
            return {'ok': True, 'msg': 'success', 'data': ret_arr}
        else:
            return {'ok': False,
                    'msg': data.relation + " not exsists" if data.relation else f"this {ent_type} has no relation",
                    'data': None}
    except Exception as e:
        return {'ok': False, "msg": str(e), 'data': None}


class CreateData(BaseModel):
    @dataclass
    class Relation:
        link_node_type: str
        link_node_properties: dict
        edge_type: str
        edge_properties: dict

    node_properties: dict
    relation: Union[None, list[Relation]]


@app.post("/create/entity&rel/{ent_type}")
def create_entity_rel(ent_type: str, data: CreateData):
    _, msg = CreateEnt(kg_mapping[ent_type], attr=data.node_properties)
    if _ is None:
        return {'ok': False, 'msg': msg, 'data': None}
    else:
        if data.relation == "None":
            return {'ok': True, 'msg': msg, 'data': None}
        elif isinstance(data.relation, list):
            flag = True
            for rela in data.relation:
                link_node_type = rela.link_node_type
                link_node_properties = rela.link_node_properties
                edge_type = rela.edge_type
                edge_properties = rela.edge_properties
                if link_node_type not in kg_mapping:
                    msg = msg + "\n" + link_node_type + "does not exist"
                    return {'ok': False, 'msg': msg, 'data': None}
                if edge_type not in kg_mapping:
                    msg = msg + "\n" + edge_type + "does not exist"
                    return {'ok': False, 'msg': msg, 'data': None}
                attr = handle_time_key(kg_mapping[link_node_type], link_node_properties)
                link_entities = kg_mapping[link_node_type].nodes.filter(**attr)
                for e in link_entities:
                    f, m = CreateRel(_, e, kg_rel_mapping[edge_type], edge_properties)
                    msg = msg + m
                    flag = flag & f
            return {'ok': flag, 'msg': msg, 'data': None}


class Rel_Data(BaseModel):
    start_node_type: str
    end_node_type: str
    start_node_properties: dict
    end_node_properties: dict
    edge_properties: dict


@app.post("/create/relationship/{rel_type}")
def create_rel(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(kg_mapping[data.start_node_type], data.start_node_properties)
    end_attr = handle_time_key(kg_mapping[data.start_node_type], data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for s in start_entities:
        for e in end_entities:
            f, m = CreateRel(s, e, kg_rel_mapping[rel_type], data.edge_properties)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}


class Update_Ent_Data(BaseModel):
    properties: dict
    new_properties: dict


@app.post("/update/entity/{ent_type}")
def update_entity(ent_type: str, data: Update_Ent_Data):
    _, msg = UpdateEnt(kg_mapping[ent_type], data.properties, data.new_properties)
    if _ is None:
        return {'ok': False, 'msg': msg, 'data': None}
    else:
        return {'ok': True, 'msg': msg, 'data': None}


@app.post("/update/relation/{rel_type}")
def update_rel(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(kg_mapping[data.start_node_type], data.start_node_properties)
    end_attr = handle_time_key(kg_mapping[data.end_node_type], data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for _ in start_entities:
        for _ in end_entities:
            f, m = UpdateRel(start_entities, end_entities, kg_rel_mapping[rel_type], data.edge_properties)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}


@app.post("/delete/relation/{rel_type}")
def delete_entity(rel_type: str, data: Rel_Data):
    if data.start_node_type not in kg_mapping:
        msg = data.start_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if data.end_node_type not in kg_mapping:
        msg = data.end_node_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    if rel_type not in kg_mapping:
        msg = rel_type + "does not exist"
        return {'ok': False, 'msg': msg, 'data': None}
    start_attr = handle_time_key(kg_mapping[data.start_node_type], data.start_node_properties)
    end_attr = handle_time_key(kg_mapping[data.start_node_type], data.end_node_properties)
    start_entities = kg_mapping[data.start_node_type].nodes.filter(**start_attr)
    end_entities = kg_mapping[data.end_node_type].nodes.filter(**end_attr)
    flag = True
    msg = ""
    for _ in start_entities:
        for _ in end_entities:
            f, m = DeleteRel(start_entities, end_entities, rel_type)
            msg = msg + m
            flag = flag & f
    return {'ok': flag, 'msg': msg, 'data': None}


@app.post("/delete/entity/{ent_type}")
def delete_rel(ent_type: str, properties: dict):
    _, msg = DeleteEnt(kg_mapping[ent_type], properties)
    if _:
        return {'ok': True, 'msg': msg, 'data': None}
    else:
        return {'ok': False, 'msg': msg, 'data': None}


class MaintenaceWorkerSearchData(BaseModel):
    key: str
    data: str


@app.post("/search/maintenance_worker/person")
def read_worker(data: MaintenaceWorkerSearchData):
    key = data.key
    data = data.data
    # @TODO: 编写错误处理代码
    check = {key: data}
    ret_arr = []
    try:
        # 人员属性字段查询
        persons: list[MaintenanceWorker] = MaintenanceWorker.nodes.filter(**check)
        # 处理时间字段
        for person in persons:
            person_dict = dict()
            for key, _ in person.__all_properties__:
                person_dict[key] = str(getattr(person, key))
            ret_arr.append({"type": type(person).__name__, "record": person_dict})

        # 查询关联信息
        cap_name = []
        rec_infos = []
        for person in persons:
            capacities = person.CapacityRate.all()
            for cap in capacities:
                if cap.name not in cap_name:
                    cap_dict = dict()
                    for key, _ in cap.__all_properties__:
                        cap_dict[key] = str(getattr(cap, key))
                    ret_arr.append({"type": "Capacity", "record": cap_dict})
                    cap_name.append(cap.name)
                    # print(ret)
                query = f"""MATCH (p:MaintenanceWorker{{uid : '{person.uid}'}})<-[r:RATE] \
                     -(c:Capacity{{name : '{cap.name}'}}) RETURN r"""
                r, _ = db.cypher_query(query)
                r = r[0][0]
                source = {"type": type(person).__name__, "majorkey": {"uid": person.uid}}
                target = {"type": type(cap).__name__, "majorkey": {"name": cap.name}}
                rel = {"type": r.type, "record": {"source": source, "target": target, "properties": r._properties}}

                ret_arr.append(rel)

            maintenancerecords = person.MaintenancePerformance.all()
            for rec in maintenancerecords:
                # print("rec", rec)
                rec_info: dict = {"malfunction": rec.malfunction, "place": rec.place, "malfunc_time": rec.malfunc_time}
                if rec_info not in rec_infos:
                    rec_dict = dict()
                    for key, _ in rec.__all_properties__:
                        rec_dict[key] = str(getattr(rec, key))
                    ret_arr.append({"type": MaintenanceRecord.__name__, "record": rec_dict})
                    rec_infos.append(rec_info)

                query = f"""MATCH (p:MaintenanceWorker{{uid: '{person.uid}'}})<-[r:PERFORMED]- \
                        (re:MaintenanceRecord{{malfunction: '{rec_info["malfunction"]}',          \
                        place: '{rec_info["place"]}',malfunc_time:'{rec_info["malfunc_time"]}'}}) \
                        RETURN r"""
                r, _ = db.cypher_query(query)
                r = r[0][0]
                source = {"type": type(person).__name__, "majorkey": {"uid": person.uid}}
                for key in rec_info.keys():
                    rec_info[key] = str(rec_info.get(key))
                target = {"type": type(rec).__name__, "majorkey": rec_info}
                rel = {"type": type(r).__name__,
                       "record": {"source": source, "target": target, "properties": r._properties}}
                ret_arr.append(rel)
        if any(ret_arr):
            return {'ok': True, 'msg': 'success', 'data': ret_arr}
        else:
            return {'ok': False, 'msg': 'person not exsists', 'data': None}

    except ValueError:
        return {'ok': False, 'msg': 'query key error', 'data': None}


@app.post("/search/maintenance_worker/capacity")
def read_worker_capacity(data: MaintenaceWorkerSearchData):
    key = data.key
    data = data.data
    # @TODO: 编写错误处理代码
    # check = {key: data}
    ret_arr = []
    try:
        if key == "capacity":
            try:
                # cap = Capacity.nodes.get(name=data)
                cap: Capacity = Capacity.nodes.filter(name=data)
            except Exception:
                return {'ok': False, 'msg': 'capacity not exsists', 'data': None}
            cap_dict = dict()
            for key, _ in cap.__all_properties__:
                cap_dict[key] = str(getattr(cap, key))
            ret_arr.append({"type": "Capacity", "record": cap_dict})

            persons = cap.rate.all()
            # 处理时间字段
            for person in persons:
                person_dict = dict()
                for key, _ in person.__all_properties__:
                    person_dict[key] = str(getattr(person, key))
                ret_arr.append({"type": type(person).__name__, "record": person_dict})
            return {'ok': True, 'msg': 'success', 'data': ret_arr}
    except ValueError:
        return {'ok': False, 'msg': 'query key error', 'data': None}


@app.get('/search/relations/{ent_type}')
def read_relations(ent_type: str):
    try:
        ret_val = getRelEnt(kg_mapping[ent_type])
    except Exception as e:
        return {'ok': False, 'msg': str(e), 'data': None}

    return {'ok': True, 'msg': 'success', 'data': ret_val}


@app.post('/search/all')
def send_all_entity_and_relations():
    ret_arr = []
    for ent_type in kg_mapping.keys():
        if issubclass(kg_mapping[ent_type], StructuredNode):
            arr = RelQueryByEnt(kg_mapping[ent_type], attr={}, rel_type=None)
            ret_arr += arr
    ret_arr = reduce(lambda x, y: x + [y] if y not in x else x, [[], ] + ret_arr)
    return {'ok': True, 'msg': 'success', 'data': ret_arr}


r"""
end post & get
----
web socket deals
"""


@dataclass
class SocketRequest:
    type: str
    data: str


@app.websocket("/search/wholeKG")
async def sendShowData(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_json()
        print(data)
        await websocket.send_json({})


malfuncs = [
    "轨道损坏"        ,
    "轮胎车轴故障"    ,
    "车门故障"        ,
    "照明损坏"        ,
    "空调故障"        ,
    "制动系统故障"    ,
    "排水沟损坏"      ,
    "排水系统堵塞"    ,
    "通风系统堵塞"    ,
    "烟雾报警器故障"  ,
    "紧急停车系统故障",
    "自动售票机故障"  , 
    "安检设备故障"    ,
    "闸机故障"        , 
    "电梯故障"        ,
    "扶梯故障"        ,
    "电力系统故障"    ,
    "地铁信号故障"    ,
    "监视系统故障"    ,
]

similary_malfuncs = {
    "轨道损坏"        :     "轨道损坏"        ,
    "轮胎车轴故障"    :     "轮胎车轴故障"    ,
    "车门故障"        :     "车门故障"        ,
    "照明损坏"        :     "照明损坏"        ,
    "空调故障"        :     "空调故障"        ,
    "制动系统故障"    :     "制动系统故障"    ,
    "排水沟损坏"      :     "排水沟损坏"      ,
    "排水系统堵塞"    :     "排水系统堵塞"    ,
    "通风系统堵塞"    :     "通风系统堵塞"    ,
    "烟雾报警器故障"  :     "烟雾报警器故障"  ,
    "紧急停车系统故障":     "紧急停车系统故障",
    "自动售票机故障"  :     "自动售票机故障"  , 
    "安检设备故障"    :     "安检设备故障"    ,
    "闸机故障"        :     "闸机故障"        , 
    "电梯故障"        :     "电梯故障"        ,
    "扶梯故障"        :     "扶梯故障"        ,
    "电力系统故障"    :     "电力系统故障"    ,
    "地铁信号故障"    :     "地铁信号故障"    ,
    "监视系统故障"    :     "监视系统故障"    ,
    "铁轨损坏": "轨道损坏",
    "铁轨破损": "轨道损坏",
    "车轮损坏": "轮胎车轴故障",
    "车辆车轮损坏": "轮胎车轴故障",
    "车辆车轮故障": "轮胎车轴故障",
    "车窗故障" : "车门故障"
}

example_outputs = [{
    "姓名": "王卫国",
    "岗位": "车辆维修技术员",
    "故障类型": "空调故障",
    "维修内容": "空调维修",
    "地点": "小寨站",
    "开始时间": "2021-05-21 08:00:00",
    "结束时间": "2021-05-21 10:00:00",
    "持续时间": "2小时"
}, {
    "姓名": "张三",
    "岗位": None,
    "故障类型": "空调故障",
    "维修内容": "辅助工作",
    "地点": "小寨站",
    "开始时间": "2021-05-21 08:00:00",
    "结束时间": "2021-05-21 10:00:00",
    "持续时间": "2小时"
}]

item_keys = {
    "姓名": "维修人员名称",
    "岗位": "人员的岗位",
    "故障类型": "维修故障",
    "维修内容": "维修内容",
    "地点": "维修所在地点",
    "开始时间": "维修开始时间",
    "结束时间": "维修结束时间",
    "持续时间": "维修持续时间"
}

prompt = InformationExtractionArrayPrompt(
    description='从地铁人员维修记录本文中提取出这个记录中包含的人物、他们的岗位、他们完成的维修内容、此次维修所在的地点、维修开始时间、维修结束时间、维修持续时间。',
    info_names=['姓名', '岗位', '故障类型', '维修内容', '地点', '开始时间', '结束时间', '持续时间'],
    item_keys=item_keys,
    constraints=[
        # PromptTypeConstraint('person', str),
        # PromptTypeConstraint('station', str),
        # PromptTypeConstraint('malfunc', str),
        # PromptTypeConstraint('content', str),
        # PromptTypeConstraint('place', str),
        PromptSelectedValueConstraint("故障类型", malfuncs),
        PromptDateFormatConstraint('%Y-%m-%d %H:%M:%S', 'begin_time'),
        PromptDateFormatConstraint('%Y-%m-%d %H:%M:%S', 'end_time'),
        # PromptTypeConstraint('duration', str),
    ],
    example_input="2021.5.21,早八点龙凤溪地铁站空调故障，车辆维修技术员王卫国前往维修，于早十点维修完成，张三参与了辅助工作。",
    example_outputs=example_outputs,
)


def GenerateMulRecordByRecord(record: dict) -> tuple[bool, str]:
    record_type = None
    if not isinstance(record, dict):
        return False, f'parse fail', None
    if "故障类型" not in record.keys():
        return False, f'parse fail', None
    if isinstance(record["故障类型"], str) and '，' in record["故障类型"]:
        record["故障类型"] = record["故障类型"].split('，')
    if isinstance(record["故障类型"], str) and ',' in record["故障类型"]:
        record["故障类型"] = record["故障类型"].split(',')
    if isinstance(record["故障类型"], list):
        for rec_t in record["故障类型"]:
            record_type = similary_malfuncs.get(rec_t, None)
            if record_type is not None:
                break
    else:
        record_type = similary_malfuncs.get(record["故障类型"], None)
    if not record_type:
        return False, f'记录中无故障类型{record["故障类型"]}', None
    if record_type not in malfuncs:
        return False, f'维修故障{record["故障类型"]}不存在', None
    if "姓名" not in record.keys():
        return False, "维修人员字段缺失", None
    person = MaintenanceWorker.nodes.filter(name=record["姓名"])
    if person.__len__() == 0:
        return False, f'维修人员{record["姓名"]}不存在'
    if person.__len__() > 1:
        return False, "维修人员不唯一", None
    attr = {"malfunction": record_type, "place": record["地点"],
            "malfunc_time": record["开始时间"], "begin_time": record["开始时间"],
            "complish_time": record["结束时间"]}
    malrecord, msg = CreateEnt(MaintenanceRecord, attr, major_keys=["malfunction", "place", "malfunc_time"])
    if malrecord is None:
        return False, f"维修记录创建失败，原因：{msg}", None
    rel: RelationshipManager = getattr(person[0], 'MaintenancePerformance')
    if_exist_edge = rel.relationship(malrecord)
    if if_exist_edge is None:
        CreateRel(person[0], malrecord, MaintenancePerformance, attr={"performance": "正常"})
    return True, "维修记录更新成功", (person[0], malrecord)


class ExtractData(BaseModel):
    text: str


@app.post('/llm/local/')
async def local_LLM_info_extraction(data: ExtractData):
    r"""
    webserver post api for calling LLM for extracting maintainance record

    Args:
        websocket: websocket
    """
    global prompt
    global local_llm_controller
    global local_capa_controller

    if not local_llm_controller:
        return {'ok': False, 'msg': 'local llm controller not init yet!', 'data': None}
    if not local_capa_controller:
        return {'ok': False, 'msg': 'local capa controller not init yet!', 'data': None}

    sentence = data.text
    # 需要很长的时间，可能开子线程处理，直接返回，后续更新完再查看会更好
    infos, _ = local_llm_controller.info_extraction(prompt, sentence)
    fail_msgs = []
    try:
        res = list()
        if not isinstance(infos, list):
            return {'ok': False, 'msg': 'parse fail', 'data': infos}
        for info in infos:
            ok, msg, ents = GenerateMulRecordByRecord(info)
            if not ok:
                fail_msgs.append(msg)
                continue
            elif ents is None:
                continue
            else:
                # attr1 = {"name": info["姓名"]}
                # attr2 = {"malfunction": info["故障类型"],
                #          "place": info["地点"],
                #          "malfunc_time": info["开始时间"]}
                # rec_ent = EntityQueryByAtt(ent_type=MaintenanceRecord, attr=attr2)[0]
                # per_ent = EntityQueryByAtt(ent_type=MaintenanceWorker, attr=attr1)[0]
                per_ent, rec_ent = ents
                attr1 = {"name": per_ent.name}
                attr2 = {"malfunction": rec_ent.malfunction, "place": rec_ent.place, "malfunc_time": rec_ent.malfunc_time }
                profiles = get_persons_profiles(
                    MaintenanceWorker,
                    attr1,
                    [ ('MaintenancePerformance', False) ],
                    [ 'complish_time'],
                    ['malfunction'],
                    ['performance'],
                )
                person_capas = local_capa_controller.predicate(profiles, local_capa_controller.get_rel_time(), True)
                person_capa = person_capas[0]
                for capa in person_capa:
                    capa_ent = Capacity.nodes.get(name=capa.capacity_name)
                    # capa should exist
                    CreateRel(per_ent, capa_ent, CapacityRate, attr={"level": capa.capacity_level})

                rel = RelQueryByEntsAttr(attr1=attr1, attr2=attr2,
                                         ent1_type=MaintenanceWorker, ent2_type=MaintenanceRecord,
                                         rel_type='MaintenancePerformance')
                
                res.append({ "type": "MaintenanceRecord", "record": {"element_id": rec_ent.element_id, "record": parse_record_to_dict(rec_ent)} })
                res.append({ "type": "MaintenanceWorker", "record": {"element_id": per_ent.element_id, "record": parse_record_to_dict(per_ent)} })
                res.append(rel)
        res = reduce(lambda x, y: x + [y] if y not in x else x, [[], ] + res)
        if len(fail_msgs) == len(infos):  # all fail
            return {'ok': False, 'msg': str(fail_msgs), 'data': infos }
        return {'ok': True, 'msg': 'update person record and capacities', 'data': res }
    except ValueError as e:
        return {'ok': False, 'msg': str(e), 'data': infos}
