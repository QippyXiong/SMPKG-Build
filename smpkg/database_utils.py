import datetime
from typing import Union, Dict, List, Optional, overload, Any

import pandas as pd
from neo4j.exceptions import ServiceUnavailable
from neomodel import db, Relationship, StructuredNode, RelationshipManager, StructuredRel, config
from neomodel.exceptions import DeflateError

from smpkg.capacity_profile import ProfileCapacity, Profile, ProfileFeature
from smpkg.logger import logger


__all__ = [
	'connect_to_neo4j',
	'handle_time_key',
	'GetEntAttribute',
	'CreateEnt',
	'CreateRel',
	'DeleteEnt',
	'DeleteRel',
	'UpdateEnt',
	'UpdateRel',
	'load_excel_file_to_graph',
	'parse_record_to_dict',
	'getRelEnt',
	'EntityQueryByAtt',
	'RelQueryByEnt',
	'RelQueryByEntsAttr',
	'RelQueryByEnts',
	'get_time_key',
	'build_database_dataset',
	'collect_attrib_values_from_db'
]


def connect_to_neo4j(address: str, username: str, password: str):
	r"""
	address is like: 'localhost:7687'
	"""

	config.DATABASE_URL = f'bolt://{ username }:{ password }@{ address }'
	db.set_connection(f'bolt://{ username }:{ password }@{ address }')


def handle_time_key(ent_cls: type[StructuredNode], attr: Dict):
	r"""
	时间类型字段处理
	"""
	ent_class = ent_cls
	time_key = get_time_key(ent_class)
	# for k in attr.keys():
	# 	if k in time_key:
	# 		attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
	attr_keys = list(attr.keys())
	for k, t in time_key:
		if k not in attr_keys:  # @TODO: optimize
			continue
		if t == 'DateProperty':
			attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d")
		elif t == 'DateTimeFormatProperty':
			attr[k] = datetime.datetime.strptime(attr[k], "%Y-%m-%d %H:%M:%S")
	return attr


def GetEntAttribute(ent_cls: type[StructuredNode]) -> list[ tuple[str, str] ]:
	r"""
	Args:
		'ent_cls': 继承StructuredNode的结点类
	Returns:
		[[属性名，属性类型名]]
	"""
	ent_class = ent_cls
	atts = ent_class.__all_properties__
	ret = list()
	for att, ty in atts:
		ret.append((att, type(ty).__name__))
	return ret


def CreateEnt(cls: type[StructuredNode], attr: dict, major_keys: Optional[List[str]] = None):
	r"""
	根据类名和属性值创建实体

	Args:
		'class_name':str   # 类名
		'attr'		:dict  # 属性值
		'major_keys':list  # 主键
	Returns:
		new_ent: StructuredNode/None
		msg    : str
	"""
	if major_keys is None:
		major_keys = ['uid']
	try:
		ent_class = cls
		major_key = {mk: attr[mk] for mk in major_keys}
		major_key = handle_time_key(cls, major_key)
	except KeyError as e:
		msg = str(e) + "not exist"
		return None, msg
	only_ent = ent_class.nodes.filter(**major_key)
	if only_ent.__nonzero__():
		msg = cls.__name__ + str(major_key) + " is already exist"
		return None, msg
	else:
		attr = handle_time_key(cls, attr)
		props = [name for name, type_name in ent_class.__all_properties__]
		for p in attr:
			if p not in props:
				msg = cls.__name__ + "has no " + p + " property."
				return None, msg
		new_ent = ent_class(**attr)
		new_ent.save()
		msg = "create " + cls.__name__ + str(major_key) + " succeed"
		return new_ent, msg


def CreateRel(start_ent: StructuredNode, end_ent: StructuredNode, rel_cls: type[StructuredRel], attr: dict):
	r"""
	Returns:
		0: bool
		1: str, message
	"""
	rel: RelationshipManager = getattr(start_ent, rel_cls.__name__)
	rel_class = rel_cls()
	rel_props = rel_class.__properties__
	for p in attr.keys():
		if p not in rel_props.keys():
			return False, rel_cls.__name__ + "has no " + p + "property."
	rel.connect(end_ent, attr)
	return True, rel_cls.__name__ + str(attr) + " create successfully."


def DeleteEnt(ent_cls: type[StructuredNode], attr: dict):
	r"""
	Returns: True/False:bool, msg:str
	"""
	try:
		ent_class = ent_cls
	except KeyError as e:
		msg = str(e) + "not exist"
		return False, msg
	attr = handle_time_key(ent_cls, attr)
	del_ent = ent_class.nodes.filter(**attr)
	if del_ent.__nonzero__():
		nums = del_ent.__len__()
		for e in del_ent:
			e.delete()
			# s = class_name+str(parse_record_to_dict(e))
			# msg.append(s)
		return True, str(nums) + " " + ent_cls.__name__ + str(attr) + " is already deleted"
	else:
		msg = ent_cls.__name__ + str(attr) + " not exist"
		return False, msg


def DeleteRel(start_ent: StructuredNode, end_ent: StructuredNode, rel_name: str):
	rel: RelationshipManager = getattr(start_ent, rel_name)
	rel.disconnect(end_ent)
	return True, rel_name + " is already deleted"


def UpdateEnt(ent_cls: type[StructuredNode], attr: dict, new_attr: dict, major_keys: Optional[list[str]] = None):
	r"""
		eg: 匹配 attr:{uid:"3456",name:"张三"}，修改 new_attr{name:"李四"}
		修改主键： 判断修改后是否重复
	"""
	if major_keys is None:
		major_keys = ['uid']
	try:
		ent_class = ent_cls
	except KeyError as e:
		msg = str(e) + "not exist"
		return msg
	try:
		# 修改主键
		# ##修改后的实体已存在
		major_key = {mk: new_attr[mk] for mk in major_keys}
		major_key = handle_time_key(ent_cls, major_key)
		only_ent = ent_class.nodes.filter(**major_key)
		if only_ent.__nonzero__():
			msg = ent_cls.__name__ + str(major_key) + " is already exist"
			return msg
		else:
			# 进行修改
			attr = handle_time_key(ent_cls, attr)
			update_ent = ent_class.nodes.filter(**attr)
			nums = update_ent.__len__()
			if nums == 1:
				# 修改单节点
				# ##更新属性值
				update_ent_ = update_ent[0]
				new_attr = handle_time_key(ent_cls, new_attr)
				msg = "The primary key of single entity has been modified"
				for key, value in new_attr.items():
					if key in ent_class.__all_properties__:
						setattr(update_ent_, key, value)
					else:
						msg = ent_cls.__name__ + "has no " + key + "property."
				update_ent_.save()
				return msg
			elif nums > 1:
				# 不能同时修改多个节点的主键
				msg = "The primary key of multiple entities cannot be modified simaltaneously"
				return msg
			else:
				# 需要修改的节点不存在
				msg = "The single entity that needs to be modified, " + ent_cls.__name__ + str(attr) + ", does not exist"
				return msg
	except KeyError:
		# 修改非主键
		attr = handle_time_key(ent_cls, attr)
		new_attr = handle_time_key(ent_cls, new_attr)
		update_ent = ent_class.nodes.filter(**attr)
		nums = update_ent.__len__()
		if nums > 0:
			for e in update_ent:
				# ##更新属性值
				for key, value in new_attr.items():
					setattr(e, key, value)
				e.save()
		return str(nums) + " " + ent_cls.__name__ + str(attr) + " is already updated"


def UpdateRel(start_ent: StructuredNode, end_ent: StructuredNode, rel_cls: type[StructuredRel], attr: dict):
	r"""
	Returns: True/False:bool, msg:str
	"""
	rel: RelationshipManager = getattr(start_ent, rel_cls.__name__)
	rel_props = rel_cls().__properties__
	for p in attr.keys():
		if p not in rel_props:
			return False, rel_cls.__name__ + "has no " + p + "property."
	edge = rel.relationship(end_ent)
	for key, value in attr.items():
		setattr(edge, key, value)
	edge.save()
	return True, rel_cls.__name__ + str(attr) + " update successfully."


def load_excel_file_to_graph(file_path: str):
	try:
		db.cypher_query(
			r"""
			MATCH(n)
			DETACH DELETE n
			"""
		)  # 删掉原先图谱中的全部内容
	except ServiceUnavailable:
		logger.error("[Neomodel Error] 未能连接到neo4j服务器，请检查neo4j服务器是否开启")
		return
	from database.maintenance_personnel import MaintenanceWorker, MaintenanceRecord, Capacity

	mapping_worker = {
		'uid' 				: '工号/志愿者编号',
		'name'				: '姓名',
		'sex' 				: '性别',
		'nation'			: '民族',
		'phone'				: '联系方式',
		'birth'				: '出生日期',
		'live_in'			: '居住地址',
		'employ_date' 		: '入职时间',
		'work_post' 		: '岗位',
		'work_level'		: '岗位级别',
		'department' 		: '部门',
	}
	# mapping_worker = { mapping_worker[key]: key for key in mapping_worker }

	mapping_record = {
		# 'uid'				: '工号',
		'malfunction' 		: '故障类型',
		'place'				: '故障位置',
		'malfunc_time'		: '故障上报时间',
		'begin_time'		: '维修开始时间',
		'complish_time'		: '维修完成时间',
		'review'			: '定期检修记录',
	}

	mapping_capacity = {
		'name' 			: '维修能力名称',
		'description' 	: '描述',
		'rule'			: '规则',
	}

	# mapping_record = { mapping_record[key]: key for key in mapping_record }

	# 处理维保人员数据
	# query = r"""CREATE CONSTRAINT MaintenanceWork_unique_key
	# 		FOR(m: MaintenanceWorker) REQUIRE(m.uid) IS UNIQUE
	# """

	worker_data = pd.read_excel(file_path, sheet_name='维保人员')

	for row in worker_data.itertuples():
		data_dict = mapping_worker.copy()
		row_dict = { worker_data.keys()[i-1]: v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			MaintenanceWorker.nodes.get(uid=data_dict['uid'])
		except Exception:
			worker = MaintenanceWorker(**data_dict)
			worker.phone = str(worker.phone)
			worker.save()

	# 处理维修记录数据
	records = pd.read_excel(file_path, sheet_name='维修记录')

	for row in records.itertuples():
		data_dict = mapping_record.copy()
		row_dict = {records.keys()[i-1]: v for i, v in enumerate(row)}
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			# 查询维修记录是否已存在
			record = MaintenanceRecord.nodes.get(
				malfunction  = data_dict['malfunction'],
				place        = data_dict['place'],
				malfunc_time = data_dict['malfunc_time'],
			)

			# 查询维修记录是否未关联此条记录的维修人员
			record2worker = record.MaintenancePerformance.all()
			ids = [w.uid for w in record2worker]
			if row_dict['工号'] not in ids:
				rel = record.MaintenancePerformance.connect(MaintenanceWorker.nodes.get(uid=row_dict['工号']), {
					'malfunc_type': record.malfunction,  # 维修记录故障内容记录故障类型
					'performance': record.review  # 维修记录返修评价记录维修效果
				})
				rel.save()
		except Exception:
			record = MaintenanceRecord(**data_dict)
			record.save()
			rel = record.MaintenancePerformance.connect(
				MaintenanceWorker.nodes.get(uid=row_dict['工号']),
				{
					'malfunc_type': record.malfunction,  # 维修记录故障内容记录故障类型
					'performance': record.review  # 维修记录返修评价记录维修效果
				}
			)
			rel.save()

	# 处理维修能力数据
	Mcapacities = pd.read_excel(file_path, sheet_name='维修能力')

	for row in Mcapacities.itertuples():
		data_dict = mapping_capacity.copy()
		row_dict = { Mcapacities.keys()[i-1] : v for i, v in enumerate(row) }
		for key in data_dict:
			data_dict[key] = row_dict[data_dict[key]]
		try:
			capacity = Capacity.nodes.get(name=data_dict['name'])
		except Exception:
			capacity = Capacity(**data_dict)
			capacity.save()
		try:
			worker2capacity = capacity.CapacityRate.connect(
				MaintenanceWorker.nodes.get(uid=row_dict['维保人员工号']),
				{
					'level': row_dict['维修能力等级'],
				}
			)
			worker2capacity.save()
		except DeflateError as e:
			logger.error(e)


def parse_record_to_dict(record: Union[Relationship, StructuredNode, StructuredRel]) -> Dict:
	r"""
	Args:
		record: entity | relationship
	将特殊属性字段转换成字符串，如时间格式
	"""
	props: Dict = record.__properties__
	props.popitem()
	for k in props:
		props[k] = str(props[k])
	return props


def getRelEnt(entity_cls: type[StructuredNode]):
	r"""
	Args:
		entity_cls: str   # 类名
	Returns:
		ret: [[]]
		[关系名，尾实体类名]
	"""
	ret = []
	start_ent_class = entity_cls
	for rel_name, _ in start_ent_class.__all_relationships__:
		rel: RelationshipManager = getattr(start_ent_class, rel_name)
		ret.append([rel_name, getattr(rel, '_raw_class')])
	return ret


def EntityQueryByAtt(
		ent_type: type[StructuredNode],
		attr: dict
	) -> list[ StructuredNode ]:
	r"""
	通过实体属性查询实体并返回实体所有属性值
	"""
	ret_arr = []
	relations = getRelEnt(ent_type)
	# try:
	attr = handle_time_key(ent_type, attr)

	entities = ent_type.nodes.filter(**attr)
	for ent in entities:
		ent_dict = parse_record_to_dict(ent)
		record = {"element_id": ent.element_id, "properties": ent_dict, "relations": relations}
		ret_arr.append({"type": type(ent).__name__, "record": record})
	return ret_arr
	# except ValueError:
	# 	msg = "property key error"
	# 	return msg


def RelQueryByEnt(ent_type: type[StructuredNode], attr: dict, rel_type: Optional[str]):
	r"""
	From a certain type entity, search the related node by relation name

	Args:	'ent_type':str,
			'attr'	:dict
	Returns:
			List[]
	"""
	ret_arr = []
	try:
		attr = handle_time_key(ent_type, attr)

		entities = ent_type.nodes.filter(**attr)
		for ent in entities:
			if rel_type is None:
				for rel_name, _ in ent.__all_relationships__:
					rel: RelationshipManager = getattr(ent, rel_name)
					ret_arr.extend(RelQueryByRel(rel))
			else:
				try:
					rel: RelationshipManager = getattr(ent, rel_type)
					ret_arr.extend(RelQueryByRel(rel))
				except AttributeError:
					# 关系类型错误
					logger.error(f"关系类型错误: { rel_type }")
		return ret_arr
	except DeflateError as e:
		raise e


def RelQueryByRel(rel: RelationshipManager):
	ret_arr = []
	for node in rel.all():
		edge = rel.relationship(node)
		source = {"type": type(edge.start_node()).__name__, "element_id": edge._start_node_element_id}
		target = {"type": type(edge.end_node()).__name__, "element_id": edge._end_node_element_id}
		properties = parse_record_to_dict(edge)
		record1 = {"source": source, "target": target, "properties": properties}
		ret_arr.append({"type": type(edge).__name__, "record": record1})

		record2 = {"element_id": node.element_id, "record": parse_record_to_dict(node)}
		ret_arr.append({"type": type(node).__name__, "record": record2})
	return ret_arr


def RelQueryByEntsAttr(
	ent1_type: type[StructuredNode], attr1: dict,
	ent2_type: type[StructuredNode], attr2: dict,
	rel_type: str
):
	r"""
	由双端实体得到关系边
	Args:
	Returns:
	"""
	attr1 = handle_time_key(ent1_type, attr1)
	ent1 = ent1_type.nodes.get(**attr1)
	if ent1 is None:
		return "ent" + str(attr1) + "doesnot exist"
	attr2 = handle_time_key(ent1_type, attr2)
	ent2 = ent2_type.nodes.get(**attr2)
	if ent2 is None:
		return "ent" + str(attr2) + "doesnot exist"
	return RelQueryByEnts(ent1, ent2, rel_type)


def RelQueryByEnts(ent1: StructuredNode, ent2: StructuredNode, rel_type: str):
	rel: RelationshipManager = getattr(ent1, rel_type)
	edge = rel.relationship(ent2)
	source = {"type": type(ent1).__name__, "element_id": edge._start_node_element_id}
	target = {"type": type(ent2).__name__, "element_id": edge._end_node_element_id}
	properties = parse_record_to_dict(edge)
	record = {"source": source, "target": target, "properties": properties}
	return {"type": type(edge).__name__, "record": record}


def get_time_key(ent_class: Union[type[StructuredNode]]):
	r"""
	得到类的时间属性字段
	"""
	attributes = ent_class.__all_properties__
	time_att = []
	for att_name, att_value in attributes:
		if type(att_value).__name__ in ['DateProperty', 'DateTimeFormatProperty']:
			time_att.append((att_name, type(att_value).__name__))
	return time_att


def build_database_dataset(
		person_cls: type[StructuredNode],
		rel_name: str,
		capacity_cls: type[StructuredNode],
		feature_rel_list: list[tuple[str, bool]],
		feature_time_node_attrib_names: Optional[Union[str, list[str]]] = 'end_date',
		feature_cls_node_attrib_names: Optional[Union[str, list[str]]] = 'name',
		# feature_value_node_attrib_name: Optional[Union[str, list[str]]] = None,
		# feature_cls_rel_attrib_name: Optional[Union[str, list[str]]] = None,
		feature_value_rel_attrib_names: Optional[Union[str, list[str]]] = 'performance',
		capacity_cls_attrib_name: str = 'name',
		capacity_level_attrib_name: str = 'level'
	) -> list[Profile]:
	r"""
	Create profiles from neo4j database
	
	Args:
		person_cls: class of evaluate person
		rel_name: name of relationship between person and capacity
		capacity_cls: class of capacity
		feature_rel_list: list of tuple(relationship name, if_static_feature), True for static feature
		feature_time_node_attrib_names: list of attrib_name lists, attrib_name in attrib_list will be returned
		feature_cls_node_attrib_names: list of tuple(class of feature node, if_static_feature), True for static feature
		feature_value_rel_attrib_names: list of attrib_name lists, attrib_name in attrib_list will be returned
		capacity_cls_attrib_name: attrib_name of capacity class, default 'name'
		capacity_level_attrib_name: attrib_name of capacity level in relation, default 'level'
	"""
	if not issubclass(person_cls, StructuredNode):
		raise TypeError(f"node_cls must be a subclass of StructuredNode, but get { person_cls.__name__ }")

	# get all capacities
	capacities = capacity_cls.nodes.all()
	# filter persons with capacities
	persons_with_capas: list[person_cls] = []
	profile_capas: list[ProfileCapacity] = []

	# align attribute names
	if isinstance(feature_cls_node_attrib_names, str):
		feature_cls_node_attrib_names = [feature_cls_node_attrib_names] * len(feature_rel_list)
	if isinstance(feature_value_rel_attrib_names, str):
		feature_value_rel_attrib_names = [feature_value_rel_attrib_names] * len(feature_rel_list)
	if isinstance(feature_time_node_attrib_names, str):
		feature_time_node_attrib_names = [feature_time_node_attrib_names] * len(feature_rel_list)

	# build profile_capas
	for capacity in capacities:
		persons_with_certain_capa = getattr(capacity, rel_name).all()
		persons_with_capas.extend(persons_with_certain_capa)

	for person in persons_with_capas:
		person_capas = []
		for capacity in capacities:
			relationship = getattr(person, rel_name).relationship(capacity)
			if relationship:
				person_capas.append(ProfileCapacity(
					getattr(capacity, capacity_cls_attrib_name),
					getattr(relationship, capacity_level_attrib_name)
				))
		profile_capas.append(person_capas)

	profile_features: list[ tuple[list[ProfileFeature], list[ProfileFeature]] ] = []  # (static, dynamic)

	for person in persons_with_capas:
		for (
			(feature_rel_name, is_static),
			node_cls_attrib_name,
			value_rel_attrib_name,
			time_attrib_name
		) in zip(
			feature_rel_list,
			feature_cls_node_attrib_names,
			feature_value_rel_attrib_names,
			feature_time_node_attrib_names
		):

			rel: RelationshipManager = getattr(person, feature_rel_name)
			static_features = []
			dynamic_features = []

			for feature_node in rel.all():
				if is_static:
					static_features.append(ProfileFeature(
						getattr(feature_node, node_cls_attrib_name),
					))
				else:
					dynamic_features.append(ProfileFeature(
						getattr(feature_node, node_cls_attrib_name),
						getattr(rel.relationship(feature_node), value_rel_attrib_name),
						getattr(feature_node, time_attrib_name)
					))

			profile_features.append((static_features, dynamic_features))

	profiles = [
		Profile(
			static_features,
			dynamic_features,
			capacity_r
		)
		for capacity_r, (static_features, dynamic_features) in zip(profile_capas, profile_features)
	]
	return profiles


@overload
def collect_attrib_values_from_db(node: StructuredNode, attrib_name: str) -> list[Any]:
	...


@overload
def collect_attrib_values_from_db(rel: StructuredRel, rel_name: str, attrib_name: str) -> list[Any]:
	...


def collect_attrib_values_from_db(
	node: type[StructuredNode],
	attrib_name_or_rel: str,
	rel_attrib_name: Optional[str] = None
) -> list[Any]:
	# if hashable
	if rel_attrib_name is None:
		attrib_name = attrib_name_or_rel
		attrib_type = dict(node.__all_properties__)[attrib_name]
	else:
		attrib_name = rel_attrib_name
		attrib_type: type = type(getattr(getattr(node, attrib_name_or_rel).definition['model'], attrib_name))

	if hasattr(attrib_type, '__hash__'):
		if getattr(attrib_type, '__hash__') is not None:
			values_set = set()
			if rel_attrib_name is None:
				[ values_set.add(getattr(item, attrib_name)) for item in node.nodes.all() ]
				return list(values_set)
			else:
				[ 
					[ 
						[
							values_set.add(getattr(relation, attrib_name))
							for relation in getattr(n, attrib_name_or_rel).all_relationships(end_node)
						] for end_node in getattr(n, attrib_name_or_rel).all()
					] for n in node.nodes.all()
				]
				return list(values_set)

	# cannot hash
	from functools import reduce
	if rel_attrib_name is None:
		all_values = [ getattr(item, attrib_name) for item in node.nodes.all() ]
		values = reduce(lambda x, y: x + [y] if y not in x else x, [[], ] + all_values)
	else:
		all_values = []
		[
			[
				[
					all_values.append(getattr(relation, attrib_name))
					for relation in getattr(n, attrib_name_or_rel).all_relationships(end_node)
				] for end_node in getattr(n, attrib_name_or_rel).all()
			] for n in node.nodes.all()
		]
		values = reduce(lambda x, y: x + [y] if y not in x else x, [[], ] + all_values)
	return values
