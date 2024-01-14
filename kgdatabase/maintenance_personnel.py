from neomodel import (
    StructuredNode, StringProperty, StructuredRel,
    UniqueIdProperty, RelationshipFrom,RelationshipTo,Relationship,
    DateProperty, BooleanProperty,DateTimeFormatProperty
	)

class MaintenancePerformance(StructuredRel):
	r"""
	维修记录表现边
	"""
	performance		= StringProperty()	# 维修效果


class MaintenanceRecord(StructuredNode):
	r"""
	维修记录实体
	"""
	malfunction 	= StringProperty()									# 故障内容
	place			= StringProperty()									# 故障位置
	malfunc_time	= DateTimeFormatProperty \
			(format="%Y-%m-%d %H:%M:%S")								# 故障上报时间
	begin_time		= DateTimeFormatProperty \
			(format="%Y-%m-%d %H:%M:%S")								# 维修开始时间
	complish_time	= DateTimeFormatProperty \
			(format="%Y-%m-%d %H:%M:%S")								# 维修完成时间
	review			= StringProperty()									# 返修评价

	MaintenancePerformance = Relationship('MaintenanceWorker', 'PERFORMED', model=MaintenancePerformance)


class VolunteerActivity(StructuredNode):
	r"""
	志愿活动记录实体
	"""
	# activity_id		=
	place			= StringProperty()									# 活动地点
	begin_time		= DateTimeFormatProperty \
			(format="%Y-%m-%d %H:%M:%S")								# 开始时间
	end_time	= DateTimeFormatProperty \
			(format="%Y-%m-%d %H:%M:%S")								# 结束时间
	review			= StringProperty()									# 返修评价

	# perform = Relationship('MaintenanceWorker', 'PERFORMED', model=MaintenancePerformance)


class SkillAssessResult(StructuredRel):
	r"""
	考核结果边
	"""
	result		= StringProperty()


class SkillAssessment(StructuredNode):
	r"""
	技能考核实体
	"""
	skill_type		= StringProperty()
	assess_date		= DateProperty()


class CapacityRate(StructuredRel):
	r"""
	关系类 维保人员-技能实体
	"""
	level		= StringProperty()


class Capacity(StructuredNode):
	r"""
	维修能力实体
	"""
	name 			= StringProperty(unique_index=True, required=True, max_length=10)    	# 能力名 唯一标识
	description 	= StringProperty(max_length=256)										# 描述
	rule			= StringProperty()														# 能力规则

	CapacityRate	= Relationship('MaintenanceWorker', 'RATE', model=CapacityRate)			# 维修能力关联的人员实体

class MaintenanceWorker(StructuredNode):
	r"""
	维修人员实体
	"""
	uid 			= StringProperty(unique_index=True, required=True, max_length=20)					# 工号 唯一标识
	name			= StringProperty(index = True, max_length=32)										# 姓名
	sex 			= StringProperty(choices={'男':'男','女':'女'})  									 # 性别
	nation			= StringProperty(max_length=20)														# 民族
	phone			= StringProperty(max_length=11)														# 联系方式
	birth			= DateProperty()																	# 出生日期
	live_in			= StringProperty(max_length=256)													# 居住地址
	employ_date 	= DateProperty()																	# 入职时间
	work_post 		= StringProperty()																	# 岗位
	work_level		= StringProperty()																	# 岗位级别
	department 		= StringProperty(max_length=20)														# 部门

	CapacityRate 	= RelationshipFrom('Capacity', 'RATE', model=CapacityRate)  							# 维修能力
	MaintenancePerformance = RelationshipFrom('MaintenanceRecord','PERFORMED', model=MaintenancePerformance)   # 维修表现

