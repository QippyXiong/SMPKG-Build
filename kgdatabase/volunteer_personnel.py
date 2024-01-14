from neomodel import (
    StructuredNode, StringProperty, StructuredRel,
    UniqueIdProperty, RelationshipFrom,RelationshipTo,Relationship,
    DateProperty, BooleanProperty,DateTimeFormatProperty
	)

class Volunteer(StructuredNode):
	r"""
	志愿者实体
	"""
	uid = StringProperty(unique_index=True, required=True, max_length=20)								# 志愿者编号 唯一标识
	name = StringProperty(index=True, max_length=32)  					  								# 姓名
	sex = StringProperty(choices={'男':'男','女':'女'})  											  	 # 性别
	nation = StringProperty(max_length=20) 								    							# 民族
	phone = StringProperty(max_length=11)  																# 联系方式
	birth = DateProperty()  																			# 出生日期
	live_in = StringProperty(max_length=256)  															# 居住地址
	apply_date = DateProperty()  																		# 申请时间