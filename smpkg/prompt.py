from dataclasses import dataclass, fields, field
from typing import Union, Optional, Any, get_args, get_origin, Sequence

import json5


__all__ = [
	'InformationExtractionArrayPrompt',
	'PromptTypeConstraint',
	'PromptDateFormatConstraint',
	'PromptSelectedValueConstraint',
	'PromptNatualLanguageConstraint',
	'extract_json'
]


@dataclass
class PromptTypeConstraint:
	key_name: str
	constraint_type: type[Union[int, float, str, bool, list, dict]]

	def __str__(self) -> str:
		return f'对于数组中单个元素的字段{ self.key_name }，你抽取到值的类型应当为{ self.constraint_type.__name__ }。'


@dataclass
class PromptDateFormatConstraint:
	format: str
	key: Optional[str] = None

	def __str__(self) -> str:
		if self.key is not None:
			return f'对于数组中单个元素的时间字段{ self.key }，你应当设置其时间格式为{ self.format }。'
		else:
			return f'你应当设置数组中时间字段的时间格式为{ self.format }。'


@dataclass
class PromptSelectedValueConstraint:
	r"""
	constraint, value of key `key_name` should only get value of `values`
	"""
	key_name: str
	values: list[str]

	def __str__(self) -> str:
		return f'对于数组中单个元素的字段{ self.key_name }，它的值只能为{ self.values }中的其中之一。'


@dataclass
class PromptNatualLanguageConstraint:
	r"""
	constraint, a natural language constraint, which may be most unuseful
	"""
	description: str

	def __str__(self) -> str:
		return self.description


ConstraintTypes = Union[
	PromptTypeConstraint, PromptDateFormatConstraint, PromptSelectedValueConstraint, PromptNatualLanguageConstraint
]


@dataclass
class InformationExtractionArrayPrompt:
	r"""
	用于生成信息抽取任务的prompt
	"""
	# 对于整个抽取任务的描述
	description: str
	# 需要抽取内容的名称
	info_names: list[str]
	# 抽取的键和对应的描述
	item_keys: dict[str, str]
	# 约束列表
	constraints: list[ConstraintTypes]
	# 示例输入
	example_input: str
	# 示例输出
	example_outputs: list[dict[str, str]]
	# 无法对于无法检测到的键对应值的填写
	unknown_value: str = 'null'

	# if the output should return extra information
	_should_return_extra: bool = field(init=False, default=True)

	def create_prompt_content(self) -> str:
		
		if hasattr(self, 'prompt_content'):
			if getattr(self, 'prompt_content') is not None:
				return getattr(self, 'prompt_content')
		# basic description
		prompt_content = f'你的任务是{ self.description }。\n'
		prompt_content += f'你需要抽取的信息有：{ ", ".join(self.info_names) }。\n'
		# add output format
		prompt_content += '你需要将你的输出结果组织成一个json数组，数组中的一个元素的格式如下：\n'
		prompt_content += '{\n'
		prompt_content += ''.join([f'\t"{key}": {desc},\n' for key, desc in self.item_keys.items() ])
		prompt_content += '}\n'
		prompt_content += f'你应当将文本中未提及的字段标记为{ self.unknown_value }。\n'
		# add constraints
		prompt_content += ''.join([ str(constraint) + '\n' for constraint in self.constraints ])
		# add exmpale
		prompt_content += '例如，对于输入：\n'
		prompt_content += self.example_input + '\n'
		prompt_content += '你的输出应当为：\n'
		prompt_content += '[\n'
		for example_output in self.example_outputs:
			prompt_content += '\t{\n'
			prompt_content += ''.join([
				'\t\t"{}": {},\n'.format(key, f'"{value}"' if value else self.unknown_value )
				for key, value in example_output.items()
			])
			prompt_content += '\t},\n'
		prompt_content += ']\n'
		if not self._should_return_extra:
			prompt_content += '除了抽取的信息，你不应当回复其他内容。\n'
		
		prompt_content += f'你需要处理的文本内容如下：\n'

		# generate content cost a lot, just cache the content
		setattr(self, 'prompt_content', prompt_content)
		return prompt_content
	
	def __len__(self) -> int:
		return len(str(self))
	
	def reset(self) -> None:
		if hasattr(self, 'prompt_content'):
			setattr(self, 'prompt_content', None)

	def __call__(self, input_text: str) -> str:
		return self.create_prompt_content() + input_text


def extract_json(string: str, encoding='UTF-8') -> Union[dict, list, None]:
	r"""
	find the json string in str and parse it

	Args:
		string: input string contains json string
		encoding: encoding of string
	
	Returns
		parsed json object, if has no json string return None
	"""
	r"""
	logic: using stack accept '{', '}' or '[', ']', parse the final and max '{-}' / '[-]' ones
	"""
	stack = []
	result = {}
	for i in range(len(str)):
		if string[i] == '{' or string[i] == '[':
			stack.append((string[i], i))
		
		if string[i] == '}':
			if not stack:  # empty list, only happen if '}' appears before json string
				continue

			char, idx = stack.pop()
			
			if char == '{':  # matched, parse
				try:
					result = json5.loads(str[idx: i+1], encoding=encoding)
				except Exception:
					stack = []  # parse fail, this not json, that mean all before i cannot be json
			else:  # that mean it's '[', all these context are errored
				stack = []
		
		if string[i] == ']':  # same as the previous one
			if not stack:
				continue

			char, idx = stack.pop()

			if char == '[':
				try:
					result = json5.loads(string[idx: i+1], encoding=encoding)
				except Exception:
					stack = []
			else:
				stack = []
	return result
