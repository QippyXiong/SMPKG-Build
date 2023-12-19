import gc
# import typing
from typing import Callable, Optional, List, Union, Any, Tuple, Generator

import openai
import torch
from openai.openai_object import OpenAIObject
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers import PreTrainedTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from smpkg.prompt import InformationExtractionArrayPrompt, extract_json

__all__ = [
	'connect_llm',
	'LocalLLMController',
	'gpt_maintainance_record_extraction',
	'HistoryType'
]


def connect_openai(api_key: str, api_base: str):
	openai.api_key = api_key
	openai.api_base = api_base


def connect_llm(llm_name: str = 'openai', **kwargs):
	r"""
	连接到llm，执行完此方法后应当能够执行任意的函数
	llm_name输入后，需要指定连接信息，用kwargs来指定，缺少会抛出KeyError
	目前只支持openai

	Args:
		llm_name: name of LLM you want to use

	list of LLM and need params:
		openai: needs api_key(openai chatgpt api key), api_base(url for access openai model)

	"""
	if llm_name == 'openai':
		connect_openai(api_key=kwargs['api_key'], api_base=kwargs['api_base'])
	else:
		raise ValueError(f"unkown LLM name { llm_name }")


def chat(content: str, num_choices: int = 1, role: str = "user") -> list[str]:
	r"""
	
	"""
	res = openai.ChatCompletion.create(
		model='gpt-3.5-turbo-0613',
		messages=[{
			'role': role,
			'content': content
		}],
		stream=False,
		n=num_choices
	)
	return [ c['message']['content'] for c in res['choices'] ]


def chat_stream(
	content: str,
	stream_handler: Callable[[OpenAIObject], None],
	num_choices: int = 1,
	role='user'
) -> None:
	res = openai.ChatCompletion.create(
		model='gpt-3.5-turbo',
		messages=[{
			'role': role,
			'content': content
		}],
		stream=True,
		n=num_choices
	)
	for event in res:
		r"""
		{
		"id": "chatcmpl-86H1dMtMox5OhUqLT0f9oPsfI3r5H",
		"object": "chat.completion.chunk",
		"created": 1696506105,
		"model": "gpt-3.5-turbo-0301",
		"choices": [
			{
			"index": 0,
			"delta": {},
			"finish_reason": "stop"
			}
		]
		}
		"""
		print('stream recieve', event)
		event: OpenAIObject
		stream_handler(event)
		

def gpt_maintainance_record_extraction(prompt: InformationExtractionArrayPrompt, content: str) -> dict:
	r"""
	Args:
		`content`: maintaince text for extracting information

	Returns:
		dict of information extracted from text

	Throws:
		`ValueError`: if extraction failed, return msg contains gpt return value
	"""
	chat_prompt = prompt(content)
	res_text = chat(chat_prompt, num_choices=1, role='user')
	res = extract_json(res_text[0])
	if not res:
		raise ValueError('extraction failed, return text is {}'.format(res_text[0]))
	return res


# self declared
HistoryType = List[Tuple[str, str]]


class QWenModelInterface:
	r"""
	qwen model declared interface
	"""

	def chat(
		self,
		tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
		query: str,
		history: Optional[HistoryType],
		system: str = "You are a helpful assistant.",
		stream: Optional[bool] = True,
		stop_words_ids: Optional[List[List[int]]] = None,
		generation_config: Optional[GenerationConfig] = None,
		**kwargs,
	) -> Tuple[str, HistoryType]:
		...

	def chat_stream(
		self,
		tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
		query: str,
		history: Optional[HistoryType],
		system: str = "You are a helpful assistant.",
		stop_words_ids: Optional[List[List[int]]] = None,
		logits_processor: Optional[LogitsProcessorList] = None,
		generation_config: Optional[GenerationConfig] = None,
		**kwargs,
	) -> Generator[str, Any, None]:
		...


class LocalLLMController:

	def __init__(self, model_dir: str, device: torch.device) -> None:
		r"""
		`model_dir`: the model directory
		`device`: can only be cuda
		"""
		# load model
		self.qwen_dir = model_dir
		if not device.type == 'cuda':
			raise ValueError('Quantitized model can only run on cuda device.')
		self.device = device
		self.model = None
		self.tokenizer = None

	def load(self):
		self.tokenizer = AutoTokenizer.from_pretrained(self.qwen_dir, trust_remote_code=True)
		self.model: QWenModelInterface = AutoModelForCausalLM.from_pretrained(
			self.qwen_dir,
			device_map=self.device,
			trust_remote_code=True
		).eval()

	@torch.no_grad()
	def info_extraction_task_stream(
		self,
		prompt: InformationExtractionArrayPrompt,
		sentence: str
	) -> Generator[str, Any, None]:
		r"""
		Args:
			`prompt`: InformationExtractionArrayPrompt
		Returns:
			generator of output string
		"""
		if self.model is None:
			raise ValueError('model not loaded')
		content = prompt(sentence)
		return self.model.chat_stream(self.tokenizer, content, None)

	@torch.no_grad()
	def info_extraction(self, prompt: InformationExtractionArrayPrompt, sentence: str) -> Optional[list[dict]]:
		r"""
		Args:
			`prompt`: InformationExtractionArrayPrompt
		Returns:
			list of information extracted from text
		Throws:
			`ValueError`: model not loaded or extraction failed
		"""
		if self.model is None:
			raise ValueError('model not loaded')
		json_str, _ = self.model.chat(self.tokenizer, prompt(sentence), None)
		return self.extract_json_array_from_str(json_str)

	@staticmethod
	def extract_json_array_from_str(content: str) -> Union[dict, list, None]:
		r = extract_json(content)
		if not r:
			return None
	
	def release(self) -> None:
		del self.model
		self.model = None
		torch.cuda.empty_cache()
		gc.collect()
		del self.tokenizer
