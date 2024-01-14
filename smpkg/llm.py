import gc
# import typing
from typing import Callable, Optional, List, Union, Any, Tuple, Generator

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers import PreTrainedTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList

from smpkg.prompt import InformationExtractionArrayPrompt, extract_json

__all__ = [
	'LocalLLMController'
]

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
