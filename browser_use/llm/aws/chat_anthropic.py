import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, overload

from anthropic import (
	NOT_GIVEN,
	APIConnectionError,
	APIStatusError,
	AsyncAnthropicBedrock,
	RateLimitError,
)
from anthropic.types import CacheControlEphemeralParam, Message, ToolParam
from anthropic.types.text_block import TextBlock
from anthropic.types.tool_choice_tool_param import ToolChoiceToolParam
from pydantic import BaseModel

from browser_use.llm.anthropic.serializer import AnthropicMessageSerializer
from browser_use.llm.aws.chat_bedrock import ChatAWSBedrock
from browser_use.llm.exceptions import ModelProviderError, ModelRateLimitError
from browser_use.llm.messages import BaseMessage
from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

if TYPE_CHECKING:
	from boto3.session import Session  # pyright: ignore


T = TypeVar('T', bound=BaseModel)


@dataclass
class ChatAnthropicBedrock(ChatAWSBedrock):
	"""
	AWS Bedrock Anthropic Claude chat model.

	This is a convenience class that provides Claude-specific defaults
	for the AWS Bedrock service. It inherits all functionality from
	ChatAWSBedrock but sets Anthropic Claude as the default model.
	"""

	# Anthropic Claude specific defaults
	model: str = 'anthropic.claude-3-5-sonnet-20240620-v1:0'
	max_tokens: int = 8192
	temperature: float | None = None
	top_p: float | None = None
	top_k: int | None = None
	stop_sequences: list[str] | None = None

	# AWS credentials and configuration
	aws_access_key: str | None = None
	aws_secret_key: str | None = None
	aws_session_token: str | None = None
	aws_region: str | None = None
	session: 'Session | None' = None

	# Client initialization parameters
	max_retries: int = 10
	default_headers: Mapping[str, str] | None = None
	default_query: Mapping[str, object] | None = None

	@property
	def provider(self) -> str:
		return 'anthropic_bedrock'

	def _get_client_params(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for Bedrock."""
		client_params: dict[str, Any] = {}

		if self.session:
			credentials = self.session.get_credentials()
			client_params.update(
				{
					'aws_access_key': credentials.access_key,
					'aws_secret_key': credentials.secret_key,
					'aws_session_token': credentials.token,
					'aws_region': self.session.region_name,
				}
			)
		else:
			# Use individual credentials
			if self.aws_access_key:
				client_params['aws_access_key'] = self.aws_access_key
			if self.aws_secret_key:
				client_params['aws_secret_key'] = self.aws_secret_key
			if self.aws_region:
				client_params['aws_region'] = self.aws_region
			if self.aws_session_token:
				client_params['aws_session_token'] = self.aws_session_token

		# Add optional parameters
		if self.max_retries:
			client_params['max_retries'] = self.max_retries
		if self.default_headers:
			client_params['default_headers'] = self.default_headers
		if self.default_query:
			client_params['default_query'] = self.default_query

		return client_params

	def _get_client_params_for_invoke(self) -> dict[str, Any]:
		"""Prepare client parameters dictionary for invoke."""
		client_params = {}

		if self.temperature is not None:
			client_params['temperature'] = self.temperature
		if self.max_tokens is not None:
			client_params['max_tokens'] = self.max_tokens
		if self.top_p is not None:
			client_params['top_p'] = self.top_p
		if self.top_k is not None:
			client_params['top_k'] = self.top_k
		if self.stop_sequences is not None:
			client_params['stop_sequences'] = self.stop_sequences

		return client_params

	def get_client(self) -> AsyncAnthropicBedrock:
		"""
		Returns an AsyncAnthropicBedrock client.

		Returns:
			AsyncAnthropicBedrock: An instance of the AsyncAnthropicBedrock client.
		"""
		client_params = self._get_client_params()
		return AsyncAnthropicBedrock(**client_params)

	@property
	def name(self) -> str:
		return str(self.model)

	def _get_usage(self, response: Message) -> ChatInvokeUsage | None:
		"""Extract usage information from the response."""
		usage = ChatInvokeUsage(
			prompt_tokens=response.usage.input_tokens
			+ (
				response.usage.cache_read_input_tokens or 0
			),  # Total tokens in Anthropic are a bit fucked, you have to add cached tokens to the prompt tokens
			completion_tokens=response.usage.output_tokens,
			total_tokens=response.usage.input_tokens + response.usage.output_tokens,
			prompt_cached_tokens=response.usage.cache_read_input_tokens,
			prompt_cache_creation_tokens=response.usage.cache_creation_input_tokens,
			prompt_image_tokens=None,
		)
		return usage

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: None = None) -> ChatInvokeCompletion[str]: ...

	@overload
	async def ainvoke(self, messages: list[BaseMessage], output_format: type[T]) -> ChatInvokeCompletion[T]: ...

	async def ainvoke(
		self, messages: list[BaseMessage], output_format: type[T] | None = None
	) -> ChatInvokeCompletion[T] | ChatInvokeCompletion[str]:
		anthropic_messages, system_prompt = AnthropicMessageSerializer.serialize_messages(messages)

		try:
			# Build base kwargs and only include 'system' when provided (avoid passing NOT_GIVEN where a str is expected)
			base_params = dict(self._get_client_params_for_invoke())
			if system_prompt is not None:
				base_params['system'] = system_prompt

			if output_format is None:
				# Normal completion without structured output
				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					**base_params,
				)

				usage = self._get_usage(response)

				# Safely extract text from the first content block (handle NotGiven and non-iterable content)
				content = getattr(response, 'content', NOT_GIVEN)
				response_text = ''
				if content is not NOT_GIVEN and isinstance(content, (list, tuple)) and len(content) > 0:
					first_content = content[0]
					if isinstance(first_content, TextBlock):
						response_text = first_content.text
					elif hasattr(first_content, 'text'):
						response_text = str(getattr(first_content, 'text'))
					else:
						# If it's not a text block, convert to string
						response_text = str(first_content)
				else:
					# Fallbacks if content missing or unexpected
					response_text = str(getattr(response, 'output', response))

				return ChatInvokeCompletion(
					completion=response_text,
					usage=usage,
				)

			else:
				# Use tool calling for structured output
				# Create a tool that represents the output format
				tool_name = output_format.__name__
				schema = output_format.model_json_schema()

				# Remove title from schema if present (Anthropic doesn't like it in parameters)
				if 'title' in schema:
					del schema['title']

				tool = ToolParam(
					name=tool_name,
					description=f'Extract information in the format of {tool_name}',
					input_schema=schema,
					cache_control=CacheControlEphemeralParam(type='ephemeral'),
				)

				# Force the model to use this tool
				tool_choice = ToolChoiceToolParam(type='tool', name=tool_name)

				# Prepare params including tools and tool_choice
				invoke_params = dict(base_params)
				invoke_params['tools'] = [tool]
				invoke_params['tool_choice'] = tool_choice

				response = await self.get_client().messages.create(
					model=self.model,
					messages=anthropic_messages,
					**invoke_params,
				)

				usage = self._get_usage(response)

				# Extract the tool use block safely
				content = getattr(response, 'content', NOT_GIVEN)
				if content is NOT_GIVEN:
					raise ValueError('Expected tool use in response but none found')

				content_iter = content if isinstance(content, (list, tuple)) else [content]
				for content_block in content_iter:
					if hasattr(content_block, 'type') and content_block.type == 'tool_use':
						# Parse the tool input as the structured output, handling various input types
						input_data = getattr(content_block, 'input', NOT_GIVEN)
						if input_data is NOT_GIVEN:
							continue
						try:
							# If it's a string, attempt JSON parse first, then validate the parsed object
							if isinstance(input_data, str):
								try:
									parsed = json.loads(input_data)
									return ChatInvokeCompletion(completion=output_format.model_validate(parsed), usage=usage)
								except Exception:
									# Not JSON: try validating the raw string
									return ChatInvokeCompletion(completion=output_format.model_validate(input_data), usage=usage)
							# If it's already a dict-like object, validate directly
							elif isinstance(input_data, dict):
								return ChatInvokeCompletion(completion=output_format.model_validate(input_data), usage=usage)
							else:
								# Attempt to coerce via to_dict() if available, otherwise try JSON from string form
								to_dict = getattr(input_data, 'to_dict', None)
								if callable(to_dict):
									return ChatInvokeCompletion(completion=output_format.model_validate(to_dict()), usage=usage)
								try:
									parsed = json.loads(str(input_data))
									return ChatInvokeCompletion(completion=output_format.model_validate(parsed), usage=usage)
								except Exception:
									# Last resort: validate string form
									return ChatInvokeCompletion(completion=output_format.model_validate(str(input_data)), usage=usage)
						except Exception as e:
							# Re-raise so outer handlers can convert to ModelProviderError
							raise e

				# If no tool use block found, raise an error
				raise ValueError('Expected tool use in response but none found')

		except APIConnectionError as e:
			raise ModelProviderError(message=e.message, model=self.name) from e
		except RateLimitError as e:
			raise ModelRateLimitError(message=e.message, model=self.name) from e
		except APIStatusError as e:
			raise ModelProviderError(message=e.message, status_code=e.status_code, model=self.name) from e
		except Exception as e:
			raise ModelProviderError(message=str(e), model=self.name) from e
