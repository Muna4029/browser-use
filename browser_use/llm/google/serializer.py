import base64

from google.genai.types import Content, Part

from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)


class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[list[Content], str | None]:
		"""
		Convert a list of BaseMessages to Google format, extracting system message.

		Google handles system instructions separately from the conversation, so we need to:
		1. Extract any system messages and return them separately as a string
		2. Convert the remaining messages to Content objects

		Args:
		    messages: List of messages to convert

		Returns:
		    A tuple of (formatted_messages, system_message) where:
		    - formatted_messages: List of Content objects for the conversation
		    - system_message: System instruction string or None
		"""

		messages = [m.model_copy(deep=True) for m in messages]

		formatted_messages: list[Content] = []
		system_message: str | None = None

		for message in messages:
			role = message.role if hasattr(message, 'role') else None

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if isinstance(message.content, str):
					system_message = message.content
				elif message.content is None or type(message.content).__name__ == 'NotGiven':
					# No usable content
					pass
				else:
					# Attempt to iterate content safely; if not iterable, coerce to string
					parts: list[str] = []
					try:
						iterator = iter(message.content)
					except TypeError:
						system_message = str(message.content)
					else:
						for part in iterator:
							ptype = getattr(part, 'type', None)
							if ptype == 'text':
								parts.append(getattr(part, 'text', ''))
							elif ptype == 'refusal':
								parts.append(f"[Refusal] {getattr(part, 'refusal', '')}")
							elif ptype == 'image_url':
								url_obj = getattr(part, 'image_url', None)
								url = getattr(url_obj, 'url', '') if url_obj is not None else ''
								parts.append(url)
							else:
								parts.append(str(part))
						if parts:
							system_message = '\n'.join(parts)
				continue

			# Determine the role for non-system messages
			if isinstance(message, UserMessage):
				role = 'user'
			elif isinstance(message, AssistantMessage):
				role = 'model'
			else:
				# Default to user for any unknown message types
				role = 'user'

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			if isinstance(message.content, str):
				# Regular text content
				message_parts = [Part.from_text(text=message.content)]
			elif message.content is None or type(message.content).__name__ == 'NotGiven':
				# No content to add
				pass
			else:
				# Attempt to iterate content safely; if not iterable, coerce to text
				try:
					iterator = iter(message.content)
				except TypeError:
					message_parts = [Part.from_text(text=str(message.content))]
				else:
					for part in iterator:
						ptype = getattr(part, 'type', None)
						if ptype == 'text':
							message_parts.append(Part.from_text(text=getattr(part, 'text', '')))
						elif ptype == 'refusal':
							message_parts.append(Part.from_text(text=f"[Refusal] {getattr(part, 'refusal', '')}"))
						elif ptype == 'image_url':
							# Handle images safely
							url_obj = getattr(part, 'image_url', None)
							url = getattr(url_obj, 'url', '') if url_obj is not None else ''
							if ',' in url:
								head, data = url.split(',', 1)
								try:
									image_bytes = base64.b64decode(data)
									image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')
								except Exception:
									# Fallback: include URL as text if decode fails
									image_part = Part.from_text(text=url)
								message_parts.append(image_part)
							else:
								# Not base64 data, store URL as text
								message_parts.append(Part.from_text(text=url))
						else:
							# Unknown part type -> convert to text
							message_parts.append(Part.from_text(text=str(part)))

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
