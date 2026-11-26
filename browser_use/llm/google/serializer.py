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
				elif message.content is not None:
					# Handle Iterable of content parts safely (skip non-iterables)
					parts = []
					try:
						iterator = iter(message.content)
					except TypeError:
						# Non-iterable or sentinel value; skip extracting parts
						iterator = ()
					for part in iterator:
						if getattr(part, 'type', None) == 'text':
							parts.append(getattr(part, 'text', ''))
					# Only join if we found parts
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
			elif message.content is not None:
				# Handle Iterable of content parts safely (skip non-iterables)
				try:
					iterator = iter(message.content)
				except TypeError:
					iterator = ()
				for part in iterator:
					ptype = getattr(part, 'type', None)
					if ptype == 'text':
						message_parts.append(Part.from_text(text=getattr(part, 'text', '')))
					elif ptype == 'refusal':
						message_parts.append(Part.from_text(text=f'[Refusal] {getattr(part, "refusal", "")}'))
					elif ptype == 'image_url':
						# Handle images
						url_obj = getattr(part, 'image_url', None)
						if url_obj is None:
							continue
						url = getattr(url_obj, 'url', '')

						# Format: data:image/png;base64,<data>
						if ',' in url:
							header, data = url.split(',', 1)
							# Decode base64 to bytes
							image_bytes = base64.b64decode(data) if data else b''
						else:
							image_bytes = b''

						# Add image part
						image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')

						message_parts.append(image_part)

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
