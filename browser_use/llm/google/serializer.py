import base64

from google.genai.types import Content, ContentListUnion, Part

from browser_use.llm.messages import (
	AssistantMessage,
	BaseMessage,
	SystemMessage,
	UserMessage,
)


class GoogleMessageSerializer:
	"""Serializer for converting messages to Google Gemini format."""

	@staticmethod
	def serialize_messages(messages: list[BaseMessage]) -> tuple[ContentListUnion, str | None]:
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

		formatted_messages: ContentListUnion = []
		system_message: str | None = None

		for message in messages:
			role = message.role if hasattr(message, 'role') else None

			# Handle system/developer messages
			if isinstance(message, SystemMessage) or role in ['system', 'developer']:
				# Extract system message content as string
				if isinstance(message.content, str):
					system_message = message.content
				elif message.content is not None:
					# Handle Iterable of content parts safely
					parts: list[str] = []
					try:
						iterable = iter(message.content)
					except TypeError:
						iterable = None
					if iterable is not None:
						for part in iterable:
							part_type = getattr(part, 'type', None)
							if part_type == 'text' and isinstance(getattr(part, 'text', None), str):
								parts.append(part.text)
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
				# Handle Iterable of content parts safely
				try:
					iterable = iter(message.content)
				except TypeError:
					iterable = None
				if iterable is not None:
					for part in iterable:
						part_type = getattr(part, 'type', None)
						# Text part
						if part_type == 'text' and isinstance(getattr(part, 'text', None), str):
							message_parts.append(Part.from_text(text=part.text))
						# Refusal part
						elif part_type == 'refusal' and isinstance(getattr(part, 'refusal', None), str):
							message_parts.append(Part.from_text(text=f'[Refusal] {part.refusal}'))
						# Image URL part (guarded)
						elif part_type == 'image_url':
							img = getattr(part, 'image_url', None)
							url = getattr(img, 'url', None) if img is not None else None
							if isinstance(url, str) and ',' in url:
								# Format: data:image/png;base64,<data>
								header, data = url.split(',', 1)
								# Attempt to decode base64 to bytes
								try:
									image_bytes = base64.b64decode(data)
								except Exception:
									image_bytes = None
								if image_bytes:
									# Infer mime type from header if possible
									mime_type = 'image/png'
									if isinstance(header, str) and header.startswith('data:'):
										mime_part = header.split(';', 1)[0]
										if mime_part.startswith('data:'):
											mime_type = mime_part[5:]
									image_part = Part.from_bytes(data=image_bytes, mime_type=mime_type)
									message_parts.append(image_part)

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
