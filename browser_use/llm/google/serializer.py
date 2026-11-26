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
			# Normalize role safely (guard against sentinels like NotGiven)
			raw_role = getattr(message, 'role', None)
			role: str | None = raw_role if isinstance(raw_role, str) else None

			# Handle system/developer messages (only if we have a proper string role or explicit SystemMessage)
			if isinstance(message, SystemMessage) or (isinstance(role, str) and role in ['system', 'developer']):
				# Extract system message content as string
				content = getattr(message, 'content', None)
				if isinstance(content, str):
					system_message = content
				elif content is not None:
					# Handle Iterable of content parts; otherwise try to coerce a sensible string
					from collections.abc import Iterable
					if isinstance(content, Iterable):
						parts: list[str] = []
						for part in content:
							part_type = getattr(part, 'type', None)
							if part_type == 'text':
								parts.append(str(getattr(part, 'text', '')))
							elif hasattr(part, 'text'):
								# Fallback: if object exposes text attribute
								parts.append(str(getattr(part, 'text')))
						system_message = '\n'.join(parts)
					else:
						# Non-iterable content: try common text-like attributes or fallback to str()
						if hasattr(content, 'text'):
							system_message = str(getattr(content, 'text'))
						elif isinstance(content, (bytes, bytearray)):
							# Binary content doesn't translate to a system string
							system_message = None
						else:
							system_message = str(content)
				continue

			# Determine the role for non-system messages
			if isinstance(message, UserMessage):
				role = 'user'
			elif isinstance(message, AssistantMessage):
				role = 'model'
			else:
				# Default to user for any unknown message types
				role = 'user'

			# Ensure role is a string before constructing Content
			if not isinstance(role, str):
				role = 'user'

			# Initialize message parts
			message_parts: list[Part] = []

			# Extract content and create parts
			content = getattr(message, 'content', None)
			if isinstance(content, str):
				# Regular text content
				message_parts = [Part.from_text(text=content)]
			elif content is not None:
				# Handle Iterable of content parts; if not iterable, coerce sensibly
				from collections.abc import Iterable
				if isinstance(content, Iterable):
					for part in content:
						part_type = getattr(part, 'type', None)
						if part_type == 'text':
							message_parts.append(Part.from_text(text=str(getattr(part, 'text', ''))))
						elif part_type == 'refusal':
							message_parts.append(Part.from_text(text=f"[Refusal] {getattr(part, 'refusal', '')}"))
						elif part_type == 'image_url':
							# Handle images
							url_obj = getattr(part, 'image_url', None)
							url = getattr(url_obj, 'url', None) if url_obj is not None else None
							if isinstance(url, str) and ',' in url:
								header, data = url.split(',', 1)
								try:
									image_bytes = base64.b64decode(data)
									# Add image part
									image_part = Part.from_bytes(data=image_bytes, mime_type='image/png')
									message_parts.append(image_part)
								except Exception:
									# If decoding fails, skip the part
									continue
				else:
					# Non-iterable content: attempt to extract text or bytes
					if hasattr(content, 'text'):
						message_parts.append(Part.from_text(text=str(getattr(content, 'text'))))
					elif isinstance(content, (bytes, bytearray)):
						message_parts.append(Part.from_bytes(data=bytes(content), mime_type='application/octet-stream'))
					else:
						# Fallback: stringify the object
						message_parts.append(Part.from_text(text=str(content)))

			# Create the Content object
			if message_parts:
				final_message = Content(role=role, parts=message_parts)
				formatted_messages.append(final_message)

		return formatted_messages, system_message
