import logging
import os
from typing import Callable, Dict, List, Optional, Union

try:
    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types
except ImportError:
    raise ImportError("The 'google-genai' library is required. Please install it using 'pip install google-genai'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.gemini import GeminiConfig
from mem0.exceptions import LLMResponseError
from mem0.llms.base import LLMBase

logger = logging.getLogger(__name__)

#: DeepMem0: called as fn(model, usage) after every response, including blocked
#: or truncated ones -- those tokens are billed too. A daily cost cap built on
#: this must see every call, so a listener error is logged, never raised.
_USAGE_LISTENERS: List[Callable[[str, Dict[str, int]], None]] = []

_USAGE_FIELDS = ("prompt_token_count", "candidates_token_count", "thoughts_token_count",
                 "cached_content_token_count", "total_token_count")

_SAFETY_CATEGORIES = ("HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
                      "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")


def add_usage_listener(fn: Callable[[str, Dict[str, int]], None]) -> None:
    if fn not in _USAGE_LISTENERS:
        _USAGE_LISTENERS.append(fn)


def remove_usage_listener(fn: Callable[[str, Dict[str, int]], None]) -> None:
    if fn in _USAGE_LISTENERS:
        _USAGE_LISTENERS.remove(fn)


def _usage_dict(usage) -> Dict[str, int]:
    out = {}
    for name in _USAGE_FIELDS:
        value = getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            out[name] = value
    return out


def _enum_name(value) -> Optional[str]:
    """Name of a google.genai enum, or of the raw string the SDK keeps for a value it does not know."""
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return value if isinstance(value, str) else None


def _http_retryable(code) -> bool:
    return isinstance(code, int) and (code in (408, 429) or code >= 500)


class GeminiLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, GeminiConfig, Dict]] = None):
        # Convert to GeminiConfig if needed
        if config is None:
            config = GeminiConfig()
        elif isinstance(config, dict):
            config = GeminiConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, GeminiConfig):
            config = GeminiConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "gemini-2.0-flash"

        self.last_usage: Dict[str, int] = {}
        http_options = self._http_options()
        client_kwargs = {"http_options": http_options} if http_options is not None else {}
        if self.config.vertexai:
            self.client = genai.Client(
                vertexai=True, project=self.config.project, location=self.config.location, **client_kwargs
            )
        else:
            api_key = self.config.api_key or os.getenv("GOOGLE_API_KEY")
            self.client = genai.Client(api_key=api_key, **client_kwargs)

    def _http_options(self):
        timeout_s = getattr(self.config, "timeout_s", None)
        attempts = getattr(self.config, "http_retry_attempts", None)
        if timeout_s is None and attempts is None:
            return None
        kwargs = {}
        if timeout_s is not None:
            kwargs["timeout"] = int(timeout_s * 1000)  # the SDK takes milliseconds
        if attempts is not None:
            kwargs["retry_options"] = types.HttpRetryOptions(attempts=attempts)
        return types.HttpOptions(**kwargs)

    def _record_usage(self, response) -> None:
        self.last_usage = _usage_dict(getattr(response, "usage_metadata", None))
        for fn in list(_USAGE_LISTENERS):
            try:
                fn(self.config.model, dict(self.last_usage))
            except Exception as exc:  # a broken listener must not fail the call
                logger.warning("Gemini usage listener %r failed: %s", fn, exc)

    @staticmethod
    def _check_complete(response) -> None:
        """Raise LLMResponseError unless the response carries a finished completion.

        Upstream returned "" for a blocked or empty response, and the extraction
        parser read "" as "no new facts". Only STOP (or no reason at all, from
        an SDK or a mock that does not set it) counts as finished.
        """
        feedback = getattr(response, "prompt_feedback", None)
        block = _enum_name(getattr(feedback, "block_reason", None)) if feedback is not None else None
        if block and block != "BLOCKED_REASON_UNSPECIFIED":
            raise LLMResponseError("gemini", "prompt_blocked", block, retryable=False)
        candidates = getattr(response, "candidates", None)
        if isinstance(candidates, (list, tuple)) and not candidates:
            raise LLMResponseError("gemini", "no_candidates", retryable=True)
        if not candidates:
            return
        finish = _enum_name(getattr(candidates[0], "finish_reason", None))
        if finish in (None, "STOP", "FINISH_REASON_UNSPECIFIED"):
            return
        if finish == "MAX_TOKENS":
            raise LLMResponseError("gemini", "truncated", finish, retryable=False)
        raise LLMResponseError("gemini", "blocked", finish, retryable=False)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        # Get parts safely — content can be None when Gemini blocks the response
        candidate = response.candidates[0] if response.candidates else None
        parts = candidate.content.parts if candidate and candidate.content else None

        if tools:
            processed_response = {
                "content": None,
                "tool_calls": [],
            }

            if parts:
                # Extract content from the first candidate
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        processed_response["content"] = part.text
                        break

                # Extract function calls
                for part in parts:
                    if hasattr(part, "function_call") and part.function_call:
                        fn = part.function_call
                        processed_response["tool_calls"].append(
                            {
                                "name": fn.name,
                                "arguments": dict(fn.args) if fn.args else {},
                            }
                        )

            return processed_response
        else:
            if parts:
                for part in parts:
                    if hasattr(part, "text") and part.text:
                        return part.text
            return ""

    def _reformat_messages(self, messages: List[Dict[str, str]]):
        """
        Reformat messages for Gemini.

        Args:
            messages: The list of messages provided in the request.

        Returns:
            tuple: (system_instruction, contents_list)
        """
        system_instruction = None
        contents = []

        for message in messages:
            if message["role"] == "system":
                system_instruction = message["content"]
            else:
                content = types.Content(
                    parts=[types.Part(text=message["content"])],
                    role=message["role"],
                )
                contents.append(content)

        return system_instruction, contents

    def _reformat_tools(self, tools: Optional[List[Dict]]):
        """
        Reformat tools for Gemini.

        Args:
            tools: The list of tools provided in the request.

        Returns:
            list: The list of tools in the required format.
        """

        def remove_additional_properties(data):
            """Recursively removes 'additionalProperties' from nested dictionaries."""
            if isinstance(data, dict):
                filtered_dict = {
                    key: remove_additional_properties(value)
                    for key, value in data.items()
                    if not (key == "additionalProperties")
                }
                return filtered_dict
            else:
                return data

        if tools:
            function_declarations = []
            for tool in tools:
                func = tool["function"].copy()
                cleaned_func = remove_additional_properties(func)

                function_declaration = types.FunctionDeclaration(
                    name=cleaned_func["name"],
                    description=cleaned_func.get("description", ""),
                    parameters=cleaned_func.get("parameters", {}),
                )
                function_declarations.append(function_declaration)

            tool_obj = types.Tool(function_declarations=function_declarations)
            return [tool_obj]
        else:
            return None

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ):
        """
        Generate a response based on the given messages using Gemini.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format for the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".

        Returns:
            str: The generated response.
        """

        # Extract system instruction and reformat messages
        system_instruction, contents = self._reformat_messages(messages)

        # Prepare generation config — only include non-None values so the
        # Gemini SDK uses its own defaults instead of rejecting None.
        config_params = {}
        if self.config.temperature is not None:
            config_params["temperature"] = self.config.temperature
        if self.config.max_tokens is not None:
            config_params["max_output_tokens"] = self.config.max_tokens
        if self.config.top_p is not None:
            config_params["top_p"] = self.config.top_p
        thinking_level = getattr(self.config, "thinking_level", None)
        thinking_budget = getattr(self.config, "thinking_budget", None)
        if thinking_level is not None or thinking_budget is not None:
            config_params["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_level, thinking_budget=thinking_budget
            )
        safety_threshold = getattr(self.config, "safety_threshold", None)
        if safety_threshold is not None:
            config_params["safety_settings"] = [
                types.SafetySetting(category=category, threshold=safety_threshold)
                for category in _SAFETY_CATEGORIES
            ]

        # Add system instruction to config if present
        if system_instruction:
            config_params["system_instruction"] = system_instruction

        if response_format is not None and response_format["type"] == "json_object":
            config_params["response_mime_type"] = "application/json"
            if "schema" in response_format:
                config_params["response_schema"] = response_format["schema"]

        if tools:
            formatted_tools = self._reformat_tools(tools)
            config_params["tools"] = formatted_tools

            if tool_choice:
                if tool_choice == "auto":
                    mode = types.FunctionCallingConfigMode.AUTO
                elif tool_choice == "any":
                    mode = types.FunctionCallingConfigMode.ANY
                else:
                    mode = types.FunctionCallingConfigMode.NONE

                tool_config = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=mode,
                        allowed_function_names=(
                            [tool["function"]["name"] for tool in tools] if tool_choice == "any" else None
                        ),
                    )
                )
                config_params["tool_config"] = tool_config

        generation_config = types.GenerateContentConfig(**config_params)

        try:
            response = self.client.models.generate_content(
                model=self.config.model, contents=contents, config=generation_config
            )
        except genai_errors.APIError as exc:
            # Tag for the caller's queue: 408/429/5xx can succeed later, a 4xx cannot.
            exc.retryable = _http_retryable(getattr(exc, "code", None))
            raise

        self._record_usage(response)
        self._check_complete(response)
        return self._parse_response(response, tools)
