"""Tool schemas (OpenAI function-calling format) and Pydantic argument models."""

from pydantic import BaseModel, Field, ValidationError

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns a list of results with title, URL, and snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return.",
                        "default": 5,
                        "maximum": 15,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a web page and extract its main content as Markdown."
                " Use for reading full articles from search results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": (
                "Ask the user a clarifying question and wait for their response."
                " The user has up to 60 minutes to respond before the request times out."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The question to ask the user."},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_status",
            "description": (
                "Set or update the current research progress TODO list."
                " Use Markdown checkboxes: '- [ ]' for pending, '- [X]' for done."
                " Call as your FIRST action to announce the plan, then update as you complete steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_list": {
                        "type": "string",
                        "description": "The research TODO list in Markdown with checkboxes.",
                    },
                },
                "required": ["todo_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Finalize the research and deliver the complete report."
                " Provide the full research result in Markdown format."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "result_markdown": {
                        "type": "string",
                        "description": "The complete research report in Markdown format.",
                    },
                },
                "required": ["result_markdown"],
            },
        },
    },
]


class WebSearchArgs(BaseModel):
    """Arguments for the web_search tool."""

    query: str = Field(min_length=1)
    max_results: int = Field(default=5, ge=1, le=15)


class WebFetchArgs(BaseModel):
    """Arguments for the web_fetch tool."""

    url: str = Field(min_length=1)


class AskUserArgs(BaseModel):
    """Arguments for the ask_user tool."""

    question: str = Field(min_length=1)


class FinishArgs(BaseModel):
    """Arguments for the finish tool."""

    result_markdown: str = Field(min_length=1)


class SetStatusArgs(BaseModel):
    """Arguments for the set_status tool."""

    todo_list: str = Field(min_length=1)


_TOOL_MODELS: dict[str, type[BaseModel]] = {
    "web_search": WebSearchArgs,
    "web_fetch": WebFetchArgs,
    "ask_user": AskUserArgs,
    "finish": FinishArgs,
    "set_status": SetStatusArgs,
}


def _validate_tool_args(name: str, args: dict) -> BaseModel | str:
    """Validate and parse tool arguments.

    Returns the parsed Pydantic model on success, or an error message string on failure.
    """
    model_cls = _TOOL_MODELS.get(name)
    if model_cls is None:
        return f"Unknown tool: {name}"
    try:
        return model_cls.model_validate(args)
    except ValidationError as e:
        errors = e.errors()
        details = "; ".join(f"{' -> '.join(str(loc) for loc in err['loc'])}: {err['msg']}" for err in errors)
        return f"Invalid arguments for {name}: {details}"
