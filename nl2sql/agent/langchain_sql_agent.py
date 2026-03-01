"""
Optional LangChain-backed SQL agent adapter.

This keeps the framework dependency isolated from the core dissertation path:
1) imports succeed without LangChain installed
2) framework imports happen only when building/running the agent
3) the adapter wraps the existing SQLAlchemy engine and local HF model
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine
else:
    Engine = Any


DEFAULT_SQL_AGENT_SYSTEM_PROMPT = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
"""

REACT_FORMAT_REMINDER = """Follow the ReAct format strictly.
When you need a tool, respond with exactly:
Thought: <short reasoning>
Action: <tool name>
Action Input: <tool input>

When you are finished, respond with exactly:
Thought: <short reasoning>
Final Answer: <final answer>

Do not output raw schema rows, observations, or tool results unless they are inside the required format.
If a parsing error occurs, correct your format on the next turn."""


@dataclass(frozen=True)
class LangChainSQLAgentConfig:
    top_k: int = 5
    max_iterations: int = 6
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.2
    top_p: float = 0.9
    verbose: bool = False
    sample_rows_in_table_info: int = 0
    system_prompt: str = DEFAULT_SQL_AGENT_SYSTEM_PROMPT


_AGENT_CACHE: dict[tuple[int, int, int, LangChainSQLAgentConfig], Any] = {}


def _require_langchain() -> tuple[Any, Any]:
    try:
        from langchain_community.agent_toolkits import create_sql_agent
        from langchain_community.utilities import SQLDatabase
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError(
            "LangChain SQL agent dependencies are not installed. "
            "Install `langchain` and `langchain-community`."
        ) from exc
    return create_sql_agent, SQLDatabase


def _truncate_on_stop(text: str, stop: list[str] | None) -> str:
    if not stop:
        return text
    cut = len(text)
    for token in stop:
        if not token:
            continue
        idx = text.find(token)
        if idx >= 0:
            cut = min(cut, idx)
    return text[:cut]


def _clean_sql_text(text: str) -> str | None:
    from ..core.llm import extract_first_select

    text = str(text or "").strip()
    if not text:
        return None

    sql = extract_first_select(text)
    if not sql:
        return None

    cut_markers = [
        "\nObservation:",
        "\nThought:",
        "\nAction:",
        "\nAction Input:",
        "\nFinal Answer:",
        "\nFor troubleshooting",
    ]
    cut = len(sql)
    for marker in cut_markers:
        idx = sql.find(marker)
        if idx >= 0:
            cut = min(cut, idx)
    sql = sql[:cut].strip()
    if not sql:
        return None
    if not sql.endswith(";"):
        sql += ";"
    return sql


def _build_langchain_llm(*, model: Any, tokenizer: Any, config: LangChainSQLAgentConfig) -> Any:
    try:
        from langchain_core.language_models.llms import LLM
        from pydantic import ConfigDict
        import torch
    except ImportError as exc:  # pragma: no cover - optional integration
        raise RuntimeError(
            "LangChain core dependencies are not installed. "
            "Install `langchain` and its required dependencies."
        ) from exc

    class LocalCausalLLM(LLM):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        hf_model: Any
        hf_tokenizer: Any
        hf_cfg: LangChainSQLAgentConfig

        @property
        def _llm_type(self) -> str:
            return "local_hf_causal_lm"

        def _call(self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any) -> str:
            tok = self.hf_tokenizer
            generation_prompt = str(prompt or "")

            if getattr(tok, "chat_template", None):
                generation_prompt = tok.apply_chat_template(
                    [
                        {"role": "system", "content": REACT_FORMAT_REMINDER},
                        {"role": "user", "content": generation_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            inputs = tok(generation_prompt, return_tensors="pt")

            try:
                device = next(self.hf_model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
            except Exception:
                pass

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": self.hf_cfg.max_new_tokens,
                "do_sample": self.hf_cfg.do_sample,
                "pad_token_id": tok.pad_token_id or tok.eos_token_id,
            }
            if self.hf_cfg.do_sample:
                gen_kwargs["temperature"] = self.hf_cfg.temperature
                gen_kwargs["top_p"] = self.hf_cfg.top_p

            with torch.no_grad():
                outputs = self.hf_model.generate(**inputs, **gen_kwargs)

            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][prompt_len:]
            text = tok.decode(generated_ids, skip_special_tokens=True)
            text = _truncate_on_stop(text, stop)
            return text.strip()

    return LocalCausalLLM(
        hf_model=model,
        hf_tokenizer=tokenizer,
        hf_cfg=config,
    )


def build_langchain_sql_agent(
    *,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    config: LangChainSQLAgentConfig | None = None,
) -> Any:
    """
    Build a LangChain SQL agent around the existing SQLAlchemy engine and HF model.
    """
    cfg = config or LangChainSQLAgentConfig()
    create_sql_agent, SQLDatabase = _require_langchain()

    llm = _build_langchain_llm(
        model=model,
        tokenizer=tokenizer,
        config=cfg,
    )
    db = SQLDatabase(
        engine,
        sample_rows_in_table_info=cfg.sample_rows_in_table_info,
    )
    prefix = cfg.system_prompt.format(dialect=db.dialect, top_k=cfg.top_k)

    return create_sql_agent(
        llm=llm,
        db=db,
        agent_type="zero-shot-react-description",
        prefix=prefix,
        top_k=cfg.top_k,
        max_iterations=cfg.max_iterations,
        verbose=cfg.verbose,
        agent_executor_kwargs={
            "handle_parsing_errors": REACT_FORMAT_REMINDER,
            "return_intermediate_steps": True,
        },
    )


def _agent_cache_key(
    *,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    config: LangChainSQLAgentConfig,
) -> tuple[int, int, int, LangChainSQLAgentConfig]:
    return (id(engine), id(model), id(tokenizer), config)


def get_cached_langchain_sql_agent(
    *,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    config: LangChainSQLAgentConfig | None = None,
) -> Any:
    cfg = config or LangChainSQLAgentConfig()
    cache_key = _agent_cache_key(
        engine=engine,
        model=model,
        tokenizer=tokenizer,
        config=cfg,
    )
    agent = _AGENT_CACHE.get(cache_key)
    if agent is None:
        agent = build_langchain_sql_agent(
            engine=engine,
            model=model,
            tokenizer=tokenizer,
            config=cfg,
        )
        _AGENT_CACHE[cache_key] = agent
    return agent


def clear_langchain_sql_agent_cache() -> None:
    _AGENT_CACHE.clear()


def run_langchain_sql_agent(
    *,
    nlq: str,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    config: LangChainSQLAgentConfig | None = None,
) -> dict[str, Any]:
    """
    Run one NL question through the optional LangChain SQL agent.
    """
    agent = get_cached_langchain_sql_agent(
        engine=engine,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    result = agent.invoke({"input": nlq})
    return {
        "output": result.get("output"),
        "intermediate_steps": result.get("intermediate_steps", []),
        "raw_result": result,
    }


def _action_sql_candidate(action: Any) -> str | None:
    tool_name = str(getattr(action, "tool", "") or "")
    if tool_name not in {"sql_db_query", "sql_db_query_checker"}:
        tool_log = str(getattr(action, "log", "") or "")
        return _clean_sql_text(tool_log)

    tool_input = getattr(action, "tool_input", None)
    if isinstance(tool_input, str):
        sql = _clean_sql_text(tool_input)
        if sql:
            return sql
    if isinstance(tool_input, dict):
        for key in ("query", "sql", "input"):
            value = tool_input.get(key)
            if isinstance(value, str) and value.strip():
                sql = _clean_sql_text(value)
                if sql:
                    return sql
    tool_log = str(getattr(action, "log", "") or "")
    return _clean_sql_text(tool_log)


def extract_sql_from_intermediate_steps(intermediate_steps: list[Any]) -> str | None:
    sql_candidates: list[str] = []
    for step in intermediate_steps or []:
        action = step[0] if isinstance(step, tuple) and step else step
        sql_candidate = _action_sql_candidate(action)
        if sql_candidate:
            sql_candidates.append(sql_candidate)
    return sql_candidates[-1] if sql_candidates else None


def extract_sql_from_langchain_result(result: dict[str, Any]) -> str | None:
    sql_from_steps = extract_sql_from_intermediate_steps(result.get("intermediate_steps", []))
    if sql_from_steps:
        return sql_from_steps

    output = str(result.get("output") or "").strip()
    return _clean_sql_text(output)


def predict_sql_with_langchain_agent(
    *,
    nlq: str,
    engine: Engine,
    model: Any,
    tokenizer: Any,
    config: LangChainSQLAgentConfig | None = None,
) -> tuple[str | None, dict[str, Any]]:
    result = run_langchain_sql_agent(
        nlq=nlq,
        engine=engine,
        model=model,
        tokenizer=tokenizer,
        config=config,
    )
    return extract_sql_from_langchain_result(result), result


__all__ = [
    "DEFAULT_SQL_AGENT_SYSTEM_PROMPT",
    "LangChainSQLAgentConfig",
    "build_langchain_sql_agent",
    "clear_langchain_sql_agent_cache",
    "extract_sql_from_intermediate_steps",
    "extract_sql_from_langchain_result",
    "get_cached_langchain_sql_agent",
    "predict_sql_with_langchain_agent",
    "run_langchain_sql_agent",
]
