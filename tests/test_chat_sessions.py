import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).parent.parent / "src").resolve()))

import types

# Minimal provider stubs for environments without optional deps
if "openai" not in sys.modules:
    mock_openai = types.ModuleType("openai")
    mock_openai.OpenAI = object
    sys.modules["openai"] = mock_openai
if "ollama" not in sys.modules:
    mock_ollama = types.ModuleType("ollama")
    mock_ollama.Client = object
    sys.modules["ollama"] = mock_ollama

from backend.services.llm.service import LLMService
from backend.services.logging.storage import (
    create_chat_session,
    create_log_database,
    delete_chat_session,
    ensure_default_chat_session,
    fetch_chat_session_messages,
    list_chat_sessions,
    save_log_record,
)


def _insert(session_id: int, origin: str, message: str, turn_id: str, created_at: float) -> None:
    save_log_record(
        created_at=created_at,
        level=20,
        logger_name=origin,
        origin=origin,
        message=message,
        formatted=message,
        session_id=session_id,
        turn_id=turn_id,
    )


def test_create_list_delete_sessions(tmp_path):
    create_log_database(tmp_path / "history.db")
    default_id = ensure_default_chat_session()
    second_id = create_chat_session("Experiment")

    sessions = list_chat_sessions()
    session_ids = [int(s["id"]) for s in sessions]
    assert default_id in session_ids
    assert second_id in session_ids

    delete_chat_session(second_id)
    sessions_after = list_chat_sessions()
    session_ids_after = [int(s["id"]) for s in sessions_after]
    assert second_id not in session_ids_after


def test_messages_persist_per_session_and_trim_enabled_by_default(tmp_path):
    create_log_database(tmp_path / "history.db")
    session_a = create_chat_session("A")
    session_b = create_chat_session("B")

    for i in range(10):
        _insert(session_a, "chat", f"user {i}", f"turn-{i}", 1000 + i)
        _insert(session_a, "llm", f"assistant {i}", f"turn-{i}", 1000 + i + 0.1)

    _insert(session_b, "chat", "other", "turn-1", 2000)
    _insert(session_b, "llm", "other reply", "turn-1", 2000.1)
    _insert(session_a, "llm_thinking", "internal reasoning", "turn-9", 2001)

    messages_a = fetch_chat_session_messages(session_a)
    assert all(int(row["session_id"]) == session_a for row in messages_a)
    assert len(messages_a) == 21

    service = LLMService()
    payload = service.build_session_payload(prompt="new question", session_id=session_a, context="system")

    assert payload[0]["role"] == "system"
    assert payload[0]["content"] == "system"
    # Keep-last-turns default is 8; with a new user prompt, oldest turns are trimmed.
    contents = [msg["content"] for msg in payload if msg["role"] != "system"]
    assert "user 0" not in contents
    assert "assistant 0" not in contents
    assert "internal reasoning" not in contents
    assert contents[-1] == "new question"


def test_list_models_dispatches_to_provider():
    class _Provider:
        name = "mock"

        def stream_chat(self, messages, **kwargs):
            return iter(())

        def list_models(self, **kwargs):
            return ["beta", "alpha", "beta", " "]

    service = LLMService()
    service.register_provider(_Provider())
    models = service.list_models(provider="mock")
    assert models == ["alpha", "beta"]
