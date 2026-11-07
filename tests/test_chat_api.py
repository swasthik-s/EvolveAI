import json

from fastapi.testclient import TestClient

import api.chat as chat


def _reset(tmp_path, monkeypatch):
    monkeypatch.setattr(chat, "DB_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setattr(chat, "STREAM_DELAY_S", 0)
    chat._agent = None
    chat._memory = None
    chat._activity = None


def _events(body: str):
    parsed = []
    for block in body.split("\n\n"):
        if not block.strip():
            continue
        event = "message"
        data = ""
        for line in block.splitlines():
            if line.startswith("event:"):
                event = line.split(":", 1)[1].strip()
            elif line.startswith("data:"):
                data += line.split(":", 1)[1].strip()
        parsed.append((event, json.loads(data)))
    return parsed


def test_health_and_json_chat_stay_compatible(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    client = TestClient(chat.app)

    health = client.get("/api/chat")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    response = client.post("/api/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert "Hey" in body["answer"]
    assert body["task_id"]
    assert body["reasoning_plan"]["intent"] == "greeting"
    assert "attachments" not in body


def test_stream_emits_stages_chunks_and_done(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    client = TestClient(chat.app)
    response = client.post("/api/chat", json={"message": "what can you do?", "stream": True})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = _events(response.text)
    kinds = [kind for kind, _ in events]
    assert "status" in kinds
    assert "delta" in kinds
    assert kinds[-1] == "done"
    answer = "".join(data["text"] for kind, data in events if kind == "delta")
    done = events[-1][1]
    assert answer == done["answer"]
    assert "Remember facts" in done["answer"]
    assert done["reasoning_plan"]["steps"]


def test_attachment_is_acknowledged_and_journaled(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    client = TestClient(chat.app)
    response = client.post("/api/chat", json={
        "message": "hello",
        "attachments": [{
            "name": "notes.txt",
            "type": "text/plain",
            "size": 11,
            "encoding": "text",
            "content": "hello world",
        }],
    })
    assert response.status_code == 200
    body = response.json()
    stored = body["attachments"][0]
    assert stored["stored"] is True
    assert stored["name"] == "notes.txt"
    assert stored["content_policy"] == "text_excerpt"
    assert stored["sha256"]
    assert "notes.txt" in body["answer"]
    assert "hello world" in body["answer"]

    journal = chat.get_journal()
    events = journal.recent(limit=20, query="notes.txt")
    assert any(event["event"] == "attachment_received" for event in events)


def test_image_attachment_is_metadata_only(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    client = TestClient(chat.app)
    response = client.post("/api/chat", json={
        "message": "hello",
        "attachments": [{
            "name": "pixel.png",
            "type": "image/png",
            "size": 12,
            "encoding": "base64",
            "content": "data:image/png;base64,aaaa",
        }],
    })
    body = response.json()
    assert body["attachments"][0]["content_policy"] == "binary_metadata"
    assert "does not interpret" in body["answer"]


def test_admin_and_feedback_contracts_remain(tmp_path, monkeypatch):
    _reset(tmp_path, monkeypatch)
    client = TestClient(chat.app)
    assert client.post("/api/chat", json={"action": "admin_overview"}).status_code == 503
    assert client.post("/api/chat", json={"action": "feedback"}).status_code == 422
    assert client.post("/api/chat", json={"message": "", "stream": True}).status_code == 422
    # Streaming is ignored for admin actions; they still return JSON errors.
    admin_stream = client.post("/api/chat", json={"action": "admin_overview", "stream": True})
    assert admin_stream.status_code == 503
    assert admin_stream.headers["content-type"].startswith("application/json")
