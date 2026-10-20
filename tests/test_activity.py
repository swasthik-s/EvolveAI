from evolveai.activity import ActivityJournal


def test_activity_journal_records_filters_and_details(tmp_path):
    journal = ActivityJournal(tmp_path / "activity.db")
    event_id = journal.record(
        "learning", "lesson_candidate", "proposed", "Check a candidate lesson",
        task_id="task-123", details={"source": "feedback", "score": 0.9},
    )
    journal.record("system", "worker_failed", "failed", "Worker unavailable")

    events = journal.recent(limit=10, category="learning", status="proposed")
    assert len(events) == 1
    assert events[0]["id"] == event_id
    assert events[0]["task_id"] == "task-123"
    assert events[0]["details"]["score"] == 0.9
    assert journal.stats()["events"] == 2
    assert journal.recent(query="unavailable")[0]["status"] == "failed"
