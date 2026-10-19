from evolveai.memory import ExperienceMemory


def test_task_rating_and_lesson(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "test.db"))
    task_id = memory.save_task("validate CSV", "done", "demo")
    assert memory.rate_task(task_id, 1.0, "good")
    assert memory.stats()["tasks"] == 1
    assert memory.stats()["rated_tasks"] == 1
    lessons = memory.search_lessons("CSV")
    assert lessons
    assert "validate CSV" in lessons[0]["task"]


def test_unknown_rating_returns_false(tmp_path):
    memory = ExperienceMemory(str(tmp_path / "test.db"))
    assert not memory.rate_task("missing", 1.0)
