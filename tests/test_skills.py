from evolveai.skills import SkillLibrary


def test_skill_lifecycle(tmp_path):
    library = SkillLibrary(str(tmp_path / "skills"))
    item = library.create_draft("CSV helper", "Validate headers", "def check(x): return x", "test")
    assert item["status"] == "draft"
    assert library.count_approved() == 0
    assert library.set_status(item["id"], "approved")
    assert library.count_approved() == 1
    assert not library.set_status("unknown", "approved")
