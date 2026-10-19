from evolveai.thinking import process_evidence


def test_process_evidence_returns_concise_answer_and_sources():
    evidence = [
        {
            "title": "Egg definition",
            "url": "https://example.org/egg",
            "source_domain": "example.org",
            "excerpt": (
                "An egg is an organic vessel produced by an animal to carry a developing embryo. "
                "The embryo develops inside the egg until it can survive on its own. "
                "Some eggs are also eaten as food by humans."
            ),
        },
        {
            "title": "Eggs as food",
            "url": "https://food.example.org/eggs",
            "source_domain": "food.example.org",
            "excerpt": "Eggs are a nutritious food commonly eaten by humans around the world.",
        },
    ]

    result = process_evidence("What is an egg?", evidence)

    assert "An egg is an organic vessel" in result["answer"]
    assert len(result["answer"]) < 1100
    assert result["sources"]
    assert result["sources"][0]["url"] == "https://example.org/egg"


def test_process_evidence_reports_when_no_relevant_text_exists():
    result = process_evidence(
        "What is an egg?",
        [{
            "title": "Random page",
            "url": "https://example.org/random",
            "excerpt": "This page discusses mountain roads and long distance travel.",
        }],
    )
    assert "couldn't extract enough relevant evidence" in result["answer"]
