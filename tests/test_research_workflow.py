import unittest

from evolveai.autonomy import GoalStatus
from evolveai.research_workflow import run_research


def source(title, url, excerpt):
    return {"title": title, "url": url, "excerpt": excerpt, "confidence": 0.7}


class AutonomousResearchWorkflowTests(unittest.TestCase):
    def test_researches_and_builds_report_from_two_domains(self):
        calls = []
        events = []

        def fake_search(query, limit):
            calls.append((query, limit))
            return [
                source("Source A", "https://alpha.example/article",
                       "This source describes the research topic with enough detail for evidence collection."),
                source("Source B", "https://beta.example/report",
                       "This independent source provides additional context about the same research topic."),
            ]

        result = run_research("test research topic", search=fake_search, event_sink=events.append)

        self.assertEqual(result["status"], GoalStatus.SUCCEEDED.value)
        self.assertEqual(len(calls), 1)
        self.assertIn("Source A", result["report"])
        self.assertIn("Source B", result["report"])
        self.assertEqual(result["verification"]["source_count"], 2)
        self.assertTrue(any(event.event == "goal_verified" for event in events))

    def test_refuses_to_call_single_domain_coverage_verified(self):
        def fake_search(query, limit):
            return [
                source("Source A", "https://same.example/a",
                       "This source contains a sufficiently long excerpt about the topic."),
                source("Source B", "https://same.example/b",
                       "This second page is still hosted on the same source domain."),
            ]

        result = run_research("test research topic", search=fake_search)
        self.assertEqual(result["status"], GoalStatus.BLOCKED.value)
        self.assertEqual(result["report"], "")
        self.assertIn("Insufficient independent-source coverage", result["last_error"])

    def test_empty_search_does_not_claim_success(self):
        result = run_research("missing topic", search=lambda query, limit: [])
        self.assertEqual(result["status"], GoalStatus.BLOCKED.value)
        self.assertEqual(result["report"], "")

    def test_rejects_empty_query(self):
        with self.assertRaises(ValueError):
            run_research("  ", search=lambda query, limit: [])


if __name__ == "__main__":
    unittest.main()
