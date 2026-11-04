import asyncio
import tempfile
import unittest
from pathlib import Path

from evolveai.parallel_data_engine import (
    ConnectorRegistry, ExperienceStore, ParallelDataEngine, SourceRecord,
)


class FakeConnector:
    def __init__(self, source_id, count=8):
        self.source_id = source_id
        self.count = count

    async def stream(self):
        for i in range(self.count):
            yield SourceRecord(
                content=f"Source {self.source_id} record {i} contains enough meaningful text for ingestion.",
                source_id=self.source_id,
                source_uri=f"https://example.test/{self.source_id}/{i}",
                domain="test",
                license="test-only",
            )


class ParallelDataEngineTests(unittest.TestCase):
    def test_many_connectors_run_through_bounded_pipeline(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                registry = ConnectorRegistry()
                for i in range(100):
                    registry.register(FakeConnector(f"source-{i}", count=2))
                engine = ParallelDataEngine(
                    registry,
                    store=ExperienceStore(str(Path(tmp) / "pde.db")),
                    queue_size=7,
                    dedupe_window=500,
                )
                status = await engine.run()
                self.assertEqual(status["sources_started"], 100)
                self.assertEqual(status["sources_completed"], 100)
                self.assertEqual(status["records_received"], 200)
                self.assertLessEqual(status["max_queue_depth"], 7)
                self.assertEqual(status["store"]["records"], 200)
        asyncio.run(scenario())

    def test_duplicate_records_are_suppressed(self):
        class DuplicateConnector:
            source_id = "repeat"
            async def stream(self):
                record = SourceRecord(
                    content="This same record contains enough meaningful information to pass normalization.",
                    source_id="repeat",
                    source_uri="https://example.test/repeat",
                )
                yield record
                yield record

        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                registry = ConnectorRegistry()
                registry.register(DuplicateConnector())
                engine = ParallelDataEngine(
                    registry,
                    store=ExperienceStore(str(Path(tmp) / "pde.db")),
                )
                status = await engine.run()
                self.assertEqual(status["records_duplicate"], 1)
                self.assertEqual(status["store"]["records"], 1)
        asyncio.run(scenario())

    def test_learner_receives_evaluated_records(self):
        async def scenario():
            with tempfile.TemporaryDirectory() as tmp:
                registry = ConnectorRegistry()
                registry.register(FakeConnector("learn-source", count=3))
                received = []
                async def learner(record, evaluation):
                    received.append((record.source_id, evaluation["decision"]))
                engine = ParallelDataEngine(
                    registry,
                    store=ExperienceStore(str(Path(tmp) / "pde.db")),
                    learner=learner,
                    learn_decisions=("learn_now", "investigate", "compare", "retain"),
                )
                status = await engine.run()
                self.assertEqual(status["learner_handoffs"], len(received))
                self.assertTrue(received)
        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()
