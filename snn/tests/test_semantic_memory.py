import tempfile, unittest
from brain.semantic_memory import SemanticMemory
class SemanticMemoryTests(unittest.TestCase):
    def test_search_provenance_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=SemanticMemory(tmp)
            self.assertTrue(m.add_document("Photosynthesis converts light energy into chemical energy in plants.",["biology"],"science"))
            self.assertTrue(m.search("photosynthesis chemical energy"))
            self.assertEqual(m.search("photosynthesis")[0]["source_ids"],["biology"])
            m.save(); self.assertEqual(len(SemanticMemory(tmp).documents),1)
    def test_duplicate_and_cooccurrence_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            m=SemanticMemory(tmp); text="Neurons communicate through electrical signals in biological networks."
            self.assertTrue(m.add_document(text,["paper"],"biology"))
            self.assertFalse(m.add_document(text,["paper"],"biology"))
            self.assertTrue(all(e["relation"]=="co_occurs_in_source" for e in m.edges.values()))
if __name__=="__main__": unittest.main()
