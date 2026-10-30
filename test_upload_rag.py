import unittest

from upload_rag import assess_construction_relevance


class ConstructionRelevanceTests(unittest.TestCase):
    def test_construction_safety_document_is_accepted(self):
        text = (
            "Construction site safety plan for scaffolding and excavation. "
            "The hazard assessment identifies fall protection, inspections, PPE, "
            "and safe work procedures for the building site. " * 4
        )
        accepted, _ = assess_construction_relevance(text)
        self.assertTrue(accepted)

    def test_unrelated_cricket_document_is_rejected(self):
        text = (
            "Cricket match report with batting, bowling, fielding, scores and players. "
            "The team reviewed performance and training plans for the next match. " * 5
        )
        accepted, _ = assess_construction_relevance(text)
        self.assertFalse(accepted)

    def test_generic_safety_document_without_construction_context_is_rejected(self):
        text = (
            "Workplace safety policy covering hazards, employee training, PPE, "
            "incident reporting, emergency procedures and inspections. " * 5
        )
        accepted, _ = assess_construction_relevance(text)
        self.assertFalse(accepted)

    def test_construction_terms_without_safety_or_task_context_is_rejected(self):
        text = (
            "Construction project overview describing concrete, masonry, roofing, "
            "scaffolding, excavation, structural steel and building materials. " * 5
        )
        accepted, _ = assess_construction_relevance(text)
        self.assertFalse(accepted)


if __name__ == "__main__":
    unittest.main()
