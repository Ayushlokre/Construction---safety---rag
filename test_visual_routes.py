import unittest
from visual_routes import known_source_visual_pages

class VisualRoutesTests(unittest.TestCase):
    def test_osha_3071_table_pages(self):
        result = known_source_visual_pages("Show me the Common Hazards and Descriptions table from OSHA 3071.")
        self.assertEqual(result["source"], "OSHA_3071_Job_Hazard_Analysis.pdf")
        self.assertEqual(result["pages"], [47, 48])

    def test_osha_3886_hierarchy_page(self):
        result = known_source_visual_pages("Show me the Hierarchy of Controls diagram from OSHA 3886.")
        self.assertEqual(result["source"], "OSHA_3886_Construction_Safety.pdf")
        self.assertEqual(result["pages"], [25])

    def test_unknown_visual_not_routed(self):
        self.assertIsNone(known_source_visual_pages("Show me a crane diagram."))

if __name__ == "__main__":
    unittest.main()
