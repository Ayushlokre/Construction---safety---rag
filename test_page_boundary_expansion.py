import unittest
from page_context import expand_page_boundary_items


class PageBoundaryExpansionTests(unittest.TestCase):
    def setUp(self):
        self.ids = ["p10_c1", "p10_c2", "p11_c1", "p11_c2", "other_p10_c2"]
        self.docs = [
            "Page 10 opening text",
            "Page 10 final sentence continues",
            "continued sentence from page 10 on page 11",
            "Page 11 final paragraph",
            "Different PDF page 10 ending",
        ]
        self.metadata = [
            {"source": "osha_a.pdf", "page": 10, "chunk": 1},
            {"source": "osha_a.pdf", "page": 10, "chunk": 2},
            {"source": "osha_a.pdf", "page": 11, "chunk": 1},
            {"source": "osha_a.pdf", "page": 11, "chunk": 2},
            {"source": "osha_b.pdf", "page": 10, "chunk": 2},
        ]

    def item(self, index):
        return {
            "id": self.ids[index],
            "text": self.docs[index],
            "metadata": self.metadata[index],
        }

    def test_last_chunk_adds_first_chunk_of_next_page_same_pdf(self):
        result = expand_page_boundary_items(
            [self.item(1)], self.ids, self.docs, self.metadata
        )
        self.assertEqual(
            [item["id"] for item in result],
            ["p10_c2", "p11_c1"],
        )

    def test_first_chunk_adds_last_chunk_of_previous_page_same_pdf(self):
        result = expand_page_boundary_items(
            [self.item(2)], self.ids, self.docs, self.metadata
        )
        self.assertEqual(
            [item["id"] for item in result],
            ["p10_c2", "p11_c1"],
        )

    def test_does_not_cross_between_different_pdfs(self):
        result = expand_page_boundary_items(
            [self.item(4)], self.ids, self.docs, self.metadata
        )
        self.assertEqual([item["id"] for item in result], ["other_p10_c2"])

    def test_preserves_page_and_chunk_metadata(self):
        result = expand_page_boundary_items(
            [self.item(1)], self.ids, self.docs, self.metadata
        )
        page_metadata = [
            (item["metadata"]["source"], item["metadata"]["page"], item["metadata"]["chunk"])
            for item in result
        ]
        self.assertEqual(
            page_metadata,
            [("osha_a.pdf", 10, 2), ("osha_a.pdf", 11, 1)],
        )


if __name__ == "__main__":
    unittest.main()
