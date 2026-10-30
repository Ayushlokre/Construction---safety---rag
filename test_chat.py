"""Regression tests for the local OSHA RAG chatbot.

Run from the project folder with: python -m unittest -v test_chat.py
These tests use stub modules only when ChromaDB/Ollama are unavailable; they do not
connect to Ollama or read/write the persistent ChromaDB database.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
CHAT_PATH = PROJECT_DIR / "chat.py"

# Keep these unit tests runnable in environments without the application packages.
if "chromadb" not in sys.modules:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        chromadb_stub = types.ModuleType("chromadb")
        chromadb_stub.PersistentClient = object
        sys.modules["chromadb"] = chromadb_stub

if "ollama" not in sys.modules:
    try:
        import ollama  # noqa: F401
    except ImportError:
        ollama_stub = types.ModuleType("ollama")
        ollama_stub.embed = lambda **kwargs: None
        ollama_stub.chat = lambda **kwargs: None
        sys.modules["ollama"] = ollama_stub

spec = importlib.util.spec_from_file_location("osha_chat_under_test", CHAT_PATH)
chat = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chat
assert spec.loader is not None
spec.loader.exec_module(chat)


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def count(self):
        return len(self.records["ids"])

    def get(self, include=None):
        return self.records


class RAGRegressionTests(unittest.TestCase):
    def test_reciprocal_rank_fusion_rewards_agreement_between_retrievers(self):
        both_ranked = chat.reciprocal_rank_fusion_score({0: 0, 1: 2}, 0, 2)
        semantic_only = chat.reciprocal_rank_fusion_score({0: 0}, None, 2)
        lexical_only = chat.reciprocal_rank_fusion_score({}, 0, 2)
        self.assertGreater(both_ranked, semantic_only)
        self.assertGreater(both_ranked, lexical_only)
        self.assertGreaterEqual(both_ranked, 0.0)
        self.assertLessEqual(both_ranked, 1.0)

    def test_osha_assistance_question_routes_to_relevant_section(self):
        routes = chat.detect_targeted_routes(
            "What kinds of assistance does OSHA describe for improving workplace safety and health programs?"
        )
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 21),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 22),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 23),
            },
        )

    def test_jha_appendix_navigation_routes_to_reference_page(self):
        routes = chat.detect_targeted_routes(
            "Where can a reader find additional hazard-control measures in the JHA booklet?"
        )
        self.assertIn(
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 18),
            {(route["source"], route["page"]) for route in routes},
        )

    def test_rotating_pulley_query_keeps_both_evidence_pages(self):
        routes = chat.detect_targeted_routes(
            "A worker reaches near a rotating pulley while clearing a snag. What kind of hazard scenario does the JHA booklet describe?"
        )
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 13),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 14),
            },
        )

    def test_reciprocal_rank_fusion_handles_empty_ranks(self):
        self.assertEqual(chat.reciprocal_rank_fusion_score({}, None, 2), 0.0)

    def test_five_jha_questions_route_to_pdf_page_12(self):
        routes = chat.detect_targeted_routes("What are the five questions a JHA should ask?")
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0]["source"], "OSHA_3071_Job_Hazard_Analysis.pdf")
        self.assertEqual(routes[0]["page"], 12)

        answer = chat.answer_jha_hazard_identification_questions(
            "What are the five questions a JHA should ask?"
        )
        self.assertIsNotNone(answer)
        self.assertEqual(answer.count("[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 12]"), 5)
        self.assertIn("What can go wrong?", answer)
        self.assertIn("How likely is it that the hazard will occur?", answer)

    def test_natural_jha_questions_query_retrieves_the_cited_page(self):
        question = "What are the five questions to ask when conducting a JHA?"
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {("OSHA_3071_Job_Hazard_Analysis.pdf", 12)},
        )
        records = {
            "ids": ["jha-page-12", "irrelevant-page-22"],
            "documents": [
                "What can go wrong? What are the consequences? How could it arise? What are other contributing factors? How likely is it that the hazard will occur?",
                "Incident investigations ask why equipment failed and whether workers were trained.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 12, "chunk": 0},
                {"source": "OSHA_3886_Construction_Safety.pdf", "page": 22, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        self.assertIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 12", context)
        self.assertNotIn("PDF page 22", context)

    def test_jha_definition_importance_retrieves_both_cited_pages(self):
        question = "What is a Job Hazard Analysis and why is it important?"
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 1),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 2),
            },
        )
        records = {
            "ids": ["jha-definition", "jha-importance", "irrelevant"],
            "documents": [
                "A job hazard analysis focuses on job tasks to identify hazards before they occur.",
                "A JHA helps establish proper work procedures, prevent injuries, and improve work methods.",
                "An unrelated consultation office directory.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 1, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 2, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 42, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        self.assertIn("PDF page 1", context)
        self.assertIn("PDF page 2", context)
        self.assertNotIn("PDF page 42", context)
        answer = chat.generate_answer(question, context)
        self.assertIn("PDF page 1", answer)
        self.assertIn("PDF page 2", answer)

    def test_natural_dropped_casting_question_routes_to_page_15_and_answers(self):
        question = "What injury can happen if a worker drops a casting?"
        routes = chat.detect_targeted_routes(question)
        self.assertIn(
            ("OSHA_3071_Job_Hazard_Analysis.pdf", 15),
            {(route["source"], route["page"]) for route in routes},
        )
        records = {
            "ids": ["jha-page-15", "unrelated"],
            "documents": [
                "Picking up a casting, the employee could drop it onto his foot. "
                "The casting's weight and height could seriously injure the worker's foot or toes. "
                "Wear steel-toe shoes with arch protection.",
                "OSHA consultation program directory.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 15, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 24, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        self.assertIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 15", context)
        answer = chat.generate_answer(question, context)
        self.assertNotEqual(
            answer,
            "The available documents do not provide enough specific information to answer this question safely.",
        )
        self.assertIn("foot", answer.lower())
        self.assertIn("PDF page 15", answer)

    def test_broad_casting_question_retrieves_and_summarizes_all_three_hazards(self):
        question = "What are the hazards associated with casting?"
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 15),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 16),
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 17),
            },
        )
        records = {
            "ids": ["casting-15", "casting-16", "casting-17", "unrelated"],
            "documents": [
                "A worker could drop a casting onto the foot or toes. Use steel-toe shoes and a handling device.",
                "Castings have sharp burrs and edges that can cause severe lacerations. Use a clamp and cut-resistant gloves.",
                "Reaching, twisting, and lifting castings from the floor could cause lower-back muscle strain.",
                "General construction chemical hazards.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 15, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 16, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 17, "chunk": 0},
                {"source": "OSHA_3886_Construction_Safety.pdf", "page": 21, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        self.assertIn("PDF page 15", context)
        self.assertIn("PDF page 16", context)
        self.assertIn("PDF page 17", context)
        self.assertNotIn("PDF page 21", context)
        answer = chat.generate_answer(question, context)
        self.assertIn("Foot injury", answer)
        self.assertIn("lacerations", answer)
        self.assertIn("lower-back muscle strain", answer.lower())
        self.assertIn("PDF page 15", answer)
        self.assertIn("PDF page 16", answer)
        self.assertIn("PDF page 17", answer)

    def test_document_comparison_retrieves_both_documents(self):
        question = "How does the construction guide differ from the JHA booklet?"
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 1),
                ("OSHA_3886_Construction_Safety.pdf", 11),
            },
        )
        context = (
            "[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 1]\nJob Hazard Analysis\n\n"
            "[Source: OSHA_3886_Construction_Safety.pdf, PDF page 11]\nCore elements of safety programs"
        )
        answer = chat.answer_document_comparison(question, context)
        self.assertIn("company-wide construction safety", answer)
        self.assertIn("task-focused method", answer)
        self.assertIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 1", answer)
        self.assertIn("OSHA_3886_Construction_Safety.pdf, PDF page 11", answer)

    def test_natural_document_comparison_wording_routes_and_answers(self):
        question = "Compare the purpose of the two OSHA documents."
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {
                ("OSHA_3071_Job_Hazard_Analysis.pdf", 1),
                ("OSHA_3886_Construction_Safety.pdf", 11),
            },
        )
        records = {
            "ids": ["jha", "construction", "other"],
            "documents": [
                "A job hazard analysis focuses on individual job tasks to identify hazards before they occur.",
                "Core elements include management leadership, worker participation, hazard identification, prevention and control, training, evaluation, and coordination for construction safety programs.",
                "An unrelated consultation directory.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 1, "chunk": 0},
                {"source": "OSHA_3886_Construction_Safety.pdf", "page": 11, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 24, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        self.assertIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 1", context)
        self.assertIn("OSHA_3886_Construction_Safety.pdf, PDF page 11", context)
        answer = chat.answer_document_comparison(question, context)
        self.assertIsNotNone(answer)
        self.assertIn("company-wide construction safety", answer)
        self.assertIn("task-focused method", answer)
        # Exercise the same answer dispatcher used by the CLI and Streamlit UI.
        dispatched_answer = chat.generate_answer(question, context)
        self.assertIn("company-wide construction safety", dispatched_answer)
        self.assertIn("task-focused method", dispatched_answer)

    def test_chemical_route_excludes_irrelevant_directory_page_42(self):
        routes = chat.detect_targeted_routes("What are the controls for chemical exposure?")
        route_keys = {(route["source"], route["page"]) for route in routes}
        self.assertIn(("OSHA_3886_Construction_Safety.pdf", 21), route_keys)
        self.assertIn(("OSHA_3071_Job_Hazard_Analysis.pdf", 46), route_keys)
        self.assertIn(("OSHA_3071_Job_Hazard_Analysis.pdf", 47), route_keys)
        self.assertNotIn(("OSHA_3071_Job_Hazard_Analysis.pdf", 42), route_keys)

        records = {
            "ids": ["directory", "controls", "chemicals"],
            "documents": [
                "OSHA Consultation Projects Anchorage Arizona regional office directory.",
                "Administrative controls include written operating procedures and safe work practices. PPE may be used when engineering controls are not feasible.",
                "Identify chemical hazards by reviewing SDSs and product labels. Consider exposure limits and ventilation.",
            ],
            "metadatas": [
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 42, "chunk": 0},
                {"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 46, "chunk": 0},
                {"source": "OSHA_3886_Construction_Safety.pdf", "page": 21, "chunk": 0},
            ],
        }
        context = chat.retrieve_documents("What are the controls for chemical exposure?", FakeCollection(records))
        self.assertNotIn("PDF page 42", context)
        self.assertIn("PDF page 46", context)
        self.assertIn("PDF page 21", context)

    def test_chemical_answer_cites_actual_pdf_page(self):
        question = "What are the controls for chemical exposure?"
        context = (
            "[Source: OSHA_3886_Construction_Safety.pdf, PDF page 21]\n"
            "chemical hazards; review SDSs and product labels.\n\n"
            "[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 46]\n"
            "Administrative controls include written operating procedures and safe work practices. "
            "PPE is acceptable when engineering controls are not feasible."
        )
        answer = chat.answer_chemical_exposure_guidance(question, context)
        self.assertIsNotNone(answer)
        self.assertIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 46", answer)
        self.assertNotIn("OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 42", answer)


    def test_natural_chemical_control_question_uses_available_page_47(self):
        question = "How should chemical exposure hazards be controlled?"
        context = (
            "[Source: OSHA_3886_Construction_Safety.pdf, PDF page 21]\n"
            "Identify chemical hazards by reviewing SDSs and product labels. "
            "Consider exposure limits, volatility, quantity, ventilation, and skin exposure.\\n\n"
            "[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 47]\n"
            "Administrative controls include written operating procedures and safe work practices. "
            "PPE may be used when engineering controls are not feasible."
        )
        answer = chat.answer_chemical_exposure_guidance(question, context)
        self.assertIsNotNone(answer)
        self.assertIn("reviewing the Safety Data Sheets (SDSs) and product labels", answer)
        self.assertIn("PDF page 47", answer)
        self.assertIn("PDF page 21", answer)


    def test_real_world_five_hazard_questions_wording_routes_to_page_12(self):
        question = "What five questions help identify workplace hazards during a JHA?"
        routes = chat.detect_targeted_routes(question)
        self.assertEqual(
            {(route["source"], route["page"]) for route in routes},
            {("OSHA_3071_Job_Hazard_Analysis.pdf", 12)},
        )
        records = {
            "ids": ["jha-page-12"],
            "documents": ["What can go wrong? What are the consequences? How could it arise? What are other contributing factors? How likely is it that the hazard will occur?"],
            "metadatas": [{"source": "OSHA_3071_Job_Hazard_Analysis.pdf", "page": 12, "chunk": 0}],
        }
        context = chat.retrieve_documents(question, FakeCollection(records))
        answer = chat.generate_answer(question, context)
        self.assertIn("What can go wrong?", answer)
        self.assertEqual(answer.count("PDF page 12"), 5)

    def test_jha_question_answer_does_not_cite_missing_page_in_production_path(self):
        question = "What are the five questions a JHA should ask?"
        context = "[Source: OSHA_3886_Construction_Safety.pdf, PDF page 11]\nGeneral program elements."
        answer = chat.generate_answer(question, context)
        self.assertEqual(
            answer,
            "The available documents do not provide enough specific information to answer this question safely.",
        )

    def test_reranker_parser_handles_json_fence_and_nested_text(self):
        parsed = chat._parse_reranker_json('```json\n{"ranked_ids":[{"id":"0","score":0.9}]}\n```')
        self.assertEqual(parsed["ranked_ids"][0]["id"], "0")

    def test_reranker_parser_rejects_truncated_json(self):
        with self.assertRaises(ValueError):
            chat._parse_reranker_json('{"ranked_ids":[{"id":"0","score":0.9}')

    def test_rare_query_terms_outweigh_generic_terms(self):
        docs = [
            "Administrative controls include general procedures and controls.",
            "Respirator selection depends on the identified chemical exposure hazard.",
        ]
        df = {"controls": 30, "chemical": 2, "exposure": 4, "respirator": 1, "selection": 2}
        generic_score = chat.weighted_keyword_score("chemical exposure controls", docs[0], df, 40)
        specific_score = chat.weighted_keyword_score("chemical exposure controls", docs[1], df, 40)
        self.assertGreater(specific_score, generic_score)

    def test_unsubstantiated_specific_respirator_question_abstains(self):
        answer = chat.generate_answer(
            "What respirator is required for a particular chemical if the documents don't specify one?",
            "[Source: OSHA_3886_Construction_Safety.pdf, PDF page 4]\nGeneral construction safety program information.",
        )
        self.assertEqual(
            answer,
            "The available documents do not provide enough specific information to answer this question safely.",
        )

    def test_query_expansion_adds_domain_terms_and_preserves_original(self):
        variants = chat.expand_query("How should chemical exposure be controlled?")
        self.assertGreaterEqual(len(variants), 2)
        self.assertEqual(variants[0], "How should chemical exposure be controlled?")
        self.assertIn("SDS", variants[1])
        self.assertIn("ventilation", variants[1])

    def test_citation_guard_rejects_page_not_present_in_context(self):
        context = "[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 12]\nWhat can go wrong?"
        fabricated = "The guide says to inspect the site. [Source: OSHA_3886_Construction_Safety.pdf, PDF page 99]"
        answer = chat.validate_answer_citations(fabricated, context)
        self.assertEqual(
            answer,
            "The available documents do not provide enough specific information to answer this question safely.",
        )

    def test_citation_guard_accepts_supported_source_page(self):
        context = "[Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 12]\nWhat can go wrong?"
        answer = "What can go wrong? [Source: OSHA_3071_Job_Hazard_Analysis.pdf, PDF page 12]"
        self.assertEqual(chat.validate_answer_citations(answer, context), answer)

    def test_reranker_falls_back_to_hybrid_order_when_local_model_fails(self):
        candidates = [
            {"id": "a", "combined_score": 0.9, "metadata": {}, "text": "Relevant"},
            {"id": "b", "combined_score": 0.2, "metadata": {}, "text": "Less relevant"},
        ]
        original_chat = chat.ollama.chat
        try:
            chat.ollama.chat = lambda **kwargs: None
            result = chat.rerank_candidates("test question", candidates)
        finally:
            chat.ollama.chat = original_chat
        self.assertEqual([item["id"] for item in result], ["a", "b"])


    def test_local_reranker_can_promote_directly_relevant_passage(self):
        from types import SimpleNamespace

        candidates = [
            {
                "id": "generic",
                "combined_score": 0.9,
                "metadata": {"source": "a", "page": 1},
                "text": "Generic safety program information",
            },
            {
                "id": "direct",
                "combined_score": 0.5,
                "metadata": {"source": "b", "page": 2},
                "text": "The five JHA hazard questions",
            },
        ]

        fake_response = SimpleNamespace(
            message=SimpleNamespace(
                content=(
                    '{"ranked_ids":['
                    '{"id":"0","score":0.1},'
                    '{"id":"1","score":0.99}'
                    ']}'
                )
            )
        )

        original_chat = chat.ollama.chat
        original_client = getattr(chat.ollama, "Client", None)
        original_enabled = chat.RERANKER_ENABLED

        def fake_chat(**kwargs):
            return fake_response

        try:
            chat.RERANKER_ENABLED = True
            # The implementation prefers Client(...).chat when Client exists.
            # Mock both paths to ensure this test never calls the real Ollama server.
            chat.ollama.chat = fake_chat
            if original_client is not None:
                chat.ollama.Client = lambda **kwargs: SimpleNamespace(chat=fake_chat)

            result = chat.rerank_candidates(
                "What are the five JHA questions?",
                candidates,
            )
        finally:
            chat.ollama.chat = original_chat
            if original_client is not None:
                chat.ollama.Client = original_client
            chat.RERANKER_ENABLED = original_enabled

        self.assertEqual(result[0]["id"], "direct")



    def test_greeting_is_handled_without_document_retrieval(self):
        self.assertTrue(chat.is_greeting("Hello!"))
        self.assertEqual(chat.retrieve_documents("Hello!", None), "")
        answer = chat.generate_answer("Hello!", "")
        self.assertIn("OSHA Safety Assistant", answer)

    def test_how_are_you_is_a_conversational_turn(self):
        self.assertTrue(chat.is_greeting("How are you?"))
        self.assertEqual(chat.retrieve_documents("How are you?", None), "")
        answer = chat.generate_answer("How are you?", "")
        self.assertIn("thanks for asking", answer.lower())

    def test_thanks_and_capability_questions_are_conversational(self):
        self.assertTrue(chat.is_greeting("Thank you"))
        self.assertIn("welcome", chat.greeting_response("Thank you").lower())
        self.assertTrue(chat.is_greeting("What can you do?"))
        self.assertIn("citations", chat.greeting_response("What can you do?").lower())

    def test_unrelated_factual_question_is_not_misclassified_as_greeting(self):
        self.assertFalse(chat.is_greeting("Who won the cricket World Cup?"))


    def test_natural_construction_program_wording_routes_to_page_11(self):
        routes = chat.detect_targeted_routes(
            "What are the main elements of an effective construction safety and health program according to OSHA 3886?"
        )
        self.assertIn(
            ("OSHA_3886_Construction_Safety.pdf", 11),
            {(route["source"], route["page"]) for route in routes},
        )

    def test_natural_construction_program_wording_returns_all_seven_elements(self):
        context = "Evidence from OSHA_3886_Construction_Safety.pdf [Source: OSHA_3886_Construction_Safety.pdf, PDF page 11]"
        answer = chat.generate_answer(
            "What are the main elements of an effective construction safety and health program according to OSHA 3886?",
            context,
        )
        expected = (
            "Management Leadership", "Worker Participation",
            "Hazard Identification and Assessment", "Hazard Prevention and Control",
            "Education and Training", "Program Evaluation and Improvement",
            "Communication and Coordination for Employers on Multiemployer Worksites",
        )
        for heading in expected:
            self.assertIn(heading, answer)
        self.assertIn("PDF page 11", answer)

    def test_hierarchy_question_naming_3071_discloses_actual_3886_source(self):
        context = "Hierarchy evidence [Source: OSHA_3886_Construction_Safety.pdf, PDF page 25]"
        answer = chat.generate_answer(
            "Show me the Hierarchy of Controls diagram from OSHA 3071.", context
        )
        self.assertIn("not OSHA_3071_Job_Hazard_Analysis.pdf", answer)
        self.assertIn("OSHA_3886_Construction_Safety.pdf", answer)
        self.assertIn("PDF page 25", answer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
