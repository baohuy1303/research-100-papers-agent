from __future__ import annotations

import asyncio
import unittest

import pandas as pd
from fastapi.testclient import TestClient

from api.core.budget import get_budget_level
from api.core.handlers.tier6_multihop import _exec_tool
from api.core.handlers.tier8_compute import _run_sandboxed
from api.core.runtime import RuntimeConfig, use_runtime
from api.core.sql_safety import validate_readonly_sql
from main import app


class RuntimeTests(unittest.TestCase):
    def test_budget_context_is_request_scoped(self) -> None:
        with use_runtime(RuntimeConfig(budget_level="$1")):
            self.assertEqual(get_budget_level(), "$1")
            with use_runtime(RuntimeConfig(budget_level="$20")):
                self.assertEqual(get_budget_level(), "$20")
            self.assertEqual(get_budget_level(), "$1")


class SqlSafetyTests(unittest.TestCase):
    def test_allows_readonly_select(self) -> None:
        self.assertEqual(validate_readonly_sql("SELECT 1;"), "SELECT 1")

    def test_blocks_write_statement(self) -> None:
        with self.assertRaises(ValueError):
            validate_readonly_sql("DROP TABLE papers")

    def test_blocks_multiple_statements(self) -> None:
        with self.assertRaises(ValueError):
            validate_readonly_sql("SELECT 1; SELECT 2")


class Tier6ToolTests(unittest.TestCase):
    def test_search_chunks_tool_executes_retriever(self) -> None:
        class FakeRetriever:
            async def search(self, query, k=None, paper_id=None):
                return {
                    "query": query,
                    "cost_usd": 0.001,
                    "chunks": [{
                        "paper_id": paper_id or "p1",
                        "section_title": "Intro",
                        "score": 0.9,
                        "text": "x" * 1000,
                    }],
                }

        result = asyncio.run(
            _exec_tool(
                "search_chunks",
                {"query": "shifted windows", "k": 3, "paper_id": "paper-1"},
                store=None,
                retriever=FakeRetriever(),
            )
        )
        self.assertEqual(result["chunks"][0]["paper_id"], "paper-1")
        self.assertLessEqual(len(result["chunks"][0]["text"]), 800)


class Tier8SandboxTests(unittest.TestCase):
    def test_allows_basic_dataframe_computation(self) -> None:
        result, err = _run_sandboxed(
            "RESULT = int(papers['citation_count'].sum())",
            {"papers": pd.DataFrame({"citation_count": [1, 2, 3]})},
        )
        self.assertIsNone(err)
        self.assertEqual(result, 6)

    def test_blocks_imports_and_file_writes(self) -> None:
        _, import_err = _run_sandboxed("import os\nRESULT = 1", {"papers": pd.DataFrame()})
        _, write_err = _run_sandboxed("papers.to_csv('out.csv')", {"papers": pd.DataFrame()})
        self.assertIn("Import", import_err or "")
        self.assertIn("to_csv", write_err or "")


class AskRouteTests(unittest.TestCase):
    def test_invalid_budget_returns_400_style_payload_without_llm(self) -> None:
        client = TestClient(app)
        response = client.post("/ask", json={"question": "hello", "budget_level": "$99"})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["tier"], 0)
        self.assertEqual(payload["error"], "invalid budget_level")


if __name__ == "__main__":
    unittest.main()
