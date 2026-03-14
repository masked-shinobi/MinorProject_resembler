"""
Planner Module
Analyzes user queries and decides the retrieval and reasoning strategy.
"""

from typing import Dict, Any, Optional


class Planner:
    """
    Analyzes the user query and creates an execution plan for the agents.
    Determines:
    - Search strategy (semantic, keyword, or hybrid)
    - Number of chunks needed
    - Whether multi-step reasoning is required
    - Special handling (e.g., comparison queries, table queries)
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for advanced query classification.
        """
        self.llm_client = llm_client

    def plan(self, query: str, available_papers: list = None) -> Dict[str, Any]:
        """
        Create an execution plan for the given query.

        Args:
            query: The user's question.
            available_papers: List of available paper IDs/titles.

        Returns:
            Execution plan dict with keys:
              - "query_type": classification of the query
              - "search_mode": "hybrid", "semantic", or "keyword"
              - "top_k": number of chunks to retrieve
              - "paper_filter": optional paper_id filter
              - "needs_summary": whether summarization agent is needed
              - "strategy_notes": human-readable explanation
        """
        query_lower = query.lower().strip()
        plan = {
            "query_type": "general",
            "search_mode": "hybrid",
            "top_k": 5,
            "paper_filter": None,
            "needs_summary": True,
            "strategy_notes": ""
        }

        # Classify query type
        plan["query_type"] = self._classify_query(query_lower)

        # Adjust strategy based on query type
        if plan["query_type"] == "factual":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["strategy_notes"] = "Factual query — targeted semantic search."

        elif plan["query_type"] == "comparative":
            plan["top_k"] = 8
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Comparative query — broader context needed."

        elif plan["query_type"] == "methodological":
            plan["top_k"] = 5
            plan["search_mode"] = "hybrid"
            plan["strategy_notes"] = "Methodology query — hybrid search for detailed results."

        elif plan["query_type"] == "summary":
            plan["top_k"] = 10
            plan["needs_summary"] = True
            plan["strategy_notes"] = "Summary request — retrieving broad context."

        elif plan["query_type"] == "definition":
            plan["top_k"] = 3
            plan["search_mode"] = "semantic"
            plan["needs_summary"] = False
            plan["strategy_notes"] = "Definition query — narrow semantic search."

        else:
            plan["strategy_notes"] = "General query — standard hybrid search."

        # Check if query mentions a specific paper
        if available_papers:
            for paper in available_papers:
                paper_name = paper if isinstance(paper, str) else paper.get("title", "")
                if paper_name.lower() in query_lower:
                    plan["paper_filter"] = paper if isinstance(paper, str) else paper.get("paper_id")
                    plan["strategy_notes"] += f" Filtered to paper: {paper_name}"
                    break

        return plan

    def _classify_query(self, query: str) -> str:
        """
        Classify the query type using keyword heuristics.

        Returns one of: "factual", "comparative", "methodological",
                        "summary", "definition", "general"
        """
        if any(w in query for w in ["compare", "difference", "versus", "vs", "better"]):
            return "comparative"

        if any(w in query for w in ["how does", "method", "approach", "algorithm", "technique"]):
            return "methodological"

        if any(w in query for w in ["summarize", "overview", "overall", "summary of"]):
            return "summary"

        if any(w in query for w in ["what is", "define", "definition", "meaning of"]):
            return "definition"

        if any(w in query for w in ["who", "when", "where", "how many", "how much"]):
            return "factual"

        return "general"
