"""
Router Module
Orchestrates the multi-agent pipeline: Retrieval → Summary → Explanation.
Controls the flow based on the Planner's execution plan.
"""

from typing import Dict, Any, Optional
import time


class Router:
    """
    Conditional router that orchestrates the sequential agent pipeline.
    Routes the query through: Planner → Retrieval Agent → Summary Agent → Explanation Agent
    """

    def __init__(
        self,
        planner,
        retrieval_agent,
        summary_agent,
        explanation_agent
    ):
        """
        Args:
            planner: An instance of reasoning.planner.Planner.
            retrieval_agent: An instance of agents.retrieval_agent.RetrievalAgent.
            summary_agent: An instance of agents.summary_agent.SummaryAgent.
            explanation_agent: An instance of agents.explanation_agent.ExplanationAgent.
        """
        self.planner = planner
        self.retrieval_agent = retrieval_agent
        self.summary_agent = summary_agent
        self.explanation_agent = explanation_agent

    def route(self, query: str, available_papers: list = None) -> Dict[str, Any]:
        """
        Route a query through the full multi-agent pipeline.

        Args:
            query: The user's question.
            available_papers: Optional list of available papers.

        Returns:
            Dict containing the full pipeline output:
              - "query": original query
              - "plan": the execution plan
              - "retrieval": retrieval agent output
              - "summary": summary agent output (if applicable)
              - "explanation": explanation agent output
              - "answer": the final answer text
              - "timing": execution time for each step (in seconds)
        """
        timing = {}
        output = {"query": query}

        # Step 1: Planning
        t0 = time.time()
        plan = self.planner.plan(query, available_papers)
        timing["planning"] = round(time.time() - t0, 3)
        output["plan"] = plan

        print(f"[Router] Query type: {plan['query_type']}")
        print(f"[Router] Strategy: {plan['strategy_notes']}")

        # Step 2: Retrieval
        t0 = time.time()
        retrieval_output = self.retrieval_agent.run(
            query=query,
            top_k=plan["top_k"],
            paper_id=plan.get("paper_filter"),
            search_mode=plan["search_mode"]
        )
        timing["retrieval"] = round(time.time() - t0, 3)
        output["retrieval"] = retrieval_output

        print(f"[Router] Retrieved {retrieval_output['num_results']} chunks")

        # Step 3: Summarization (conditional)
        if plan["needs_summary"] and retrieval_output["num_results"] > 0:
            t0 = time.time()
            summary_output = self.summary_agent.run(retrieval_output, query=query)
            timing["summarization"] = round(time.time() - t0, 3)
            output["summary"] = summary_output
        else:
            # Skip summarization — pass retrieval context directly
            output["summary"] = {
                "query": query,
                "summary": retrieval_output.get("context", ""),
                "source_count": retrieval_output.get("num_results", 0),
                "original_context": retrieval_output.get("context", "")
            }
            timing["summarization"] = 0

        # Step 4: Explanation (final answer)
        t0 = time.time()
        explanation_output = self.explanation_agent.run(
            output["summary"], query=query
        )
        timing["explanation"] = round(time.time() - t0, 3)
        output["explanation"] = explanation_output

        # Final answer
        output["answer"] = explanation_output["answer"]
        output["confidence"] = explanation_output.get("confidence", "unknown")
        output["timing"] = timing

        total_time = sum(timing.values())
        print(f"[Router] Total time: {total_time:.2f}s | "
              f"Confidence: {output['confidence']}")

        return output
