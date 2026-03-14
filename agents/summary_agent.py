"""
Summary Agent Module
Summarizes retrieved context for the final explanation agent.
"""

from typing import Optional


class SummaryAgent:
    """
    Agent responsible for condensing retrieved passages into a coherent summary.
    Acts as the second step in the multi-agent pipeline.
    """

    def __init__(self, llm_client):
        """
        Args:
            llm_client: An instance of reasoning.llm_client.LLMClient.
        """
        self.llm_client = llm_client

    def run(self, retrieval_output: dict, query: str = "") -> dict:
        """
        Execute the summary agent on retrieval results.

        Args:
            retrieval_output: Output dict from RetrievalAgent.run().
            query: The original user query for context.

        Returns:
            Dict with keys:
              - "query": original query
              - "summary": condensed summary of retrieved context
              - "source_count": number of source chunks
              - "original_context": the original context (for reference)
        """
        context = retrieval_output.get("context", "")
        num_results = retrieval_output.get("num_results", 0)

        if not context.strip():
            return {
                "query": query,
                "summary": "No relevant context was found for this query.",
                "source_count": 0,
                "original_context": ""
            }

        system_prompt = (
            "You are an expert academic summarizer. Your task is to condense "
            "multiple retrieved passages into a coherent, well-organized summary "
            "that preserves key information and is relevant to the user's query."
        )

        prompt = (
            f"User Query: {query}\n\n"
            f"Retrieved Passages:\n{context}\n\n"
            "Please provide a comprehensive summary of the above passages that "
            "addresses the user's query. Organize the information logically and "
            "note which sources support each point."
        )

        try:
            summary = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=800,
                temperature=0.2
            )
        except Exception as e:
            print(f"[SummaryAgent] Error: {e}")
            summary = f"Error generating summary: {e}"

        return {
            "query": query,
            "summary": summary,
            "source_count": num_results,
            "original_context": context
        }
