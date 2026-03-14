"""
RAG Metrics Module
Evaluates RAG system performance: Precision, Recall, Faithfulness, and Latency.
Implements custom evaluation functions simulating RAGAS-like metrics.
"""

from typing import List, Dict, Any, Optional
import time
import re


class RAGMetrics:
    """
    Evaluates the quality of RAG system outputs.

    Metrics:
    - Context Precision: Are the retrieved chunks relevant to the query?
    - Context Recall: Did we retrieve all relevant information?
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Does the answer address the user's query?
    - Latency: How fast is the pipeline?
    """

    def __init__(self, llm_client=None):
        """
        Args:
            llm_client: Optional LLM client for LLM-based evaluation.
        """
        self.llm_client = llm_client

    def evaluate(
        self,
        query: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
        timing: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics.

        Args:
            query: The user's question.
            answer: The system-generated answer.
            context: The retrieved context used to generate the answer.
            ground_truth: Optional ground truth answer for precision/recall.
            timing: Optional timing dict from the router.

        Returns:
            Dict with metric scores (0.0 to 1.0) and explanations.
        """
        results = {}

        # Context Precision
        results["context_precision"] = self.context_precision(query, context)

        # Faithfulness
        results["faithfulness"] = self.faithfulness(answer, context)

        # Answer Relevancy
        results["answer_relevancy"] = self.answer_relevancy(query, answer)

        # Context Recall (only if ground truth provided)
        if ground_truth:
            results["context_recall"] = self.context_recall(
                ground_truth, context
            )

        # Latency
        if timing:
            results["latency"] = {
                "total_seconds": sum(timing.values()),
                "breakdown": timing
            }

        # Overall score (average of available scores)
        scores = [
            v["score"] for v in results.values()
            if isinstance(v, dict) and "score" in v
        ]
        results["overall_score"] = round(sum(scores) / len(scores), 3) if scores else 0.0

        return results

    def context_precision(self, query: str, context: str) -> Dict[str, Any]:
        """
        Measure how relevant the retrieved context is to the query.
        Uses keyword overlap as a proxy metric.
        """
        query_keywords = set(self._extract_keywords(query))
        context_keywords = set(self._extract_keywords(context))

        if not query_keywords:
            return {"score": 0.0, "explanation": "No keywords in query."}

        overlap = query_keywords.intersection(context_keywords)
        score = len(overlap) / len(query_keywords)

        return {
            "score": round(min(score, 1.0), 3),
            "matched_keywords": list(overlap),
            "explanation": f"{len(overlap)}/{len(query_keywords)} query keywords found in context."
        }

    def context_recall(self, ground_truth: str, context: str) -> Dict[str, Any]:
        """
        Measure how much of the ground truth is covered by the retrieved context.
        """
        truth_keywords = set(self._extract_keywords(ground_truth))
        context_keywords = set(self._extract_keywords(context))

        if not truth_keywords:
            return {"score": 0.0, "explanation": "No keywords in ground truth."}

        overlap = truth_keywords.intersection(context_keywords)
        score = len(overlap) / len(truth_keywords)

        return {
            "score": round(min(score, 1.0), 3),
            "matched_keywords": list(overlap),
            "explanation": f"{len(overlap)}/{len(truth_keywords)} ground truth keywords found in context."
        }

    def faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Measure whether the answer is grounded in (faithful to) the context.
        Checks if key claims in the answer can be traced to the context.
        """
        if self.llm_client:
            return self._llm_faithfulness(answer, context)

        # Heuristic: check sentence overlap
        answer_sentences = self._split_sentences(answer)
        context_lower = context.lower()

        grounded_count = 0
        total = len(answer_sentences)

        for sent in answer_sentences:
            # Check if key phrases from the sentence appear in context
            key_phrases = self._extract_keywords(sent)
            match_count = sum(1 for kp in key_phrases if kp in context_lower)
            if key_phrases and match_count / len(key_phrases) > 0.3:
                grounded_count += 1

        score = grounded_count / total if total > 0 else 0.0

        return {
            "score": round(score, 3),
            "grounded_sentences": grounded_count,
            "total_sentences": total,
            "explanation": f"{grounded_count}/{total} answer sentences are grounded in context."
        }

    def answer_relevancy(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Measure whether the answer is relevant to the query.
        """
        query_keywords = set(self._extract_keywords(query))
        answer_keywords = set(self._extract_keywords(answer))

        if not query_keywords:
            return {"score": 0.0, "explanation": "No keywords in query."}

        overlap = query_keywords.intersection(answer_keywords)
        score = len(overlap) / len(query_keywords)

        return {
            "score": round(min(score, 1.0), 3),
            "matched_keywords": list(overlap),
            "explanation": f"{len(overlap)}/{len(query_keywords)} query keywords addressed in answer."
        }

    def _llm_faithfulness(self, answer: str, context: str) -> Dict[str, Any]:
        """LLM-based faithfulness check."""
        prompt = (
            "Given the following context and answer, evaluate the faithfulness "
            "of the answer. Score from 0.0 (hallucinated) to 1.0 (fully grounded).\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Answer:\n{answer}\n\n"
            "Respond with ONLY a JSON object: {\"score\": <float>, \"explanation\": \"<text>\"}"
        )

        try:
            response = self.llm_client.generate(prompt, max_tokens=200, temperature=0.1)
            import json
            result = json.loads(response)
            return {
                "score": round(float(result.get("score", 0)), 3),
                "explanation": result.get("explanation", "LLM evaluation.")
            }
        except Exception:
            # Fall back to heuristic
            return self.faithfulness.__wrapped__(self, answer, context) if hasattr(self.faithfulness, '__wrapped__') else {"score": 0.5, "explanation": "LLM evaluation failed, defaulting."}

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        stopwords = {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "and",
            "or", "but", "in", "on", "at", "to", "for", "of", "with",
            "by", "from", "as", "it", "its", "this", "that", "not"
        }
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stopwords]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
