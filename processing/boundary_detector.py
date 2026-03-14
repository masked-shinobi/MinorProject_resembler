"""
Boundary Detector Module
Splits sections into logical chunks for embedding and retrieval.
Uses a combination of section boundaries and paragraph-level splitting.
"""

from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class TextChunk:
    """Represents a single text chunk for embedding and retrieval."""
    chunk_id: str                   # Unique identifier: "{paper_id}_{section_id}_chunk_{n}"
    paper_id: str                   # Source paper identifier
    section_heading: str            # Section this chunk belongs to
    content: str                    # The actual chunk text
    char_start: int = 0             # Start position in the section content
    char_end: int = 0               # End position in the section content
    page_numbers: List[int] = field(default_factory=list)
    token_estimate: int = 0         # Rough token count (~words * 1.3)

    def __post_init__(self):
        # Rough token estimation (1 token ≈ 4 chars for English text)
        self.token_estimate = len(self.content) // 4


class BoundaryDetector:
    """
    Splits text into chunks suitable for embedding.

    Strategy:
    1. Respect section boundaries from StructureAnalyzer.
    2. Within sections, split at paragraph boundaries.
    3. If a paragraph is too large, split at sentence boundaries.
    4. Maintain overlap between consecutive chunks for context continuity.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50
    ):
        """
        Args:
            max_chunk_size: Maximum number of characters per chunk.
            min_chunk_size: Minimum number of characters per chunk (merge small ones).
            overlap_size: Number of overlapping characters between consecutive chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size

    def chunk_section(
        self,
        content: str,
        paper_id: str,
        section_id: str,
        section_heading: str
    ) -> List[TextChunk]:
        """
        Split a section's content into chunks.

        Args:
            content: The section text content.
            paper_id: Identifier for the source paper.
            section_id: Section identifier.
            section_heading: Section heading for metadata.

        Returns:
            List of TextChunk objects.
        """
        if not content.strip():
            return []

        # Split into paragraphs first
        paragraphs = self._split_paragraphs(content)

        # Merge small paragraphs and split large ones
        raw_chunks = self._balance_chunks(paragraphs)

        # Create TextChunk objects with overlap
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(raw_chunks):
            chunk = TextChunk(
                chunk_id=f"{paper_id}_{section_id}_chunk_{i}",
                paper_id=paper_id,
                section_heading=section_heading,
                content=chunk_text.strip(),
                char_start=char_offset,
                char_end=char_offset + len(chunk_text)
            )
            chunks.append(chunk)
            char_offset += len(chunk_text)

        return chunks

    def chunk_document(
        self,
        sections: list,
        paper_id: str
    ) -> List[TextChunk]:
        """
        Chunk all sections of a document.

        Args:
            sections: List of Section objects from StructureAnalyzer.
            paper_id: Identifier for the source paper.

        Returns:
            List of all TextChunk objects across all sections.
        """
        all_chunks = []
        for section in sections:
            section_chunks = self.chunk_section(
                content=section.content,
                paper_id=paper_id,
                section_id=section.section_id,
                section_heading=section.heading
            )
            all_chunks.extend(section_chunks)
        return all_chunks

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs (double newline separated)."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    def _balance_chunks(self, paragraphs: List[str]) -> List[str]:
        """
        Balance paragraphs into chunks:
        - Merge short adjacent paragraphs.
        - Split overly long paragraphs by sentence.
        """
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(para) > self.max_chunk_size:
                # Flush current buffer
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                # Split long paragraph by sentences
                sentences = self._split_sentences(para)
                sent_buffer = ""
                for sent in sentences:
                    if len(sent_buffer) + len(sent) + 1 > self.max_chunk_size:
                        if sent_buffer:
                            chunks.append(sent_buffer)
                        sent_buffer = sent
                    else:
                        sent_buffer = (sent_buffer + " " + sent).strip()
                if sent_buffer:
                    chunks.append(sent_buffer)
            elif len(current_chunk) + len(para) + 2 > self.max_chunk_size:
                # Current buffer would overflow — flush it
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = para
            else:
                # Merge paragraph into current buffer
                current_chunk = (current_chunk + "\n\n" + para).strip()

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitter based on period + space."""
        import re
        # Split on sentence-ending punctuation followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
