"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re


VAGUE_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "what", "whats", "how", "why", "who", "when", "where", "which",
    "do", "does", "did", "can", "could", "would", "should", "will",
    "i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
    "help", "please", "hi", "hello", "hey", "up", "about", "info",
    "tell", "explain", "know", "want", "need", "thing", "stuff", "things",
    "and", "or", "but", "of", "to", "for", "on", "in", "with", "some",
}


class DocuBot:
    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.documents)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, documents):
        """
        Build a tiny inverted index mapping lowercase words to the documents
        they appear in.
        """
        index = {}
        for filename, text in documents:
            words = re.findall(r"[a-z0-9]+", text.lower())
            for word in set(words):
                if word not in index:
                    index[word] = set()
                index[word].add(filename)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        Return a simple relevance score for how well the text matches the query.
        The score rewards exact matches and also handles common synonym-style
        overlaps such as 'generated' matching 'generation' or 'token' matching
        'tokens'.
        """
        query_words = re.findall(r"[a-z0-9]+", query.lower())
        text_words = re.findall(r"[a-z0-9]+", text.lower())

        if not query_words:
            return 0

        score = 0
        for word in query_words:
            if word in text_words:
                score += 2
            elif word.endswith("s") and word[:-1] in text_words:
                score += 1
            elif word.endswith("ed") and word[:-2] in text_words:
                score += 1
            elif word.endswith("ing") and word[:-3] in text_words:
                score += 1

        return score

    def retrieve(self, query, top_k=3):
        """
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text, score) sorted by score descending.
        """
        results = []
        for filename, text in self.documents:
            score = self.score_document(query, text)
            if score > 0:
                results.append((filename, text, score))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]

    # -----------------------------------------------------------
    # Guardrails
    # -----------------------------------------------------------

    def is_too_vague(self, query):
        """
        Heuristic check for whether a query is too vague to answer from
        these docs: either it has too few meaningful words, or none of
        its words appear anywhere in the inverted index.
        """
        words = re.findall(r"[a-z0-9]+", query.lower())
        meaningful = [w for w in words if w not in VAGUE_STOPWORDS]

        if len(meaningful) < 2:
            return True

        if not any(w in self.index for w in meaningful):
            return True

        return False

    VAGUE_MESSAGE = (
        "Your question seems too vague for me to answer from these docs. "
        "Could you mention a specific feature, file, or topic "
        "(e.g. 'How is the auth token generated?')?"
    )

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns compact snippets and filenames with no LLM involved.
        """
        if self.is_too_vague(query):
            return self.VAGUE_MESSAGE

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text, _score in snippets:
            cleaned_text = re.sub(r"\s+", " ", text).strip()
            paragraph = cleaned_text.split("\n\n", 1)[0]
            if len(paragraph) > 220:
                paragraph = paragraph[:217].rstrip() + "..."
            formatted.append(f"[{filename}]\n{paragraph}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        if self.is_too_vague(query):
            return self.VAGUE_MESSAGE

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
