"""LLM client using LiteLLM for multi-provider support."""

from __future__ import annotations

import json
from typing import Any

import litellm

from .config import get_settings
from .edx_client import Resource, Submission


class LLMClient:
    """Multi-provider LLM client using LiteLLM."""

    def __init__(self):
        self.settings = get_settings()
        self.model = self.settings.get_llm_model()

        # Ensure we have at least one provider configured
        if not self.settings.get_available_provider():
            raise ValueError(
                "No LLM provider API key configured. "
                "Set at least one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
                "GOOGLE_API_KEY, or XAI_API_KEY"
            )

    async def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """Make an async completion request via LiteLLM."""
        response = await litellm.acompletion(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def interpret_search_query(self, natural_query: str) -> dict[str, Any]:
        """
        Convert a natural language query into EDX search parameters.

        Args:
            natural_query: User's natural language search query

        Returns:
            Dictionary with search parameters:
            - query: The search text
            - tags: List of relevant tags
            - format_filter: File format filter (if applicable)
        """
        system_prompt = """You are a search query interpreter for the NETL Energy Data eXchange (EDX) CLAIMM database.
CLAIMM focuses on Critical Minerals and Materials data, including mine waste, mineral resources, and related datasets.

Given a natural language query, extract structured search parameters.

Respond with a JSON object containing:
- "query": The main search text (keywords for searching titles/descriptions)
- "tags": A list of relevant tags (e.g., ["lithium", "rare earth", "coal ash"])
- "format_filter": File format if the user wants specific types (e.g., "CSV", "JSON", "PDF", "XLSX", or null)
- "explanation": Brief explanation of your interpretation

Common topics in CLAIMM:
- Critical minerals (lithium, cobalt, rare earth elements, etc.)
- Mine waste and tailings
- Coal combustion residuals
- Mineral characterization data
- Geochemical analysis
- Resource assessments

Only include the JSON in your response, no other text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": natural_query},
        ]

        response = await self._complete(messages, temperature=0.1)

        try:
            # Parse the JSON response
            result = json.loads(response.strip())
            return {
                "query": result.get("query", natural_query),
                "tags": result.get("tags", []),
                "format_filter": result.get("format_filter"),
                "explanation": result.get("explanation", ""),
            }
        except json.JSONDecodeError:
            # Fallback to simple interpretation
            return {
                "query": natural_query,
                "tags": [],
                "format_filter": None,
                "explanation": "Using query as-is (could not parse LLM response)",
            }

    async def summarize_search_results(
        self,
        results: list[Submission],
        original_query: str,
    ) -> str:
        """
        Generate a human-friendly summary of search results.

        Args:
            results: List of submissions from EDX search
            original_query: The user's original query

        Returns:
            A formatted summary of the results
        """
        if not results:
            return f"No results found for '{original_query}' in the CLAIMM database."

        # Build context from results
        results_context = []
        for i, sub in enumerate(results[:10], 1):  # Limit to top 10
            result_info = f"""
{i}. **{sub.title or sub.name}**
   - ID: {sub.id}
   - Description: {(sub.notes or "No description")[:300]}...
   - Tags: {", ".join(sub.tags[:5]) if sub.tags else "None"}
   - Resources: {len(sub.resources)} file(s)
   - Formats: {", ".join(set(r.format for r in sub.resources if r.format)) or "Unknown"}
"""
            results_context.append(result_info)

        context_text = "\n".join(results_context)

        system_prompt = """You are a helpful assistant summarizing search results from the CLAIMM (Critical Minerals and Materials) database.

Provide a concise but informative summary that:
1. Highlights the most relevant results for the user's query
2. Groups similar datasets if applicable
3. Notes the types of data available (formats, size, etc.)
4. Suggests which datasets might be most useful

Keep the summary focused and actionable."""

        user_prompt = f"""User searched for: "{original_query}"

Found {len(results)} results:
{context_text}

Provide a helpful summary of these results."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self._complete(messages, temperature=0.5)

    async def answer_about_resource(
        self,
        resource: Resource,
        submission: Submission | None,
        question: str,
    ) -> str:
        """
        Answer a question about a specific resource.

        Args:
            resource: The resource metadata
            submission: The parent submission metadata (if available)
            question: User's question

        Returns:
            AI-generated answer based on metadata
        """
        context = f"""
Resource Information:
- Name: {resource.name}
- ID: {resource.id}
- Format: {resource.format or "Unknown"}
- Size: {resource.size or "Unknown"} bytes
- Description: {resource.description or "No description available"}
- Created: {resource.created or "Unknown"}
- Last Modified: {resource.last_modified or "Unknown"}
- Download URL: {resource.url or "Not available"}
"""

        if submission:
            context += f"""
Parent Dataset Information:
- Title: {submission.title or submission.name}
- Description: {submission.notes or "No description"}
- Author: {submission.author or "Unknown"}
- Organization: {submission.organization or "Unknown"}
- Tags: {", ".join(submission.tags) if submission.tags else "None"}
- Total Resources: {len(submission.resources)}
"""

        system_prompt = """You are a helpful assistant answering questions about datasets in the CLAIMM (Critical Minerals and Materials) database.
Answer based only on the provided metadata. If the information isn't available in the metadata, say so.
Be concise but helpful."""

        user_prompt = f"""{context}

User question: {question}"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self._complete(messages, temperature=0.3)

    async def suggest_related_searches(
        self,
        query: str,
        results: list[Submission],
    ) -> list[str]:
        """
        Suggest related search queries based on initial results.

        Args:
            query: Original search query
            results: Results from the search

        Returns:
            List of suggested related queries
        """
        if not results:
            return []

        # Gather tags and titles from results
        all_tags = set()
        keywords = []
        for sub in results[:5]:
            all_tags.update(sub.tags)
            if sub.title:
                keywords.append(sub.title)

        context = f"""
Original query: {query}
Found tags: {", ".join(list(all_tags)[:20])}
Sample titles: {"; ".join(keywords[:5])}
"""

        system_prompt = """Based on a search in the CLAIMM (Critical Minerals and Materials) database, suggest 3-5 related search queries that might help the user find more relevant data.

Return ONLY a JSON array of strings, no other text.
Example: ["lithium extraction data", "rare earth processing", "mine tailings analysis"]"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": context},
        ]

        response = await self._complete(messages, temperature=0.7)

        try:
            suggestions = json.loads(response.strip())
            return suggestions if isinstance(suggestions, list) else []
        except json.JSONDecodeError:
            return []
