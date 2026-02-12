"""EDX API client for NETL's Energy Data eXchange."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from .config import get_settings


class Resource(BaseModel):
    """EDX Resource model."""

    id: str
    name: str
    description: str | None = None
    format: str | None = None
    size: int | None = None
    url: str | None = None
    created: str | None = None
    last_modified: str | None = None
    package_id: str | None = None


class SearchResult(BaseModel):
    """Search result containing resources."""

    count: int
    resources: list[Resource]


class Submission(BaseModel):
    """EDX Submission (dataset) model."""

    id: str
    name: str
    title: str | None = None
    notes: str | None = None
    author: str | None = None
    organization: str | None = None
    tags: list[str] = []
    resources: list[Resource] = []
    metadata_created: str | None = None
    metadata_modified: str | None = None


class EDXClient:
    """Async client for NETL EDX API."""

    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.edx_base_url
        self.headers = {
            "X-CKAN-API-Key": self.settings.edx_api_key,
            "User-Agent": "EDX-USER",
            "Content-Type": "application/json",
        }

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an async request to the EDX API."""
        url = f"{self.base_url}/{endpoint}"
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data,
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success", False):
                error = result.get("error", {})
                raise Exception(f"EDX API error: {error}")

            return result.get("result", {})

    async def search_resources(
        self,
        query: str | None = None,
        format_filter: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResult:
        """
        Search for resources in EDX.

        Args:
            query: Search query string
            format_filter: Filter by file format (e.g., 'CSV', 'JSON', 'PDF')
            limit: Maximum number of results to return
            offset: Number of results to skip

        Returns:
            SearchResult containing matching resources
        """
        # Build query parameters for resource_search
        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        # Build the query string
        query_parts = []
        if query:
            query_parts.append(f"name:{query}")
        if format_filter:
            query_parts.append(f"format:{format_filter}")

        if query_parts:
            params["query"] = " ".join(query_parts)

        result = await self._request("GET", "resource_search", params=params)

        resources = [
            Resource(
                id=r.get("id", ""),
                name=r.get("name", ""),
                description=r.get("description"),
                format=r.get("format"),
                size=r.get("size"),
                url=r.get("url"),
                created=r.get("created"),
                last_modified=r.get("last_modified"),
                package_id=r.get("package_id"),
            )
            for r in result.get("results", [])
        ]

        return SearchResult(
            count=result.get("count", len(resources)),
            resources=resources,
        )

    async def get_resource(self, resource_id: str) -> Resource:
        """
        Get detailed metadata for a specific resource.

        Args:
            resource_id: The resource ID

        Returns:
            Resource with full metadata
        """
        result = await self._request("GET", "resource_show", params={"id": resource_id})

        return Resource(
            id=result.get("id", ""),
            name=result.get("name", ""),
            description=result.get("description"),
            format=result.get("format"),
            size=result.get("size"),
            url=result.get("url"),
            created=result.get("created"),
            last_modified=result.get("last_modified"),
            package_id=result.get("package_id"),
        )

    async def get_submission(self, submission_id: str) -> Submission:
        """
        Get detailed metadata for a submission (dataset).

        Args:
            submission_id: The submission/package ID or name

        Returns:
            Submission with full metadata and resources
        """
        result = await self._request("GET", "package_show", params={"id": submission_id})

        resources = [
            Resource(
                id=r.get("id", ""),
                name=r.get("name", ""),
                description=r.get("description"),
                format=r.get("format"),
                size=r.get("size"),
                url=r.get("url"),
                created=r.get("created"),
                last_modified=r.get("last_modified"),
                package_id=result.get("id"),
            )
            for r in result.get("resources", [])
        ]

        tags = [t.get("name", "") for t in result.get("tags", [])]

        return Submission(
            id=result.get("id", ""),
            name=result.get("name", ""),
            title=result.get("title"),
            notes=result.get("notes"),
            author=result.get("author"),
            organization=(result.get("organization") or {}).get("title"),
            tags=tags,
            resources=resources,
            metadata_created=result.get("metadata_created"),
            metadata_modified=result.get("metadata_modified"),
        )

    async def list_group_submissions(
        self,
        group: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Submission]:
        """
        List submissions in a group (like CLAIMM).

        Args:
            group: Group name/ID (defaults to CLAIMM group from settings)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of submissions in the group
        """
        group_name = group or self.settings.claimm_group

        # First get the group to get package list
        result = await self._request(
            "GET",
            "group_show",
            params={
                "id": group_name,
                "include_datasets": True,
                "limit": limit,
                "offset": offset,
            },
        )

        submissions = []
        for pkg in result.get("packages", []):
            resources = [
                Resource(
                    id=r.get("id", ""),
                    name=r.get("name", ""),
                    description=r.get("description"),
                    format=r.get("format"),
                    size=r.get("size"),
                    url=r.get("url"),
                    created=r.get("created"),
                    last_modified=r.get("last_modified"),
                )
                for r in pkg.get("resources", [])
            ]

            tags = [t.get("name", "") for t in pkg.get("tags", [])]

            submissions.append(
                Submission(
                    id=pkg.get("id", ""),
                    name=pkg.get("name", ""),
                    title=pkg.get("title"),
                    notes=pkg.get("notes"),
                    author=pkg.get("author"),
                    organization=pkg.get("organization", {}).get("title")
                    if isinstance(pkg.get("organization"), dict)
                    else None,
                    tags=tags,
                    resources=resources,
                    metadata_created=pkg.get("metadata_created"),
                    metadata_modified=pkg.get("metadata_modified"),
                )
            )

        return submissions

    async def search_submissions(
        self,
        query: str | None = None,
        tags: list[str] | None = None,
        groups: list[str] | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[Submission]:
        """
        Search for submissions (datasets) in EDX.

        Args:
            query: Free text search query
            tags: Filter by tags
            groups: Filter by groups (if None, searches CLAIMM-related data)
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching submissions
        """
        # Build filter query (fq) for CKAN
        fq_parts = []

        # If specific groups provided, filter by them
        if groups:
            for g in groups:
                fq_parts.append(f"groups:{g}")

        if tags:
            for tag in tags:
                fq_parts.append(f"tags:{tag}")

        params: dict[str, Any] = {
            "rows": limit,
            "start": offset,
        }

        if query:
            params["q"] = query

        if fq_parts:
            params["fq"] = " AND ".join(fq_parts)

        result = await self._request("GET", "package_search", params=params)

        submissions = []
        for pkg in result.get("results", []):
            resources = [
                Resource(
                    id=r.get("id", ""),
                    name=r.get("name", ""),
                    description=r.get("description"),
                    format=r.get("format"),
                    size=r.get("size"),
                    url=r.get("url"),
                    created=r.get("created"),
                    last_modified=r.get("last_modified"),
                )
                for r in pkg.get("resources", [])
            ]

            tags_list = [t.get("name", "") for t in pkg.get("tags", [])]

            submissions.append(
                Submission(
                    id=pkg.get("id", ""),
                    name=pkg.get("name", ""),
                    title=pkg.get("title"),
                    notes=pkg.get("notes"),
                    author=pkg.get("author"),
                    organization=pkg.get("organization", {}).get("title")
                    if isinstance(pkg.get("organization"), dict)
                    else None,
                    tags=tags_list,
                    resources=resources,
                    metadata_created=pkg.get("metadata_created"),
                    metadata_modified=pkg.get("metadata_modified"),
                )
            )

        return submissions

    async def create_submission(
        self,
        name: str,
        title: str,
        notes: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
        groups: list[str] | None = None,
        private: bool = False,
        extras: dict[str, str] | None = None,
    ) -> Submission:
        """
        Create a new submission (dataset) in EDX.

        Args:
            name: Unique identifier for the dataset (lowercase, no spaces, use hyphens)
            title: Human-readable title for the dataset
            notes: Description of the dataset (supports Markdown)
            author: Author name
            tags: List of tags to apply to the dataset
            groups: List of group names/IDs to add the dataset to (e.g., ["claimm"])
            private: Whether the dataset should be private (default: False)
            extras: Additional metadata as key-value pairs

        Returns:
            Created Submission object
        """
        data: dict[str, Any] = {
            "name": name,
            "title": title,
            "private": private,
        }

        if notes:
            data["notes"] = notes
        if author:
            data["author"] = author
        if tags:
            data["tags"] = [{"name": tag} for tag in tags]
        if groups:
            data["groups"] = [{"name": g} for g in groups]
        if extras:
            data["extras"] = [{"key": k, "value": v} for k, v in extras.items()]

        result = await self._request("POST", "package_create", data=data)

        resources = [
            Resource(
                id=r.get("id", ""),
                name=r.get("name", ""),
                description=r.get("description"),
                format=r.get("format"),
                size=r.get("size"),
                url=r.get("url"),
                created=r.get("created"),
                last_modified=r.get("last_modified"),
                package_id=result.get("id"),
            )
            for r in result.get("resources", [])
        ]

        tags_list = [t.get("name", "") for t in result.get("tags", [])]

        return Submission(
            id=result.get("id", ""),
            name=result.get("name", ""),
            title=result.get("title"),
            notes=result.get("notes"),
            author=result.get("author"),
            organization=(result.get("organization") or {}).get("title"),
            tags=tags_list,
            resources=resources,
            metadata_created=result.get("metadata_created"),
            metadata_modified=result.get("metadata_modified"),
        )

    async def update_submission(
        self,
        submission_id: str,
        title: str | None = None,
        notes: str | None = None,
        author: str | None = None,
        tags: list[str] | None = None,
        private: bool | None = None,
    ) -> Submission:
        """
        Update an existing submission (dataset) in EDX.

        Args:
            submission_id: The submission/package ID or name
            title: New title (optional)
            notes: New description (optional)
            author: New author name (optional)
            tags: New list of tags (replaces existing tags)
            private: Change privacy setting (optional)

        Returns:
            Updated Submission object
        """
        data: dict[str, Any] = {"id": submission_id}

        if title is not None:
            data["title"] = title
        if notes is not None:
            data["notes"] = notes
        if author is not None:
            data["author"] = author
        if tags is not None:
            data["tags"] = [{"name": tag} for tag in tags]
        if private is not None:
            data["private"] = private

        result = await self._request("POST", "package_update", data=data)

        resources = [
            Resource(
                id=r.get("id", ""),
                name=r.get("name", ""),
                description=r.get("description"),
                format=r.get("format"),
                size=r.get("size"),
                url=r.get("url"),
                created=r.get("created"),
                last_modified=r.get("last_modified"),
                package_id=result.get("id"),
            )
            for r in result.get("resources", [])
        ]

        tags_list = [t.get("name", "") for t in result.get("tags", [])]

        return Submission(
            id=result.get("id", ""),
            name=result.get("name", ""),
            title=result.get("title"),
            notes=result.get("notes"),
            author=result.get("author"),
            organization=(result.get("organization") or {}).get("title"),
            tags=tags_list,
            resources=resources,
            metadata_created=result.get("metadata_created"),
            metadata_modified=result.get("metadata_modified"),
        )

    async def upload_resource(
        self,
        package_id: str,
        file_path: str | Path,
        name: str | None = None,
        description: str | None = None,
        format: str | None = None,
    ) -> Resource:
        """
        Upload a file as a new resource to an existing submission.

        Args:
            package_id: The submission/package ID to add the resource to
            file_path: Path to the file to upload
            name: Resource name (defaults to filename)
            description: Description of the resource
            format: File format (e.g., 'CSV', 'JSON'). Auto-detected if not provided.

        Returns:
            Created Resource object
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect format from extension if not provided
        if format is None:
            format = file_path.suffix.lstrip(".").upper() or None

        # Use filename as name if not provided
        if name is None:
            name = file_path.name

        url = f"{self.base_url}/resource_create"

        # For file uploads, we need multipart form data
        async with httpx.AsyncClient(timeout=120.0) as client:
            with open(file_path, "rb") as f:
                files = {"upload": (file_path.name, f, "application/octet-stream")}
                data = {
                    "package_id": package_id,
                    "name": name,
                }
                if description:
                    data["description"] = description
                if format:
                    data["format"] = format

                response = await client.post(
                    url,
                    headers={"X-CKAN-API-Key": self.settings.edx_api_key},
                    data=data,
                    files=files,
                )
                response.raise_for_status()
                result = response.json()

                if not result.get("success", False):
                    error = result.get("error", {})
                    raise Exception(f"EDX API error: {error}")

                r = result.get("result", {})

        return Resource(
            id=r.get("id", ""),
            name=r.get("name", ""),
            description=r.get("description"),
            format=r.get("format"),
            size=r.get("size"),
            url=r.get("url"),
            created=r.get("created"),
            last_modified=r.get("last_modified"),
            package_id=r.get("package_id"),
        )

    async def upload_resource_from_bytes(
        self,
        package_id: str,
        file_content: bytes,
        filename: str,
        name: str | None = None,
        description: str | None = None,
        format: str | None = None,
    ) -> Resource:
        """
        Upload bytes as a new resource to an existing submission.

        Args:
            package_id: The submission/package ID to add the resource to
            file_content: The file content as bytes
            filename: The filename to use for the upload
            name: Resource name (defaults to filename)
            description: Description of the resource
            format: File format (e.g., 'CSV', 'JSON'). Auto-detected if not provided.

        Returns:
            Created Resource object
        """
        # Auto-detect format from extension if not provided
        if format is None:
            suffix = Path(filename).suffix.lstrip(".").upper()
            format = suffix if suffix else None

        # Use filename as name if not provided
        if name is None:
            name = filename

        url = f"{self.base_url}/resource_create"

        async with httpx.AsyncClient(timeout=120.0) as client:
            files = {"upload": (filename, file_content, "application/octet-stream")}
            data = {
                "package_id": package_id,
                "name": name,
            }
            if description:
                data["description"] = description
            if format:
                data["format"] = format

            response = await client.post(
                url,
                headers={"X-CKAN-API-Key": self.settings.edx_api_key},
                data=data,
                files=files,
            )
            response.raise_for_status()
            result = response.json()

            if not result.get("success", False):
                error = result.get("error", {})
                raise Exception(f"EDX API error: {error}")

            r = result.get("result", {})

        return Resource(
            id=r.get("id", ""),
            name=r.get("name", ""),
            description=r.get("description"),
            format=r.get("format"),
            size=r.get("size"),
            url=r.get("url"),
            created=r.get("created"),
            last_modified=r.get("last_modified"),
            package_id=r.get("package_id"),
        )

    async def update_resource(
        self,
        resource_id: str,
        name: str | None = None,
        description: str | None = None,
        format: str | None = None,
        file_path: str | Path | None = None,
    ) -> Resource:
        """
        Update an existing resource's metadata or replace its file.

        Args:
            resource_id: The resource ID to update
            name: New resource name (optional)
            description: New description (optional)
            format: New format (optional)
            file_path: Path to new file to replace existing (optional)

        Returns:
            Updated Resource object
        """
        url = f"{self.base_url}/resource_update"

        data: dict[str, Any] = {"id": resource_id}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if format is not None:
            data["format"] = format

        async with httpx.AsyncClient(timeout=120.0) as client:
            if file_path:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")

                with open(file_path, "rb") as f:
                    files = {"upload": (file_path.name, f, "application/octet-stream")}
                    response = await client.post(
                        url,
                        headers={"X-CKAN-API-Key": self.settings.edx_api_key},
                        data=data,
                        files=files,
                    )
            else:
                response = await client.post(
                    url,
                    headers={
                        "X-CKAN-API-Key": self.settings.edx_api_key,
                        "Content-Type": "application/json",
                    },
                    json=data,
                )

            response.raise_for_status()
            result = response.json()

            if not result.get("success", False):
                error = result.get("error", {})
                raise Exception(f"EDX API error: {error}")

            r = result.get("result", {})

        return Resource(
            id=r.get("id", ""),
            name=r.get("name", ""),
            description=r.get("description"),
            format=r.get("format"),
            size=r.get("size"),
            url=r.get("url"),
            created=r.get("created"),
            last_modified=r.get("last_modified"),
            package_id=r.get("package_id"),
        )

    async def delete_resource(self, resource_id: str) -> bool:
        """
        Delete a resource from EDX.

        Args:
            resource_id: The resource ID to delete

        Returns:
            True if deletion was successful
        """
        await self._request("POST", "resource_delete", data={"id": resource_id})
        return True

    async def delete_submission(self, submission_id: str) -> bool:
        """
        Delete a submission (dataset) from EDX.

        Args:
            submission_id: The submission/package ID or name to delete

        Returns:
            True if deletion was successful
        """
        await self._request("POST", "package_delete", data={"id": submission_id})
        return True

    def get_download_url(self, resource_id: str) -> str:
        """
        Get the download URL for a resource.

        Args:
            resource_id: The resource ID

        Returns:
            Direct download URL
        """
        return f"https://edx.netl.doe.gov/resource/{resource_id}/download"
