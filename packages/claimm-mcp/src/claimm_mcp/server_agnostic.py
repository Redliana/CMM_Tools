"""LLM-Agnostic MCP Server for NETL EDX CLAIMM data.

This server provides direct access to CLAIMM data without any LLM dependencies.
Suitable for collaborators who want to use their own LLM or no LLM at all.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .edx_client import EDXClient
from .header_detector import HeaderDetector

# Initialize MCP server
mcp = FastMCP(
    "CLAIMM-Agnostic",
    instructions="""LLM-Agnostic CLAIMM Data Server.

Provides direct access to NETL EDX CLAIMM (Critical Minerals and Materials) datasets.
No LLM processing - returns raw data for your own analysis or LLM integration.

Data includes:
- Rare Earth Elements in coal and byproducts
- Produced water geochemistry
- Mine waste characterization
- Geological surveys
- Research publications

All resources include direct download URLs.""",
)

# Initialize clients
edx = EDXClient()
header_detector = HeaderDetector()


# ============================================================================
# Dataset Search & Discovery
# ============================================================================


@mcp.tool()
async def search_claimm_datasets(
    query: str | None = None,
    tags: str | None = None,
    limit: int = 20,
) -> dict:
    """Search CLAIMM datasets in NETL EDX.

    Args:
        query: Search query (e.g., "rare earth", "produced water", "coal ash")
        tags: Comma-separated tags to filter (e.g., "REE,Coal")
        limit: Maximum datasets to return (default 20)

    Returns datasets with titles, descriptions, tags, and resource download URLs.
    """
    # Always include claimm in search
    search_query = f"claimm {query}" if query else "claimm"

    tag_list = [t.strip() for t in tags.split(",")] if tags else None

    submissions = await edx.search_submissions(
        query=search_query,
        tags=tag_list,
        limit=limit,
    )

    datasets = []
    for sub in submissions:
        resources = [
            {
                "id": r.id,
                "name": r.name,
                "format": r.format,
                "size": r.size,
                "download_url": edx.get_download_url(r.id),
            }
            for r in sub.resources
        ]

        datasets.append(
            {
                "id": sub.id,
                "title": sub.title or sub.name,
                "description": sub.notes,
                "author": sub.author,
                "organization": sub.organization,
                "tags": sub.tags,
                "resource_count": len(resources),
                "resources": resources,
                "created": sub.metadata_created,
                "modified": sub.metadata_modified,
            }
        )

    return {
        "count": len(datasets),
        "query": query,
        "datasets": datasets,
    }


@mcp.tool()
async def get_dataset_details(dataset_id: str) -> dict:
    """Get full details for a specific CLAIMM dataset.

    Args:
        dataset_id: Dataset ID from search results

    Returns complete metadata including all resources with download URLs.
    """
    sub = await edx.get_submission(dataset_id)

    resources = [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "format": r.format,
            "size": r.size,
            "download_url": edx.get_download_url(r.id),
            "created": r.created,
            "modified": r.last_modified,
        }
        for r in sub.resources
    ]

    return {
        "id": sub.id,
        "name": sub.name,
        "title": sub.title,
        "description": sub.notes,
        "author": sub.author,
        "organization": sub.organization,
        "tags": sub.tags,
        "resources": resources,
        "resource_count": len(resources),
        "created": sub.metadata_created,
        "modified": sub.metadata_modified,
    }


@mcp.tool()
async def list_claimm_datasets(limit: int = 50) -> dict:
    """List all CLAIMM datasets.

    Args:
        limit: Maximum datasets to return (default 50)

    Returns list of all datasets in the CLAIMM collection.
    """
    submissions = await edx.search_submissions(query="claimm", limit=limit)

    datasets = []
    for sub in submissions:
        datasets.append(
            {
                "id": sub.id,
                "title": sub.title or sub.name,
                "tags": sub.tags,
                "resource_count": len(sub.resources),
                "formats": list(set(r.format for r in sub.resources if r.format)),
            }
        )

    return {
        "count": len(datasets),
        "datasets": datasets,
    }


# ============================================================================
# Resource Search & Access
# ============================================================================


@mcp.tool()
async def search_resources(
    query: str | None = None,
    format_filter: str | None = None,
    limit: int = 20,
) -> dict:
    """Search for specific resources (files) in CLAIMM.

    Args:
        query: Search query for resource names
        format_filter: Filter by format (e.g., "CSV", "XLSX", "PDF", "JSON")
        limit: Maximum resources to return

    Returns matching resources with download URLs.
    """
    result = await edx.search_resources(
        query=query,
        format_filter=format_filter,
        limit=limit,
    )

    resources = [
        {
            "id": r.id,
            "name": r.name,
            "description": r.description,
            "format": r.format,
            "size": r.size,
            "download_url": edx.get_download_url(r.id),
            "dataset_id": r.package_id,
        }
        for r in result.resources
    ]

    return {
        "count": result.count,
        "returned": len(resources),
        "resources": resources,
    }


@mcp.tool()
async def get_resource_details(resource_id: str) -> dict:
    """Get detailed metadata for a specific resource.

    Args:
        resource_id: Resource ID

    Returns full resource metadata including download URL.
    """
    r = await edx.get_resource(resource_id)

    return {
        "id": r.id,
        "name": r.name,
        "description": r.description,
        "format": r.format,
        "size": r.size,
        "download_url": edx.get_download_url(r.id),
        "dataset_id": r.package_id,
        "created": r.created,
        "modified": r.last_modified,
    }


@mcp.tool()
def get_download_url(resource_id: str) -> str:
    """Get direct download URL for a resource.

    Args:
        resource_id: Resource ID

    Returns the direct download URL.
    """
    return edx.get_download_url(resource_id)


# ============================================================================
# Schema Detection
# ============================================================================


@mcp.tool()
async def detect_file_schema(
    resource_id: str,
    format: str | None = None,
) -> dict:
    """Detect column headers and data types from a CSV or Excel file.

    Uses HTTP Range requests to fetch only file headers without full download.

    Args:
        resource_id: Resource ID of CSV or Excel file
        format: File format (CSV, XLSX) - auto-detected if not provided

    Returns column names, detected types, and sample values.
    """
    result = await header_detector.detect_headers(resource_id, format)

    if result.get("success"):
        result["download_url"] = edx.get_download_url(resource_id)

    return result


@mcp.tool()
async def detect_dataset_schemas(dataset_id: str) -> dict:
    """Detect schemas for all tabular files in a dataset.

    Args:
        dataset_id: Dataset ID

    Returns schema information for all CSV/Excel files in the dataset.
    """
    sub = await edx.get_submission(dataset_id)

    # Find tabular resources
    tabular_formats = {"CSV", "XLSX", "XLS", "TSV"}
    tabular_resources = [
        r for r in sub.resources if r.format and r.format.upper() in tabular_formats
    ]

    if not tabular_resources:
        return {
            "dataset_id": dataset_id,
            "title": sub.title,
            "message": "No tabular files (CSV/Excel) found in this dataset",
            "resource_formats": [r.format for r in sub.resources],
        }

    # Detect schemas
    schemas = []
    for r in tabular_resources:
        result = await header_detector.detect_headers(r.id, r.format)
        result["resource_name"] = r.name
        result["download_url"] = edx.get_download_url(r.id)
        schemas.append(result)

    return {
        "dataset_id": dataset_id,
        "title": sub.title,
        "tabular_files": len(schemas),
        "schemas": schemas,
    }


# ============================================================================
# Statistics & Categories
# ============================================================================


@mcp.tool()
async def get_claimm_statistics() -> dict:
    """Get statistics about CLAIMM datasets.

    Returns counts by format, tag frequency, and other metadata.
    """
    submissions = await edx.search_submissions(query="claimm", limit=200)

    # Aggregate statistics
    format_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    total_resources = 0

    for sub in submissions:
        for r in sub.resources:
            total_resources += 1
            fmt = r.format or "Unknown"
            format_counts[fmt] = format_counts.get(fmt, 0) + 1

        for tag in sub.tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    # Sort by count
    top_formats = sorted(format_counts.items(), key=lambda x: x[1], reverse=True)
    top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    return {
        "total_datasets": len(submissions),
        "total_resources": total_resources,
        "formats": dict(top_formats),
        "top_tags": dict(top_tags),
    }


@mcp.tool()
async def get_datasets_by_category() -> dict:
    """Get CLAIMM datasets organized by category.

    Categories include: Rare Earth Elements, Produced Water, Coal, Geology, etc.
    """
    submissions = await edx.search_submissions(query="claimm", limit=200)

    # Define categories with keywords
    categories = {
        "Rare Earth Elements": ["rare earth", "ree", "lanthanide", "critical mineral"],
        "Produced Water": ["produced water", "brine", "newts", "flowback"],
        "Coal & Coal Byproducts": ["coal", "coal ash", "fly ash", "bottom ash"],
        "Mine Waste": ["mine waste", "tailings", "mining waste"],
        "Lithium": ["lithium"],
        "Geology": ["geology", "geological", "geophysic", "basin"],
        "Geochemistry": ["geochemistry", "geochemical", "chemical analysis"],
    }

    categorized: dict[str, list] = {cat: [] for cat in categories}
    categorized["Other"] = []

    for sub in submissions:
        text = f"{sub.title or ''} {sub.notes or ''} {' '.join(sub.tags)}".lower()

        matched = False
        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                categorized[category].append(
                    {
                        "id": sub.id,
                        "title": sub.title or sub.name,
                        "resource_count": len(sub.resources),
                    }
                )
                matched = True
                break

        if not matched:
            categorized["Other"].append(
                {
                    "id": sub.id,
                    "title": sub.title or sub.name,
                    "resource_count": len(sub.resources),
                }
            )

    # Create summary
    summary = {cat: len(datasets) for cat, datasets in categorized.items()}

    return {
        "total_datasets": len(submissions),
        "category_counts": summary,
        "categories": categorized,
    }


# ============================================================================
# Entry Point
# ============================================================================


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
