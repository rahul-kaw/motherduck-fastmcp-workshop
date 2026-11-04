"""Your MCP server for San Francisco 311 data.

FastMCP Gateway runs this file. Its entrypoint setting is `server.py:mcp`.
"""

import logging
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from sf311.data import BY_ID, LOCAL_NAMES, NEIGHBORHOODS, REQUESTS, SERVICES, Request, closest
from sf311.kit import caller_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sf311")

mcp = FastMCP("sf-311")

MAX_ROWS = 50
MIN_CLOSED_FOR_MEDIAN = 10

GROUPS = {
    "neighborhood": lambda r: r.neighborhood,
    "service": lambda r: r.service,
    "agency": lambda r: r.agency,
    "source": lambda r: r.source,
    "weekday": lambda r: r.opened.strftime("%A"),
    "status": lambda r: r.status,
}


HOUSE_NUMBER = re.compile(r"^(\d+)\s+([^,]+)")


def _mask_address(address: str) -> str:
    """Reduce a street address to its block: "994 HARRISON ST, ..." -> "900 block of HARRISON ST".

    Addresses without a leading house number, such as intersections, are returned unchanged.
    """
    match = HOUSE_NUMBER.match(address)
    if not match:
        return address
    block = int(match.group(1)) // 100 * 100
    return f"{block} block of {match.group(2).strip()}"


def _public(request: Request) -> dict:
    """Return a request as a dict with its address masked to the block."""
    row = request.as_dict()
    row["address"] = _mask_address(row["address"])
    return row


def _check_name(value: str | None, choices: list[str], kind: str) -> None:
    if value is None or value in choices:
        return
    suggestions = closest(value, choices)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ToolError(f"Unknown {kind} {value!r}.{hint} Call dataset_info for the exact {kind} names.")


def _matching(service: str | None = None, neighborhood: str | None = None, status: str | None = None) -> list[Request]:
    _check_name(service, SERVICES, "service")
    _check_name(neighborhood, NEIGHBORHOODS, "neighborhood")
    return [
        r
        for r in REQUESTS
        if (service is None or r.service == service)
        and (neighborhood is None or r.neighborhood == neighborhood)
        and (status is None or r.status == status)
    ]


@mcp.tool(annotations={"readOnlyHint": True})
def dataset_info() -> dict:
    """Describe the 311 dataset this server answers questions about.

    Call this first to learn the date range and the exact service and
    neighborhood names that other tools accept.
    """
    return {
        "description": "San Francisco 311 service requests filed in one week.",
        "first_opened": min(request.opened for request in REQUESTS).isoformat(),
        "last_opened": max(request.opened for request in REQUESTS).isoformat(),
        "requests": len(REQUESTS),
        "services": SERVICES,
        "neighborhoods": NEIGHBORHOODS,
        "local_neighborhood_names": len(LOCAL_NAMES),
    }


@mcp.tool(annotations={"readOnlyHint": True})
def count_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    group_by: Literal["neighborhood", "service", "agency", "source", "weekday", "status"] | None = None,
) -> int | dict[str, int]:
    """Count 311 requests, optionally filtered by service and/or official neighborhood.

    Without group_by, returns one number. With group_by, returns a dict of
    group -> count, largest first, at most 50 groups. Use this for any
    "how many", "which has the most", or ranking question; chain calls to
    drill down (e.g. find the top neighborhood for a service, then group
    that neighborhood by service). Weekday groups are day names like "Monday".

    Use median_close_hours for how fast requests close, and search_requests
    only when you need to see individual requests.
    """
    rows = _matching(service, neighborhood)
    if group_by is None:
        return len(rows)
    key = GROUPS[group_by]
    counts = Counter(key(r) for r in rows if key(r))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return dict(ranked[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def median_close_hours(
    service: str | None = None,
    group_by: Literal["neighborhood", "agency"] | None = None,
) -> float | None | dict[str, float]:
    """Median hours from opened to closed for closed 311 requests, rounded to 0.1.

    Optionally filter by service. Without group_by, returns one number (null
    if nothing has closed). With group_by, returns a dict of group -> median
    hours, fastest first, leaving out groups with fewer than 10 closed
    requests. Open requests are ignored.

    Use count_requests for volumes, not this tool.
    """
    rows = [r for r in _matching(service) if r.hours_to_close is not None]
    if group_by is None:
        return round(median(r.hours_to_close for r in rows), 1) if rows else None
    key = GROUPS[group_by]
    hours: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if key(r):
            hours[key(r)].append(r.hours_to_close)
    medians = {group: round(median(values), 1) for group, values in hours.items() if len(values) >= MIN_CLOSED_FOR_MEDIAN}
    return dict(sorted(medians.items(), key=lambda item: (item[1], item[0]))[:MAX_ROWS])


@mcp.tool(annotations={"readOnlyHint": True})
def search_requests(
    service: str | None = None,
    neighborhood: str | None = None,
    status: Literal["Open", "Closed"] | None = None,
    limit: int = 20,
) -> list[dict]:
    """List individual 311 requests matching the filters, newest first.

    Returns up to `limit` requests (1 to 50, default 20), each with its ID,
    open/close times, status, service, subtype, agency, neighborhood, local
    area name, address (to the block only, e.g. "900 block of HARRISON ST"),
    and source. Use this to show examples or inspect
    specific cases. Do not use it to count or rank: results are capped at
    50, so use count_requests or median_close_hours for totals.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. For totals, use count_requests instead.")
    rows = sorted(_matching(service, neighborhood, status), key=lambda r: r.opened, reverse=True)
    return [_public(r) for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def slowest_closes(
    service: str | None = None,
    neighborhood: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """List the closed 311 requests that took longest to close, slowest first.

    Optionally filter by service and/or official neighborhood. Returns up to
    `limit` requests (1 to 50, default 5), each with all its fields (address
    to the block only) plus `hours_to_close`, the exact hours from opened to
    closed, rounded to 0.1. Use limit=1 for "the longest time any request
    took to close". Open requests are ignored.

    Use median_close_hours for typical close times, and count_requests for
    volumes.
    """
    if not 1 <= limit <= MAX_ROWS:
        raise ToolError(f"limit must be between 1 and {MAX_ROWS}. For typical close times, use median_close_hours instead.")
    rows = [r for r in _matching(service, neighborhood) if r.hours_to_close is not None]
    rows.sort(key=lambda r: (-r.hours_to_close, r.request_id))
    return [{**_public(r), "hours_to_close": round(r.hours_to_close, 1)} for r in rows[:limit]]


@mcp.tool(annotations={"readOnlyHint": True})
def get_request(request_id: str) -> dict:
    """Fetch one 311 request by its ID, with all its fields.

    The address is given only to the block, e.g. "900 block of HARRISON ST".

    Use this when you already have a request ID (for example from
    search_requests). To find requests by service or neighborhood, use
    search_requests instead.
    """
    log.info("get_request %s by %s", request_id, caller_email() or "local")
    request = BY_ID.get(request_id.strip())
    if request is None:
        raise ToolError(f"No request with ID {request_id!r}. Use search_requests to find request IDs.")
    return _public(request)


if __name__ == "__main__":
    mcp.run()
