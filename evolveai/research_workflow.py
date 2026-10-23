"""A bounded, no-LLM research workflow built on the symbolic autonomy engine.

Searches public sources using EvolveAI's existing heuristic search adapter,
checks basic evidence coverage, and produces a traceable report. It does not
claim semantic verification or guarantee source truthfulness.
"""
from __future__ import annotations

from typing import Any, Callable
from urllib.parse import urlparse

from .autonomy import ActionRegistry, ActionResult, ActionSpec, AutonomousEngine, ExecutionPolicy, Goal, GoalStatus, state_to_dict
from .research import research


def _distinct_domains(items: list[dict[str, Any]]) -> set[str]:
    domains: set[str] = set()
    for item in items:
        host = (urlparse(str(item.get("url") or "")).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            domains.add(host)
    return domains


def build_research_engine(search: Callable[[str, int], list[dict[str, Any]]] | None = None,
                          event_sink=None, approval=None) -> AutonomousEngine:
    """Create an engine with read-only public web research actions.

    The search function is injectable so tests run without network access.
    """
    search_fn = search or research
    registry = ActionRegistry()

    def collect_sources(goal: Goal, state) -> ActionResult:
        query = goal.description.strip()
        if not query:
            return ActionResult(False, error="Research query is empty.", cost=0.1)
        items = search_fn(query, 5)
        safe_items = []
        for item in items[:5]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()[:240]
            url = str(item.get("url") or "").strip()[:2000]
            excerpt = str(item.get("excerpt") or "").strip()[:1600]
            if not title or not url.startswith(("https://", "http://")) or len(excerpt) < 30:
                continue
            safe_items.append({
                "title": title, "url": url, "excerpt": excerpt,
                "confidence": item.get("confidence"),
                "source_domain": urlparse(url).hostname or "",
            })
        if not safe_items:
            return ActionResult(False, error="Search returned no usable source excerpts.", cost=1.0)
        state.artifacts["research_sources"] = safe_items
        return ActionResult(
            success=True, observations=["sources_collected"],
            output={"count": len(safe_items)}, cost=1.0,
            evidence=[{"title": x["title"], "url": x["url"]} for x in safe_items],
        )

    def verify_and_report(goal: Goal, state) -> ActionResult:
        sources = state.artifacts.get("research_sources", [])
        domains = _distinct_domains(sources)
        # Coverage check only; this is not semantic fact verification.
        if len(sources) < 2 or len(domains) < 2:
            return ActionResult(
                success=False,
                error="Insufficient independent-source coverage: need at least two usable sources from distinct domains.",
                cost=0.25,
                evidence=[{"title": x["title"], "url": x["url"]} for x in sources],
            )
        lines = [
            "# Research report: " + goal.description, "",
            "## Evidence collected", "",
            f"Collected {len(sources)} usable excerpts across {len(domains)} distinct domains.", "",
        ]
        for index, item in enumerate(sources, start=1):
            lines.extend([
                f"### Source {index}: {item['title']}",
                "URL: " + item["url"], "", item["excerpt"], "",
            ])
        lines.extend([
            "## Verification status", "",
            "- Passed: minimum source count and distinct-domain coverage.",
            "- Not established: factual agreement, source authority, publication freshness, or causal correctness.",
            "- Treat excerpts as evidence to inspect, not independently proven facts.",
        ])
        report = "\n".join(lines)
        state.artifacts["research_report"] = report
        state.artifacts["research_verification"] = {
            "source_count": len(sources),
            "distinct_domains": sorted(domains),
            "checks_passed": ["minimum_source_count", "distinct_domain_coverage"],
            "checks_not_performed": [
                "semantic_claim_comparison", "source_authority_validation",
                "freshness_validation", "independent_fact_checking",
            ],
        }
        return ActionResult(
            success=True, observations=["report_verified"],
            output={"report": report, "verification": state.artifacts["research_verification"]},
            cost=0.5,
            evidence=[{"title": x["title"], "url": x["url"]} for x in sources],
        )

    registry.register(
        ActionSpec(name="collect_sources", description="Search public web sources for the research goal",
                   effects=("sources_collected",), cost=1.0, risk=0.05),
        collect_sources,
    )
    registry.register(
        ActionSpec(name="verify_and_report", description="Check source coverage and compile a traceable report",
                   preconditions=("sources_collected",), effects=("report_verified",), cost=0.5, risk=0.02),
        verify_and_report,
    )
    return AutonomousEngine(
        registry=registry,
        policy=ExecutionPolicy(approval=approval),
        verifier=lambda goal, state: (
            "report_verified" in state.facts
            and isinstance(state.artifacts.get("research_report"), str)
            and bool(state.artifacts["research_report"].strip())
        ),
        event_sink=event_sink,
    )


def research_goal(query: str, max_steps: int = 4, max_cost: float = 3.0) -> Goal:
    """Build a bounded research goal from a query."""
    query = str(query).strip()
    if not query:
        raise ValueError("A non-empty research query is required.")
    return Goal(
        description=query,
        success_criteria=("report_verified",),
        constraints={"mode": "read_only_public_research", "llm": False},
        max_steps=max(1, min(int(max_steps), 8)),
        max_cost=max(0.5, min(float(max_cost), 10.0)),
    )


def run_research(query: str, search: Callable[[str, int], list[dict[str, Any]]] | None = None,
                 event_sink=None) -> dict[str, Any]:
    """Run bounded public research and return a serializable execution trace."""
    state = build_research_engine(search=search, event_sink=event_sink).run(research_goal(query))
    payload = state_to_dict(state)
    payload["report"] = state.artifacts.get("research_report", "")
    payload["verification"] = state.artifacts.get("research_verification", {})
    payload["sources"] = state.artifacts.get("research_sources", [])
    return payload
