"""
Structured data builder — fallback that constructs solver-ready
structured_data from the LLM's semantic parse when the LLM doesn't
produce it in the right format.

This bridges the gap between:
  - LLM output: {"entities": ["Job A", "Job B"], "constraints": [...]}
  - Solver input: {"type": "scheduling", "jobs": [{"name": "Job A", "duration": 3, ...}]}
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def ensure_structured_data(task_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if structured_data has solver-ready content.
    If not, attempt to build it from entities, constraints, description, and raw_input.

    Modifies task_dict in-place and returns it.
    """
    sd = task_dict.get("structured_data", {})
    domain = task_dict.get("domain", "unknown")

    # Scheduling: always try regex extraction — it's more reliable than LLM output
    if domain == "scheduling":
        built = _build_scheduling_data(task_dict)
        if built and built.get("jobs"):
            task_dict["structured_data"] = {**sd, **built}

    # Optimization: need items with 'weight' and 'value', or a valid type
    elif domain == "optimization":
        items = sd.get("items", [])
        items_valid = items and all(
            isinstance(it, dict) and "weight" in it and "value" in it
            for it in items
        )
        if not items_valid and sd.get("type") not in ("knapsack", "assignment"):
            built = _build_optimization_data(task_dict)
            if built:
                task_dict["structured_data"] = {**sd, **built}

    # Planning: need subgoals
    elif domain == "planning" and not sd.get("subgoals"):
        built = _build_planning_data(task_dict)
        if built:
            task_dict["structured_data"] = {**sd, **built}

    return task_dict


def _build_scheduling_data(task_dict: Dict) -> Optional[Dict]:
    """
    Extract job definitions from raw_input text.
    Looks for patterns like: "Job A takes 3 hours" / "task X duration 5"
    """
    text = task_dict.get("raw_input", "") or task_dict.get("description", "")
    entities = task_dict.get("entities", [])

    # Pattern: "Job/Task X takes/duration N hours/units"
    job_pattern = re.compile(
        r'(?:job|task)\s+(\w+)\s+(?:takes|duration|requires|needs|is)\s+(\d+)\s*(?:hours?|units?|h)?',
        re.IGNORECASE
    )

    jobs_by_name: Dict[str, Dict] = {}

    # Extract durations
    for match in job_pattern.finditer(text):
        name = match.group(1).strip()
        duration = int(match.group(2))
        jobs_by_name[name] = {
            "name": f"job_{name}",
            "duration": duration,
            "dependencies": [],
        }

    # Extract dependencies by splitting text into per-job segments
    # to avoid greedy regex matching across jobs.
    # Split on "Job X" boundaries, then look for dependency keywords in each segment.
    known_names = set(jobs_by_name.keys())
    segments = re.split(r'(?=(?:job|task)\s+\w)', text, flags=re.IGNORECASE)
    for seg in segments:
        # Which job does this segment describe?
        seg_match = re.match(r'(?:job|task)\s+(\w+)', seg, re.IGNORECASE)
        if not seg_match:
            continue
        seg_job = seg_match.group(1).strip()
        if seg_job not in jobs_by_name:
            continue
        # Look for dependency keywords in this segment
        dep_match = re.search(
            r'(?:must come after|comes? after|depends on|after|requires|follows)\s+(?:both\s+)?(.+)',
            seg, re.IGNORECASE
        )
        if not dep_match:
            continue
        deps_str = dep_match.group(1).strip()
        # Extract dependency names: single uppercase letters that are known job names
        for candidate in known_names:
            if candidate == seg_job:
                continue
            # Check if the candidate name appears in the deps string as a word
            if re.search(r'\b' + re.escape(candidate) + r'\b', deps_str):
                dep_ref = f"job_{candidate}"
                if dep_ref not in jobs_by_name[seg_job]["dependencies"]:
                    jobs_by_name[seg_job]["dependencies"].append(dep_ref)

    if not jobs_by_name:
        # Fallback: create jobs from entities with default durations
        for i, entity in enumerate(entities):
            name = re.sub(r'[^a-zA-Z0-9_]', '_', str(entity))
            jobs_by_name[name] = {
                "name": name,
                "duration": 1,
                "dependencies": [],
            }

    if not jobs_by_name:
        return None

    jobs = list(jobs_by_name.values())
    total_duration = sum(j["duration"] for j in jobs)

    return {
        "type": "scheduling",
        "jobs": jobs,
        "horizon": total_duration + 10,
        "resources": {},
    }


def _build_optimization_data(task_dict: Dict) -> Optional[Dict]:
    """
    Extract items/capacity from raw_input text.
    Looks for patterns like: "Package 1: 10kg, value $15" / "capacity 50kg"
    """
    text = task_dict.get("raw_input", "") or task_dict.get("description", "")

    # Extract capacity
    cap_match = re.search(r'capacity\s+(\d+)\s*(?:kg|units?|lbs?)?', text, re.IGNORECASE)
    capacity = int(cap_match.group(1)) if cap_match else 100

    # Extract items: "Package 1: 10kg, value $15" or "Item A: 5 kg, worth $10"
    # Handles concatenated units (10kg) and dollar signs ($15)
    item_pattern = re.compile(
        r'(?:package|item)\s+(\w+)\s*[:\-]\s*(\d+)\s*(?:kg|units?|lbs?)?\s*,?\s*(?:value|worth)\s+\$?(\d+)',
        re.IGNORECASE
    )

    items = []
    for match in item_pattern.finditer(text):
        name = f"item_{match.group(1)}"
        weight = int(match.group(2))
        value = int(match.group(3))
        items.append({"name": name, "weight": weight, "value": value})

    if not items:
        return None

    return {
        "type": "knapsack",
        "items": items,
        "capacity": capacity,
    }


def _build_planning_data(task_dict: Dict) -> Optional[Dict]:
    """Build subgoals from entities with sequential dependencies."""
    entities = task_dict.get("entities", [])
    if not entities:
        return None

    subgoals = []
    for i, entity in enumerate(entities):
        name = f"handle_{re.sub(r'[^a-zA-Z0-9_]', '_', str(entity))}"
        deps = [f"handle_{re.sub(r'[^a-zA-Z0-9_]', '_', str(entities[i-1]))}"] if i > 0 else []
        subgoals.append({
            "name": name,
            "description": f"Process {entity}",
            "dependencies": deps,
            "params": {"entity": str(entity)},
        })

    return {"subgoals": subgoals}
