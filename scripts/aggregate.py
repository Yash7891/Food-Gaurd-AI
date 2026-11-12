from collections import defaultdict
from typing import List, Dict


def summarise_detections(detections: List[Dict]) -> Dict:
    """Group per‑item results by produce type and compute a concise summary.

    Returns a dict of the form:
    {
        "apple": {
            "total_items": 3,
            "fresh_items": 2,
            "stale_items": 1,
            "fresh_percentage": 66.7,
            "recommendation": "Keep 2 fresh, discard 1 spoiled"
        },
        ...
    }
    """
    summary = defaultdict(lambda: {"total": 0, "fresh": 0, "stale": 0})
    for d in detections:
        ptype = d.get("produce_type", "unknown")
        freshness = d.get("freshness", "unknown")
        summary[ptype]["total"] += 1
        if freshness == "fresh":
            summary[ptype]["fresh"] += 1
        elif freshness == "stale":
            summary[ptype]["stale"] += 1
    result = {}
    for ptype, stats in summary.items():
        total = stats["total"]
        fresh = stats["fresh"]
        stale = stats["stale"]
        fresh_pct = round(100 * fresh / total, 1) if total else 0.0
        if stale == 0:
            recommendation = "Keep all"
        elif fresh == 0:
            recommendation = "Discard all (all items are spoiled)"
        else:
            recommendation = f"Keep {fresh} fresh, discard {stale} spoiled"
        result[ptype] = {
            "total_items": total,
            "fresh_items": fresh,
            "stale_items": stale,
            "fresh_percentage": fresh_pct,
            "recommendation": recommendation,
        }
    return result

