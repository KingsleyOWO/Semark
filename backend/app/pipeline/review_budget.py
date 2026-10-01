"""Bounded document budgets and estimates for a complete, verified recovery."""
import math
import statistics

from app.pipeline.recovery_models import MAX_REGION_CONTEXT_READS


def document_budget(limits, pages):
    """User limits are hard ceilings, including for existing run configurations."""
    calls, seconds = limits.max_model_calls, limits.max_seconds
    if limits.adaptive_budget:
        # Allow both page audit passes plus actual complete-tool work. The old
        # fixed 600s/12-call margin could not admit a cold image repair at all.
        actions = min(pages, limits.max_recovery_actions) if limits.recovery_enabled else 0
        recovery = tool_cost({}, "read_regions")
        calls = min(calls, max(24, pages * 2 + max(12, actions * (recovery["model_calls"] + 1))))
        seconds = min(seconds, max(900, pages * 75 + max(600, actions * (
            recovery["seconds"] + call_seconds({}, "recovery_plan")))))
    return {"policy": "adaptive" if limits.adaptive_budget else "fixed", "source_pages": pages,
            "max_model_calls": calls, "max_seconds": seconds,
            "hard_max_model_calls": limits.max_model_calls, "hard_max_seconds": limits.max_seconds}


def call_seconds(state, task):
    samples = [c["duration_seconds"] for c in state.get("calls", [])
               if c.get("task") == task and c.get("duration_seconds") and
               (not c.get("error") or str(c["error"]).startswith(("TimeoutError", "CancelledError")))]
    # A fast text-only planner must not lower the allowance for reading images.
    baseline = {"recovery_plan": 45, "read_source": 45, "verify_native_reading": 45,
                "revise_image_regions": 60, "resolve_image_evidence": 60, "read_image_parts": 45, "verify_image_parts": 45, "audit_image_parts": 75, "read_region": 60, "audit_page": 75,
                "read_image_relations": 90, "read_image_relations_alternate": 90,
                "match_source": 45, "verify_match": 45, "repair_delivery": 60}.get(task, 60)
    return max(baseline, statistics.median(samples[-5:]) * 1.25) if samples else baseline


def tool_cost(state, action, *, pictures=0, layout="horizontal2", word_pictures=0, image_read_calls=(3, 3),
              layout_cached=False, detail_regions=0, image_geometry=False):
    """Includes candidate audit AND post-rebuild audit, not just generation."""
    tasks = ["audit_page", "audit_page"]
    overhead = 15
    if action == "ocr_image":
        tasks = []
        overhead = 90
    elif action == "inspect_layout":
        # Reserve the next plan and a full regional repair, including both audits.
        # The query itself performs no reviewer calls and cannot certify a page.
        tasks += ["recovery_plan"] + ["read_region"] * (2 + MAX_REGION_CONTEXT_READS)
        overhead += (0 if layout_cached else 120) + 45
    elif action == "read_source":
        # Each original image uses separate scene/text observations. One bounded
        # split per reader permits at most three calls, including failed format.
        tasks += (["read_image_parts"] * 3 + ["verify_image_parts"] * 3) * pictures
    elif action == "repair_image" and image_geometry:
        tasks = ["read_image_relations"] * 2 + ["audit_image_parts"] * 2 + ["audit_page"] * 2
    elif action == "repair_image":
        # Source reads, bounded evidence batches, one proposal with format repair,
        # three candidate audits and the final page audit all share the hard budget.
        from app.pipeline.image_evidence import MAX_IMAGE_RESOLUTION_CALLS
        tasks = (["read_image_parts"] * image_read_calls[0] + ["verify_image_parts"] * image_read_calls[1]
                 + ["resolve_image_evidence"] * MAX_IMAGE_RESOLUTION_CALLS
                 + ["revise_image_regions", "revise_image_regions", "audit_image_parts", "audit_image_parts", "audit_image_parts", "audit_page"])
    elif action == "read_regions":
        tasks += ["read_region"] * ((2 if layout == "horizontal2" else 4) + MAX_REGION_CONTEXT_READS)
        overhead += 15 * detail_regions
    elif action == "rebuild_delivery":
        tasks += ["repair_delivery"]
    elif action == "resolve_source":
        tasks += ["match_source", "verify_match"]
    elif action == "reparse_pdf":
        overhead += 120
    if action in {"rebuild_delivery", "resolve_source"}:
        # Native pictures survive these tools; source-only reads are reused by the second audit.
        from app.pipeline.image_source_review import WORD_PICTURE_BATCH_SIZE
        tasks += ["verify_native_reading"] * (2 * math.ceil(word_pictures / WORD_PICTURE_BATCH_SIZE))
    estimate = overhead + sum(call_seconds(state, t) for t in tasks)
    completed = [step["duration_seconds"] for step in state.get("recovery", {}).get("steps", [])
                 if step.get("action") == action and step.get("layout", "horizontal2") == layout
                 and step.get("duration_seconds")]
    if completed and not (action == "repair_image" and (image_geometry or sum(image_read_calls) < 4) or action == "inspect_layout" and layout_cached):
        estimate = max(estimate, statistics.median(completed[-5:]) * 1.25)
    return {"model_calls": len(tasks), "seconds": round(estimate, 1),
            "format_fallback": "bounded splits use remaining shared budget; never override limits"}


def timing_summary(calls):
    result = {}
    for call in calls:
        item = result.setdefault(call["task"], {"calls": 0, "seconds": 0, "tokens": 0, "errors": 0})
        item["calls"] += 1
        item["seconds"] = round(item["seconds"] + call.get("duration_seconds", 0), 3)
        item["tokens"] += call.get("tokens", 0)
        item["errors"] += bool(call.get("error"))
    return result
