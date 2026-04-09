"""
Rule-based pricing engine for vehicle body damage assessments.

This module takes a validated DamageAnalysis and produces a deterministic
ILS price estimate using shop pricing tables — no AI inference, no free-form
guessing.

────────────────────────────────────────────────────────────────────────────────
DESIGN PRINCIPLE: TWO CLASSES OF EVIDENCE
────────────────────────────────────────────────────────────────────────────────

  CONFIRMED DAMAGE  (from visible_confirmed_damage)
  ─────────────────
  The AI saw it in the photos or the customer stated it explicitly.
  This is the billable foundation of the estimate.
  → Contributes to BOTH minimum and maximum.
  → Each item is priced individually from the PRICE_TABLE or fallback.
  → If the part appears in replacement_likely_parts, severity is upgraded
    to "high" regardless of the AI's per-item severity — because replacement
    is always more expensive than repair.

  POSSIBLE RELATED DAMAGE  (from possible_related_affected_parts)
  ───────────────────────
  The AI suspects these parts may be affected based on collision physics
  but cannot confirm them from the available photos.
  → Contributes ONLY to the maximum (upper bound), never the minimum.
  → Represented as a percentage uplift on the confirmed maximum.
  → Never listed as separate billable line items — they are uncertainty buffer.
  → Each possible part adds 10% to the maximum, capped at 40%.

  WHY THIS SPLIT MATTERS
  ──────────────────────
  Showing a customer a range of ₪2,000–3,500 (confirmed only) communicates
  confidence. Showing ₪2,000–5,500 (with possible parts widening) is honest
  about what might be found during inspection. The estimator uses the
  physical inspection to move items from "possible" to "confirmed" and narrow
  the range to a final quote.

────────────────────────────────────────────────────────────────────────────────
ALGORITHM
────────────────────────────────────────────────────────────────────────────────

  STEP 1 — Resolve each confirmed damage item
    • Map free-text part name → price table key (fuzzy keyword matching)
    • Upgrade severity to "high" if part is in replacement_likely_parts
    • Handle "unclear" severity as medium_min / high_max
    • Look up from PRICE_TABLE → EXTENDED_PRICE_TABLE → _DEFAULT_FALLBACK
    • Accumulate: confirmed_min += row_min, confirmed_max += row_max

  STEP 2 — Apply possible-parts widening to max only
    • Each possible part: +10% to confirmed_max
    • Cap: +40% maximum total uplift
    • widened_max = confirmed_max × (1 + uplift_pct / 100)

  STEP 3 — Panel misalignment surcharge
    • If panel_misalignment_visible:
        min += ALIGNMENT_SURCHARGE_MIN  (₪300)
        max += ALIGNMENT_SURCHARGE_MAX  (₪1,500)
    • Rationale: alignment work (jig measurement, structural adjustment) is
      always additional work on top of the cosmetic repair.

  STEP 4 — Confidence-based range widening
    • confidence ≥ 0.70 → no adjustment
    • 0.50 ≤ confidence < 0.70 → max × 1.10
    • 0.35 ≤ confidence < 0.50 → min × 0.90, max × 1.20
    • confidence < 0.35 → min × 0.80, max × 1.35
    • Rationale: low-quality images mean the visible damage may be
      underestimated. The minimum shrinks (could be less than feared)
      and the maximum grows (could be worse than visible).

  STEP 5 — Minimum floor
    • Ensure min ≥ ₪500 (no body shop visit is free)
    • Ensure max ≥ min + ₪500

────────────────────────────────────────────────────────────────────────────────
PUBLIC INTERFACE
────────────────────────────────────────────────────────────────────────────────

    from app.services.pricing_service import price_damage, PriceEstimate
    from app.prompts.damage_analysis_prompt import DamageAnalysis

    estimate: PriceEstimate = price_damage(analysis)
    stored = estimate.model_dump()   # JSON-safe dict for DB / API response

    # Or from a raw dict (e.g. from Conversation.ai_analysis):
    estimate = price_from_dict(conv.ai_analysis)
"""

from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field

from app.prompts.damage_analysis_prompt import (
    ConfirmedDamageItem,
    DamageAnalysis,
    validate_analysis,
)
from app.services.pricing import (
    LABOR_RATE_ILS,
    PAINT_COST_PER_PANEL,
    PRICE_TABLE,
    get_price_range,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# Minimum floor for any estimate — no intake assessment is free
_ESTIMATE_FLOOR_MIN: int = 500
_ESTIMATE_FLOOR_GAP: int = 500   # max must always exceed min by at least this

# Panel misalignment surcharge (₪) — added when misalignment is visually confirmed
# This covers jig measurement, structural re-alignment, and re-fitting work.
_ALIGNMENT_SURCHARGE_MIN: int = 300
_ALIGNMENT_SURCHARGE_MAX: int = 1_500

# Possible-parts widening — each unconfirmed related part adds this % to the max
_POSSIBLE_PART_UPLIFT_PCT: int = 10
_POSSIBLE_PART_UPLIFT_CAP: int = 40   # never more than 40% from possible parts

# Confidence adjustment thresholds
_CONF_GOOD:   float = 0.70   # no adjustment above this
_CONF_OK:     float = 0.50   # mild widening
_CONF_POOR:   float = 0.35   # moderate widening (needs_human_review floor)


# ══════════════════════════════════════════════════════════════════════════════
# EXTENDED PRICE TABLE
# ══════════════════════════════════════════════════════════════════════════════
#
# Parts that are in PART_LABELS but not in the main PRICE_TABLE in pricing.py,
# plus a _default fallback for any part the resolver cannot identify.
#
# Structure: part_key → severity → (min_ils, max_ils)
# All values include parts + labor + paint (same accounting as PRICE_TABLE).

_EXTENDED: dict[str, dict[str, tuple[int, int]]] = {

    "taillight_left": {
        "low":    (500,  1_200),   # lens polish or minor chip/crack repair
        "medium": (1_000, 3_000),  # taillight unit replacement (standard)
        "high":   (2_000, 6_000),  # full LED/OLED taillight assembly
        "unclear": (800, 4_000),
    },
    "taillight_right": {
        "low":    (500,  1_200),
        "medium": (1_000, 3_000),
        "high":   (2_000, 6_000),
        "unclear": (800, 4_000),
    },
    "wheel_arch": {
        "low":    (200,  700),     # trim clip / minor arch liner repair
        "medium": (500,  1_500),   # arch trim or liner replacement
        "high":   (1_000, 3_000),  # structural arch metal repair
        "unclear": (400, 2_000),
    },

    # ── Catch-all fallback — used when part cannot be matched ─────────────────
    # Priced conservatively: broad range, skewed toward medium Israeli market rates.
    "_default": {
        "low":    (600,  2_000),
        "medium": (1_200, 4_000),
        "high":   (2_500, 8_000),
        "unclear": (1_000, 5_000),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PART NAME → PRICE TABLE KEY RESOLVER
# ══════════════════════════════════════════════════════════════════════════════
#
# The AI returns free-text part names like "front bumper cover", "left front
# fender", "rear left door".  These must map to keys in PRICE_TABLE or
# _EXTENDED so the engine can look up a price.
#
# Strategy: ordered list of (keyword_set, price_table_key) pairs.
# A part name matches the first entry where ALL keywords in the set appear
# as substrings of the normalised (lowercased) part name.
#
# Ordering rules:
#   1. More specific patterns (more keywords) come BEFORE generic ones.
#   2. Side-qualified patterns come before side-agnostic ones.
#   3. Hebrew patterns are checked last (AI is instructed to use English).
#
# Example walk for "rear right door":
#   ("front", "left", "door")  → "front" not present → skip
#   ("rear", "right", "door")  → all present → match "door_rear_right"  ✓

_PART_PATTERNS: list[tuple[tuple[str, ...], str]] = [

    # ── Doors (most specific: position + side) ────────────────────────────────
    (("front", "left",  "door"), "door_front_left"),
    (("front", "right", "door"), "door_front_right"),
    (("rear",  "left",  "door"), "door_rear_left"),
    (("rear",  "right", "door"), "door_rear_right"),
    (("back",  "left",  "door"), "door_rear_left"),
    (("back",  "right", "door"), "door_rear_right"),
    (("driver",    "door"),      "door_front_left"),
    (("passenger", "door"),      "door_front_right"),
    (("left",  "door"),          "door_front_left"),    # side only → assume front
    (("right", "door"),          "door_front_right"),
    (("door",),                  "door_front_left"),    # no qualifier → generic

    # ── Fenders / wings ───────────────────────────────────────────────────────
    (("front", "left",  "fender"), "fender_left"),
    (("front", "right", "fender"), "fender_right"),
    (("left",  "fender"),          "fender_left"),
    (("right", "fender"),          "fender_right"),
    (("left",  "wing"),            "fender_left"),
    (("right", "wing"),            "fender_right"),
    (("fender",),                  "fender_left"),      # generic → left as stand-in

    # ── Bumpers ───────────────────────────────────────────────────────────────
    (("front", "bumper"),          "bumper_front"),
    (("rear",  "bumper"),          "bumper_rear"),
    (("back",  "bumper"),          "bumper_rear"),
    (("bumper",),                  "bumper_front"),     # unspecified → front (most common)

    # ── Headlights ────────────────────────────────────────────────────────────
    (("left",  "headlight"),       "headlight_left"),
    (("right", "headlight"),       "headlight_right"),
    (("left",  "head", "light"),   "headlight_left"),
    (("right", "head", "light"),   "headlight_right"),
    (("left",  "front", "light"),  "headlight_left"),
    (("right", "front", "light"),  "headlight_right"),
    (("headlight",),               "headlight_left"),   # generic → left as stand-in

    # ── Taillights ────────────────────────────────────────────────────────────
    (("left",  "taillight"),       "taillight_left"),
    (("right", "taillight"),       "taillight_right"),
    (("left",  "tail", "light"),   "taillight_left"),
    (("right", "tail", "light"),   "taillight_right"),
    (("left",  "rear", "light"),   "taillight_left"),
    (("right", "rear", "light"),   "taillight_right"),
    (("taillight",),               "taillight_left"),   # generic → left as stand-in

    # ── Mirrors ───────────────────────────────────────────────────────────────
    (("left",  "mirror"),          "mirror_left"),
    (("right", "mirror"),          "mirror_right"),
    (("driver",    "mirror"),      "mirror_left"),
    (("passenger", "mirror"),      "mirror_right"),
    (("mirror",),                  "mirror_left"),      # generic → left as stand-in

    # ── Quarter panels / rear panels ──────────────────────────────────────────
    (("left",  "quarter"),         "quarter_left"),
    (("right", "quarter"),         "quarter_right"),
    (("rear",  "left",  "panel"),  "quarter_left"),
    (("rear",  "right", "panel"),  "quarter_right"),
    (("quarter",),                 "quarter_left"),     # generic → left as stand-in

    # ── Sills / rockers ───────────────────────────────────────────────────────
    (("left",  "sill"),            "sill_left"),
    (("right", "sill"),            "sill_right"),
    (("left",  "rocker"),          "sill_left"),
    (("right", "rocker"),          "sill_right"),
    (("left",  "side", "skirt"),   "sill_left"),
    (("right", "side", "skirt"),   "sill_right"),
    (("sill",),                    "sill_left"),        # generic → left as stand-in
    (("rocker",),                  "sill_left"),

    # ── Single-sided parts ────────────────────────────────────────────────────
    (("hood",),                    "hood"),
    (("bonnet",),                  "hood"),
    (("trunk",),                   "trunk_lid"),
    (("boot",),                    "trunk_lid"),
    (("tailgate",),                "trunk_lid"),
    (("liftgate",),                "trunk_lid"),
    (("roof",),                    "roof"),
    (("grille",),                  "grille"),
    (("grill",),                   "grille"),
    (("wheel", "arch"),            "wheel_arch"),

    # ── Hebrew part names (AI sometimes uses Hebrew despite instructions) ──────
    (("פגוש קדמי",),               "bumper_front"),
    (("פגוש אחורי",),              "bumper_rear"),
    (("פגוש",),                    "bumper_front"),
    (("מכסה מנוע",),               "hood"),
    (("תא מטען",),                 "trunk_lid"),
    (("דלת קדמית שמאל",),         "door_front_left"),
    (("דלת קדמית ימין",),          "door_front_right"),
    (("דלת אחורית שמאל",),        "door_rear_left"),
    (("דלת אחורית ימין",),         "door_rear_right"),
    (("כנף שמאל",),                "fender_left"),
    (("כנף ימין",),                "fender_right"),
    (("פנס קדמי שמאל",),          "headlight_left"),
    (("פנס קדמי ימין",),           "headlight_right"),
    (("פנס אחורי שמאל",),         "taillight_left"),
    (("פנס אחורי ימין",),          "taillight_right"),
    (("מראה שמאל",),               "mirror_left"),
    (("מראה ימין",),               "mirror_right"),
    (("רכסית שמאל",),              "quarter_left"),
    (("רכסית ימין",),              "quarter_right"),
    (("ספת דלת שמאל",),           "sill_left"),
    (("ספת דלת ימין",),            "sill_right"),
    (("גג",),                      "roof"),
    (("רשת אוויר",),               "grille"),
]


def _resolve_part_key(part_name: str) -> str | None:
    """
    Map a free-text AI part name to a price table key.

    Returns None if no pattern matches.  The caller falls through to
    _EXTENDED["_default"] when this returns None.

    Examples:
        "front bumper cover" → "bumper_front"
        "left front fender"  → "fender_left"
        "rear left door"     → "door_rear_left"
        "left taillight"     → "taillight_left"
        "completely unknown" → None
    """
    norm = part_name.lower().strip()
    for keywords, key in _PART_PATTERNS:
        if all(kw.lower() in norm for kw in keywords):
            return key
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PRICE LOOKUP
# ══════════════════════════════════════════════════════════════════════════════

def _lookup_price(part_key: str | None, severity: str) -> tuple[int, int]:
    """
    Look up a (min, max) ILS price for a resolved part key and severity.

    Resolution order:
      1. PRICE_TABLE (main pricing.py table)
      2. _EXTENDED (taillights, wheel arch, etc.)
      3. _EXTENDED["_default"] (catch-all)

    "unclear" severity is handled by taking medium_min and high_max —
    a conservative range that acknowledges the uncertainty.
    """
    effective = severity

    # Normalise "unclear" before table lookup
    if severity == "unclear":
        # Use medium min + high max as a wide, conservative bracket
        if part_key and part_key in PRICE_TABLE:
            entry_med = PRICE_TABLE[part_key].get("medium")
            entry_high = PRICE_TABLE[part_key].get("high")
            if entry_med and entry_high:
                avg_paint_low_m,  _    = PAINT_COST_PER_PANEL["medium"]
                _,  avg_paint_high_h   = PAINT_COST_PER_PANEL["high"]
                p_min = (
                    entry_med.parts_low
                    + int(entry_med.labor_hours_low * LABOR_RATE_ILS)
                    + entry_med.paint_panels * avg_paint_low_m
                )
                p_max = (
                    entry_high.parts_high
                    + int(entry_high.labor_hours_high * LABOR_RATE_ILS)
                    + entry_high.paint_panels * avg_paint_high_h
                )
                return p_min, p_max
        # No PRICE_TABLE entry → fall through to _EXTENDED / _default
        ext_key = part_key if (part_key and part_key in _EXTENDED) else "_default"
        lo, hi = _EXTENDED[ext_key]["unclear"]
        return lo, hi

    # Standard severity: try PRICE_TABLE first
    if part_key and part_key in PRICE_TABLE:
        result = get_price_range(part_key, effective)  # type: ignore[arg-type]
        if result:
            return result

    # PRICE_TABLE miss or None part_key: try _EXTENDED
    ext_key = part_key if (part_key and part_key in _EXTENDED) else "_default"
    row = _EXTENDED[ext_key]
    if effective in row:
        return row[effective]

    # Severity not in extended entry either → use _default
    return _EXTENDED["_default"].get(effective, _EXTENDED["_default"]["medium"])


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

class LineItem(BaseModel):
    """
    Pricing breakdown for one confirmed damaged part.

    Used in PriceEstimate.line_items.  The complete set of line items
    shows the estimator exactly how the total was built.
    """
    part_name:          str            # original AI text, e.g. "front bumper cover"
    part_key:           str | None     # resolved table key, e.g. "bumper_front"
    damage_type:        str            # from ConfirmedDamageItem.damage_type
    item_severity:      str            # as reported by the AI
    effective_severity: str            # used for pricing (may be upgraded)
    is_replacement:     bool           # severity was upgraded because part is in replacement_likely_parts
    price_min_ils:      int
    price_max_ils:      int
    table_source:       Literal["price_table", "extended_table", "fallback"]
    note:               str = ""       # AI observation note, passed through for context


class PriceEstimate(BaseModel):
    """
    Deterministic price estimate produced by the pricing engine.

    The primary output fields (estimated_min_price_ils, estimated_max_price_ils,
    estimate_explanation, recommended_next_step) are the four values
    the caller needs for display and storage.

    All other fields are the engine's working data — useful for the admin
    dashboard and for debugging estimate logic without re-running the AI.

    Serialise with .model_dump() for storage in Conversation.ai_analysis
    or a separate DB column.
    """

    # ── Primary output (as specified) ─────────────────────────────────────────
    estimated_min_price_ils: int = Field(
        description="Lower bound of the ILS estimate, based on confirmed damage only."
    )
    estimated_max_price_ils: int = Field(
        description=(
            "Upper bound of the ILS estimate. Includes confirmed damage "
            "plus possible-parts widening and confidence adjustment."
        )
    )
    estimate_explanation:    str = Field(
        description="Hebrew breakdown of how the estimate was built."
    )
    recommended_next_step:   str = Field(
        description="Hebrew recommendation for the customer's next action."
    )

    # ── Line items (per confirmed part) ───────────────────────────────────────
    line_items: list[LineItem] = Field(
        description="One entry per confirmed damaged part, showing individual pricing."
    )

    # ── Intermediate totals (for transparency) ────────────────────────────────
    confirmed_min_ils: int = Field(
        description="Sum of confirmed damage line items, before any adjustments."
    )
    confirmed_max_ils: int = Field(
        description="Sum of confirmed damage line items, before any adjustments."
    )

    # ── Adjustments applied ────────────────────────────────────────────────────
    possible_parts_count:       int   = Field(description="Number of possible related parts.")
    possible_uplift_pct:        int   = Field(description="% added to max for possible parts.")
    misalignment_surcharge_min: int   = Field(description="₪ added to min for panel misalignment.")
    misalignment_surcharge_max: int   = Field(description="₪ added to max for panel misalignment.")
    confidence_widened:         bool  = Field(description="True if confidence triggered range widening.")
    confidence_used:            float = Field(description="Confidence value from the analysis.")

    # ── Unresolved parts (for admin visibility) ───────────────────────────────
    parts_unresolved: list[str] = Field(
        description=(
            "Part names that could not be matched to the price table. "
            "Priced using the default fallback range."
        )
    )

    # ── Meta ──────────────────────────────────────────────────────────────────
    needs_human_review: bool = Field(
        description="Propagated from the analysis — True means physical inspection is essential."
    )
    is_preliminary:     bool = Field(
        default=True,
        description="Always True for intake assessments. Final price requires physical inspection."
    )


# ══════════════════════════════════════════════════════════════════════════════
# SEVERITY UPGRADE
# ══════════════════════════════════════════════════════════════════════════════

def _effective_severity(item: ConfirmedDamageItem, replacement_parts: list[str]) -> tuple[str, bool]:
    """
    Determine the severity to use for pricing an item.

    If the item's part name appears (case-insensitive substring) in
    replacement_likely_parts, upgrade to "high" regardless of the AI's
    per-item severity.

    Returns (effective_severity, is_replacement).

    Why force "high" for replacement parts?
    The AI flags a part as likely-to-replace when it sees cracks, shattering,
    or deformation that makes repair impractical. Pricing a replacement part
    at "low" or "medium" severity would understate the cost — those tiers
    assume the part can be repaired.  "High" in the price table corresponds
    to replacement cost + labor.
    """
    part_lower = item.part.lower()
    for rp in replacement_parts:
        if part_lower in rp.lower() or rp.lower() in part_lower:
            return "high", True
    return item.severity, False


# ══════════════════════════════════════════════════════════════════════════════
# CONFIRMATION: SPECIAL CASE PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
#
# These flags annotate the estimate explanation with scenario-specific language.
# They do NOT change the numeric output — the price table already encodes the
# economics. They exist to make the explanation more specific and professional.

def _detect_scenario(analysis: DamageAnalysis) -> list[str]:
    """
    Return a list of active scenario labels for the current analysis.
    Used to inject scenario-specific language into the explanation.

    Scenarios are mutually non-exclusive — a single assessment can trigger
    several (e.g. "bumper_cracked" + "light_damage" + "misalignment").
    """
    scenarios: list[str] = []

    damage_types = set(analysis.damage_types)
    parts_lower  = {p.lower() for p in analysis.primary_affected_parts}
    replacement  = {r.lower() for r in analysis.replacement_likely_parts}

    # ── Bumper scenarios ──────────────────────────────────────────────────────

    has_bumper = any("bumper" in p for p in parts_lower)

    if has_bumper and damage_types <= {"scratch", "paint_transfer", "abrasion", "chip"}:
        # Only surface damage — no structural concern
        scenarios.append("bumper_surface_only")

    if has_bumper and ("dent" in damage_types or "bumper_deformation" in damage_types
                       or "deformation" in damage_types):
        if analysis.paint_damage:
            scenarios.append("bumper_dent_with_paint")

    if has_bumper and ("crack" in damage_types or "broken_plastic" in damage_types
                       or "tear" in damage_types):
        scenarios.append("bumper_cracked_plastic")

    # ── Fender scenarios ──────────────────────────────────────────────────────

    has_fender = any("fender" in p or "wing" in p or "כנף" in p for p in parts_lower)

    if has_fender and ("dent" in damage_types or "crease" in damage_types):
        if analysis.paint_damage:
            scenarios.append("fender_dent_with_paint")

    # ── Door scenarios ────────────────────────────────────────────────────────

    has_door = any("door" in p or "דלת" in p for p in parts_lower)

    if has_door:
        scenarios.append("door_damage")

    # ── Light scenarios ───────────────────────────────────────────────────────

    if analysis.light_damage:
        scenarios.append("light_damage")

    # ── Trunk / tailgate alignment ────────────────────────────────────────────

    has_trunk = any(
        kw in p for p in parts_lower for kw in ("trunk", "tailgate", "boot", "תא מטען")
    )
    if has_trunk and analysis.panel_misalignment_visible:
        scenarios.append("trunk_alignment")

    # ── Multiple panels ───────────────────────────────────────────────────────

    if len(analysis.primary_affected_parts) >= 3:
        scenarios.append("multi_panel")

    # ── Panel misalignment (general) ──────────────────────────────────────────

    if analysis.panel_misalignment_visible and "trunk_alignment" not in scenarios:
        scenarios.append("panel_misalignment")

    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# HEBREW TEXT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

_DAMAGE_TYPE_HE: dict[str, str] = {
    "scratch":            "שריטה",
    "paint_transfer":     "העברת צבע",
    "paint_peeling":      "קילוף צבע",
    "dent":               "שקע",
    "crease":             "קמט",
    "crack":              "סדק",
    "broken_plastic":     "פלסטיק שבור",
    "bumper_deformation": "עיוות פגוש",
    "misalignment":       "חוסר יישור",
    "gap":                "פער בין פנלים",
    "deformation":        "עיוות",
    "abrasion":           "שחיקה",
    "chip":               "ניסוק צבע",
    "tear":               "קרע",
    "other":              "נזק",
}

_SEVERITY_HE: dict[str, str] = {
    "low":     "נמוכה",
    "medium":  "בינונית",
    "high":    "גבוהה",
    "unclear": "לא ברורה",
}

_SEVERITY_EMOJI: dict[str, str] = {
    "low":     "🟢",
    "medium":  "🟡",
    "high":    "🔴",
    "unclear": "⚪",
}


def _ils(amount: int) -> str:
    """Format an integer as a Hebrew-market price string: "₪2,500"."""
    return f"₪{amount:,}"


def _build_explanation(
    line_items:              list[LineItem],
    confirmed_min:           int,
    confirmed_max:           int,
    possible_parts:          list[str],
    possible_uplift_pct:     int,
    align_min:               int,
    align_max:               int,
    final_min:               int,
    final_max:               int,
    confidence:              float,
    confidence_widened:      bool,
    missing_angles:          list[str],
    scenarios:               list[str],
) -> str:
    """
    Build a professional Hebrew explanation of the estimate.

    Structure:
      1. Confirmed damage line items with individual ranges
      2. Sum of confirmed items
      3. Possible related parts (if any)
      4. Panel misalignment surcharge (if any)
      5. Confidence note (if widened)
      6. Final range
      7. Preliminary disclaimer
    """
    lines: list[str] = []

    # ── SECTION 1: Confirmed damage ───────────────────────────────────────────
    lines.append("*בסיס ההערכה — נזק מאושר ויזואלית:*")

    for item in line_items:
        dmg_he  = _DAMAGE_TYPE_HE.get(item.damage_type, item.damage_type)
        sev_he  = _SEVERITY_HE.get(item.effective_severity, item.effective_severity)
        sev_em  = _SEVERITY_EMOJI.get(item.effective_severity, "")
        replace = " _(החלפה סבירה)_" if item.is_replacement else ""
        lines.append(
            f"• {item.part_name} — {dmg_he} — חומרה {sev_he} {sev_em}{replace}: "
            f"{_ils(item.price_min_ils)}–{_ils(item.price_max_ils)}"
        )

    lines.append(f"\nסה\"כ נזק מאושר: *{_ils(confirmed_min)}–{_ils(confirmed_max)}*")

    # ── SECTION 2: Possible related parts ────────────────────────────────────
    if possible_parts:
        lines.append("\n*אזורים לבדיקה בשטח — לא מאושרים ויזואלית:*")
        for pp in possible_parts:
            lines.append(f"• {pp}")
        lines.append(
            f"חלקים אלו עשויים להרחיב את היקף העבודה. "
            f"אי-ודאות זו הוסיפה *{possible_uplift_pct}%* לתקרת ההערכה."
        )

    # ── SECTION 3: Misalignment surcharge ────────────────────────────────────
    if align_min > 0 or align_max > 0:
        lines.append(
            f"\n*תוספת יישור פנלים:* "
            f"{_ils(align_min)}–{_ils(align_max)}"
            f" _(עבודות כיוון ומדידה בג'יג)_"
        )

    # ── SECTION 4: Confidence note ────────────────────────────────────────────
    if confidence_widened:
        conf_pct = int(confidence * 100)
        lines.append(
            f"\n⚠️ *ביטחון הניתוח: {conf_pct}%* — "
            "הטווח הורחב בשל איכות תמונות או זוויות חסרות."
        )

    if missing_angles:
        lines.append("זוויות שהיו מסייעות לאומדן מדויק יותר:")
        for angle in missing_angles:
            lines.append(f"  • {angle}")

    # ── SECTION 5: Scenario notes ─────────────────────────────────────────────
    scenario_notes: list[str] = []

    if "bumper_surface_only" in scenarios:
        scenario_notes.append("נזק שטחי לפגוש — ייתכן שיספיקו ליטוש וצביעה מקומית ללא פירוק.")
    if "bumper_cracked_plastic" in scenarios:
        scenario_notes.append("פלסטיק סדוק/שבור — סביר שיידרש החלפת פגוש ולא תיקון.")
    if "trunk_alignment" in scenarios:
        scenario_notes.append("חוסר יישור מכסה תא מטען — ייתכן נזק לצירים, מנגנון נעילה, או נקודות עיגון.")
    if "multi_panel" in scenarios:
        scenario_notes.append(
            f"נזק ל-{len(line_items)} פנלים — עבודה מרובת פנלים לעיתים מייעלת את זמן הצביעה."
        )

    if scenario_notes:
        lines.append("\n*הערות ספציפיות:*")
        for note in scenario_notes:
            lines.append(f"• {note}")

    # ── SECTION 6: Final range ────────────────────────────────────────────────
    lines.append(f"\n*טווח הערכה סופי: {_ils(final_min)}–{_ils(final_max)}*")
    lines.append("_(הערכה ראשונית בלבד — המחיר הסופי ייקבע לאחר בדיקה פיזית של הרכב)_")

    return "\n".join(lines)


def _build_next_step(
    analysis:    DamageAnalysis,
    scenarios:   list[str],
    final_min:   int,
    final_max:   int,
) -> str:
    """
    Build a Hebrew next-step recommendation tailored to the analysis result.

    The recommendation becomes more urgent as severity and complexity increase.
    needs_human_review always results in a direct inspection request.
    """
    if analysis.needs_human_review:
        return (
            "⚠️ מומלץ להגיע לבדיקה פיזית בהקדם — "
            "לא ניתן להפיק אומדן אמין מהתמונות הקיימות בלבד. "
            "נציג המוסך יצור אתכם קשר לתיאום."
        )

    if analysis.severity == "high" or "bumper_cracked_plastic" in scenarios:
        return (
            "🔧 נזק משמעותי — מומלץ לתאם בדיקה פיזית בהקדם האפשרי. "
            "לאחר הבדיקה נפיק אומדן מחייב ונסכם תאריך לתחילת עבודה."
        )

    if "panel_misalignment" in scenarios or "trunk_alignment" in scenarios:
        return (
            "📐 נצפתה הגחה בין פנלים — יש לבדוק את נקודות העיגון ומנגנוני הנעילה פיזית. "
            "מומלץ לתאם הגעה לבדיקת הנזק ואומדן מחייב."
        )

    if analysis.severity == "medium" or len(analysis.primary_affected_parts) >= 2:
        return (
            "📋 מומלץ לתאם הגעה לבדיקה ואומדן מחייב. "
            f"הערכת העלות הראשונית: {_ils(final_min)}–{_ils(final_max)}. "
            "המחיר הסופי ייקבע לאחר בדיקה פיזית."
        )

    # Low severity — likely cosmetic only
    return (
        "✨ נזק נראה קוסמטי בעיקרו. "
        f"הערכה ראשונית: {_ils(final_min)}–{_ils(final_max)}. "
        "ניתן לתאם הגעה לאומדן מחייב ולביצוע העבודה."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def price_damage(analysis: DamageAnalysis) -> PriceEstimate:
    """
    Produce a deterministic ILS price estimate from a validated DamageAnalysis.

    This function is the primary entry point. It accepts the Pydantic model
    returned by damage_analysis_service.analyze_body_damage().

    The estimate is built in five steps:
      1. Price each confirmed damaged part individually
      2. Widen the max for possible related parts (buffer only, not billable)
      3. Add misalignment surcharge if visually confirmed
      4. Widen the range further if confidence is low
      5. Apply minimum floor

    Parameters
    ──────────
    analysis
        A validated DamageAnalysis instance. Must have at least one
        item in visible_confirmed_damage.

    Returns
    ───────
    PriceEstimate
        Fully populated estimate. Call .model_dump() to store or serialize.

    Raises
    ──────
    ValueError
        If analysis.visible_confirmed_damage is empty.
        (The AI should always produce at least one confirmed item if it reaches
        the pricing step — empty means something went wrong upstream.)
    """
    if not analysis.visible_confirmed_damage:
        raise ValueError(
            "Cannot price an analysis with no confirmed damage items. "
            "Check that the AI analysis completed successfully."
        )

    logger.info(
        "pricing_started | confirmed_items=%d possible_parts=%d "
        "severity=%s confidence=%.2f needs_review=%s",
        len(analysis.visible_confirmed_damage),
        len(analysis.possible_related_affected_parts),
        analysis.severity,
        analysis.confidence,
        analysis.needs_human_review,
    )

    # ── STEP 1: Price each confirmed item ─────────────────────────────────────
    line_items:      list[LineItem] = []
    parts_unresolved: list[str]     = []

    for item in analysis.visible_confirmed_damage:
        part_key = _resolve_part_key(item.part)
        effective_sev, is_replacement = _effective_severity(
            item, analysis.replacement_likely_parts
        )

        p_min, p_max = _lookup_price(part_key, effective_sev)

        # Determine table source for transparency
        if part_key and part_key in PRICE_TABLE:
            source: Literal["price_table", "extended_table", "fallback"] = "price_table"
        elif part_key and part_key in _EXTENDED:
            source = "extended_table"
        else:
            source = "fallback"
            if part_key is None:
                parts_unresolved.append(item.part)

        line_items.append(LineItem(
            part_name=item.part,
            part_key=part_key,
            damage_type=item.damage_type,
            item_severity=item.severity,
            effective_severity=effective_sev,
            is_replacement=is_replacement,
            price_min_ils=p_min,
            price_max_ils=p_max,
            table_source=source,
            note=item.note,
        ))

        logger.debug(
            "pricing_line_item | part=%r key=%s sev=%s→%s replacement=%s "
            "min=%d max=%d source=%s",
            item.part, part_key, item.severity, effective_sev,
            is_replacement, p_min, p_max, source,
        )

    confirmed_min = sum(li.price_min_ils for li in line_items)
    confirmed_max = sum(li.price_max_ils for li in line_items)

    # ── STEP 2: Possible-parts widening (max only) ────────────────────────────
    #
    # Each possible related part adds _POSSIBLE_PART_UPLIFT_PCT % to the max.
    # The min is NOT touched — possible parts are not confirmed billable work.
    n_possible       = len(analysis.possible_related_affected_parts)
    raw_uplift_pct   = n_possible * _POSSIBLE_PART_UPLIFT_PCT
    uplift_pct       = min(raw_uplift_pct, _POSSIBLE_PART_UPLIFT_CAP)
    widened_max      = int(confirmed_max * (1 + uplift_pct / 100))

    running_min = confirmed_min
    running_max = widened_max

    # ── STEP 3: Panel misalignment surcharge ──────────────────────────────────
    align_min = 0
    align_max = 0
    if analysis.panel_misalignment_visible:
        align_min = _ALIGNMENT_SURCHARGE_MIN
        align_max = _ALIGNMENT_SURCHARGE_MAX
        running_min += align_min
        running_max += align_max

    # ── STEP 4: Confidence-based widening ─────────────────────────────────────
    conf = analysis.confidence
    confidence_widened = False

    if conf < _CONF_POOR:
        # Very low confidence — significant uncertainty in both directions
        running_min = int(running_min * 0.80)
        running_max = int(running_max * 1.35)
        confidence_widened = True
    elif conf < _CONF_OK:
        # Poor quality images — widen both sides modestly
        running_min = int(running_min * 0.90)
        running_max = int(running_max * 1.20)
        confidence_widened = True
    elif conf < _CONF_GOOD:
        # Acceptable images, some angles missing — widen max only
        running_max = int(running_max * 1.10)
        confidence_widened = True

    # ── STEP 5: Floor ─────────────────────────────────────────────────────────
    final_min = max(running_min, _ESTIMATE_FLOOR_MIN)
    final_max = max(running_max, final_min + _ESTIMATE_FLOOR_GAP)

    # Round to nearest 50 for cleaner display
    final_min = int(round(final_min / 50) * 50)
    final_max = int(round(final_max / 50) * 50)

    # ── Build output ──────────────────────────────────────────────────────────
    scenarios = _detect_scenario(analysis)

    explanation = _build_explanation(
        line_items=line_items,
        confirmed_min=confirmed_min,
        confirmed_max=confirmed_max,
        possible_parts=analysis.possible_related_affected_parts,
        possible_uplift_pct=uplift_pct,
        align_min=align_min,
        align_max=align_max,
        final_min=final_min,
        final_max=final_max,
        confidence=conf,
        confidence_widened=confidence_widened,
        missing_angles=analysis.missing_angles,
        scenarios=scenarios,
    )

    next_step = _build_next_step(analysis, scenarios, final_min, final_max)

    estimate = PriceEstimate(
        estimated_min_price_ils=final_min,
        estimated_max_price_ils=final_max,
        estimate_explanation=explanation,
        recommended_next_step=next_step,
        line_items=line_items,
        confirmed_min_ils=confirmed_min,
        confirmed_max_ils=confirmed_max,
        possible_parts_count=n_possible,
        possible_uplift_pct=uplift_pct,
        misalignment_surcharge_min=align_min,
        misalignment_surcharge_max=align_max,
        confidence_widened=confidence_widened,
        confidence_used=conf,
        parts_unresolved=parts_unresolved,
        needs_human_review=analysis.needs_human_review,
    )

    logger.info(
        "pricing_done | final_min=%d final_max=%d "
        "confirmed_min=%d confirmed_max=%d "
        "uplift_pct=%d align_min=%d align_max=%d "
        "confidence_widened=%s parts_unresolved=%d",
        final_min, final_max,
        confirmed_min, confirmed_max,
        uplift_pct, align_min, align_max,
        confidence_widened, len(parts_unresolved),
    )

    return estimate


def price_from_dict(raw: dict) -> PriceEstimate:
    """
    Convenience wrapper: validate a raw dict (from Conversation.ai_analysis)
    as a DamageAnalysis and then price it.

    Use this when you have the stored JSON from the DB and want to
    (re-)compute the price estimate without re-running the AI.

    Raises
    ──────
    pydantic.ValidationError
        If raw does not conform to the DamageAnalysis schema.
    ValueError
        If the validated analysis has no confirmed damage items.
    """
    analysis = validate_analysis(raw)
    return price_damage(analysis)
