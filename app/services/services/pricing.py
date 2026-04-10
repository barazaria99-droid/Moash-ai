"""
Rule-based price table for vehicle bodywork in the Israeli market.

This module defines the data structures and lookup functions for pricing.
It does NOT call any API or perform AI analysis — it is purely deterministic.

Architecture note
─────────────────
This table is intentionally kept separate from the AI analysis so the shop
owner can edit prices without touching any AI code. When the full analysis
engine is ready, it will call get_price_range() with the detected part +
damage type + severity and get back an ILS range.

Usage (future)
──────────────
    from app.services.pricing import get_price_range, LABOR_RATE_ILS

    low, high = get_price_range(
        part="bumper_front",
        damage_type="dent_with_scratch",
        severity="medium",
    )
    # → (3500, 6000)

Price sources
─────────────
Prices are based on Israeli bodywork market rates as of early 2026:
  - Labor: ₪180–220/hour (we use ₪200 as the midpoint)
  - Paint: ₪400–700 per panel (single-stage urethane)
  - Parts: local supplier pricing (Genuine / OEM-equivalent)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

SeverityLevel = Literal["low", "medium", "high"]

# ── Shop labor rate ────────────────────────────────────────────────────────────

LABOR_RATE_ILS: int = 200  # ₪ per hour — update this when shop rates change

# ── Damage type definitions ────────────────────────────────────────────────────
# These map to what the AI's affected_area field describes.

DAMAGE_TYPES = {
    "scratch":              "שריטה",
    "dent":                 "שקע",
    "dent_with_scratch":    "שקע ושריטה",
    "crack":                "סדק",
    "break":                "שבר / ניתוק",
    "deformation":          "עיוות",
    "paint_fade":           "דהיית צבע",
    "rust":                 "חלודה",
}

# ── Part definitions ───────────────────────────────────────────────────────────

PART_LABELS = {
    # Hebrew name for admin display
    "bumper_front":     "פגוש קדמי",
    "bumper_rear":      "פגוש אחורי",
    "hood":             "מכסה מנוע",
    "trunk_lid":        "מכסה תא מטען",
    "fender_left":      "כנף שמאל",
    "fender_right":     "כנף ימין",
    "door_front_left":  "דלת קדמית שמאל",
    "door_front_right": "דלת קדמית ימין",
    "door_rear_left":   "דלת אחורית שמאל",
    "door_rear_right":  "דלת אחורית ימין",
    "roof":             "גג",
    "quarter_left":     "רכסית שמאל",
    "quarter_right":    "רכסית ימין",
    "sill_left":        "סף דלת שמאל",
    "sill_right":       "סף דלת ימין",
    "mirror_left":      "מראה שמאל",
    "mirror_right":     "מראה ימין",
    "headlight_left":   "פנס קדמי שמאל",
    "headlight_right":  "פנס קדמי ימין",
    "taillight_left":   "פנס אחורי שמאל",
    "taillight_right":  "פנס אחורי ימין",
    "grille":           "רשת אוויר",
    "wheel_arch":       "שלדת גלגל",
    "underbody":        "תחתית רכב",
}


@dataclass(frozen=True)
class PriceEntry:
    """
    Price range and labor estimate for a single (part, severity) combination.
    All prices in ILS (₪).

    parts_low / parts_high   — cost of replacement/repair parts
    labor_hours_low / high   — estimated bodywork hours
    paint_panels             — number of panels requiring respray

    The total estimate is: parts + (labor_hours * LABOR_RATE_ILS) + (paint_panels * avg_paint_cost)
    """
    parts_low: int
    parts_high: int
    labor_hours_low: float
    labor_hours_high: float
    paint_panels: int = 1
    notes: str = ""


# ── Price table ───────────────────────────────────────────────────────────────
#
# Structure: PRICE_TABLE[part_key][severity] = PriceEntry
#
# "low"    — surface scratch / minor dent, paint touch-up or panel blend
# "medium" — dent requiring filler, full panel repaint, possible part replacement
# "high"   — structural damage, full part replacement, multiple panels

PRICE_TABLE: dict[str, dict[SeverityLevel, PriceEntry]] = {

    "bumper_front": {
        "low":    PriceEntry(parts_low=0,    parts_high=500,  labor_hours_low=2,  labor_hours_high=4,  paint_panels=1, notes="שריטה שטחית / צביעה מקומית"),
        "medium": PriceEntry(parts_low=500,  parts_high=1500, labor_hours_low=4,  labor_hours_high=8,  paint_panels=1, notes="פגוש ניתן לתיקון + צביעה"),
        "high":   PriceEntry(parts_low=1200, parts_high=3500, labor_hours_low=6,  labor_hours_high=12, paint_panels=2, notes="החלפת פגוש + רשת + עבודות נוספות"),
    },

    "bumper_rear": {
        "low":    PriceEntry(parts_low=0,    parts_high=500,  labor_hours_low=2,  labor_hours_high=4,  paint_panels=1),
        "medium": PriceEntry(parts_low=500,  parts_high=1500, labor_hours_low=4,  labor_hours_high=8,  paint_panels=1),
        "high":   PriceEntry(parts_low=1000, parts_high=3000, labor_hours_low=6,  labor_hours_high=12, paint_panels=2),
    },

    "hood": {
        "low":    PriceEntry(parts_low=0,    parts_high=300,  labor_hours_low=1,  labor_hours_high=3,  paint_panels=1),
        "medium": PriceEntry(parts_low=300,  parts_high=800,  labor_hours_low=4,  labor_hours_high=8,  paint_panels=1, notes="פחחות + מילוי + צביעה"),
        "high":   PriceEntry(parts_low=1500, parts_high=4000, labor_hours_low=4,  labor_hours_high=8,  paint_panels=1, notes="החלפת מכסה מנוע"),
    },

    "trunk_lid": {
        "low":    PriceEntry(parts_low=0,    parts_high=300,  labor_hours_low=1,  labor_hours_high=3,  paint_panels=1),
        "medium": PriceEntry(parts_low=300,  parts_high=800,  labor_hours_low=4,  labor_hours_high=8,  paint_panels=1),
        "high":   PriceEntry(parts_low=1500, parts_high=4500, labor_hours_low=4,  labor_hours_high=8,  paint_panels=1),
    },

    "fender_left": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=4,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1000, labor_hours_low=5,  labor_hours_high=10, paint_panels=1),
        "high":   PriceEntry(parts_low=1200, parts_high=3000, labor_hours_low=5,  labor_hours_high=10, paint_panels=2),
    },

    "fender_right": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=4,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1000, labor_hours_low=5,  labor_hours_high=10, paint_panels=1),
        "high":   PriceEntry(parts_low=1200, parts_high=3000, labor_hours_low=5,  labor_hours_high=10, paint_panels=2),
    },

    "door_front_left": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=5,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1200, labor_hours_low=6,  labor_hours_high=12, paint_panels=1),
        "high":   PriceEntry(parts_low=2000, parts_high=5000, labor_hours_low=8,  labor_hours_high=16, paint_panels=2, notes="פגיעה מבנית — ייתכן צורך בהחלפה"),
    },

    "door_front_right": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=5,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1200, labor_hours_low=6,  labor_hours_high=12, paint_panels=1),
        "high":   PriceEntry(parts_low=2000, parts_high=5000, labor_hours_low=8,  labor_hours_high=16, paint_panels=2),
    },

    "door_rear_left": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=5,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1200, labor_hours_low=6,  labor_hours_high=12, paint_panels=1),
        "high":   PriceEntry(parts_low=2000, parts_high=5000, labor_hours_low=8,  labor_hours_high=16, paint_panels=2),
    },

    "door_rear_right": {
        "low":    PriceEntry(parts_low=0,    parts_high=400,  labor_hours_low=2,  labor_hours_high=5,  paint_panels=1),
        "medium": PriceEntry(parts_low=400,  parts_high=1200, labor_hours_low=6,  labor_hours_high=12, paint_panels=1),
        "high":   PriceEntry(parts_low=2000, parts_high=5000, labor_hours_low=8,  labor_hours_high=16, paint_panels=2),
    },

    "roof": {
        "low":    PriceEntry(parts_low=0,    parts_high=500,  labor_hours_low=3,  labor_hours_high=6,  paint_panels=1),
        "medium": PriceEntry(parts_low=500,  parts_high=2000, labor_hours_low=8,  labor_hours_high=16, paint_panels=1, notes="פחחות גג — עבודה מורכבת"),
        "high":   PriceEntry(parts_low=3000, parts_high=8000, labor_hours_low=12, labor_hours_high=24, paint_panels=3, notes="נזק מבני — ייתכן החלפת גג"),
    },

    "quarter_left": {
        "low":    PriceEntry(parts_low=0,    parts_high=500,  labor_hours_low=3,  labor_hours_high=6,  paint_panels=1),
        "medium": PriceEntry(parts_low=500,  parts_high=1500, labor_hours_low=8,  labor_hours_high=16, paint_panels=2),
        "high":   PriceEntry(parts_low=3000, parts_high=8000, labor_hours_low=16, labor_hours_high=30, paint_panels=3, notes="רכסית — עבודת גוף מורכבת"),
    },

    "quarter_right": {
        "low":    PriceEntry(parts_low=0,    parts_high=500,  labor_hours_low=3,  labor_hours_high=6,  paint_panels=1),
        "medium": PriceEntry(parts_low=500,  parts_high=1500, labor_hours_low=8,  labor_hours_high=16, paint_panels=2),
        "high":   PriceEntry(parts_low=3000, parts_high=8000, labor_hours_low=16, labor_hours_high=30, paint_panels=3),
    },

    "mirror_left": {
        "low":    PriceEntry(parts_low=200,  parts_high=500,  labor_hours_low=0.5, labor_hours_high=1, paint_panels=0, notes="החלפת כיסוי מראה"),
        "medium": PriceEntry(parts_low=400,  parts_high=900,  labor_hours_low=1,   labor_hours_high=2, paint_panels=1),
        "high":   PriceEntry(parts_low=800,  parts_high=2500, labor_hours_low=1,   labor_hours_high=2, paint_panels=1, notes="החלפת מראה כולל מנוע / חיישנים"),
    },

    "mirror_right": {
        "low":    PriceEntry(parts_low=200,  parts_high=500,  labor_hours_low=0.5, labor_hours_high=1, paint_panels=0),
        "medium": PriceEntry(parts_low=400,  parts_high=900,  labor_hours_low=1,   labor_hours_high=2, paint_panels=1),
        "high":   PriceEntry(parts_low=800,  parts_high=2500, labor_hours_low=1,   labor_hours_high=2, paint_panels=1),
    },

    "headlight_left": {
        "low":    PriceEntry(parts_low=200,  parts_high=600,  labor_hours_low=0.5, labor_hours_high=1, paint_panels=0, notes="ליטוש פנס"),
        "medium": PriceEntry(parts_low=600,  parts_high=1500, labor_hours_low=1,   labor_hours_high=2, paint_panels=0),
        "high":   PriceEntry(parts_low=1500, parts_high=6000, labor_hours_low=1,   labor_hours_high=3, paint_panels=0, notes="החלפת פנס LED / קסנון"),
    },

    "headlight_right": {
        "low":    PriceEntry(parts_low=200,  parts_high=600,  labor_hours_low=0.5, labor_hours_high=1, paint_panels=0),
        "medium": PriceEntry(parts_low=600,  parts_high=1500, labor_hours_low=1,   labor_hours_high=2, paint_panels=0),
        "high":   PriceEntry(parts_low=1500, parts_high=6000, labor_hours_low=1,   labor_hours_high=3, paint_panels=0),
    },

    "sill_left": {
        "low":    PriceEntry(parts_low=0,    parts_high=300,  labor_hours_low=2,   labor_hours_high=4, paint_panels=1),
        "medium": PriceEntry(parts_low=300,  parts_high=800,  labor_hours_low=4,   labor_hours_high=8, paint_panels=1),
        "high":   PriceEntry(parts_low=800,  parts_high=2500, labor_hours_low=6,   labor_hours_high=12, paint_panels=2),
    },

    "sill_right": {
        "low":    PriceEntry(parts_low=0,    parts_high=300,  labor_hours_low=2,   labor_hours_high=4, paint_panels=1),
        "medium": PriceEntry(parts_low=300,  parts_high=800,  labor_hours_low=4,   labor_hours_high=8, paint_panels=1),
        "high":   PriceEntry(parts_low=800,  parts_high=2500, labor_hours_low=6,   labor_hours_high=12, paint_panels=2),
    },

    "grille": {
        "low":    PriceEntry(parts_low=200,  parts_high=600,  labor_hours_low=0.5, labor_hours_high=1, paint_panels=0),
        "medium": PriceEntry(parts_low=400,  parts_high=1200, labor_hours_low=1,   labor_hours_high=2, paint_panels=0),
        "high":   PriceEntry(parts_low=800,  parts_high=2500, labor_hours_low=1,   labor_hours_high=2, paint_panels=0),
    },
}

# ── Paint cost lookup ─────────────────────────────────────────────────────────

PAINT_COST_PER_PANEL: dict[SeverityLevel, tuple[int, int]] = {
    "low":    (300,  600),   # spot repair / blend
    "medium": (500,  900),   # full panel single-stage
    "high":   (700, 1400),   # full panel, metallic / pearl, multi-stage
}


# ── Public lookup function ────────────────────────────────────────────────────

def get_price_range(
    part: str,
    severity: SeverityLevel,
) -> tuple[int, int] | None:
    """
    Return a (low, high) ILS price range for a given part + severity combination.
    Returns None if the part is not in the table (caller should fall back to AI range).

    Total = parts + labor + paint
    """
    entry = PRICE_TABLE.get(part, {}).get(severity)
    if entry is None:
        return None

    avg_paint_low,  avg_paint_high  = PAINT_COST_PER_PANEL[severity]

    total_low  = (
        entry.parts_low
        + int(entry.labor_hours_low  * LABOR_RATE_ILS)
        + entry.paint_panels * avg_paint_low
    )
    total_high = (
        entry.parts_high
        + int(entry.labor_hours_high * LABOR_RATE_ILS)
        + entry.paint_panels * avg_paint_high
    )

    return total_low, total_high


def get_all_parts() -> dict[str, str]:
    """Return {part_key: hebrew_label} for all parts in the table."""
    return {k: PART_LABELS.get(k, k) for k in PRICE_TABLE}
