"""
Vehicle body damage analysis — system prompt and output schema.

This module is the single source of truth for:
  • The AI system prompt sent to GPT-4o on every damage analysis call.
  • The Pydantic schema that validates every AI response before it touches the DB.
  • The user-turn prompt template that packages customer text + image count.

────────────────────────────────────────────────────────────────────────────────
DESIGN PHILOSOPHY
────────────────────────────────────────────────────────────────────────────────

Why a dedicated prompts/ module instead of embedding the prompt in the service?

  1. Prompt is domain logic, not infrastructure.
     The system prompt encodes years of body shop intake knowledge — which parts
     are adjacent, which impact patterns suggest hidden damage, what "conservative
     professional judgment" means in this industry. That belongs beside the schema
     that validates its output, not buried in a service function.

  2. The schema IS the contract the prompt enforces.
     The Pydantic model and the SCHEMA section of the prompt must stay in sync.
     Keeping them in the same file makes that relationship impossible to miss and
     easy to maintain. When you add a field to the schema, you update both here.

  3. Testability.
     The prompt can be reviewed, linted, and version-controlled independently of
     the HTTP/async machinery in the service. Mock responses can be validated
     against the schema without starting a server.

  4. Reusability.
     Other services (e.g. a web API endpoint, a batch re-analysis job) can import
     SYSTEM_PROMPT and DamageAnalysis directly without pulling in Twilio or DB deps.

────────────────────────────────────────────────────────────────────────────────
PUBLIC EXPORTS
────────────────────────────────────────────────────────────────────────────────

  SYSTEM_PROMPT          str            — pass as the system role message to GPT-4o
  USER_PROMPT_TEMPLATE   str            — format with customer_text + image_count
  DamageAnalysis         Pydantic model — validate AI JSON output before DB storage
  ConfirmedDamageItem    Pydantic model — one row in visible_confirmed_damage list
  AffectedSide           Literal type   — controlled vocabulary for affected_side
  SeverityLevel          Literal type   — controlled vocabulary for severity fields
  VALID_DAMAGE_TYPES     frozenset      — allowed tokens in damage_types / damage_type
  validate_analysis()    function       — parse + validate raw AI dict in one call

Usage (in damage_analysis_service.py)
──────────────────────────────────────
    from app.prompts.damage_analysis_prompt import (
        SYSTEM_PROMPT, USER_PROMPT_TEMPLATE,
        DamageAnalysis, validate_analysis,
    )

    result = validate_analysis(json.loads(raw_ai_text))
    conv.ai_analysis = result.model_dump()
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ══════════════════════════════════════════════════════════════════════════════
# CONTROLLED VOCABULARIES
# ══════════════════════════════════════════════════════════════════════════════
#
# Why controlled vocabularies instead of free strings?
#
# Downstream consumers of the analysis (admin dashboard, pricing engine, future
# routing logic) need to filter and branch on these values. "front-left",
# "front left", and "Front Left" are the same thing to a human but three
# different strings to code. Normalisation here — enforced by both the prompt
# and the Pydantic validators — means the rest of the system never needs to
# handle variant spellings.

AffectedSide = Literal[
    "front",        # damage only at front of vehicle
    "rear",         # damage only at rear
    "left",         # driver's side (in Israel: right-hand traffic, so left = passenger side)
    "right",        # passenger side (in Israel: right = driver's side)
    "front_left",   # front corner, left side
    "front_right",  # front corner, right side
    "rear_left",    # rear corner, left side
    "rear_right",   # rear corner, right side
    "multiple",     # damage visible on more than one distinct side/corner
    "unknown",      # cannot determine from available images
]

SeverityLevel = Literal[
    "low",      # surface damage, likely paintwork and minor repair only
    "medium",   # clear deformation, may require panel work or part replacement
    "high",     # significant deformation, multiple parts, likely replacement
    "unclear",  # insufficient information to rate severity
]

# Tokens allowed in damage_types and ConfirmedDamageItem.damage_type.
# When the model returns an unrecognised value, it normalises to "other".
#
# Why not a free-text field?
# Because "deep scratch", "heavy scratch", "severe scratch" all mean scratch + high severity.
# Separating the *type* from the *severity* keeps the schema consistent and makes
# filtering by damage type possible without NLP.
VALID_DAMAGE_TYPES: frozenset[str] = frozenset({
    "scratch",            # surface-level line, clear coat or paint layer
    "paint_transfer",     # foreign paint deposited on the vehicle surface
    "paint_peeling",      # paint lifting or flaking from substrate
    "dent",               # inward deformation, surface intact
    "crease",             # sharp-edged dent running along a body line
    "crack",              # structural break through material (metal or plastic)
    "broken_plastic",     # fractured / shattered plastic component
    "bumper_deformation", # bumper cover pushed in, torn, or bent out of shape
    "misalignment",       # panel no longer flush or parallel with its neighbours
    "gap",                # abnormal visible gap between panels or components
    "deformation",        # general shape distortion (use when no specific type fits)
    "abrasion",           # surface material worn away, deeper than a scratch
    "chip",               # small paint chip, primer or bare metal exposed
    "tear",               # material torn (common on soft bumper covers)
    "other",              # fallback for anything not covered above
})

# Internal set for fast membership testing; mirrors the Literal type values
_VALID_SIDES: frozenset[str] = frozenset(AffectedSide.__args__)  # type: ignore[attr-defined]


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SCHEMA  (Pydantic v2)
# ══════════════════════════════════════════════════════════════════════════════
#
# The schema is the contract between the prompt and the rest of the system.
# Every AI response is validated against it before anything is stored or sent.
#
# Design choices:
#
#   • Separate sub-model for confirmed damage items (ConfirmedDamageItem)
#     so that each observation carries its own severity and note, not just a
#     part name. This is the richest field in the schema — it's what an estimator
#     would actually write in a damage report.
#
#   • Boolean flags (paint_damage, plastic_damage, light_damage,
#     panel_misalignment_visible) duplicate information already in
#     visible_confirmed_damage but in a form that is trivially queryable.
#     The admin dashboard and future routing rules can filter on these without
#     parsing the confirmed damage list.
#
#   • needs_human_review is FORCED True when confidence < 0.35 by a model
#     validator. The AI is allowed to set it True for any reason; it cannot
#     set it False when confidence is below the actionable threshold.
#
#   • summary is required to be in Hebrew. It is the only field that humans
#     read directly — shop staff see it in the admin dashboard and it may be
#     included in the WhatsApp response. All structured fields are in English
#     (controlled vocabulary) for consistency and filterability.

class ConfirmedDamageItem(BaseModel):
    """
    One itemised damage observation tied to a specific vehicle part.

    Used in DamageAnalysis.visible_confirmed_damage.
    Every entry here represents something visible and unambiguous in the photos
    or explicitly confirmed by the customer's description.

    Fields
    ──────
    part         The affected component using standard body shop names.
                 Examples: "front bumper cover", "left front fender",
                           "rear left door", "right headlight housing".

    damage_type  What kind of damage it is — one of the VALID_DAMAGE_TYPES tokens.
                 Normalised to lowercase + underscores; unknown values → "other".

    severity     How severe this specific damage is on this specific part.
                 Independent of the overall severity — a dented door can coexist
                 with a scratched bumper where each has its own severity.

    note         Optional free-text clarification in English or Hebrew.
                 Use this for anything the controlled fields can't express:
                 extent ("spans full width of bumper"), location ("upper left corner"),
                 or uncertainty ("appears to be crease but angle unclear").
    """

    part:        str           = Field(..., min_length=1)
    damage_type: str           = Field(..., min_length=1)
    severity:    SeverityLevel
    note:        str           = Field(default="")

    @field_validator("damage_type", mode="before")
    @classmethod
    def normalise_damage_type(cls, v: str) -> str:
        """
        Accept any casing and spacing (e.g. "Paint Transfer", "paint-transfer").
        Falls back to "other" for unrecognised tokens rather than raising.
        Falling back is intentional: a validation error on a single damage item
        would reject the entire response, which is worse than an "other" token.
        """
        normalised = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return normalised if normalised in VALID_DAMAGE_TYPES else "other"

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_item_severity(cls, v: str) -> str:
        return str(v).strip().lower()


class DamageAnalysis(BaseModel):
    """
    Validated output of one AI vehicle body damage assessment.

    Every instance has passed full Pydantic validation — call .model_dump()
    to get a JSON-safe dict for storage in Conversation.ai_analysis (JSON column).

    Never instantiate directly from untrusted input; use validate_analysis() which
    wraps model_validate() and handles pydantic.ValidationError cleanly.

    Convenience properties
    ──────────────────────
    is_low_confidence     True when confidence < 0.35 (human review threshold)
    confirmed_part_names  Flat list of part names from visible_confirmed_damage
    all_damage_types      Unique set of damage_type tokens across all items
    """

    # ── Location ──────────────────────────────────────────────────────────────
    #
    # affected_side is the first routing field: a rear impact gets different
    # handling than a front one (different adjacent parts, different estimate path).
    affected_side: AffectedSide

    # ── Parts ─────────────────────────────────────────────────────────────────
    #
    # The three-tier separation is the core of the epistemic framework:
    #
    #   primary_affected_parts          → confirmed in images
    #   possible_related_affected_parts → suspected based on impact physics
    #   (cannot determine)              → implied by absence from both lists
    #
    # This split lets estimators quickly see what is certain vs. what they need
    # to inspect, without the AI overstating its confidence.
    primary_affected_parts:          list[str] = Field(
        ...,
        min_length=1,
        description="Parts with clearly visible, confirmed damage. Never empty.",
    )
    possible_related_affected_parts: list[str] = Field(
        default_factory=list,
        description=(
            "Parts that may be affected based on impact type/location but are "
            "not visibly confirmed. e.g. brackets, clips, mounts, adjacent panels."
        ),
    )

    # ── Itemised damage ───────────────────────────────────────────────────────
    #
    # visible_confirmed_damage is the richest field: part × type × severity × note.
    # damage_types is a flat deduplicated projection — useful for quick filtering
    # without parsing the nested list.
    visible_confirmed_damage: list[ConfirmedDamageItem] = Field(
        ...,
        min_length=1,
        description="One entry per damaged part. Must match primary_affected_parts.",
    )
    damage_types: list[str] = Field(
        ...,
        min_length=1,
        description="Deduplicated list of all damage_type tokens across all items.",
    )

    # ── Material / component flags ────────────────────────────────────────────
    #
    # Boolean flags duplicate information in visible_confirmed_damage but in a
    # form that is trivially queryable — no list parsing needed.
    # panel_misalignment_visible is the most critical: it signals that the
    # repair scope may extend beyond what's visible on the surface.
    paint_damage:               bool = Field(description="Any paint-layer damage visible")
    plastic_damage:             bool = Field(description="Any plastic component damaged")
    light_damage:               bool = Field(description="Any lighting unit damaged or at risk")
    panel_misalignment_visible: bool = Field(
        description="Gap, step, or offset between adjacent panels visible in images"
    )

    # ── Repair planning ───────────────────────────────────────────────────────
    replacement_likely_parts: list[str] = Field(
        default_factory=list,
        description=(
            "Parts where visible deformation, cracks, or breaks suggest "
            "replacement is more likely than repair. Helps estimators triage."
        ),
    )

    # ── Assessment metadata ───────────────────────────────────────────────────
    severity: SeverityLevel = Field(
        description="Overall damage severity across all confirmed damage."
    )
    missing_angles: list[str] = Field(
        default_factory=list,
        description=(
            "Specific camera angles that would improve assessment accuracy. "
            "e.g. 'underside of front bumper', 'door jamb with door open', "
            "'wheel arch from below'. Empty if coverage is sufficient."
        ),
    )
    needs_human_review: bool = Field(
        description=(
            "True when: confidence < 0.35, image quality is poor, damage pattern "
            "is complex, or the AI cannot draw reliable conclusions from available input."
        ),
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "AI confidence based on image quality and information completeness. "
            "Reflects HOW GOOD the evidence is, not how confident the AI feels."
        ),
    )

    # ── Human-readable summary (Hebrew required) ──────────────────────────────
    #
    # The only field displayed directly to shop staff and potentially to customers.
    # Requiring Hebrew here (enforced by the prompt) keeps the rest of the schema
    # in English (consistent, filterable) while the narrative is in the working
    # language of the shop.
    summary: str = Field(
        ...,
        min_length=10,
        description=(
            "2–4 sentence professional summary IN HEBREW. Written as a qualified "
            "body shop assessor would write it. Uses proper Israeli automotive "
            "terminology. States confirmed findings, notes uncertainties."
        ),
    )

    # ── Validators ────────────────────────────────────────────────────────────

    @field_validator("affected_side", mode="before")
    @classmethod
    def normalise_side(cls, v: str) -> str:
        """
        Normalise any casing / spacing to the controlled vocabulary.
        Falls back to 'unknown' rather than raising — a bad side value should
        not invalidate an otherwise correct analysis.
        """
        normalised = str(v).strip().lower().replace(" ", "_").replace("-", "_")
        return normalised if normalised in _VALID_SIDES else "unknown"

    @field_validator("severity", mode="before")
    @classmethod
    def normalise_severity(cls, v: str) -> str:
        return str(v).strip().lower()

    @field_validator("damage_types", mode="before")
    @classmethod
    def normalise_damage_types(cls, values: list) -> list[str]:
        """
        Normalise each token and deduplicate while preserving first-seen order.
        Unknown tokens → "other". Empty input → ["other"] (schema requires min 1).
        """
        seen: set[str] = set()
        result: list[str] = []
        for v in values:
            tok_raw = str(v).strip().lower().replace(" ", "_").replace("-", "_")
            tok = tok_raw if tok_raw in VALID_DAMAGE_TYPES else "other"
            if tok not in seen:
                seen.add(tok)
                result.append(tok)
        return result or ["other"]

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v) -> float:
        """
        Accept string floats (GPT sometimes returns "0.8" instead of 0.8).
        Clamp strictly to [0.0, 1.0]. On parse failure, default to 0.5 — mid-range
        is safer than 0 (would force human review on every failure) or 1 (overconfident).
        """
        try:
            f = float(v)
        except (TypeError, ValueError):
            return 0.5
        return max(0.0, min(1.0, f))

    @model_validator(mode="after")
    def enforce_low_confidence_review(self) -> "DamageAnalysis":
        """
        Safety floor: if confidence is below the actionable threshold, force
        needs_human_review = True regardless of what the model said.

        Why 0.35?
        At < 0.35 the analysis is based on insufficient visual evidence. Routing
        to a human is cheaper than generating an estimate from bad data.
        The AI may not always self-assess its confidence accurately — the Pydantic
        layer enforces the floor as a deterministic guarantee.
        """
        if self.confidence < 0.35:
            self.needs_human_review = True
        return self

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_low_confidence(self) -> bool:
        """True when confidence is below the human-review threshold."""
        return self.confidence < 0.35

    @property
    def confirmed_part_names(self) -> list[str]:
        """Flat list of part names from the confirmed damage items."""
        return [item.part for item in self.visible_confirmed_damage]

    @property
    def all_damage_types(self) -> set[str]:
        """Unique set of all damage_type tokens across all confirmed items."""
        return {item.damage_type for item in self.visible_confirmed_damage}


# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════════════════
#
# STRUCTURAL DECISIONS — read before editing:
#
#   1. ROLE comes first.
#      The model's persona is set before any rules or schema. Research on system
#      prompts consistently shows that role framing shapes everything that follows.
#      "Professional intake assessor" is specific enough to anchor domain knowledge
#      without overclaiming (e.g. "certified mechanic" would invite mechanical
#      speculation we explicitly don't want).
#
#   2. EPISTEMIC FRAMEWORK before domain knowledge.
#      The three-tier classification (confirmed / possible / cannot determine)
#      is the most critical concept in the entire prompt. It must be established
#      before the model encounters collision patterns — otherwise it might apply
#      the patterns too aggressively and claim confirmed damage from inference alone.
#
#   3. COLLISION PATTERN KNOWLEDGE BASE.
#      LLMs have general automotive knowledge but do not reliably apply collision
#      physics during intake scenarios. Explicit patterns prevent two failure modes:
#        • Under-reporting: missing that a bumper impact typically affects clips and
#          brackets behind the cover — which matters for repair scope.
#        • Over-reporting: claiming confirmed damage to the radiator support from
#          a photo that only shows a scraped bumper cover.
#      The patterns instruct the model to list these as POSSIBLE, not confirmed.
#
#   4. ANALYSIS RULES after the conceptual framework.
#      By the time the model reads "never hallucinate", it already understands WHY —
#      the three-tier framework has established that unconfirmed damage belongs in a
#      specific place, not nowhere and not in the confirmed list.
#
#   5. OUTPUT FORMAT after all rules.
#      The model must understand its constraints before seeing the schema. If the
#      schema comes first, it tends to fill fields mechanically without applying the
#      epistemic rules. Placing it last means the rules are the active context when
#      the model constructs its response.
#
#   6. CONFIDENCE is calibrated to IMAGE QUALITY, not analysis confidence.
#      This is a subtle but important distinction. "How confident are you in your
#      assessment?" is circular. "How good is the visual evidence you have?" is
#      objective and actionable. A high-confidence analysis from a blurry photo
#      is worthless; a lower-confidence assessment from clear photos of ambiguous
#      damage is useful.
#
#   7. Written in English for technical precision.
#      The rules, schema, and controlled vocabularies are in English.
#      Only the summary field is required in Hebrew — it is the only output
#      that shop staff and customers read directly. Keeping everything else in
#      English makes the schema consistent and prevents translation drift.
#
#   8. No pricing, anywhere.
#      The rule is stated once, plainly, and the schema has no price field.
#      Pricing is the estimator's job. An AI-generated price from intake photos
#      would either anchor the customer's expectations incorrectly or expose the
#      shop to liability. The intake assessment's job is to determine what needs
#      to be inspected, not what it costs.

SYSTEM_PROMPT: str = """\
ROLE
════
You are a professional vehicle body damage intake assessor working for an Israeli \
auto body and paint repair shop. Your role is to evaluate customer-submitted photos \
and damage descriptions at the point of first contact — before any physical inspection \
has taken place.

Your output supports the estimator who will conduct the in-person assessment. \
You are providing a qualified first reading of the damage, not a final diagnosis.


EPISTEMIC FRAMEWORK — READ THIS BEFORE ANYTHING ELSE
══════════════════════════════════════════════════════

Every piece of damage information you produce must fall into exactly one of three tiers:

  TIER 1 — CONFIRMED VISIBLE DAMAGE
  ───────────────────────────────────
  Damage you can see clearly and directly in the provided images, or that the customer
  has explicitly described AND is consistent with what you can see.
  → Goes into: primary_affected_parts + visible_confirmed_damage

  TIER 2 — POSSIBLE RELATED DAMAGE
  ────────────────────────────────────
  Parts that are commonly affected by this type of impact at this location, based on
  known collision patterns, but which you cannot confirm from the available images.
  → Goes into: possible_related_affected_parts ONLY
  → Never state these as confirmed. Use language like "may also be affected".

  TIER 3 — CANNOT DETERMINE
  ──────────────────────────
  Damage you suspect may exist but have no visual or textual evidence for.
  → Omit entirely from all damage lists. Mention in missing_angles if a specific
    photo angle would help confirm or rule it out.

This framework is the most important rule in this prompt. When in doubt, go one tier
lower. An under-stated intake assessment is professionally corrected at inspection.
An overstated one damages trust and wastes the customer's time.


COLLISION PATTERN KNOWLEDGE BASE
═════════════════════════════════

Use the following patterns to identify POSSIBLE RELATED DAMAGE (Tier 2) for common
impact scenarios. These are not to be reported as confirmed unless visually evident.

PATTERN A — FRONTAL BUMPER IMPACT
  Primary: front bumper cover
  Possible related: bumper absorber/foam, bumper mounting brackets, front clips and
    fasteners, lower grille trim, front license plate bracket, hood latch area (if
    impact is high), front fender lower edges (if lateral spread is visible), fog
    light housings (if corner involved), wheel arch trim lower section.
  Key question: Is the bumper pushed back far enough to suggest absorber or
    crossmember contact? Look for hood or fender edge step/misalignment.

PATTERN B — FENDER DEFORMATION NEAR DOOR GAP
  Primary: front or rear fender
  Possible related: door hinge side (adjacent door may not close flush), door edge
    trim, wheel arch trim/liner, A-pillar or B-pillar lower section (if crease
    extends toward pillar), fender mounting bolts / inner fender liner.
  Key question: Is the door gap visibly uneven? If yes, door-to-fender alignment
    damage is confirmed, not just possible.

PATTERN C — REAR IMPACT
  Primary: rear bumper cover, trunk lid / tailgate
  Possible related: rear bumper absorber, bumper mounting brackets, trunk lid
    alignment (hinge or striker), rear quarter panel lower section, tail light
    housings and mounting tabs, rear valance panel, tow hook cover or area,
    exhaust tip position (may indicate subframe movement — flag for physical check).
  Key question: Does the trunk/tailgate still close and align correctly?
    Visible gap change → alignment damage confirmed.

PATTERN D — LATERAL SIDE IMPACT (door / rocker)
  Primary: one or more doors, rocker panel / sill
  Possible related: side skirt / sill extension, adjacent door edges (gap change),
    B-pillar or C-pillar lower section, door glass run channel (if door is deformed),
    mirror mounting (if forward door is affected), quarter panel forward edge
    (if rear door affected), floor pan edges (not confirmable from exterior photos —
    flag as needing physical inspection).
  Key question: Are multiple panels misaligned relative to each other? Rocker panel
    crease extending under doors is a higher-severity signal.

PATTERN E — FRONT CORNER IMPACT
  Primary: bumper corner section, front fender
  Possible related: headlight housing and mounting tabs, wheel arch trim / splash
    guard, front clip and crossmember area, inner fender liner, hood edge alignment
    (front corner impacts often rotate the fender slightly), tyre and wheel (if
    impact is low and close to wheel — flag for physical alignment check, not
    confirmable from photos), A-pillar lower section.
  Key question: Is the headlight still seated correctly in its housing? Any step
    between hood edge and fender upper surface?


ANALYSIS RULES
══════════════

1. VISIBLE EVIDENCE ONLY
   Analyze only what you can see in the images OR what the customer has explicitly
   described. Do not infer, assume, or extrapolate beyond Tier 2.

2. NO HALLUCINATION
   Never state structural damage, frame damage, mechanical damage, or internal
   component damage as confirmed unless you can see it in the photos. These require
   physical inspection and cannot be assessed from exterior images.

3. CONSERVATIVE OVER EXPANSIVE
   When deciding between Tier 1 and Tier 2 for a borderline observation,
   choose Tier 2. An estimator can confirm on inspection. You cannot.

4. SPECIFICITY OVER VAGUENESS
   Avoid "some damage to the front area". Say "dent and paint transfer on the
   front bumper cover, lower centre section" instead. Specific observations are
   more useful to estimators than general ones.

5. MISALIGNMENT IS OBJECTIVE
   Panel misalignment is visible or it isn't. If you can see a gap, step, or
   offset between adjacent panels that should be flush, set
   panel_misalignment_visible = true and name the specific panels. Do not infer
   misalignment from a single panel photo where no reference panel is visible.

6. FLAG INSUFFICIENT IMAGES
   If key areas of the damage are not visible, or if the only photos are low
   resolution, heavily backlit, or taken from angles that obscure the damage
   extent, set needs_human_review = true and populate missing_angles with
   specific descriptions of what additional photos would help.

7. REPLACEMENT VS REPAIR SIGNAL
   List a part in replacement_likely_parts when you can see structural deformation
   (crease that breaks the panel geometry), cracking, shattering, or tearing that
   makes repair impractical. Surface dents and scratches alone do not trigger this.

8. NO PRICING
   Never include, estimate, imply, or reference repair costs or part prices.
   Pricing is determined at physical inspection. This field does not exist in
   your output. Do not mention it in notes or the summary.

9. PROFESSIONAL TONE
   Write notes and summary as a qualified body shop estimator would: precise,
   direct, and free of hedging language ("might be", "could possibly", "perhaps").
   If something is uncertain, use the tier system — not hedging adjectives.


OUTPUT FORMAT
═════════════

Return ONLY a valid JSON object.
No markdown code fences. No explanatory text. No preamble or closing remarks.
The first character of your response must be { and the last must be }.


SCHEMA
══════

Return exactly these keys. Types and allowed values are specified inline.

{
  "affected_side": "<one of: front | rear | left | right | front_left | front_right | rear_left | rear_right | multiple | unknown>",

  "primary_affected_parts": [
    "<part name — confirmed visible damage — e.g. 'front bumper cover', 'left front fender', 'rear left door'>"
  ],

  "visible_confirmed_damage": [
    {
      "part":        "<exact part name, consistent with primary_affected_parts>",
      "damage_type": "<one of: scratch | paint_transfer | paint_peeling | dent | crease | crack | broken_plastic | bumper_deformation | misalignment | gap | deformation | abrasion | chip | tear | other>",
      "severity":    "<one of: low | medium | high | unclear>",
      "note":        "<optional: specific observation, location, or extent — English or Hebrew — empty string if nothing to add>"
    }
  ],

  "possible_related_affected_parts": [
    "<part that may be affected but is not visibly confirmed — e.g. 'bumper mounting brackets', 'inner fender liner', 'trunk alignment'>"
  ],

  "damage_types": [
    "<flat deduplicated list of all damage_type values present in visible_confirmed_damage>"
  ],

  "paint_damage":               <true | false>,
  "plastic_damage":             <true | false>,
  "light_damage":               <true | false>,
  "panel_misalignment_visible": <true | false>,

  "replacement_likely_parts": [
    "<part where replacement is more likely than repair based on visible deformation or structural damage>"
  ],

  "severity": "<overall severity: one of: low | medium | high | unclear>",

  "missing_angles": [
    "<specific camera angle that would improve assessment — e.g. 'underside of front bumper to assess absorber', 'door gap with door open', 'wheel arch from below'>"
  ],

  "needs_human_review": <true | false>,

  "confidence": <float 0.0–1.0, see calibration table below>,

  "summary": "<MUST BE IN HEBREW. 2–4 sentences. Professional, direct, written as a qualified assessor. State confirmed damage and affected parts. Note any significant uncertainties or missing information. Use correct Israeli body shop terminology: פגיעה / נזק / שריטה / שקע / קמט / סדק / עיוות / פגוש / כנף / מכסה מנוע / תא מטען / גג / רכסית / דלת / פנס / מראה / ספת דלת / רכסית.>"
}


DAMAGE TYPE REFERENCE
═════════════════════

Use these definitions when selecting damage_type. When multiple types apply to
one part, create one ConfirmedDamageItem per damage type for that part.

  scratch           Surface-level line damage to clear coat or paint. Does not
                    deform the substrate. Usually repaired with polishing or spot paint.

  paint_transfer    Paint or coating from another object (wall, vehicle, barrier)
                    deposited on the vehicle surface. Often appears as a colour stripe
                    that does not match the vehicle. May be removable without repainting.

  paint_peeling     Paint or primer lifting, bubbling, or flaking from the surface.
                    Indicates adhesion failure, often from impact stress, moisture, or
                    prior repair failure.

  dent              Inward deformation of a panel without breaking the surface.
                    Panel metal or plastic is displaced but intact.

  crease            A sharp-edged dent that runs along a body line or across a panel.
                    Harder to repair than a round dent; often requires panel replacement.

  crack             A structural break through the material thickness — metal or plastic.
                    For plastic parts this often means the part cannot be repaired.

  broken_plastic    A plastic component that is fractured, shattered, or has pieces
                    missing. Replacement is typically required.

  bumper_deformation The bumper cover is pushed in, torn, bent out of shape, or
                    separated from its mounting points. Covers a wide range of bumper
                    damage not better described by another type.

  misalignment      A panel is no longer flush or parallel with adjacent panels.
                    Visible as a step, offset, or uneven gap at a panel junction.

  gap               An abnormal visible gap between panels, trim, or components that
                    should be tight or evenly spaced.

  deformation       General shape distortion not better described by dent, crease,
                    or bumper_deformation. Use for complex or compound deformation.

  abrasion          Surface material worn or ground away, typically from contact with
                    pavement, kerb, or another surface. Deeper than a scratch.

  chip              A small, localised paint chip that exposes primer or bare metal.
                    Common on leading edges (hood front, bumper lower, door sill).

  tear              Material physically torn — common on soft bumper covers and
                    flexible plastic trim. The material is separated, not just bent.


CONFIDENCE CALIBRATION
═══════════════════════

Confidence measures the quality of available visual evidence, not your certainty
about your conclusions. Ask: "How well do the provided images support this analysis?"

  0.90 – 1.00  Multiple clear, well-lit, high-resolution photos from multiple angles.
               All damaged areas are fully visible. Damage extent is unambiguous.

  0.70 – 0.89  Photos are good quality. Most damaged areas are visible. One or two
               angles are missing but do not significantly affect the assessment.

  0.50 – 0.69  Photos are acceptable but missing important angles (underside, close-up,
               door jamb). Assessment is reasonable but some uncertainty about extent.

  0.30 – 0.49  Photos are poor quality (blurry, backlit, distant), or key areas are
               obscured. Assessment is possible but unreliable. Human review recommended.

  0.00 – 0.29  Insufficient images, unidentifiable damage, or images do not show the
               reported damage at all. Set needs_human_review = true.
"""


# ══════════════════════════════════════════════════════════════════════════════
# USER PROMPT TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════
#
# Why a separate user-turn template rather than embedding everything in the system
# prompt?
#
# The system prompt defines the model's role, rules, and schema — it is static
# across every analysis call. The user turn provides the specific intake context
# for this call. Keeping them separate means:
#   • The system prompt can be cached by the API (stable across requests).
#   • The user turn is the natural place for per-call dynamic content.
#   • It mirrors the actual conversation structure the model expects.

USER_PROMPT_TEMPLATE: str = """\
INTAKE INFORMATION
──────────────────
Customer description : {customer_text}
Images provided      : {image_count}

Instructions:
Apply all rules from the system prompt.
Analyze the attached images alongside the customer description above.
Return only the JSON object — no other text.\
"""


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPER
# ══════════════════════════════════════════════════════════════════════════════

def validate_analysis(raw: dict) -> DamageAnalysis:
    """
    Parse and validate a raw dict (from AI JSON output) into a DamageAnalysis.

    This is the canonical entry point for consuming AI output.
    All validators run automatically via Pydantic — normalisation, clamping,
    and the low-confidence human-review floor are all applied here.

    Parameters
    ──────────
    raw
        A Python dict parsed from the AI's JSON response.

    Returns
    ───────
    DamageAnalysis
        Fully validated instance. Call .model_dump() for a JSON-safe dict.

    Raises
    ──────
    pydantic.ValidationError
        If the raw dict is missing required fields or has values that cannot
        be normalised. The caller (damage_analysis_service.py) catches this
        and logs it as damage_analysis_invalid before re-raising as ValueError.
    """
    return DamageAnalysis.model_validate(raw)
