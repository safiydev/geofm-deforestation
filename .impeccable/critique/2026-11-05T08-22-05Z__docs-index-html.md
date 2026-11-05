---
target: github-pages landing
total_score: 18
max_score: 32
na_heuristics: 7,10
p0_count: 2
p1_count: 2
target_identity: "file:/data/data/com.termux/files/home/geofm_deforestation/docs/index.html"
target_fingerprint: "sha256:d0e51a2f96dec50f598fcd14ef354f233563e639e70f7160bb6a0656f0cc07cf"
target_path: /data/data/com.termux/files/home/geofm_deforestation/docs/index.html
timestamp: 2026-11-05T08-22-05Z
slug: docs-index-html
---
### Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|:-----:|-----------|
| 1 | Visibility of System Status | 2 | Reactive numbers update on slider, but header navigation lacks active scroll-spy indicators; code blocks lack copy confirmation. |
| 2 | Match Between System & Real World | 3 | Rigorous domain terminology (Sentinel-1 SAR, Sentinel-2, LoRA, Dec 31 2020 cutoff, Olofsson et al.), but lacks spatial cartographic grounding. |
| 3 | User Control and Freedom | 2 | Navigation links (`#pipeline`, `#eudr`) are dead ends without targets in the DOM; primary CTA "Launch Web Studio" jumps to terminal CLI installation. |
| 4 | Consistency and Standards | 2 | Discrepancies with product ground truth (badge claims 87 tests vs. 94 in repo/PRODUCT.md; Card 3 claims 9 spectral layers vs. 12 in PRODUCT.md); shield SVG icon is duplicated for brand logo and EUDR card. |
| 5 | Error Prevention | 3 | Range input safely clamped `[0.10, 0.90]` with `0.05` step; external links lack copy buttons and `rel="noopener noreferrer"`. |
| 6 | Recognition Rather Than Recall | 2 | Slider updates raw floating-point metrics in isolation without showing visual impact on satellite imagery or clearing polygons. |
| 7 | Flexibility and Efficiency of Use | n/a | Persuade mode landing page: accelerators and power-user shortcuts are not applicable to public landing page. |
| 8 | Aesthetic and Minimalist Design | 2 | Monotonous 3-column grid with AI-template icon-tile stacks; lacks visual satellite imagery; typography feels like generic SaaS. |
| 9 | Help Users Recognize, Diagnose, & Recover from Errors | 2 | Dead anchor links fail silently without feedback; code box renders as single run-on line in browser. |
| 10 | Help and Documentation | n/a | Persuade mode landing page: operational documentation deferred to repository. |
| **Total** | | **18 / 32** | **Acceptable (56.3%)** |

### Design Specificity Verdict

**Verdict**: Category-Interchangeable SaaS Shell Masking Deep Geospatial Rigor.

- **LLM Assessment**: The copy speaks with domain authority (Sentinel-1 SAR, Sentinel-2 optical, Hansen GFC 2024, EUDR EU 2023/1115), but visual composition and typography betray it. Without satellite imagery, bitemporal split-sliders, or map previews, it reads as an unskinned DevOps dashboard.
- **Deterministic Scan**: 7 warnings flagged by `impeccable detect` (100% true positives):
  - 6x `icon-tile-stack`: 40x40px rounded box directly above `h3` on Cards 1–6 in `#studio`.
  - 1x `overused-font`: Primary font Inter used across generic AI templates.
- **Visual Overlays**: Skipped (headless environment without active browser automation or CDP debugging port).

### Overall Impression
GeoFM has rock-solid product truth and scientific defensibility, but the landing page currently presents this breakthrough as a text-heavy developer readme. The biggest opportunity is transforming it into an empirical satellite mission briefing anchored by an interactive bitemporal before/after visual slider and real imagery.

### What's Working
1. **Defensible Value Proposition Copy**: The hero statement explicitly rejects predictive hype in favor of empirical bitemporal detection.
2. **Zero-Dependency Architecture**: Native CSS tokens, pure vanilla JS, and zero npm bloat ensure instant load times.
3. **Structured Design System Variables**: Clean `:root` tokens for obsidian, emerald, cyan, and crimson.

### Priority Issues
- **[P0] Broken Navigation Anchors and Misleading Primary CTA**: `#pipeline` and `#eudr` nav links have no DOM targets; "Launch Web Studio" jumps to `#quickstart` instead of `#studio`.
- **[P0] Critical Code Box Rendering Defect**: `.code-box` is a `<div>` without `white-space: pre-wrap;`, collapsing multi-line terminal commands into a single run-on sentence.
- **[P1] Total Absence of Earth Observation Visual Proof**: Zero satellite chips, radar backscatter plots, or Leaflet control room screenshots.
- **[P1] SVG & Form Accessibility Failures**: 0 of 8 SVGs accessible (`aria-hidden="true"` missing); range slider unassociated with `<label>`; missing `rel="noopener noreferrer"`.
- **[P2] Category-Interchangeable AI Card Slop & Generic Typography**: 6x icon-tile-stack boilerplate cards; generic Inter font lacking cartographic character.

### Persona Red Flags
- **Alex (ML Researcher)**: Skeptical of mismatched test counts (87 in badge vs. 94 in repo); notices linear simulation formula.
- **Jordan (GIS Analyst)**: Intimidated by "Launch Web Studio" CTA jumping straight to terminal bash commands; no map screenshots.
- **Elena (EUDR Auditor)**: Encounters dead link on "EUDR Engine"; lacks interactive Due Diligence Statement preview.

### Minor Observations
- Missing OpenGraph meta tags (`og:image`, `og:title`) for social previews.
- Mobile header overflow on viewports under 600px.
- Shield SVG icon duplicated across brand logo and EUDR card.

### Questions to Consider
- What if the hero featured an interactive before/after split slider comparing $T_1$ baseline against $T_2$ canopy loss?
- What if users could toggle between an optical-only cloudy view and Sentinel-1 SAR penetrating clouds?
- What if Elena could inspect an interactive EUDR Due Diligence Statement preview right in the browser?
