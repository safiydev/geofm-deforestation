---
name: "GeoFM Deforestation Suite"
description: "Empirical bitemporal satellite change detection and statutory EUDR compliance interface"
colors:
  bg: "#FBFBF9"
  card-bg: "#FFFFFF"
  card-dark: "#F4F6F3"
  header-bg: "rgba(251, 251, 249, 0.92)"
  text: "#0F1914"
  text-muted: "#526059"
  border: "rgba(15, 25, 20, 0.08)"
  border-bright: "rgba(15, 25, 20, 0.16)"
  primary: "#047857"
  primary-hover: "#065F46"
  cyan: "#0369A1"
  crimson: "#B91C1C"
  amber: "#B45309"
  code-bg: "#111A16"
  code-text: "#A7F3D0"
typography:
  display:
    fontFamily: "Outfit, sans-serif"
    fontSize: "clamp(2.8rem, 7vw, 5.2rem)"
    fontWeight: 800
    lineHeight: 1.05
    letterSpacing: "-0.03em"
  headline:
    fontFamily: "Outfit, sans-serif"
    fontSize: "clamp(2rem, 4vw, 2.75rem)"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "-0.02em"
  subheadline:
    fontFamily: "Outfit, sans-serif"
    fontSize: "1.75rem"
    fontWeight: 700
    lineHeight: 1.25
  title:
    fontFamily: "Outfit, sans-serif"
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.3
  nav-title:
    fontFamily: "Outfit, sans-serif"
    fontSize: "1.35rem"
    fontWeight: 800
  body-lg:
    fontFamily: "Figtree, sans-serif"
    fontSize: "1.15rem"
    fontWeight: 400
    lineHeight: 1.6
  body-mid:
    fontFamily: "Figtree, sans-serif"
    fontSize: "1.05rem"
    fontWeight: 400
    lineHeight: 1.6
  body:
    fontFamily: "Figtree, sans-serif"
    fontSize: "0.95rem"
    fontWeight: 400
    lineHeight: 1.6
  nav-link:
    fontFamily: "Figtree, sans-serif"
    fontSize: "0.90rem"
    fontWeight: 500
  body-sm:
    fontFamily: "Figtree, sans-serif"
    fontSize: "0.88rem"
    fontWeight: 400
    lineHeight: 1.5
  stream-sm:
    fontFamily: "Figtree, sans-serif"
    fontSize: "0.85rem"
    fontWeight: 400
  sub-caption:
    fontFamily: "Figtree, sans-serif"
    fontSize: "0.82rem"
    fontWeight: 400
  mono-lg:
    fontFamily: "Outfit, sans-serif"
    fontSize: "1.85rem"
    fontWeight: 800
  synth-lg:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "1.75rem"
    fontWeight: 700
  synth-mid:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "1.4rem"
    fontWeight: 700
  code:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.88rem"
    fontWeight: 400
  mono:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.78rem"
    fontWeight: 500
  mono-sm:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.76rem"
    fontWeight: 500
  telemetry:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.75rem"
    fontWeight: 500
  hud-badge:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.72rem"
    fontWeight: 600
  micro:
    fontFamily: "JetBrains Mono, monospace"
    fontSize: "0.70rem"
    fontWeight: 600
rounded:
  xs: "2px"
  sm: "4px"
  btn: "6px"
  card-sm: "8px"
  md: "10px"
  base: "12px"
  lg: "16px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "36px"
  2xl: "56px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "#FFFFFF"
    rounded: "6px"
    padding: "0.85rem 2rem"
  button-primary-hover:
    backgroundColor: "{colors.primary-hover}"
  button-outline:
    backgroundColor: "transparent"
    textColor: "{colors.text}"
    rounded: "6px"
    padding: "0.85rem 2rem"
  card:
    backgroundColor: "{colors.card-bg}"
    textColor: "{colors.text}"
    rounded: "{rounded.base}"
    padding: "24px"
---

# Design System: GeoFM Deforestation Suite (Friendly Botanical Observatory)

## Overview

**Creative North Star: "Friendly Botanical Observatory" (Warm, Accessible & Professional)**

GeoFM pairs the approachable warmth and clarity of **Outfit** (headings) and **Figtree** (body) with the precision of **JetBrains Mono** (telemetry). The typography feels open, inviting, and human without sacrificing scientific credibility or legal authority.

Set against an alabaster daylight canvas (`#FBFBF9`), the typography uses obsidian ink (`#0F1914`) and rich botanical emerald (`#047857`) to create an inspiring, accessible experience for auditors, environmental teams, and global businesses.

Scroll-driven entrance animations (`cubic-bezier(0.16, 1, 0.3, 1)`) guide the visitor's attention, while delicate hairline borders and natural card elevation replace heavy dark frames.

## Core Rules & Principles

### The Human Voice Rule
User-facing copy speaks directly to the mission: living trees, ethical global trade, and legal certainty. Developer vanity metrics (passing test numbers, compiler flags, framework version strings) belong in test pipelines, not on landing pages. Every sentence must be clear, welcoming, and free of unnecessary academic jargon.

### The Botanical Contrast Rule
Foreground ink (`#0F1914`) and deep botanical emerald (`#047857`) maintain minimum 5.4:1 contrast on all light surfaces (`#FBFBF9` and `#FFFFFF`), ensuring effortless readability and full WCAG AA compliance.

### The Purposeful Motion Rule
Motion is subtle, smooth, and informative. The scroll progress bar communicates viewport location, while content reveals animate gently with `cubic-bezier(0.16, 1, 0.3, 1)`. All motion is fully disabled when the user specifies `prefers-reduced-motion`.

## Component Guidelines

### Primary Actions
- Solid botanical emerald (`#047857`) background with white foreground text (`#FFFFFF`).
- 6px border radius, subtle neutral lift on hover (`translateY(-2px)`).

### Impact Cards
- Crisp white surface (`#FFFFFF`) with 1px border (`rgba(15, 25, 20, 0.08)`).
- 12px border radius, generous 1.75rem padding.
- Emotive botanical icons paired with clean Outfit headings and clear 2-line descriptions.

### Interactive Synthesizer
- Clean white console container with soft elevation shadow.
- Emerald slider thumb with smooth live numerical feedback in JetBrains Mono.
- Instant tabular recalculation of IoU, F1, and Precision/Recall.
