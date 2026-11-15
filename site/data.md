---
title: Data
layout: default
nav_order: 3
description: "USGS gauge cameras, where the labels come from, and how splits keep whole months apart."
---

# Data
{: .no_toc }

1. TOC
{:toc}

{% assign n = site.data.numbers %}

## Source

Frames come from the public USGS NIMS image bucket. Labels are USGS gauge heights. Both are US Government works in the public domain and need no account or key. No camera-derived value is ever used as a label.

{: .caution }
NWIS was not reachable from my environment while preparing the main camera, so its gauge values come from the committed result tables of two open-source projects (listed in `data/usgs/README.md`) and were not re-checked against NWIS. The other cameras' labels were fetched from the USGS API directly.

## Cameras

| Camera | What it sees | Role |
|---|---|---|
| `in_overland` (USGS 411229084541101, Indiana) | Shallow overland flow past a cream staff plate | Main camera: {{ n.nFrames }} frames over {{ n.nBlocks }} months, {{ n.yearFirst }}&ndash;{{ n.yearLast }} |
| `in_north` | Cream plate | Added after the method was frozen; evaluation only |
| `oh_wetland` | White plate | Added after the method was frozen; evaluation only |
| `wi_mt1` | Orange plate | Added after the method was frozen; evaluation only |
| `hancock` (USGS 01421500, New York) | River plate | Added after the method was frozen; one flood ({{ n.hancN }} gated frames over {{ n.hancRangeCm }} cm) |

The three added staff-plate cameras have about 1400 frames each from July to November 2026.

## The main camera

- Median spacing between frames is 21 minutes.
- Gauge height ranges from &minus;0.6 to {{ n.gaugeMax }} cm. {{ n.pctDry }}% of frames are dry and {{ n.pctIR }}% are infrared night images.
- The plate spans 0&ndash;{{ n.plateTopCm }} cm and the gauge sits beside it. The plate bottom is at {{ n.plateBottomCm }} cm, so shallower water cannot be observed.
- The pose changes between months. Images show fog, mud, droplets, glare and over-exposure.

## Splits

The split unit is the calendar month, never the frame. Consecutive frames are near duplicates, and lighting and camera pose are shared within a month, so a frame-level split would leak.

| Set | Months | Frames | Used for |
|---|---|---|---|
| Development | {{ n.nDevBlocks }} | {{ n.nDevFrames }} | Leave-one-month-out cross-validation and all tuning |
| Locked test | {{ n.nLockedBlocks }} ({{ n.lockedBlocks }}) | {{ n.nLockedFrames }} | Final evaluation, fixed before any model was trained |
| Hancock flood | one event | {{ n.hancFrames }} | Locked; no template, offset or training uses it |

The split files are committed as file IDs only (`data/splits/`). The script that evaluates the locked months refuses to run twice without an explicit flag and logs every run to `results/real/final_test_log.jsonl`.

## Leakage control

Everything learned is fitted on training months only: the dry template, calibration, classifier, baseline threshold and gate thresholds. Registration references come from development months and carry geometry, not labels.

## Licences and credit

USGS imagery and gauge data are public domain. The 68 demo frames on this site are USGS images from one event in September 2026, shown at 640 px.
