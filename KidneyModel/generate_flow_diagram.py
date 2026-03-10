"""Generate an interactive HTML call-graph / flow-diagram report."""

# ---------------------------------------------------------------------------
# Node definitions  (id, label, group, level, title-tooltip)
# group → colour class:
#   main=0, init=1, qnewton=2, fcn=3, jacobi=4, qflux=5, compute=6, config=7
# level → vertical position in the hierarchical layout (0 = top)
# ---------------------------------------------------------------------------
NODES = [
    # ── Orchestrator ────────────────────────────────────────────────────────
    {"id": "main",              "label": "main.py",               "group": 0, "level": 0,
     "title": "Top-level orchestrator — runs all segments in sequence"},
    {"id": "out_data_PT",       "label": "out_data_PT",           "group": 0, "level": 2,
     "title": "Prints PT solute deliveries + O₂ consumption (TNa/QO₂)"},

    # ── Config / globals ────────────────────────────────────────────────────
    {"id": "values",            "label": "values.py",             "group": 7, "level": 7,
     "title": "All simulation parameters (NZ, NS, boundary conditions, flags)"},
    {"id": "glo",               "label": "glo.py",                "group": 7, "level": 7,
     "title": "Global variables shared across modules"},
    {"id": "defs",              "label": "defs.py",               "group": 7, "level": 7,
     "title": "Membrane class + named compartment/solute index constants"},

    # ── Initialisation ───────────────────────────────────────────────────────
    {"id": "initPT",            "label": "initPT",                "group": 1, "level": 1,
     "title": "Initialise Proximal Tubule membrane — geometry, permeabilities, transporters"},
    {"id": "initSDL",           "label": "initSDL",               "group": 1, "level": 1,
     "title": "Initialise Short Descending Limb"},
    {"id": "initA",             "label": "initA\n(mTAL)",         "group": 1, "level": 1,
     "title": "Initialise medullary Thick Ascending Limb"},
    {"id": "initT",             "label": "initT\n(cTAL)",         "group": 1, "level": 1,
     "title": "Initialise cortical Thick Ascending Limb"},
    {"id": "initD",             "label": "initD\n(DCT)",          "group": 1, "level": 1,
     "title": "Initialise Distal Convoluted Tubule"},
    {"id": "initC",             "label": "initC\n(CNT)",          "group": 1, "level": 1,
     "title": "Initialise Connecting Tubule"},
    {"id": "initCCD",           "label": "initCCD\n(CCD)",        "group": 1, "level": 1,
     "title": "Initialise Cortical Collecting Duct"},
    {"id": "initOMC",           "label": "initOMC\n(OMCD)",       "group": 1, "level": 1,
     "title": "Initialise Outer Medullary Collecting Duct"},
    {"id": "initIMC",           "label": "initIMC\n(IMCD)",       "group": 1, "level": 1,
     "title": "Initialise Inner Medullary Collecting Duct"},

    # ── Newton solvers ───────────────────────────────────────────────────────
    {"id": "qnewton1PT",        "label": "qnewton1PT\n(Pass 1)",  "group": 2, "level": 2,
     "title": "Newton-Raphson solver — PT first pass (inlet BCs)"},
    {"id": "qnewton2PT",        "label": "qnewton2PT\n(Pass 2)",  "group": 2, "level": 2,
     "title": "Newton-Raphson solver — PT second pass (full segment)"},
    {"id": "qnewton2SDL",       "label": "qnewton2SDL",           "group": 2, "level": 2,
     "title": "Newton-Raphson solver — Short Descending Limb"},
    {"id": "qnewton2b",         "label": "qnewton2b\n(mTAL/cTAL/DCT/IMCD)", "group": 2, "level": 2,
     "title": "Shared Newton solver for mTAL, cTAL, DCT, IMCD segments"},
    {"id": "qnewton2icb",       "label": "qnewton2icb\n(CNT/CCD/OMCD)", "group": 2, "level": 2,
     "title": "Shared Newton solver for CNT, CCD, OMCD segments (4-compartment epithelium)"},

    # ── Residual (fcn) functions ─────────────────────────────────────────────
    {"id": "fcn1PT",            "label": "fcn1PT",                "group": 3, "level": 3,
     "title": "Residual vector — PT pass 1"},
    {"id": "fcn2PT",            "label": "fcn2PT",                "group": 3, "level": 3,
     "title": "Residual vector — PT pass 2"},
    {"id": "fcn2SDL",           "label": "fcn2SDL",               "group": 3, "level": 3,
     "title": "Residual vector — SDL"},
    {"id": "fcn2b",             "label": "fcn2b\n(mTAL/cTAL/DCT/IMCD)", "group": 3, "level": 3,
     "title": "Shared residual function — dispatches to qflux2A/T/D/IMC by segment id"},
    {"id": "fcn2C",             "label": "fcn2C\n(CNT/CCD)",      "group": 3, "level": 3,
     "title": "Residual vector — CNT and CCD; dispatches to qflux2C or qflux2CCD"},
    {"id": "fcn2OMC",           "label": "fcn2OMC\n(OMCD)",       "group": 3, "level": 3,
     "title": "Residual vector — OMCD"},

    # ── Jacobian evaluators ──────────────────────────────────────────────────
    {"id": "jacobi2_1PT",       "label": "jacobi2_1PT",           "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — PT pass 1"},
    {"id": "jacobi2_2PT",       "label": "jacobi2_2PT",           "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — PT pass 2 (bug was fixed here in v0)"},
    {"id": "jacobi2_2SDL",      "label": "jacobi2_2SDL",          "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — SDL"},
    {"id": "jacobi2_2b",        "label": "jacobi2_2b",            "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — mTAL/cTAL/DCT/IMCD"},
    {"id": "jacobi2_2icb",      "label": "jacobi2_2icb",          "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — CNT/CCD"},
    {"id": "jacobi2_2icbOMC",   "label": "jacobi2_2icbOMC",       "group": 4, "level": 3,
     "title": "Finite-difference Jacobian — OMCD"},

    # ── Flux evaluators ──────────────────────────────────────────────────────
    {"id": "qflux1PT",          "label": "qflux1PT",              "group": 5, "level": 4,
     "title": "All transmembrane fluxes — PT pass 1 (NHE3, SGLT, water, ECD)"},
    {"id": "qflux2PT",          "label": "qflux2PT",              "group": 5, "level": 4,
     "title": "All transmembrane fluxes — PT pass 2"},
    {"id": "qflux2A",           "label": "qflux2A\n(mTAL)",       "group": 5, "level": 4,
     "title": "Fluxes for mTAL: NKCC2 F/A, NHE3, NHE1, AE, KCC4, Na-K-ATPase"},
    {"id": "qflux2T",           "label": "qflux2T\n(cTAL)",       "group": 5, "level": 4,
     "title": "Fluxes for cTAL: NKCC2 B/A, NHE3, NHE1, AE, KCC4, Na-K-ATPase"},
    {"id": "qflux2D",           "label": "qflux2D\n(DCT)",        "group": 5, "level": 4,
     "title": "Fluxes for DCT: ENaC, NCC, TRPV5, NCX, PMCA, fatpase"},
    {"id": "qflux2C",           "label": "qflux2C\n(CNT)",        "group": 5, "level": 4,
     "title": "Fluxes for CNT: ENaC, ROMK, NHE3, NCX, PMCA, H-ATPase"},
    {"id": "qflux2CCD",         "label": "qflux2CCD\n(CCD)",      "group": 5, "level": 4,
     "title": "Fluxes for CCD: ENaC, ROMK, H-ATPase"},
    {"id": "qflux2OMC",         "label": "qflux2OMC\n(OMCD)",     "group": 5, "level": 4,
     "title": "Fluxes for OMCD: H-ATPase, water, ECD"},
    {"id": "qflux2IMC",         "label": "qflux2IMC\n(IMCD)",     "group": 5, "level": 4,
     "title": "Fluxes for IMCD: ENaC, ROMK, H-K-ATPase"},

    # ── Shared compute utilities ─────────────────────────────────────────────
    {"id": "compute_water_fluxes",      "label": "compute_water_fluxes",      "group": 6, "level": 5,
     "title": "Osmotic + hydrostatic water flux (Starling equation); @njit"},
    {"id": "compute_ecd_fluxes",        "label": "compute_ecd_fluxes",        "group": 6, "level": 5,
     "title": "Electrodiffusive (GHK) + convective solute fluxes; @njit"},
    {"id": "compute_nkcc2_flux",        "label": "compute_nkcc2_flux",        "group": 6, "level": 5,
     "title": "NKCC2 cotransporter kinetics (Na-K-2Cl); @njit"},
    {"id": "compute_kcc_fluxes",        "label": "compute_kcc_fluxes",        "group": 6, "level": 5,
     "title": "KCC4 cotransporter kinetics (K-Cl); @njit"},
    {"id": "compute_nhe3_fluxes",       "label": "compute_nhe3_fluxes",       "group": 6, "level": 5,
     "title": "NHE3 exchanger kinetics (Na-H); @njit"},
    {"id": "compute_ncx_fluxes",        "label": "compute_ncx_fluxes",        "group": 6, "level": 5,
     "title": "NCX exchanger kinetics (3Na:1Ca); @njit"},
    {"id": "compute_sdl_water_fluxes",  "label": "compute_sdl_water_fluxes",  "group": 6, "level": 5,
     "title": "SDL-specific water fluxes (different geometry from generic); @njit"},
    {"id": "compute_o2_consumption",    "label": "compute_o2_consumption",    "group": 6, "level": 5,
     "title": "Na transport (TNa) and O2 consumption (QO2) along a segment; @njit"},
    {"id": "sglt",                      "label": "sglt",                      "group": 6, "level": 5,
     "title": "SGLT1/2 glucose cotransporter kinetics; @njit"},
    {"id": "fatpase",                   "label": "fatpase",                   "group": 6, "level": 5,
     "title": "H-K-ATPase kinetic model (14 enzyme species); @njit"},
    {"id": "set_intconc",               "label": "set_intconc",               "group": 6, "level": 5,
     "title": "Interpolates interstitial concentrations from cortex to papilla"},
]

# ---------------------------------------------------------------------------
# Edge definitions  (from, to, label, dashes)
# ---------------------------------------------------------------------------
EDGES = [
    # main -> init
    ("main", "initPT",        ""),
    ("main", "initSDL",       ""),
    ("main", "initA",         ""),
    ("main", "initT",         ""),
    ("main", "initD",         ""),
    ("main", "initC",         ""),
    ("main", "initCCD",       ""),
    ("main", "initOMC",       ""),
    ("main", "initIMC",       ""),
    # main -> solvers
    ("main", "qnewton1PT",    ""),
    ("main", "qnewton2PT",    ""),
    ("main", "qnewton2SDL",   ""),
    ("main", "qnewton2b",     ""),
    ("main", "qnewton2icb",   ""),
    ("main", "out_data_PT",   ""),
    # PT solver -> fcn/jacobi
    ("qnewton1PT",   "fcn1PT",          ""),
    ("qnewton1PT",   "jacobi2_1PT",     ""),
    ("qnewton2PT",   "fcn2PT",          ""),
    ("qnewton2PT",   "jacobi2_2PT",     ""),
    ("jacobi2_1PT",  "fcn1PT",          "perturb"),
    ("jacobi2_2PT",  "fcn2PT",          "perturb"),
    # SDL solver
    ("qnewton2SDL",  "fcn2SDL",         ""),
    ("qnewton2SDL",  "jacobi2_2SDL",    ""),
    ("jacobi2_2SDL", "fcn2SDL",         "perturb"),
    # TAL/DCT/IMCD solver
    ("qnewton2b",    "fcn2b",           ""),
    ("qnewton2b",    "jacobi2_2b",      ""),
    ("jacobi2_2b",   "fcn2b",           "perturb"),
    # CNT/CCD/OMCD solver
    ("qnewton2icb",  "fcn2C",           ""),
    ("qnewton2icb",  "fcn2OMC",         ""),
    ("qnewton2icb",  "jacobi2_2icb",    ""),
    ("qnewton2icb",  "jacobi2_2icbOMC", ""),
    ("jacobi2_2icb",    "fcn2C",        "perturb"),
    ("jacobi2_2icbOMC", "fcn2OMC",      "perturb"),
    # fcn -> qflux
    ("fcn1PT",  "qflux1PT",   ""),
    ("fcn2PT",  "qflux2PT",   ""),
    ("fcn2SDL", "compute_sdl_water_fluxes", ""),
    ("fcn2b",   "qflux2A",    "mTAL"),
    ("fcn2b",   "qflux2T",    "cTAL"),
    ("fcn2b",   "qflux2D",    "DCT/IMCD"),
    ("fcn2C",   "qflux2C",    "CNT"),
    ("fcn2C",   "qflux2CCD",  "CCD"),
    ("fcn2OMC", "qflux2OMC",  ""),
    # qflux -> compute utilities
    ("qflux1PT",   "compute_water_fluxes",  ""),
    ("qflux1PT",   "compute_ecd_fluxes",    ""),
    ("qflux1PT",   "sglt",                  ""),
    ("qflux1PT",   "compute_nhe3_fluxes",   ""),
    ("qflux2PT",   "compute_water_fluxes",  ""),
    ("qflux2PT",   "compute_ecd_fluxes",    ""),
    ("qflux2PT",   "sglt",                  ""),
    ("qflux2PT",   "compute_nhe3_fluxes",   ""),
    ("qflux2A",    "compute_water_fluxes",  ""),
    ("qflux2A",    "compute_ecd_fluxes",    ""),
    ("qflux2A",    "compute_nkcc2_flux",    ""),
    ("qflux2A",    "compute_nhe3_fluxes",   ""),
    ("qflux2A",    "compute_kcc_fluxes",    ""),
    ("qflux2A",    "fatpase",               ""),
    ("qflux2T",    "compute_water_fluxes",  ""),
    ("qflux2T",    "compute_ecd_fluxes",    ""),
    ("qflux2T",    "compute_nkcc2_flux",    ""),
    ("qflux2T",    "compute_nhe3_fluxes",   ""),
    ("qflux2T",    "compute_kcc_fluxes",    ""),
    ("qflux2T",    "fatpase",               ""),
    ("qflux2D",    "compute_water_fluxes",  ""),
    ("qflux2D",    "compute_ecd_fluxes",    ""),
    ("qflux2D",    "compute_nhe3_fluxes",   ""),
    ("qflux2D",    "compute_ncx_fluxes",    ""),
    ("qflux2D",    "fatpase",               ""),
    ("qflux2C",    "compute_water_fluxes",  ""),
    ("qflux2C",    "compute_ecd_fluxes",    ""),
    ("qflux2C",    "compute_ncx_fluxes",    ""),
    ("qflux2C",    "fatpase",               ""),
    ("qflux2CCD",  "compute_water_fluxes",  ""),
    ("qflux2CCD",  "fatpase",               ""),
    ("qflux2OMC",  "compute_water_fluxes",  ""),
    ("qflux2OMC",  "compute_ecd_fluxes",    ""),
    ("qflux2OMC",  "fatpase",               ""),
    ("qflux2IMC",  "compute_water_fluxes",  ""),
    ("qflux2IMC",  "compute_ecd_fluxes",    ""),
    ("qflux2IMC",  "fatpase",               ""),
    # init -> set_intconc
    ("initPT",   "set_intconc", ""),
    ("initSDL",  "set_intconc", ""),
    ("initA",    "set_intconc", ""),
    ("initC",    "set_intconc", ""),
    ("initCCD",  "set_intconc", ""),
    ("initOMC",  "set_intconc", ""),
    ("initIMC",  "set_intconc", ""),
    # out_data_PT -> O2
    ("out_data_PT", "compute_o2_consumption", ""),
    # config (shown as dashed background references, not arrows in main graph)
    # values/glo/defs used by almost everything — shown in legend, not as edges
    # to avoid visual clutter
]

# ---------------------------------------------------------------------------
# Segment pipeline description (for the static pipeline table)
# ---------------------------------------------------------------------------
PIPELINE = [
    ("PT",   "Proximal Tubule",              "initPT",   "qnewton1PT + qnewton2PT", "qflux1PT + qflux2PT",
     "compute_water_fluxes, compute_ecd_fluxes, sglt, compute_nhe3_fluxes"),
    ("SDL",  "Short Descending Limb",        "initSDL",  "qnewton2SDL",             "compute_sdl_water_fluxes",
     "compute_sdl_water_fluxes (SDL-specific)"),
    ("mTAL", "medullary Thick Ascending Limb","initA",   "qnewton2b",               "qflux2A",
     "compute_water_fluxes, compute_ecd_fluxes, compute_nkcc2_flux, compute_nhe3_fluxes, compute_kcc_fluxes, fatpase"),
    ("cTAL", "cortical Thick Ascending Limb","initT",    "qnewton2b",               "qflux2T",
     "compute_water_fluxes, compute_ecd_fluxes, compute_nkcc2_flux, compute_nhe3_fluxes, compute_kcc_fluxes, fatpase"),
    ("DCT",  "Distal Convoluted Tubule",     "initD",    "qnewton2b",               "qflux2D",
     "compute_water_fluxes, compute_ecd_fluxes, compute_nhe3_fluxes, compute_ncx_fluxes, fatpase"),
    ("CNT",  "Connecting Tubule",            "initC",    "qnewton2icb",             "qflux2C",
     "compute_water_fluxes, compute_ecd_fluxes, compute_ncx_fluxes, fatpase"),
    ("CCD",  "Cortical Collecting Duct",     "initCCD",  "qnewton2icb",             "qflux2CCD",
     "compute_water_fluxes, fatpase"),
    ("OMCD", "Outer Medullary Collecting Duct","initOMC","qnewton2icb",             "qflux2OMC",
     "compute_water_fluxes, compute_ecd_fluxes, fatpase"),
    ("IMCD", "Inner Medullary Collecting Duct","initIMC", "qnewton2b",              "qflux2IMC",
     "compute_water_fluxes, compute_ecd_fluxes, fatpase"),
]

import json

nodes_js = json.dumps(NODES, indent=4)
edges_js = json.dumps([
    {"from": f, "to": t, "label": lbl, "dashes": (lbl == "perturb")}
    for f, t, lbl in EDGES
], indent=4)

pipeline_rows = ""
for seg, name, init, solver, flux, utils in PIPELINE:
    pipeline_rows += f"""
      <tr>
        <td><strong>{seg}</strong></td>
        <td>{name}</td>
        <td><code>{init}</code></td>
        <td><code>{solver}</code></td>
        <td><code>{flux}</code></td>
        <td style="font-size:9pt">{utils}</td>
      </tr>"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Kidney Model — Call Graph &amp; Flow Diagram</title>
<script src="https://unpkg.com/vis-network@9.1.9/standalone/umd/vis-network.min.js"></script>
<style>
  /* ── Page chrome ────────────────────────────────────────────────────── */
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11pt;
         color: #1a1a1a; max-width: 1100px; margin: 40px auto; line-height: 1.55;
         padding: 0 20px; }}
  h1   {{ font-size: 20pt; border-bottom: 3px solid #2c5f8a; padding-bottom: 8px;
         color: #1a3a5c; margin-top: 40px; }}
  h2   {{ font-size: 15pt; color: #2c5f8a; border-bottom: 1px solid #aac4de;
         padding-bottom: 4px; margin-top: 36px; }}
  h3   {{ font-size: 12pt; color: #1a3a5c; margin-top: 26px; margin-bottom: 6px; }}
  p    {{ margin: 6px 0 10px; }}
  code {{ background:#f0f0f0; padding:1px 5px; border-radius:3px;
         font-family:"Courier New",monospace; font-size:9.5pt; }}

  /* ── Tables ─────────────────────────────────────────────────────────── */
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 10pt; }}
  th    {{ background: #2c5f8a; color: #fff; padding: 7px 10px; text-align: left;
          font-weight: 600; }}
  td    {{ padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #dde5ef; }}
  tr:nth-child(even) td {{ background: #f3f7fb; }}

  /* ── Vis.js network container ────────────────────────────────────────── */
  #network-container {{
    width: 100%; height: 820px;
    border: 1px solid #aac4de; border-radius: 8px;
    background: #fafcff;
    margin: 16px 0 24px;
  }}

  /* ── Legend ─────────────────────────────────────────────────────────── */
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 14px 0 20px; }}
  .leg-item {{ display: flex; align-items: center; gap: 6px; font-size: 10pt; }}
  .leg-dot {{
    width: 18px; height: 18px; border-radius: 50%;
    border: 2px solid rgba(0,0,0,0.2); flex-shrink: 0;
  }}

  /* ── Pipeline strip ─────────────────────────────────────────────────── */
  .pipeline {{
    display: flex; align-items: center; gap: 0;
    margin: 18px 0 28px; flex-wrap: wrap;
  }}
  .seg-box {{
    background: #2c5f8a; color: #fff;
    padding: 10px 14px; border-radius: 6px;
    text-align: center; font-weight: 700;
    font-size: 10.5pt; min-width: 72px;
  }}
  .seg-box span {{ display: block; font-size: 8pt; font-weight: 400;
                    opacity: 0.85; margin-top: 2px; }}
  .arrow {{
    font-size: 20pt; color: #2c5f8a; padding: 0 4px;
    flex-shrink: 0;
  }}

  /* ── Call-chain cards ────────────────────────────────────────────────── */
  .chain-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 16px; margin: 14px 0 24px;
  }}
  .chain-card {{
    border: 1px solid #aac4de; border-radius: 7px;
    padding: 14px 16px; background: #fafcff;
  }}
  .chain-card h4 {{ margin: 0 0 10px; color: #1a3a5c; font-size: 11pt; }}
  .chain-step {{
    display: flex; align-items: flex-start; gap: 8px;
    margin-bottom: 6px; font-size: 9.5pt;
  }}
  .step-dot {{
    width: 14px; height: 14px; border-radius: 50%;
    flex-shrink: 0; margin-top: 3px;
  }}
  .chain-indent {{ margin-left: 22px; }}

  /* ── Tip ─────────────────────────────────────────────────────────────── */
  .tip {{
    background: #fef9e7; border: 1px solid #f0c040; border-radius: 5px;
    padding: 10px 16px; margin: 0 0 18px; font-size: 10pt;
  }}
  .section-intro {{ color: #444; font-size: 10.5pt; margin-bottom: 10px; }}
</style>
</head>
<body>

<h1>Kidney Model — Call Graph &amp; Flow Diagram</h1>
<p style="color:#555; font-size:10pt;">
  Repository: <code>kidney-models-females_age/PaperCodes/</code> &nbsp;|&nbsp;
  Generated: 2026-03-04 &nbsp;|&nbsp;
  Non-v0 files only &nbsp;|&nbsp; 16 solutes &nbsp;|&nbsp; 9 tubule segments
</p>

<!-- =========================================================== -->
<h2>1. Pipeline Overview</h2>
<p class="section-intro">
  <code>main.py</code> runs each segment sequentially. The outlet of each segment
  becomes the inlet of the next. Each segment follows the same pattern:
  <strong>init &rarr; qnewton (solver) &rarr; fcn (residual) &rarr; qflux (fluxes) &rarr; compute_* (transporters)</strong>.
</p>

<div class="pipeline">
  <div class="seg-box">PT<span>Proximal<br>Tubule</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">SDL<span>Short Desc.<br>Limb</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">mTAL<span>med. Thick<br>Asc. Limb</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">cTAL<span>cort. Thick<br>Asc. Limb</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">DCT<span>Distal Conv.<br>Tubule</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">CNT<span>Connecting<br>Tubule</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">CCD<span>Cort. Coll.<br>Duct</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">OMCD<span>Outer Med.<br>Coll. Duct</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box">IMCD<span>Inner Med.<br>Coll. Duct</span></div>
  <div class="arrow">&#8594;</div>
  <div class="seg-box" style="background:#1a3a5c;">Outlet<br>_SS<span>urinary<br>output</span></div>
</div>

<table>
  <tr>
    <th>Seg.</th><th>Full Name</th><th>Init module</th>
    <th>Solver</th><th>Flux module</th><th>Compute utilities called</th>
  </tr>
  {pipeline_rows}
</table>

<!-- =========================================================== -->
<h2>2. Interactive Call Graph</h2>
<p class="section-intro">
  Every node is a Python file. Edges show function calls / imports.
  <strong>Hover</strong> over a node for a description.
  <strong>Drag</strong> nodes to rearrange. <strong>Scroll</strong> to zoom.
  Dashed edges = Jacobian finite-difference perturbation calls.
</p>
<div class="tip">
  &#128161; <strong>Tip:</strong> Use the scroll wheel to zoom in on a region, and
  drag the background to pan. Click any node to highlight its direct connections.
</div>

<!-- Legend -->
<div class="legend">
  <div class="leg-item"><div class="leg-dot" style="background:#1a3a5c"></div> main / output</div>
  <div class="leg-item"><div class="leg-dot" style="background:#27ae60"></div> init*</div>
  <div class="leg-item"><div class="leg-dot" style="background:#2980b9"></div> qnewton* (solver)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#e67e22"></div> fcn* (residual)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#8e44ad"></div> jacobi* (Jacobian)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#c0392b"></div> qflux* (fluxes)</div>
  <div class="leg-item"><div class="leg-dot" style="background:#d4a017"></div> compute_* / sglt / fatpase</div>
  <div class="leg-item"><div class="leg-dot" style="background:#7f8c8d"></div> values / glo / defs</div>
</div>

<div id="network-container"></div>

<!-- =========================================================== -->
<h2>3. Call Chain by Segment Family</h2>
<p class="section-intro">
  Three solver families serve all nine segments. Shared solvers dispatch dynamically
  to the correct flux module based on a segment identifier.
</p>

<div class="chain-grid">

  <!-- PT -->
  <div class="chain-card">
    <h4>&#9679; Proximal Tubule (PT)</h4>
    <div class="chain-step"><div class="step-dot" style="background:#1a3a5c"></div>
      <div><code>main.py</code></div></div>
    <div class="chain-indent">
      <div class="chain-step"><div class="step-dot" style="background:#27ae60"></div>
        <div><code>initPT</code> &rarr; <code>set_intconc</code></div></div>
      <div class="chain-step"><div class="step-dot" style="background:#2980b9"></div>
        <div><strong>Pass 1:</strong> <code>qnewton1PT</code></div></div>
      <div class="chain-indent">
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn1PT</code> &rarr; <code>qflux1PT</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_1PT</code> &rarr; <code>fcn1PT</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
          <code>qflux1PT</code>: water, ecd, sglt, nhe3</div>
      </div>
      <div class="chain-step"><div class="step-dot" style="background:#2980b9"></div>
        <div><strong>Pass 2:</strong> <code>qnewton2PT</code></div></div>
      <div class="chain-indent">
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn2PT</code> &rarr; <code>qflux2PT</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_2PT</code> &rarr; <code>fcn2PT</code> <em>(bug fixed)</em></div>
        <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
          <code>qflux2PT</code>: water, ecd, sglt, nhe3</div>
      </div>
      <div class="chain-step"><div class="step-dot" style="background:#1a3a5c"></div>
        <div><code>out_data_PT</code> &rarr; <code>compute_o2_consumption</code></div></div>
    </div>
  </div>

  <!-- SDL -->
  <div class="chain-card">
    <h4>&#9679; Short Descending Limb (SDL)</h4>
    <div class="chain-step"><div class="step-dot" style="background:#1a3a5c"></div>
      <div><code>main.py</code></div></div>
    <div class="chain-indent">
      <div class="chain-step"><div class="step-dot" style="background:#27ae60"></div>
        <code>initSDL</code> &rarr; <code>set_intconc</code></div>
      <div class="chain-step"><div class="step-dot" style="background:#2980b9"></div>
        <code>qnewton2SDL</code></div>
      <div class="chain-indent">
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn2SDL</code> &rarr; <code>compute_sdl_water_fluxes</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_2SDL</code> &rarr; <code>fcn2SDL</code></div>
      </div>
    </div>
    <p style="font-size:9pt; color:#555; margin-top:8px;">
      SDL uses a dedicated water-flux function because its geometry (descending thin limb)
      differs from the generic module.
    </p>
  </div>

  <!-- qnewton2b family -->
  <div class="chain-card">
    <h4>&#9679; mTAL / cTAL / DCT / IMCD &mdash; <code>qnewton2b</code> family</h4>
    <div class="chain-step"><div class="step-dot" style="background:#1a3a5c"></div>
      <div><code>main.py</code></div></div>
    <div class="chain-indent">
      <div class="chain-step"><div class="step-dot" style="background:#27ae60"></div>
        <code>initA</code> / <code>initT</code> / <code>initD</code> / <code>initIMC</code>
        &rarr; <code>set_intconc</code></div>
      <div class="chain-step"><div class="step-dot" style="background:#2980b9"></div>
        <code>qnewton2b</code> (one call per segment)</div>
      <div class="chain-indent">
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn2b</code> (dispatches by <code>idid</code>):</div>
        <div class="chain-indent">
          <div class="chain-step"><div class="step-dot" style="background:#c0392b"></div>
            &rarr; <code>qflux2A</code> (mTAL)</div>
          <div class="chain-step"><div class="step-dot" style="background:#c0392b"></div>
            &rarr; <code>qflux2T</code> (cTAL)</div>
          <div class="chain-step"><div class="step-dot" style="background:#c0392b"></div>
            &rarr; <code>qflux2D</code> (DCT + IMCD)</div>
        </div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_2b</code> &rarr; <code>fcn2b</code></div>
      </div>
      <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
        <code>qflux2A/T</code>: water, ecd, nkcc2, nhe3, kcc, fatpase</div>
      <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
        <code>qflux2D</code>: water, ecd, nhe3, ncx, fatpase</div>
    </div>
  </div>

  <!-- qnewton2icb family -->
  <div class="chain-card">
    <h4>&#9679; CNT / CCD / OMCD &mdash; <code>qnewton2icb</code> family</h4>
    <div class="chain-step"><div class="step-dot" style="background:#1a3a5c"></div>
      <div><code>main.py</code></div></div>
    <div class="chain-indent">
      <div class="chain-step"><div class="step-dot" style="background:#27ae60"></div>
        <code>initC</code> / <code>initCCD</code> / <code>initOMC</code>
        &rarr; <code>set_intconc</code></div>
      <div class="chain-step"><div class="step-dot" style="background:#2980b9"></div>
        <code>qnewton2icb</code> (one call per segment)</div>
      <div class="chain-indent">
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn2C</code> (CNT/CCD):</div>
        <div class="chain-indent">
          <div class="chain-step"><div class="step-dot" style="background:#c0392b"></div>
            &rarr; <code>qflux2C</code> (CNT)</div>
          <div class="chain-step"><div class="step-dot" style="background:#c0392b"></div>
            &rarr; <code>qflux2CCD</code> (CCD)</div>
        </div>
        <div class="chain-step"><div class="step-dot" style="background:#e67e22"></div>
          <code>fcn2OMC</code> &rarr; <code>qflux2OMC</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_2icb</code> &rarr; <code>fcn2C</code></div>
        <div class="chain-step"><div class="step-dot" style="background:#8e44ad"></div>
          <code>jacobi2_2icbOMC</code> &rarr; <code>fcn2OMC</code></div>
      </div>
      <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
        <code>qflux2C</code>: water, ecd, ncx, fatpase</div>
      <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
        <code>qflux2CCD</code>: water, fatpase</div>
      <div class="chain-step"><div class="step-dot" style="background:#d4a017"></div>
        <code>qflux2OMC</code>: water, ecd, fatpase</div>
    </div>
  </div>

</div>

<!-- =========================================================== -->
<h2>4. Shared Utilities</h2>
<p class="section-intro">These modules are imported by multiple segments. Every compute_* function is JIT-compiled with Numba <code>@njit</code>.</p>

<table>
  <tr><th style="width:26%">Module</th><th style="width:16%">Called by</th><th>Purpose</th></tr>
  <tr><td><code>compute_water_fluxes</code></td>
      <td>qflux1PT, qflux2PT, qflux2A, qflux2T, qflux2D, qflux2C, qflux2CCD, qflux2OMC, qflux2IMC (9 modules)</td>
      <td>Osmotic and hydrostatic water fluxes across each membrane (Starling equation). Used by every segment.</td></tr>
  <tr><td><code>compute_ecd_fluxes</code></td>
      <td>qflux1PT, qflux2PT, qflux2A, qflux2T, qflux2D, qflux2C, qflux2OMC, qflux2IMC (8 modules)</td>
      <td>Electrodiffusive (Goldman-Hodgkin-Katz) + convective solute fluxes across tight junctions and membranes.</td></tr>
  <tr><td><code>fatpase</code></td>
      <td>qflux2A, qflux2T, qflux2D, qflux2C, qflux2CCD, qflux2OMC, qflux2IMC (7 modules)</td>
      <td>H-K-ATPase (proton pump) kinetics — 14-state enzyme model. Used by all TAL and collecting duct segments.</td></tr>
  <tr><td><code>compute_nkcc2_flux</code></td>
      <td>qflux2A, qflux2T (2 modules)</td>
      <td>NKCC2 (Na-K-2Cl) cotransporter kinetics. Only in TAL segments.</td></tr>
  <tr><td><code>compute_kcc_fluxes</code></td>
      <td>qflux2A, qflux2T (2 modules)</td>
      <td>KCC4 (K-Cl) cotransporter kinetics. Only in TAL segments.</td></tr>
  <tr><td><code>compute_nhe3_fluxes</code></td>
      <td>qflux1PT, qflux2PT, qflux2A, qflux2T, qflux2D (5 modules)</td>
      <td>NHE3 (Na-H) exchanger kinetics. Used in PT, TAL, DCT.</td></tr>
  <tr><td><code>compute_ncx_fluxes</code></td>
      <td>qflux2D, qflux2C (2 modules)</td>
      <td>NCX (3Na:1Ca) exchanger kinetics. Used in DCT and CNT.</td></tr>
  <tr><td><code>compute_sdl_water_fluxes</code></td>
      <td>fcn2SDL only</td>
      <td>SDL-specific water flux — different membrane area/permeability structure from generic.</td></tr>
  <tr><td><code>sglt</code></td>
      <td>qflux1PT, qflux2PT (2 modules)</td>
      <td>SGLT1/2 glucose cotransporter. Only in PT.</td></tr>
  <tr><td><code>compute_o2_consumption</code></td>
      <td>out_data_PT only</td>
      <td>Computes TNa (Na transport rate) and QO₂ (O₂ consumption) for PT, PCT, S3.</td></tr>
  <tr><td><code>set_intconc</code></td>
      <td>initPT, initSDL, initA, initC, initCCD, initOMC, initIMC (7 modules)</td>
      <td>Interpolates interstitial bath concentrations between cortical and medullary boundaries at each spatial node.</td></tr>
</table>

<!-- =========================================================== -->
<h2>5. Configuration Layer</h2>
<p class="section-intro">These three modules are imported by nearly every file in the codebase.</p>
<table>
  <tr><th style="width:20%">Module</th><th>Role</th><th>Key contents</th></tr>
  <tr>
    <td><code>values.py</code></td>
    <td>All simulation parameters</td>
    <td><code>NZ</code> (grid), <code>NS=16</code> (solutes), <code>NC=6</code> (compartments),
        <code>sngfr0</code>, <code>bdiabetes</code>, <code>furo</code>, boundary concentrations</td>
  </tr>
  <tr>
    <td><code>glo.py</code></td>
    <td>Global runtime variables</td>
    <td>Shared mutable state (e.g. current interstitial concentrations, dimensional factors)
        passed implicitly between modules via <code>from glo import *</code></td>
  </tr>
  <tr>
    <td><code>defs.py</code></td>
    <td>Class definition + named index constants</td>
    <td><code>Membrane</code> class; compartment constants
        <code>LUM=0, P=1, A=2, B=3, LIS=4, BATH=5</code>;
        solute constants <code>NA=0, K=1, CL=2, HCO3=3 ...</code></td>
  </tr>
</table>

<p style="margin-top:30px; font-size:9pt; color:#888;">
  Generated by generate_flow_diagram.py &mdash; kidney-models-females_age/PaperCodes/ &mdash; 2026-03-04
</p>

<!-- =========================================================== -->
<script>
// ── Node colour palette ──────────────────────────────────────────────────
const GROUP_COLORS = {{
  0: {{ background:"#1a3a5c", border:"#0d1f33", font:"#ffffff" }},   // main/output
  1: {{ background:"#27ae60", border:"#1a7a42", font:"#ffffff" }},   // init
  2: {{ background:"#2980b9", border:"#1a5c8a", font:"#ffffff" }},   // qnewton
  3: {{ background:"#e67e22", border:"#b55a00", font:"#ffffff" }},   // fcn
  4: {{ background:"#8e44ad", border:"#6c2d87", font:"#ffffff" }},   // jacobi
  5: {{ background:"#c0392b", border:"#8e1a12", font:"#ffffff" }},   // qflux
  6: {{ background:"#d4a017", border:"#a07800", font:"#1a1a1a" }},   // compute / utils
  7: {{ background:"#95a5a6", border:"#6b7f80", font:"#1a1a1a" }},   // config
}};

const raw_nodes = {nodes_js};
const raw_edges = {edges_js};

// Build vis DataSets
const nodesData = raw_nodes.map(n => ({{
  id:    n.id,
  label: n.label,
  title: n.title,
  level: n.level,
  color: {{
    background: GROUP_COLORS[n.group].background,
    border:     GROUP_COLORS[n.group].border,
    highlight: {{ background: "#f39c12", border: "#d68910" }},
    hover:     {{ background: "#f9ca24", border: "#d68910" }},
  }},
  font:  {{ color: GROUP_COLORS[n.group].font, size: 12, face: "Helvetica Neue, Arial, sans-serif" }},
  shape: "box",
  margin: 8,
  shadow: {{ enabled: true, size: 4, x: 2, y: 2, color: "rgba(0,0,0,0.15)" }},
}}));

const edgesData = raw_edges.map((e, i) => ({{
  id:     i,
  from:   e.from,
  to:     e.to,
  label:  (e.label && e.label !== "perturb") ? e.label : "",
  dashes: e.dashes,
  arrows: {{ to: {{ enabled: true, scaleFactor: 0.7 }} }},
  color:  {{ color: e.dashes ? "#8e44ad" : "#7f8c8d",
             highlight: "#f39c12", hover: "#f39c12" }},
  font:   {{ size: 9, color: "#555", align: "middle" }},
  width:  e.dashes ? 1 : 1.5,
  smooth: {{ type: "cubicBezier", forceDirection: "vertical", roundness: 0.4 }},
}}));

const container = document.getElementById("network-container");
const data = {{
  nodes: new vis.DataSet(nodesData),
  edges: new vis.DataSet(edgesData),
}};

const options = {{
  layout: {{
    hierarchical: {{
      direction: "UD",           // top-to-bottom
      sortMethod: "directed",
      levelSeparation: 110,
      nodeSpacing: 155,
      treeSpacing: 200,
      blockShifting: true,
      edgeMinimization: true,
    }},
  }},
  physics: {{ enabled: false }},
  interaction: {{
    hover: true,
    navigationButtons: true,
    keyboard: true,
    tooltipDelay: 200,
  }},
  nodes: {{ borderWidth: 2, borderWidthSelected: 3 }},
  edges: {{ selectionWidth: 2 }},
}};

const network = new vis.Network(container, data, options);

// Highlight neighbours on click
network.on("click", function(params) {{
  if (params.nodes.length === 0) {{
    // Deselect — reset all
    nodesData.forEach(n => {{
      data.nodes.update({{ id: n.id, opacity: 1 }});
    }});
    edgesData.forEach(e => {{
      data.edges.update({{ id: e.id, opacity: 1 }});
    }});
    return;
  }}
  const selectedId = params.nodes[0];
  const connected = new Set(network.getConnectedNodes(selectedId));
  connected.add(selectedId);
  nodesData.forEach(n => {{
    data.nodes.update({{ id: n.id, opacity: connected.has(n.id) ? 1 : 0.15 }});
  }});
  edgesData.forEach(e => {{
    const visible = connected.has(e.from) && connected.has(e.to);
    data.edges.update({{ id: e.id, opacity: visible ? 1 : 0.06 }});
  }});
}});
</script>
</body>
</html>
"""

out = "/Users/sofiapolychroniadou/Desktop/edwardslab/kidney-models-females_age/PaperCodes/call_graph.html"
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"Written: {out}")
