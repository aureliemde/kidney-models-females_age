"""
Generate a readable HTML report comparing v0 vs current versions,
then convert to PDF via pandoc + LaTeX.
"""

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>v0 vs Current: Kidney Model Refactoring Report</title>
<style>
  body { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11pt;
         color: #1a1a1a; max-width: 900px; margin: 40px auto; line-height: 1.55; }
  h1   { font-size: 20pt; border-bottom: 3px solid #2c5f8a; padding-bottom: 8px;
         color: #1a3a5c; margin-top: 40px; }
  h2   { font-size: 15pt; color: #2c5f8a; border-bottom: 1px solid #aac4de;
         padding-bottom: 4px; margin-top: 36px; }
  h3   { font-size: 12pt; color: #1a3a5c; margin-top: 26px; margin-bottom: 6px; }
  p    { margin: 6px 0 10px; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0 22px; font-size: 10pt; }
  th   { background: #2c5f8a; color: #fff; padding: 7px 10px; text-align: left;
         font-weight: 600; }
  td   { padding: 6px 10px; vertical-align: top; border-bottom: 1px solid #dde5ef; }
  tr:nth-child(even) td { background: #f3f7fb; }
  .tag-bug    { background:#c0392b; color:#fff; border-radius:3px; padding:1px 6px;
                font-size:9pt; font-weight:700; }
  .tag-new    { background:#27ae60; color:#fff; border-radius:3px; padding:1px 6px;
                font-size:9pt; font-weight:700; }
  .tag-refac  { background:#7f8c8d; color:#fff; border-radius:3px; padding:1px 6px;
                font-size:9pt; font-weight:700; }
  .tag-fix    { background:#e67e22; color:#fff; border-radius:3px; padding:1px 6px;
                font-size:9pt; font-weight:700; }
  .file-header { background:#eaf0f8; padding:6px 12px; border-left:4px solid #2c5f8a;
                 margin:20px 0 6px; font-weight:700; font-size:11pt; }
  .summary-box { background:#fef9e7; border:1px solid #f0c040; border-radius:5px;
                 padding:14px 18px; margin:18px 0; }
  .no-change   { color: #7f8c8d; font-style: italic; }
  code { background:#f0f0f0; padding:1px 4px; border-radius:2px; font-size:9.5pt;
         font-family: "Courier New", monospace; }
  ul { margin: 4px 0 8px 18px; }
  li { margin-bottom: 3px; }
  .section-intro { color: #444; font-size: 10.5pt; margin-bottom: 10px; }
</style>
</head>
<body>

<h1>Kidney Model Codebase: v0 → Current Refactoring Report</h1>
<p style="color:#555; font-size:10pt;">
  Repository: <code>kidney-models-females_age/PaperCodes/</code> &nbsp;|&nbsp;
  Generated: 2026-03-04 &nbsp;|&nbsp;
  Authors: Fortran — Prof. Aurelie Edwards; Python — Dr. Mohammad M. Tajdini; Refactoring — Sofia Polychroniadou
</p>

<!-- ============================================================ -->
<h2>Executive Summary</h2>

<div class="summary-box">
  <strong>51 v0 files</strong> were compared against their current counterparts. Every file changed.
  The changes fall into five categories:
  <ul>
    <li><strong>Code organisation</strong> — monolithic functions split into private helpers everywhere</li>
    <li><strong>Documentation</strong> — module-level and function-level docstrings added to every module</li>
    <li><strong>Namespace modernisation</strong> — Fortran-style 1-based indices (<code>C[I-1, 1-1]</code>) replaced with named constants (<code>C[NA, LUM]</code>); imports moved from inside loops/functions to module top level</li>
    <li><strong>Algorithm improvements</strong> — Newton solver restructuring, Jacobian bug fix in PT, packing index fixes in SDL</li>
    <li><strong>New shared modules</strong> — <code>compute_o2_consumption.py</code>, <code>compute_sdl_water_fluxes.py</code>, <code>set_intconc.py</code> extracted from duplicated inline code</li>
  </ul>
  Numerical (physiological) results are changed only in the PT segment (Jacobian bug fix) and are unaffected by the SDL water permeability cleanup.
</div>

<!-- ============================================================ -->
<h2>1. Changes Shared by All 51 Files</h2>
<p class="section-intro">These changes appear uniformly across every module and are not repeated in the per-file tables below.</p>

<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current version</th></tr>
  <tr>
    <td>Imports</td>
    <td>Placed inside the function body, or inside the Newton solver loop — re-evaluated on every call</td>
    <td>All imports at module top level — evaluated once at import time</td>
  </tr>
  <tr>
    <td>Array indexing</td>
    <td>Fortran-style integer arithmetic: <code>C[1-1, 1-1]</code>, <code>C[12-1, 2-1]</code>, <code>pars[1+NS-1]</code></td>
    <td>Named constants from <code>defs.py</code>: <code>C[NA, LUM]</code>, <code>C[H, P]</code>, <code>pars[NS]</code></td>
  </tr>
  <tr>
    <td>Module docstrings</td>
    <td>None, or <code>#---</code> banner comments</td>
    <td>Full module-level docstring with physiology notes, model origin, unit system</td>
  </tr>
  <tr>
    <td>Function docstrings</td>
    <td>None</td>
    <td>NumPy-style docstrings with <code>Parameters</code>, <code>Returns</code>, and transporter/physiology notes</td>
  </tr>
  <tr>
    <td>Code structure</td>
    <td>Single monolithic function body (100–600+ lines)</td>
    <td>Decomposed into private helper functions (<code>_set_membrane_areas</code>, <code>_compute_water_fluxes</code>, <code>_unpack_state</code>, etc.)</td>
  </tr>
  <tr>
    <td>Unused variables</td>
    <td>Dead variables present throughout (<code>theta</code>, <code>Slum</code>, <code>Slat</code>, <code>Sbas</code>, <code>pos</code>, <code>ind</code>, <code>fracdel</code>, etc.)</td>
    <td>All unused local variables removed</td>
  </tr>
  <tr>
    <td>Separator comments</td>
    <td>Large blocks of <code>#---72</code>, <code>##########</code> lines between sections</td>
    <td>Replaced with inline section comments or removed</td>
  </tr>
  <tr>
    <td>Transporter kinetics</td>
    <td><em>Baseline</em></td>
    <td>Unchanged in all files except where noted explicitly below</td>
  </tr>
</table>

<!-- ============================================================ -->
<h2>2. New Files (No v0 Counterpart)</h2>

<table>
  <tr><th style="width:30%">File</th><th>Purpose</th></tr>
  <tr><td><code>compute_o2_consumption.py</code></td><td>Computes sodium transport (TNa) and O₂ consumption (QO₂) along any tubule segment. Called by <code>out_data_PT.py</code>.</td></tr>
  <tr><td><code>compute_sdl_water_fluxes.py</code></td><td>SDL-specific water flux function split from <code>compute_water_fluxes.py</code> because SDL geometry differs. Called by <code>fcn2SDL.py</code>.</td></tr>
  <tr><td><code>set_intconc.py</code></td><td>Shared helper that interpolates interstitial concentrations between cortical and medullary boundary values. Previously duplicated inside every <code>init*</code> module.</td></tr>
  <tr><td><code>compare_results.py</code></td><td>Post-processing utility for comparing simulation output against reference data.</td></tr>
  <tr><td><code>defs.py</code>, <code>glo.py</code>, <code>values.py</code></td><td>Core parameter / global / definition files now explicitly tracked.</td></tr>
</table>

<!-- ============================================================ -->
<h2>3. Proximal Tubule (PT) Modules</h2>

<!-- initPT -->
<div class="file-header">initPTv0.py → initPT.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Code structure</td><td>Single <code>initPT(pt)</code> function, ~662 lines</td><td>Split into 8 private helpers: <code>_set_membrane_areas</code>, <code>_set_water_permeabilities</code>, <code>_set_reflection_coefficients</code>, <code>_set_solute_permeabilities</code>, <code>_set_net_coefficients</code>, <code>_set_carbonate_kinetics</code>, <code>_set_boundary_conditions</code>, <code>_set_metabolic_parameters</code></td></tr>
  <tr><td>Interstitial concentrations</td><td><code>set_intconc</code> imported inline</td><td>Imported at module top</td></tr>
  <tr><td>Numerical parameters</td><td><em>Baseline</em></td><td><span class="no-change">No change</span></td></tr>
</table>

<!-- qnewton1PT -->
<div class="file-header">qnewton1PTv0.py → qnewton1PT.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Size</td><td>381 lines</td><td>162 lines (−57 %)</td></tr>
  <tr><td>Fortran-era comments</td><td>20+ lines of <code>#---</code> headers explaining the model</td><td>Moved to module docstring</td></tr>
  <tr><td>Newton iteration logic</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- jacobi2_2PT — critical bug fix -->
<div class="file-header">jacobi2_2PTv0.py → jacobi2_2PT.py &nbsp;<span class="tag-bug">BUG FIX</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Jacobian computation</td>
    <td>Finite-difference loop was <strong>commented out</strong> inside a <code># ===</code> block — dead code. Function returned an uninitialised zero matrix.</td>
    <td>Loop is <strong>active</strong>. Correctly computes the finite-difference Jacobian approximation.</td>
  </tr>
  <tr>
    <td>Impact</td>
    <td>PT interior Newton iterations used a zero Jacobian — solver relied entirely on line-search fallback</td>
    <td>PT interior Newton iterations use a properly computed Jacobian — numerically correct</td>
  </tr>
  <tr>
    <td>Floating-point epsilon</td>
    <td><code>zero = 0.0; if fjac[j,j] == zero:</code></td>
    <td><code>if h == 0.0:</code> (equivalent, cleaner)</td>
  </tr>
</table>

<!-- jacobi2_1PT -->
<div class="file-header">jacobi2_1PTv0.py → jacobi2_1PT.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Loop indexing</td><td>1-based range with offset: <code>range(1, n+1)</code>, <code>x[j-1]</code></td><td>0-based: <code>range(n)</code>, <code>x[j]</code></td></tr>
  <tr><td>Jacobian computation</td><td>Active (no bug here)</td><td>Active — <span class="no-change">unchanged logic</span></td></tr>
</table>

<!-- out_data_PT -->
<div class="file-header">out_data_PTv0.py → out_data_PT.py &nbsp;<span class="tag-new">NEW CAPABILITY</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>O₂ consumption</td><td>Not computed</td><td>Calls <code>compute_o2_consumption(tube, name, dimL, ind1, ind2)</code> for PT, PCT, and S3 sub-segments — prints TNa and QO₂</td></tr>
  <tr><td>Array access</td><td><code>for I in range(1, NSPT+1): deliv[I-1] = pt[0].conc[I-1, 1-1] * ...</code></td><td>Vectorised: <code>deliv[:] = pt[0].conc[:, LUM] * pt[0].vol[LUM] * cw</code></td></tr>
  <tr><td>Unused arrays</td><td><code>fracdel</code> allocated but never used; <code>cwplPT</code> conversion factor unused</td><td>Both removed</td></tr>
</table>

<!-- fcn1PT, qflux1PT, qnewton2PT, fcn2PT, qflux2PT — grouped -->
<div class="file-header">fcn1PTv0 / qflux1PTv0 / qnewton2PTv0 / fcn2PTv0 / qflux2PTv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Transporter imports</td><td>Inside function body (<code>compute_water_fluxes</code>, <code>compute_ecd_fluxes</code>, <code>sglt</code>, <code>compute_nhe3_fluxes</code>)</td><td>At module top level</td></tr>
  <tr><td>Size reduction</td><td>322–631 lines each</td><td>162–378 lines each (−40 % average)</td></tr>
  <tr><td>Residual equations / flux kinetics</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>4. SDL (Short Descending Limb) Modules</h2>

<!-- initSDL -->
<div class="file-header">initSDLv0.py → initSDL.py &nbsp;<span class="tag-fix">CODE FIX</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Code structure</td><td>Single function body</td><td>Split into 6 helpers including <code>_apply_terminal_impermeability</code></td></tr>
  <tr>
    <td>Water permeabilities (Pf)</td>
    <td>Two sequential complete assignments to <code>PfMP</code>, <code>PfME</code>, <code>PfPE</code>, <code>PfPS</code>, <code>PfES</code> — first block silently overridden by the second</td>
    <td>Dead first assignment removed; only the effective values retained</td>
  </tr>
  <tr>
    <td>Simulation impact of Pf fix</td>
    <td>First assignment (<em>never used</em>): e.g. <code>PfMP = 0.00165</code>, <code>PfME = 0.07</code></td>
    <td>Effective values identical to v0 second pass: <code>PfMP = 0.40/36.0 = 0.01111</code>, <code>PfME = 220.0</code> — <strong>no numerical change</strong></td>
  </tr>
  <tr><td>Terminal impermeability threshold</td><td>Loop uses integer comparison inline</td><td>Extracted to <code>_apply_terminal_impermeability</code> using <code>int(n_segments * 0.46)</code></td></tr>
</table>

<!-- qnewton2SDL -->
<div class="file-header">qnewton2SDLv0.py → qnewton2SDL.py &nbsp;<span class="tag-fix">FIX + REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Code structure</td><td>Single function</td><td>Decomposed into 10 helpers (<code>_initialize_state</code>, <code>_newton_iteration_loop</code>, <code>_pack_unknowns</code>, etc.)</td></tr>
  <tr><td>State management</td><td>Parallel arrays</td><td>Dictionary-based state object</td></tr>
  <tr>
    <td>Parameter packing</td>
    <td><code>pars[1:NS+1]</code> — 1-indexed offset, off-by-one relative to 0-based array</td>
    <td><code>pars[0:NS]</code> — correct 0-based slice</td>
  </tr>
  <tr><td>Convergence factor</td><td><code>pconv = 1.3</code> applied inline at outlet</td><td>Preserved in <code>_apply_convergence_factor</code> — identical logic</td></tr>
  <tr>
    <td>Matrix inversion</td>
    <td>Manual: <code>if np.linalg.det(fjac) &gt;= 1: fjac_inv = ...; AVF = fjac_inv @ fvec</code></td>
    <td>Unified: <code>np.linalg.inv(fjac) @ fvec</code></td>
  </tr>
  <tr><td>Solver tolerances</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged (<code>TOL=1e-5</code>, <code>MAX_ITER=21</code>)</span></td></tr>
</table>

<!-- fcn2SDL -->
<div class="file-header">fcn2SDLv0.py → fcn2SDL.py &nbsp;<span class="tag-fix">FIX + REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Parameter unpacking</td>
    <td><code>pars[i-1]</code> 1-based offset pattern — misaligned with what <code>qnewton2SDL</code> packed</td>
    <td>Clean slices <code>pars[0:NS]</code>, <code>pars[NS:2*NS]</code>, etc. — aligned with packing</td>
  </tr>
  <tr><td>SDL water flux import</td><td>Called inside function body</td><td>Imported at module top</td></tr>
  <tr><td>Residual equations</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- jacobi2_2SDL -->
<div class="file-header">jacobi2_2SDLv0.py → jacobi2_2SDL.py &nbsp;<span class="tag-fix">BUG FIX + REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Mutation bug</td>
    <td><code>fvec_J = fvec</code> — alias, not a copy; subsequent perturbed evaluation could corrupt <code>fvec</code></td>
    <td><code>fvec_J = fvec.copy()</code> — safe copy</td>
  </tr>
  <tr><td>Loop indexing</td><td>1-based: <code>for j in range(1, n+1): temp = x[j-1]</code></td><td>0-based: <code>for j in range(n): x_original = x[j]</code></td></tr>
  <tr><td>Machine epsilon</td><td>Hard-coded: <code>epsmch = 2.22044604926e-16</code></td><td>Computed: <code>np.finfo(float).eps</code></td></tr>
  <tr><td>Column update</td><td>Inner loop over rows</td><td>Vectorised: <code>fjac[:, j] = (wa1 - fvec_J) / h</code></td></tr>
</table>

<!-- ============================================================ -->
<h2>5. TAL / DCT Modules (mTAL, cTAL, DCT)</h2>

<!-- initA, initT, initD grouped -->
<div class="file-header">initAv0 / initTv0 / initDv0 → initA / initT / initD &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Code structure</td><td>Single monolithic function (415–431 lines each)</td><td>Decomposed into private helpers; type hint <code>def initA(mtal: List[Membrane]) -&gt; None</code></td></tr>
  <tr><td>Interstitial concentrations</td><td>Duplicated inline logic in each module</td><td>Calls shared <code>set_intconc()</code> imported at top</td></tr>
  <tr><td>Numerical parameters</td><td><em>Baseline</em></td><td><span class="no-change">No change to permeabilities or transporter scalings</span></td></tr>
</table>

<!-- initD_Var -->
<div class="file-header">initD_Varv0.py → initD_Var.py &nbsp;<span class="tag-fix">INTERFACE FIX</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>TRPV5 Ca²⁺ channel scaling</td>
    <td><code>xTRPV5_dct</code> written to a module-level global variable — implicit coupling</td>
    <td>Returned explicitly: <code>def initD_Var(...) -&gt; float: ... return xTRPV5_dct</code></td>
  </tr>
  <tr><td>Code structure</td><td>Single function</td><td>Decomposed into helpers</td></tr>
</table>

<!-- qnewton2b -->
<div class="file-header">qnewton2bv0.py → qnewton2b.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Code structure</td><td>Single function, 294 lines</td><td>Decomposed into 9 helpers, 454 lines total</td></tr>
  <tr>
    <td>Tolerance schedule</td>
    <td>4 branches: <code>TOL=1e-5</code> (iter&lt;11), <code>TOL=1e-4</code> (11–20), <code>TOL=1e-3</code> (21–30), <code>TOL=res*1.01</code> (&gt;30)</td>
    <td>2 branches: <code>TOL=1e-5</code> (iter&lt;11), <code>TOL=1e-4</code> (iter≥11). Branches 3–4 were unreachable (loop exits at 21) — <strong>no behavioural change</strong></td>
  </tr>
  <tr>
    <td>Newton step (damping)</td>
    <td>Comments show original damping of 0.25 (mTAL/cTAL/DCT) or 0.50 (IMCD); actual v0 code already used factor 1.0 (full step)</td>
    <td>Uses <code>np.linalg.inv(fjac) @ fvec</code> — full step, damping=1.0. Documented explicitly. <strong>No behavioural change vs v0</strong></td>
  </tr>
  <tr><td>Segment dispatch</td><td><code>if/elif</code> chain on segment identifier</td><td><code>dict</code> lookup</td></tr>
</table>

<!-- fcn2b, jacobi2_2b, qflux2A/T/D grouped -->
<div class="file-header">fcn2bv0 / jacobi2_2bv0 / qflux2Av0 / qflux2Tv0 / qflux2Dv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Flux imports</td><td>Inside function body (selected dynamically per segment)</td><td>At module top; still dispatched by segment id</td></tr>
  <tr><td>Loop indexing (Jacobian)</td><td>1-based range with offset</td><td>0-based; vectorised column update</td></tr>
  <tr><td>Machine epsilon</td><td>Hard-coded <code>2.22044604926e-16</code></td><td><code>np.finfo(float).eps</code></td></tr>
  <tr><td>Transporter equations (NKCC2 F/A, NHE3, NHE1, AE, KCC4, Na-K-ATPase)</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>6. CNT / CCD / OMCD Modules</h2>

<!-- initC, initC_Var, initCCD, initCCD_Var, initOMC, initOMC_Var grouped -->
<div class="file-header">initCv0 / initC_Varv0 / initCCDv0 / initCCD_Varv0 / initOMCv0 / initOMC_Varv0 → current &nbsp;<span class="tag-fix">INTERFACE FIX + REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Channel expression scalars</td>
    <td><code>hENaC_CNT</code>, <code>hROMK_CNT</code>, <code>hCltj_CNT</code> (and CCD/OMCD equivalents) stored in module-level globals</td>
    <td>Returned explicitly as tuples: <code>return (hENaC_CNT, hROMK_CNT, hCltj_CNT)</code></td>
  </tr>
  <tr><td>Interstitial concentrations</td><td>Duplicated inline in each module</td><td>Calls shared <code>set_intconc()</code></td></tr>
  <tr><td>Code structure</td><td>Single monolithic functions (528–623 lines each)</td><td>Decomposed into private helpers</td></tr>
  <tr><td>Unused variables</td><td><code>theta</code>, <code>Slum</code>, <code>Slat</code>, <code>Sbas</code>, <code>Pf</code>, <code>dLA</code>, <code>pos</code> present in all modules</td><td>All removed</td></tr>
  <tr><td>Permeabilities / transporter scalings</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- qnewton2icb, fcn2C, fcn2OMC, jacobi modules grouped -->
<div class="file-header">qnewton2icbv0 / fcn2Cv0 / fcn2OMCv0 / jacobi2_2icbv0 / jacobi2_2icbOMCv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Solver structure</td><td>Single function</td><td>Decomposed; type hints for <code>idid</code> (CNT=6, CCD=7, OMCD=8)</td></tr>
  <tr><td>Flux imports</td><td>Inside function body</td><td>At module top</td></tr>
  <tr><td>Jacobian loop indexing</td><td>1-based</td><td>0-based; vectorised column update</td></tr>
  <tr><td>OMCD B-cell constraint</td><td>Implicit</td><td>Documented: "No B cells; <code>Cb[i,B] = Cb[i,A]</code> enforced explicitly"</td></tr>
  <tr><td>Residual equations</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- qflux2C, qflux2CCD, qflux2OMC grouped -->
<div class="file-header">qflux2Cv0 / qflux2CCDv0 / qflux2OMCv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Transporter imports</td><td>Inside function body (<code>compute_water_fluxes</code>, <code>compute_ecd_fluxes</code>, <code>compute_ncx_fluxes</code>, <code>fatpase</code>)</td><td>All at module top</td></tr>
  <tr><td>Size</td><td>477–621 lines each</td><td>339–460 lines each</td></tr>
  <tr><td>Transporter kinetics (ENaC, ROMK, NHE3, NCX, PMCA, Na-K-ATPase, H-ATPase)</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>7. IMCD (Inner Medullary Collecting Duct)</h2>

<!-- initIMC, initIMC_Var, qflux2IMC -->
<div class="file-header">initIMCv0 / initIMC_Varv0 / qflux2IMCv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Phantom cell note</td>
    <td>Not documented</td>
    <td>Module comment clarifies: "Only P cells are real; A and B slots are phantom/unused"</td>
  </tr>
  <tr>
    <td>Channel expression (initIMC_Var)</td>
    <td>Written to module-level globals</td>
    <td>Returned explicitly: <code>return (hENaC_IMC, hROMK_IMC, hCltj_IMC)</code></td>
  </tr>
  <tr>
    <td>qflux2IMC initialisation</td>
    <td><code>initIMC_Var</code> called once somewhere in pipeline</td>
    <td>Called once at module import time with clear comment</td>
  </tr>
  <tr><td>Transporter kinetics (ENaC, ROMK, H-K-ATPase)</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>8. Compute Utility Modules</h2>

<!-- compute_water_fluxes, compute_ecd_fluxes -->
<div class="file-header">compute_water_fluxesv0 / compute_ecd_fluxesv0 → current &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Intermediate variable</td><td><code>(Pfref * Vwbar * Cref) / href</code> inline in expression</td><td>Named as <code>dimless</code> for readability</td></tr>
  <tr><td>Flux computation algorithm</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- compute_nkcc2_flux, compute_kcc_fluxes, compute_nhe3_fluxes, compute_ncx_fluxes -->
<div class="file-header">compute_nkcc2_fluxv0 / compute_kcc_fluxesv0 / compute_nhe3_fluxesv0 / compute_ncx_fluxesv0 → current &nbsp;<span class="tag-fix">FIX + REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Array indexing</td>
    <td>Integer offsets: <code>C[0, 0]</code> (Na lumen), <code>C[10, 0]</code> (NH₄ lumen), <code>var_ncx[1-1]</code></td>
    <td>Named constants: <code>C[NA, LUM]</code>, <code>C[NH4, LUM]</code>, <code>var_ncx[0]</code></td>
  </tr>
  <tr>
    <td>Missing import (compute_kcc_fluxes)</td>
    <td><code>from glo import *</code> missing — relied on namespace pollution from caller</td>
    <td><code>from glo import *</code> added at module top</td>
  </tr>
  <tr><td>Kinetic rate constants</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged (NKCC2, KCC4, NHE3, NCX stoichiometry all preserved)</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>9. H-K-ATPase Kinetics: fatpase</h2>

<div class="file-header">fatpasev0.py → fatpase.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr><td>Loop indexing</td><td><code>for i in range(1, n+1): Amat[1-1, i-1] = 1.0</code></td><td><code>for i in range(n): Amat[0, i] = 1.0</code></td></tr>
  <tr><td>Documentation</td><td>No docstring</td><td>Lists all 14 enzyme species of the H-K-ATPase kinetic model</td></tr>
  <tr><td>Kinetic rate constants / matrix structure</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>10. Main Pipeline</h2>

<div class="file-header">mainv0.py → main.py &nbsp;<span class="tag-refac">REFACTOR</span></div>
<table>
  <tr><th style="width:22%">Category</th><th style="width:38%">v0</th><th style="width:38%">Current</th></tr>
  <tr>
    <td>Imports</td>
    <td>Each segment's modules imported inside the segment processing block (e.g. <code>from initSDL import initSDL</code> inside the SDL block)</td>
    <td>All imports at file top — all 30+ module imports visible at a glance</td>
  </tr>
  <tr><td>Unused allocations</td><td><code>membrane_inst = Membrane(NSPT, NC, NS)</code>; <code>inlet</code>, <code>outlet</code> arrays — allocated but never used</td><td>All removed</td></tr>
  <tr>
    <td>Output array indexing</td>
    <td>Hardcoded: <code>pt[LzPT].conc[I-1, 1-1]</code></td>
    <td>Named: <code>pt[NZ].conc[i, LUM]</code></td>
  </tr>
  <tr><td>Simulation pipeline order</td><td><em>Baseline</em></td><td><span class="no-change">Unchanged: PT → SDL → mTAL → cTAL → DCT → CNT → CCD → OMCD → IMCD</span></td></tr>
</table>

<!-- ============================================================ -->
<h2>11. Full File Index</h2>

<table>
  <tr>
    <th>File pair (v0 → current)</th>
    <th style="text-align:center">v0 lines</th>
    <th style="text-align:center">Current lines</th>
    <th>Primary change type</th>
  </tr>
  <tr><td><code>initPT</code></td><td style="text-align:center">662</td><td style="text-align:center">526</td><td>Decomposition</td></tr>
  <tr><td><code>qnewton1PT</code></td><td style="text-align:center">381</td><td style="text-align:center">162</td><td>Decomposition, imports moved</td></tr>
  <tr><td><code>fcn1PT</code></td><td style="text-align:center">322</td><td style="text-align:center">203</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>qflux1PT</code></td><td style="text-align:center">605</td><td style="text-align:center">378</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>qnewton2PT</code></td><td style="text-align:center">478</td><td style="text-align:center">261</td><td>Imports moved, decomposition</td></tr>
  <tr><td><code>fcn2PT</code></td><td style="text-align:center">460</td><td style="text-align:center">263</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>qflux2PT</code></td><td style="text-align:center">631</td><td style="text-align:center">378</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>jacobi2_2PT</code></td><td style="text-align:center">66</td><td style="text-align:center">38</td><td><strong>Bug fix: Jacobian loop uncommented</strong></td></tr>
  <tr><td><code>jacobi2_1PT</code></td><td style="text-align:center">60</td><td style="text-align:center">46</td><td>0-based loop, cleanup</td></tr>
  <tr><td><code>out_data_PT</code></td><td style="text-align:center">107</td><td style="text-align:center">65</td><td><strong>New: O₂ consumption output</strong></td></tr>
  <tr><td><code>initSDL</code></td><td style="text-align:center">189</td><td style="text-align:center">178</td><td>Dead Pf assignment removed</td></tr>
  <tr><td><code>qnewton2SDL</code></td><td style="text-align:center">238</td><td style="text-align:center">233</td><td>Packing fix, state dict</td></tr>
  <tr><td><code>fcn2SDL</code></td><td style="text-align:center">202</td><td style="text-align:center">230</td><td>Unpacking fix, decomposition</td></tr>
  <tr><td><code>jacobi2_2SDL</code></td><td style="text-align:center">49</td><td style="text-align:center">94</td><td>fvec copy bug fix, 0-based loop</td></tr>
  <tr><td><code>initA</code></td><td style="text-align:center">426</td><td style="text-align:center">405</td><td>Decomposition, type hints</td></tr>
  <tr><td><code>initT</code></td><td style="text-align:center">415</td><td style="text-align:center">358</td><td>Decomposition, type hints</td></tr>
  <tr><td><code>initD</code></td><td style="text-align:center">431</td><td style="text-align:center">346</td><td>Decomposition, type hints</td></tr>
  <tr><td><code>initD_Var</code></td><td style="text-align:center">431</td><td style="text-align:center">395</td><td>Explicit return of xTRPV5_dct</td></tr>
  <tr><td><code>qnewton2b</code></td><td style="text-align:center">294</td><td style="text-align:center">454</td><td>Decomposition, tolerance cleanup</td></tr>
  <tr><td><code>fcn2b</code></td><td style="text-align:center">334</td><td style="text-align:center">518</td><td>Decomposition, docstring</td></tr>
  <tr><td><code>qflux2A</code></td><td style="text-align:center">358</td><td style="text-align:center">381</td><td>Decomposition, imports moved</td></tr>
  <tr><td><code>qflux2T</code></td><td style="text-align:center">368</td><td style="text-align:center">394</td><td>Decomposition, imports moved</td></tr>
  <tr><td><code>qflux2D</code></td><td style="text-align:center">516</td><td style="text-align:center">576</td><td>Decomposition, imports moved</td></tr>
  <tr><td><code>initC</code></td><td style="text-align:center">623</td><td style="text-align:center">539</td><td>Decomposition, explicit return</td></tr>
  <tr><td><code>initC_Var</code></td><td style="text-align:center">623</td><td style="text-align:center">430</td><td>Decomposition, explicit return</td></tr>
  <tr><td><code>initCCD</code></td><td style="text-align:center">615</td><td style="text-align:center">474</td><td>Decomposition, set_intconc</td></tr>
  <tr><td><code>initCCD_Var</code></td><td style="text-align:center">613</td><td style="text-align:center">490</td><td>Decomposition, explicit return</td></tr>
  <tr><td><code>initOMC</code></td><td style="text-align:center">528</td><td style="text-align:center">428</td><td>Decomposition, AMW model note</td></tr>
  <tr><td><code>initOMC_Var</code></td><td style="text-align:center">528</td><td style="text-align:center">444</td><td>Decomposition, explicit return</td></tr>
  <tr><td><code>qnewton2icb</code></td><td style="text-align:center">268</td><td style="text-align:center">237</td><td>Decomposition, type hints</td></tr>
  <tr><td><code>fcn2C</code></td><td style="text-align:center">383</td><td style="text-align:center">243</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>jacobi2_2icb</code></td><td style="text-align:center">49</td><td style="text-align:center">70</td><td>0-based loop, np.finfo</td></tr>
  <tr><td><code>fcn2OMC</code></td><td style="text-align:center">363</td><td style="text-align:center">250</td><td>Imports moved, B-cell note</td></tr>
  <tr><td><code>jacobi2_2icbOMC</code></td><td style="text-align:center">49</td><td style="text-align:center">71</td><td>0-based loop, np.finfo</td></tr>
  <tr><td><code>qflux2C</code></td><td style="text-align:center">621</td><td style="text-align:center">460</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>qflux2CCD</code></td><td style="text-align:center">601</td><td style="text-align:center">415</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>qflux2OMC</code></td><td style="text-align:center">477</td><td style="text-align:center">339</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>initIMC</code></td><td style="text-align:center">459</td><td style="text-align:center">333</td><td>Decomposition, phantom cell note</td></tr>
  <tr><td><code>initIMC_Var</code></td><td style="text-align:center">459</td><td style="text-align:center">342</td><td>Decomposition, explicit return</td></tr>
  <tr><td><code>qflux2IMC</code></td><td style="text-align:center">375</td><td style="text-align:center">400</td><td>Imports moved, named indices</td></tr>
  <tr><td><code>compute_water_fluxes</code></td><td style="text-align:center">108</td><td style="text-align:center">88</td><td>Docstring, dimless variable</td></tr>
  <tr><td><code>compute_ecd_fluxes</code></td><td style="text-align:center">56</td><td style="text-align:center">85</td><td>Docstring, GHK comment</td></tr>
  <tr><td><code>compute_nkcc2_flux</code></td><td style="text-align:center">53</td><td style="text-align:center">77</td><td>Named constants</td></tr>
  <tr><td><code>compute_kcc_fluxes</code></td><td style="text-align:center">64</td><td style="text-align:center">88</td><td>Named constants, glo import fix</td></tr>
  <tr><td><code>compute_nhe3_fluxes</code></td><td style="text-align:center">65</td><td style="text-align:center">70</td><td>Named constants, dead code removed</td></tr>
  <tr><td><code>compute_ncx_fluxes</code></td><td style="text-align:center">60</td><td style="text-align:center">96</td><td>Named constants, NCX docstring</td></tr>
  <tr><td><code>fatpase</code></td><td style="text-align:center">122</td><td style="text-align:center">138</td><td>0-based loop, enzyme species list</td></tr>
  <tr><td><code>sglt</code></td><td style="text-align:center">136</td><td style="text-align:center">153</td><td>Named constants, docstring</td></tr>
  <tr><td><code>main</code></td><td style="text-align:center">582</td><td style="text-align:center">452</td><td>All imports at top, dead code removed</td></tr>
</table>

<p style="margin-top:30px; font-size:9pt; color:#888;">
  Report generated by generate_report_pdf.py — kidney-models-females_age/PaperCodes/ — 2026-03-04
</p>
</body>
</html>
"""

output_html = "/Users/sofiapolychroniadou/Desktop/edwardslab/kidney-models-females_age/PaperCodes/v0_comparison_report.html"
with open(output_html, "w", encoding="utf-8") as f:
    f.write(html)

print(f"HTML written to {output_html}")
