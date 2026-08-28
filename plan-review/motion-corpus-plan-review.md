# Adversarial validation — "Generalized multimodal motion corpus and latent-action experiment"

Review of the implementation brief against `roahmlab/feasible-motion-decoder`
@ `0b68fb4` (read in full: `dfed/*`, `scripts/*`, `gates/*`, `docs/*`,
committed `results/*.json`). Numeric claims below are anchored to file:line or
to a reproducible probe (`probe_family2_geometry.py`, alongside this file).

**Verdict.** The plan is architecturally right — it reuses the repo's actual
seams (inverse design, fixed tube executor, differentiable layer, independent
judge) and most of its thresholds are the repo's own registered constants. But
it is **not implementable as written**: the `sequential_two_barrier` family is
geometrically near-infeasible under the repo's registered keepout at the
specced travel/deviation ranges (measured yield ≈ 0.1–0.2% under assumptions
*optimistic* by ~6×), and four spec-level defects (mode/style identifiability,
graze-vs-style interaction, family-mix enforcement, split hashing) would
silently corrupt the corpus even where generation succeeds. All are fixable
with parameter and spec changes; none require touching F_eps or the keepout.

---

## 1. What the plan gets right (verified against the code)

- The module map is real: `dfed.inverse` (make_pair, grow_barrier,
  sweep_points), `dfed.tube` (demo_batch, tube_batch, solve), `dfed.operator`,
  `dfed.judge`, `dfed.robot` all exist with the interfaces the plan assumes.
  `N=20, nq=7, T=8.0 s` (provenance blocks in `results/*.json`), so
  trajectories are `[21, 14]` — the schema shapes are consistent.
- Motions-first, obstacles-grown-after is the repo's hard-won law
  (`docs/FINDINGS.md` #1: three place-then-hope families had **zero**
  solutions). The plan preserves it. Good.
- Thresholds mostly match registered constants: terminal EE ≤ 0.010
  (`Tolerances.ee_tol_m`), clearance ≥ 0.028 (`GrazeFamily.demo_clr_floor`),
  graze band 3–5.5 cm (`place_lo/place_hi`), tracking ≤ 25 mm
  (`a1_track_med_mm`), solver feasibility = `esc_feas ≤ 0` (never
  `converged` — FINDINGS #3).
- Per-attempt `SeedSequence([global_seed, attempt_id])` fixes a documented
  real defect: `data/README.md` admits the current pool sampler is
  "host-order-sensitive" and regeneration is not byte-identical.
- The Part B/C architecture is grounded: `ArmV3` already has the relational
  per-node conditioning, d_z=32, VAE encoder, and residual-on-naive decoder;
  `tube_layer.tube_solve` is FD-verified (cos 0.9999); the curriculum gate
  and row masking exist in `train_arms.py`. Encoding demos at the posterior
  mean and freezing E/D/F is a clean extension, and the ≥32-samples,
  judge-verified, matched-budget baseline protocol matches the repo's
  discipline.
- Atomic shards, resumability, rejection-reason counters, manifest with
  provenance: all consistent with how this repo already records evidence.

---

## 2. FATAL — `sequential_two_barrier` as specced has ~zero yield

**Claim under test** (plan §9): at t1≈1/3 and t2≈2/3 of a 0.30–0.75 m EE
travel (plan §5), deviations of 0.25–0.45 m per stage, grow two barriers on
the naive path clearing all eight protected route sweeps under the registered
`KEEPOUT = 0.165 m` (`dfed/inverse.py:40`), which the plan (correctly)
forbids lowering.

**Evidence 1 — the repo's own registered sweep** (`dfed/inverse.py:33-39`):
one barrier between **two** routes needs 45–70 cm bulges for 12% fit at
margin 0.06; 30–50 cm bulges give **2.7%**. The plan specs 25–45 cm bulges,
below the range the repo measured as workable, and needs **two** barriers
against **eight** sweeps.

**Evidence 2 — direct probe.** I ported `grow_barrier` verbatim and ran an
EE-space Monte-Carlo of the specced construction
(`probe_family2_geometry.py`; EE polylines only, deviations orthogonal to
travel — both *optimistic*, since the real builder protects whole-arm capsule
sweeps and FINDINGS #4 measured the arm-to-arm gap at ~55% of the hand-path
gap). Calibration: the same proxy on the existing family gives 71% where the
real builder measures 12–13% — i.e. the proxy overestimates by ~5.7×.

| configuration | EE-proxy yield | dominant failure |
|---|---|---|
| family 1 (current, calibration) | 71.0% | seed keepout 29% |
| **family 2 as specced** | **0.15%** | barrier-1 seed keepout **98.2%** |
| family 2, no style variants | 0.75% | seed keepout 93.7% |
| dev → 0.45–0.70 (repo scale) | 2.05% | seed 77%, barrier overlap 20% |
| travel → 0.60–1.00 | 3.00% | seed keepout 93% |
| dev 0.45–0.70 **and** travel 0.60–1.00 | 18.45% | seed 49%, overlap 28% |

Applying the ≥5.7× calibration, the as-specced real yield is ≲ 0.03% —
roughly ≥10 M attempts for the 3,000 contexts the 30% mix implies, and the
few survivors would be an extreme geometric tail (the "uncharacterised
selection effect" FINDINGS #4 already flags at 12% acceptance, squared).

**Evidence 3 — the mechanism is structural, not bad luck.** The chord from
q0 to the stage-1 knot (lateral offset d at longitudinal distance ℓ = L/3)
passes within `d·ℓ/√(ℓ²+d²)` of the barrier-1 seed. With L ≤ 0.75 → ℓ ≤ 0.25
and d ≤ 0.45, that is ≤ 0.145–0.22 m against a 0.165 m keepout — **the
family's own same-sign routes kill the seed** (measured: same-sign routes
alone kill 70% of seeds; crossing routes 72%; median clearance 0.14 m).
Putting the decision at 1/3 of a ≤0.75 m path leaves too little longitudinal
lever arm for any deviation magnitude to open a 0.165 m corridor. No amount
of retry or seed search fixes an inequality.

**What to change** (keepout untouched):

1. Family-specific ranges: travel 0.60–0.95 m and per-stage deviations
   0.45–0.70 m for this family (probe: 18.45% EE-proxy → ~3% expected real),
   with t1/t2 at ~0.30/0.70. §5's global 0.30–0.75 travel must become a
   per-family parameter. Interactions to re-measure: the 1.2 rad joint-move
   cap and the 0.80 speed fraction at the fixed T=8 s / N=20 horizon —
   FINDINGS #2 shows speed-limit violations are exactly how this family
   design dies at training time.
2. Mutual barrier keepout **during growth** (overlap is the #2 failure at
   workable scales, 20–28%): include barrier 1's faces in barrier 2's
   keep-clear with a small box-to-box margin, instead of grow-then-reject.
3. Alternatively redesign the mechanism: (a) both decisions at ONE stage —
   four openings in a single grown wall complex at the midpoint (full L/2
   lever arm, one keepout corridor); or (b) hierarchical — barrier 1 at
   midpoint as today, decision 2 realized by an obstacle placed in the
   post-crossing half where routes have already separated (blocking the two
   greedy continuations, not the naive path).
4. Whatever is chosen: add a **registered family-design gate** — a whole-arm
   construction-yield probe (pure CPU, `dfed.robot` + `sweep_points`, ~5k
   attempts) with a pre-registered floor (e.g. ≥ 2%) that must pass BEFORE
   the corpus machinery is built. That is this repo's own discipline; the
   plan currently discovers this failure only when `--contexts 10000` never
   terminates. Also give the builder per-family attempt caps that abort with
   the rejection histogram instead of spinning.

---

## 3. Major spec defects (silent corpus corruption)

### 3.1 Mode/style identifiability is not enforced — the bands overlap

§16 allows style siblings up to **0.10 m** apart while requiring mode pairs
only ≥ **0.08 m** (two-barrier stages) or ≥ **0.12 m** (single barrier).
A "style pair" may legitimately be farther apart than a "mode pair" — the
labels the whole experiment ranks on (coverage, mode entropy) then mean
nothing at the boundary. Worse at the bottom: the style floor of **0.010 m
is below the trained decoder's own reconstruction error** (9.2 mm median /
16.1 mm p90 on an easier 2-motion family, `results/arms/eval_implicit_*`),
so styles near the floor are unresolvable by the model the corpus is for.

Fix: scale-separate the bands and register them — e.g. styles in
[0.03, 0.08] m, modes ≥ 0.20 m in their decision window (construction
geometry gives ≥ 0.5 m; 0.08/0.12 is needlessly loose), and sample style
scales antithetically (s1 ∈ U(0.85,0.95), s2 ∈ U(1.05,1.15)) so the style
floor holds by construction instead of by rejection. Also pin the node
window (mid-third, as `MID` in the repo) — §16's "mean EE distance" doesn't
say over which nodes, and lead-in/out dilution changes the number.

### 3.2 Graze placement ignores the same-mode style sibling

§14 requires clearance from the *other modes* (`other_route_clear` exists,
`build_family_v2.py:69`) but says nothing about the **other style of the
same mode**, which may sit up to 10 cm from the representative route the box
is placed 3–5.5 cm from — i.e. potentially **through** the graze box. The
current code never faced this (one motion per route). Consequence: the §14.8
re-solve either rejects the context or the solver shoves the sibling
around the box, breaking the style-distance band or the 0.028 clearance
floor — systematically, on ~half of graze contexts. Fix: place graze boxes
against the union of that mode's style sweeps (clearance band satisfied for
*every* style of the own mode), and state the fallback when placement or
re-verification fails (drop the box and keep the pre-graze context, recording
`graze_attempted/failed` — noting the residual bias: graze prevalence then
correlates with easy geometry, which matters for §17's stratified use).

### 3.3 Family 0 has no mode-separation gate at all

§16 gates separations for families 1 and 2 only. Family 0 has no barrier
enforcing its two modes and no registered post-solve separation floor —
nothing stops the two "modes" from being 2 cm apart after the demo solve, or
a style variant of mode 0 drifting into mode 1's half-space. The demo solve
preserves routes only through the tracking weight. Fix: register the same
mid-third mode floor (≥ 0.20 m as above, or at minimum the 0.12 m used for
family 1) for family 0, and verify each demo's realized mode with the
measured mode axis (as `phase_assemble` does today with `label_mode`) rather
than trusting construction intent.

### 3.4 Split leakage through the shared configuration pool

Splitting by context hash is necessary but not sufficient: all contexts draw
(q0, qg) from ONE `config_pool`. With ~10k accepted contexts over a 60k pool,
~800+ context pairs are expected to share the exact same start row (birthday
bound), some spanning train/test — near-duplicate contexts differing only in
clutter. And §23's downstream requirement ("scene IDs disjoint from corpus
pretraining") is vacuous across different generators — worse, if the
downstream family is regenerated with the documented `--seed 0`, it draws
from the same pool stream and can share exact endpoint pairs with corpus
contexts. Fix: (a) record pool indices per context; forbid test contexts
from sharing (i0, ig) — or the exact q0 row — with any train context;
(b) replace §23's scene-ID clause with a concrete rule: downstream scenes
regenerated from a reserved seed range + an explicit near-duplicate audit of
(X0, p_goal) against the pretraining corpus.

### 3.5 Family mix, quotas, and determinism of *which* contexts exist

The 20/50/30 mix is specced at sampling time, but acceptance rates differ
per family by orders of magnitude, so the accepted mix will drift wherever
the builder stops at "10,000 contexts" — and *which* contexts make the
cutoff depends on worker completion order unless the plan says otherwise.
Fix: per-family attempt streams (seed with `[global_seed, family_id, k]`),
per-family accepted quotas (2,000/5,000/3,000) as **required** final
conditions in §21 (currently the report only *counts* by family — a builder
that quietly ships 0 family-2 contexts satisfies every stated done
condition), and a selection rule of "lowest attempt_id wins" so the corpus
is a pure function of the seed, not of scheduling.

### 3.6 The split hash is an implementation trap

§18 says "hash of context_id" — Python's builtin `hash()` is salted per
process; using it silently reshuffles splits every run (the costliest
possible leak: undetectable). Pin a stable keyed hash (e.g.
`sha256(global_seed || context_id) mod 100`, thresholds 90/95) and set
`context_id := attempt_id` so split membership is decided at attempt time,
independent of acceptance history. Add the §19 test asserting split
assignment survives process restart.

---

## 4. Architecture and determinism risks

### 4.1 The worker model fights the GPU's economics

The plan's per-context framing (28 workers, "batch all motions from a
context where practical") collides with measured solver economics: batched
solves cost ~22–30 ms/row at chunk 1008 (`REPRODUCE.md`: 24k rows in ~9 min)
but ~300 ms/row at batch 16 (`results/profile_step.json`) — per-context
launches make the GPU phase 10–15× dearer, and 28 processes cannot share one
CUDA solver anyway. The repo already contains the right pattern:
`build_dataset.py`'s staged phases. Spec the builder as deterministic
**waves** keyed by attempt-id ranges: construct (CPU-parallel) → demo solve
(one GPU stream, large batches) → screen → judge (process pool) → graze →
re-solve → re-judge → shard. This also resolves determinism: the quick gate
proves byte-determinism only for an **identical batch** solved twice
(`gates/quick_gate.py:81-86`); if resume changes batch composition, solver
outputs can drift at float level — and graze placement branches on solver
output (the pull side), so a float-level drift becomes a *different scene*.
Deterministic wave composition (derived from attempt ids, not completion
order) is therefore load-bearing, and the §20 "deterministic regeneration"
check must pin it. Judge determinism has the same caveat: a `killed_worker`
row (crash-tolerant path in `dfed/judge.py`) is an honest failure but a
nondeterministic one; the regeneration check must re-run rather than
compare such rows.

### 4.2 Budget the judge; order the checks

Measured: judge 0.73 s/row wall at 28 workers (`build_dataset.py:7`), ~60×
a batched solve (`dfed/eval.py:91`). Strict all-motion acceptance means a
context rejected by its last-judged row wastes 4–8 judge rows. At ~10k
accepted + rejects, expect ~70–120k judged rows ≈ **14–24 h** — the binding
accepted cost. Keep it near the floor: screen in cost order (speed/clearance
screens → `esc_feas` → separations → judge last), judge a context only when
everything cheaper passed, and don't spin a judge pool per context (worker
init imports pinocchio+trimesh; amortize pools across waves). Note the
plan's own numbers make 8-row graze contexts expensive to certify: v2
measured ~0.90–0.93 per-row pass on clearance+feasibility after graze
re-solve (82% for 2-row scenes) → ~0.5–0.6 all-pass at 8 rows; that is
tolerable, but only if rejects happen before the judge where possible.

### 4.3 Box-count ceilings conflict

Family 2 worst case = 2 barriers + 10 clutter + 4 graze = **16 boxes**;
the existing dataset cap is `MAX_BOXES = 14` (`build_dataset.py:34`), and —
more important — the repo's measured healthy-gradient regime is **≤ 10 total
obstacles** ("75% of rows return a learning signal", `dfed/scenes.py:229`).
The plan's §13 clutter ranges push most family-2 (and some family-1)
contexts past the regime the training signal was validated in. Fix: set
Mmax = 16 explicitly in the schema, but cap *total* boxes ≈ 10–12 (family-2
clutter 3–6, not 4–10), or pre-register a gradient-health measurement at the
new box counts before committing 60k motions to it.

### 4.4 Provenance gaps the manifest must close

`solver_provenance()` exists, but the judge's repo (`CPU_HEAD_ROOT`,
`dfed/config.py:21`) is an **unpinned path** — no SHA is recorded anywhere.
For a 10k-context corpus whose acceptance is defined by that code, the
manifest must pin the CPU-head commit (and the dfed commit, per §3's
"generator commit"). Also: per-worker manifest appends race — write
per-shard sidecar JSONs and let `merge`/`summarize` build the manifest;
record pool seed/size; record per-shard host+GPU (cross-machine resume
breaks byte-determinism and should at least be visible).

### 4.5 The demo weight is currently ambiguous in the repo

§15 says demo weights are "fixed across the corpus, read from registered
configuration" — but today they are script-local literals that *disagree*:
`train_arms.py:34` uses w_q=30 ("v2 gate-A1 selection"); `build_family_v2.py`
uses w_q=100 ("v1 record"). The plan must decide which value defines the
corpus (recommend 30, matching the downstream demos it will be compared to
in Part C) and register it in `dfed/config.py` — otherwise the corpus and
the downstream family fork silently on tracking tightness.

---

## 5. Compute budget (measured anchors, RTX A6000 + 28 cores)

| item | basis | estimate |
|---|---|---|
| construction, fam 0+1 (~7k ctx) | 18 ms/attempt, 12%± | < 1 h |
| construction, fam 2 **as specced** | ≲0.03% yield | ≥10 M attempts: **days–weeks, pathological tail** |
| construction, fam 2 re-specced | ~3% yield | ~100k attempts ≈ 3 h on 28 cores |
| GPU solves (demos+pulls+resolves+§17) | 22–30 ms/row batched | ~150–250k rows ≈ 1.5–2.5 h (10–15× if per-context batches) |
| judge | 0.73 s/row | ~70–120k rows ≈ **14–24 h** (binding) |
| Part B implicit training | 10 h @ 6.6k rows / 120 ep | ~8× rows → tens of hours–days per arm; plan is silent, should budget/schedule |
| Part C eval per model | 0.73 s/row | ~500 scenes × 32 × 2 models ≈ 7–9 h |

Feasible end-to-end in ~2–3 days *only* with §2 fixed and the wave
architecture of §4.1; unbounded otherwise.

---

## 6. Minor findings

1. **Speed headroom is ~zero at the corners.** Route cap 0.80 (§5) ×
   dev-scale 1.15 × time-warp 1.10 ≈ 0.92 > the 0.90 acceptance bar (§16) —
   style variants near the warp corner get rejected only after full
   solve+judge. Enforce the 0.80 cap on the *post-style* route (§10's helper
   says "rejects speed-limit violations" without naming the bar — name it).
   Note also the deployment solver runs bounds-mode "tau" (v-limits widened
   100×, `dfed/operator.py:55`), so demo speed is only ever enforced by this
   acceptance bar — it is load-bearing, not redundant.
2. **"Clearance" needs a definition.** The 0.028 floor in the repo is the
   capsule-screen metric (`rob.traj_min_dist`); the judge separately reports
   mesh `min_clearance`. Pin `clearance_m` to the capsule screen (matching
   `demo_clr_floor` semantics) and store the judge's mesh value in metadata.
3. **Graze windows are family-relative.** `place_box` ranks pull over the
   mid-third bulge (`MID`); family-2 decisions live near t1/t2 — the
   placement window must be per-stage, or graze boxes cluster mid-path.
4. **Joint-limit margin** (§5, ≥ 0.10 rad "where possible"): 4 of the Gen3's
   7 joints are continuous (clamped to ±π in `dfed/robot.py:27-28`) — the
   margin is meaningful only for joints 2/4/6; say so, and drop "where
   possible" (untestable).
5. **Schema omissions:** store per-context `mode_axis`/`mode_origin` (or
   per-stage equivalents) as first-class arrays — §26's coverage metrics and
   3.3's label verification need them, and today's `label_mode` consumes
   them; add a global `context_id` per motion row (the local shard index
   alone invites merge bugs); `judge_ok` (always-true) is fine as an
   invariant column.
6. **Unit tests (§19) and imports:** route construction currently reaches
   into the solver package for v_max/URDF (`inverse.py:133`,
   `robot.py:15`); imports are CPU-safe (the CUDA build is lazy,
   INSTALL.md), but keep `corpus_routes` consumable with injected
   (lo, hi, v_max, capsule) data so the pure-geometry tests don't require a
   solver checkout at all.
7. **§20 gate:** cap attempts per family and fail with the rejection
   histogram; "deterministic regeneration" should assert byte-identical
   construction artifacts per attempt and byte-identical solves for
   identical wave composition (the quick-gate stage-2 precedent), tolerating
   judge `killed_worker` rows only via re-run.
8. **§22:** the reconstruction distance d is unspecified — inherit the
   registered refine-2 form (mid-third hand-path via `torchfk` + small
   full-state anchor, `ArmsConfig.ee_loss_*`) rather than inventing a new
   one; note `ARMS.d_z=16` belongs to the v1 `Arm`, ArmV3 is already 32.
9. **§25 honesty note:** at N=20 the "full trajectory" is 280-dim vs a
   32-dim latent — the "lower-dimensional and faster diffusion" advantage
   (§26) is weak at this horizon; expect the case to rest on coverage /
   judge-verified validity, and say so up front so a null speed result
   isn't spun.
10. **Naive-blocked (§8):** today `build_scene` records `naive_blocked` but
    does not reject on it; the plan makes it a hard condition — fine
    (grow_barrier's reach check makes it ~always true), just note the
    small yield cost and keep G0 unchanged.

## 7. What this review could not verify

No GPU/solver/judge in this environment, so nothing here re-ran the real
pipeline. The §2 probe is an EE-space proxy calibrated against the repo's
committed whole-arm measurements (12–13% real vs 71% proxy on family 1);
its *direction* (optimistic for the plan) is certain, its correction factor
(≈6× on family 1) is not guaranteed to transfer exactly to eight-route
scenes — which is precisely why §2's recommendation is a registered
whole-arm yield gate before corpus code, not a specific yield number.
