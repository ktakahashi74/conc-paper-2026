# Rebuttal to Reviewer 2

We thank the reviewer for the careful reading and constructive feedback. Below we address each weakness point by point.

---

## W1. Circularity of evaluation metrics

> The evaluation metrics (harmonicity, roughness, consonance) are derived from the same psychoacoustic model that defines the fitness landscape, raising circularity concerns.

We agree that this is a legitimate methodological concern and have addressed it through three complementary strategies in the revised manuscript.

**Terrain controls (Discussion §Terrain Validity; Supplementary S7).** We introduced matched terrain controls that operate on terrain-agnostic metrics—unique pitch bins, nearest-neighbour distance, and interval entropy—none of which reference the consonance model. The critical result comes from the shuffled-landscape control: a fixed random permutation of the consonance-score bins preserves the marginal score distribution identically while destroying spatial coherence. Under this condition, pitch spread remains intact (19.2 vs. 19.5 unique bins, *p* = 0.54), but interval entropy rises from 3.93 ± 0.16 to 4.65 ± 0.10 nats (*p* < 0.0001), approaching the uniform ceiling. This demonstrates that it is not merely the presence of a fitness signal but the coherent gradient structure of the psychoacoustic landscape that drives self-organised polyphony. A coefficient sweep over 16 kernel settings (Supplementary Table 6) further shows that polyphonic diversity is robust to parameter choice (unique bins ≥ 19 across the entire grid).

**Model-external probe.** The ratio-complexity measure JI_score (Experiment 1) is defined independently of the consonance model—it quantifies proximity to simple integer ratios via number-theoretic complexity. It returned uniformly low values (≈ 0.11) in Experiment 1, confirming that this external metric does not merely echo the training objective, while still capturing the large heredity–random differentiation in Experiment 4.

**Explicit acknowledgment.** The Limitations section now clearly separates what the terrain controls establish (dependence on coherent psychoacoustic topology) from what remains open (whether emergent configurations predict human aesthetic judgements), framing behavioural listening tests as a distinct future question.

---

## W2. Scale limits and open-ended evolution

> The system operates at a fixed scale (24 agents, 100 steps) and does not demonstrate open-ended evolution.

We acknowledge this and have addressed it explicitly in the Limitations section (§5):

> "Experiment 4's hereditary adaptation converges to fixed-landscape attractors and does not constitute open-ended evolution (Bedau, 2000); whether evolvable timbres or coupling topologies could support open-ended dynamics remains open."

We want to emphasise that the paper's claims are deliberately scoped to avoid overstating the system's evolutionary capacity. The four experiments demonstrate self-organisation, selection, synchronisation, and hereditary adaptation—each validated against ablation controls—but we do not claim open-ended evolution. The fixed consonance landscape defines a finite set of attractors, and we state this transparently.

The 24-agent, 100-step scale was chosen to enable rigorous statistical comparison (20 seeds per condition, Welch *t*-tests with effect sizes) within a tractable computational budget. We note that the emergent phenomena reported (polyphonic structure, consonance-lifetime correlation, entrainment bias, hereditary convergence) are qualitatively robust across the parameter range tested (Supplementary S7.3), suggesting that scale is not the binding constraint for the phenomena under study.

Open-ended dynamics would require fundamentally different design choices (evolvable timbres, variable coupling topologies, open-ended niche construction), which we identify as promising future directions rather than limitations of the current contribution.

---

## W3. Computational scalability (Zen6, AVX-512, JSON5)

> Suggestions regarding Zen6 CPU support, AVX-512 SIMD optimisation, and JSON5 configuration format.

We appreciate the practical suggestions. However, we believe these concern implementation engineering rather than the scientific contribution of the paper.

**Zen6/AVX-512.** The current Rust implementation already leverages SIMD through the compiler's auto-vectorisation (compiled with `--release` and appropriate target features). The core DSP operations (convolutions for harmonicity and roughness) operate on contiguous f32 arrays that auto-vectorise well. Explicit AVX-512 intrinsics would be a worthwhile optimisation for a production deployment but would not change any experimental result, as all experiments complete in seconds to minutes on current hardware.

**JSON5.** The system uses Rhai scripts for scenario configuration, which already provides a more expressive and programmable configuration layer than JSON5 would offer (with comments, variables, and control flow).

These are valuable engineering considerations for the open-source release but are orthogonal to the paper's scientific claims about psychoacoustic landscapes as ALife terrain.

---

## W4. Continuous vs. discrete formulation gap for roughness (Eq. 3)

> Equation 3 presents roughness as a continuous integral, but the implementation necessarily operates on a discrete grid, and this gap is not discussed.

Equation 3 presents the roughness computation in continuous form for clarity of exposition, following the standard mathematical convention in psychoacoustics literature (cf. Plomp & Levelt, 1965; Glasberg & Moore, 1990). The implementation operates on a discrete log₂-frequency grid at 400 bins per octave—a resolution of 3 cents per bin, which is well below the just-noticeable difference for frequency (~5–10 cents in the relevant register) and far below the width of the Plomp–Levelt interference kernel (~0.25 ERB ≈ 40–80 cents in the mid-register).

At this resolution, the discrete convolution faithfully approximates the continuous integral: the kernel spans dozens of bins at any frequency, ensuring smooth interpolation with negligible aliasing. The 400-bins/octave resolution is stated in the technical documentation and code; we can add a brief note to the paper if the reviewer considers this helpful, but we believe the continuous formulation correctly conveys the mathematical intent without burdening the reader with implementation-level discretisation details that do not affect the qualitative or quantitative results.

We note that the same continuous-to-discrete convention is standard practice in computational auditory modelling (e.g., Patterson et al., 1992; de Cheveigné, 2005) and does not represent a methodological gap.
