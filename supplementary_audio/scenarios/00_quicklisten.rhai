// 00_quicklisten.rhai — Orientation montage
// Audio supplement for Conchordal (ALIFE 2026)
//
// Curated highlights from evidence tracks (same seed, same conditions).
// Contrast the best-case emergence with ablation controls.
//
// 4 segments (8s each, 1s gaps):
//   1. E2 baseline  — hill-climbing + repulsion → polyphony
//   2. E6 both      — heredity + hill-climbing → integration
//   3. E2 no-hill   — repulsion only → no landscape guidance
//   4. E6 neither   — no mechanisms → random control

let anchor_spec = derive(harmonic)
    .freq(220.0)
    .brain("drone")
    .amp(0.3)
    .pitch_mode("lock");

let agent_random = derive(harmonic)
    .brain("entrain")
    .metabolism(0.5)
    .respawn_random()
    .pitch_core("hill_climb")
    .amp(0.15);

let agent_hereditary = derive(harmonic)
    .brain("entrain")
    .metabolism(0.5)
    .respawn_hereditary(0.003)
    .pitch_core("hill_climb")
    .amp(0.15);

// ── Segment 1: E2 Baseline (seed 0) ────────────────────────────
seed(12648430);
scene("ql1_e2_baseline", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(1.0)
        .repulsion(0.15, 72.0);
    wait(8.0);
});
wait(1.0);

// ── Segment 2: E6 Both (seed 0) ────────────────────────────────
seed(12648490);
scene("ql2_e6_both", || {
    create(anchor_spec, 1);
    create(agent_hereditary, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.5);
    wait(8.0);
});
wait(1.0);

// ── Segment 3: E2 No hill-climbing (seed 0) ────────────────────
seed(12648430);
scene("ql3_e2_nohill", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0)
        .repulsion(0.15, 72.0);
    wait(8.0);
});
wait(1.0);

// ── Segment 4: E6 Neither (seed 0) ─────────────────────────────
seed(12648490);
scene("ql4_e6_neither", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0);
    wait(8.0);
});
