// 10_exp1_polyphony.rhai — E2: Hill-climbing + repulsion → emergent polyphony
// Audio supplement for Conchordal (ALIFE 2026)
//
// 6 segments: 3 conditions × 2 seeds (E2_SEEDS index 0 and 10)
//   baseline : landscape_weight=1.0 + repulsion(0.15, 72)
//   no-hill  : landscape_weight=0.0 + repulsion(0.15, 72)
//   no-rep   : landscape_weight=1.0 + repulsion off
//
// E2_SEEDS[0]  = 0xC0FFEE    = 12648430
// E2_SEEDS[10] = 0x0F0F0F0F  = 252645135

let anchor_spec = derive(harmonic)
    .freq(220.0)
    .brain("drone")
    .amp(0.3)
    .pitch_mode("lock")
    .timbre(1.0, 0.0)
    .sustain_drive(0.02)
    .pitch_smooth(0.05)
    .adsr(0.5, 0.2, 0.9, 1.5);

let agent_spec = derive(harmonic)
    .brain("drone")
    .metabolism(0.01)
    .respawn_random()
    .pitch_core("hill_climb")
    .loo(true)
    .move_cost(0.05)
    .improvement_threshold(0.0)
    .exploration(0.8)
    .anneal_temp(0.3)
    .timbre(1.0, 0.0)
    .amp(0.15)
    .sustain_drive(0.02)
    .pitch_smooth(0.05)
    .adsr(0.3, 0.3, 0.7, 1.2);

// ── Segment 1: Seed 0, Baseline ──────────────────────────────────
seed(12648430);
scene("s1_seed0_baseline", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(1.0)
        .repulsion(0.15, 72.0);
    wait(12.0);
});
wait(1.0);

// ── Segment 2: Seed 0, No hill-climbing ──────────────────────────
seed(12648430);
scene("s2_seed0_nohill", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0)
        .repulsion(0.15, 72.0);
    wait(12.0);
});
wait(1.0);

// ── Segment 3: Seed 0, No repulsion ─────────────────────────────
seed(12648430);
scene("s3_seed0_norep", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(1.0)
        .repulsion(0.0);
    wait(12.0);
});
wait(1.0);

// ── Segment 4: Seed 10, Baseline ─────────────────────────────────
seed(252645135);
scene("s4_seed10_baseline", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(1.0)
        .repulsion(0.15, 72.0);
    wait(12.0);
});
wait(1.0);

// ── Segment 5: Seed 10, No hill-climbing ─────────────────────────
seed(252645135);
scene("s5_seed10_nohill", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0)
        .repulsion(0.15, 72.0);
    wait(12.0);
});
wait(1.0);

// ── Segment 6: Seed 10, No repulsion ────────────────────────────
seed(252645135);
scene("s6_seed10_norep", || {
    create(anchor_spec, 1);
    create(agent_spec, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(1.0)
        .repulsion(0.0);
    wait(12.0);
});
