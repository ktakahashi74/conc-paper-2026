// 20_integration.rhai — E6: Heredity × hill-climbing integration
// Audio supplement for Conchordal (ALIFE 2026)
//
// 8 segments: 4 conditions × 2 seeds (E6_SEEDS index 0 and 10)
//   neither   : respawn_random()       + landscape_weight=0.0
//   H-only    : respawn_hereditary()   + landscape_weight=0.0
//   hill-only : respawn_random()       + landscape_weight=0.5
//   both      : respawn_hereditary()   + landscape_weight=0.5
//
// E6_SEEDS[0]  = 0xC0FFEE + 60  = 12648490
// E6_SEEDS[10] = 0xC0FFEE + 70  = 12648500

let anchor_spec = derive(harmonic)
    .freq(220.0)
    .brain("drone")
    .amp(0.3)
    .pitch_mode("lock")
    .sustain_drive(0.02)
    .pitch_smooth(0.3)
    .adsr(0.5, 0.2, 0.9, 1.5);

let agent_random = derive(harmonic)
    .brain("drone")
    .metabolism(0.01)
    .respawn_random()
    .pitch_core("hill_climb")
    .exploration(0.8)
    .persistence(0.2)
    .anneal_temp(0.5)
    .proposal_interval(0.3)
    .amp(0.15)
    .sustain_drive(0.02)
    .pitch_smooth(0.3)
    .adsr(0.3, 0.3, 0.7, 1.2);

let agent_hereditary = derive(harmonic)
    .brain("drone")
    .metabolism(0.01)
    .respawn_hereditary(0.003)
    .pitch_core("hill_climb")
    .exploration(0.8)
    .persistence(0.2)
    .anneal_temp(0.5)
    .proposal_interval(0.3)
    .amp(0.15)
    .sustain_drive(0.02)
    .pitch_smooth(0.3)
    .adsr(0.3, 0.3, 0.7, 1.2);

// ── Segment 1: Seed 0, Neither ──────────────────────────────────
seed(12648490);
scene("s1_seed0_neither", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0);
    wait(10.0);
});
wait(1.0);

// ── Segment 2: Seed 0, H-only ──────────────────────────────────
seed(12648490);
scene("s2_seed0_honly", || {
    create(anchor_spec, 1);
    create(agent_hereditary, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0);
    wait(10.0);
});
wait(1.0);

// ── Segment 3: Seed 0, Hill-only ────────────────────────────────
seed(12648490);
scene("s3_seed0_hillonly", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.5);
    wait(10.0);
});
wait(1.0);

// ── Segment 4: Seed 0, Both ────────────────────────────────────
seed(12648490);
scene("s4_seed0_both", || {
    create(anchor_spec, 1);
    create(agent_hereditary, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.5);
    wait(10.0);
});
wait(1.0);

// ── Segment 5: Seed 10, Neither ─────────────────────────────────
seed(12648500);
scene("s5_seed10_neither", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0);
    wait(10.0);
});
wait(1.0);

// ── Segment 6: Seed 10, H-only ─────────────────────────────────
seed(12648500);
scene("s6_seed10_honly", || {
    create(anchor_spec, 1);
    create(agent_hereditary, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.0);
    wait(10.0);
});
wait(1.0);

// ── Segment 7: Seed 10, Hill-only ───────────────────────────────
seed(12648500);
scene("s7_seed10_hillonly", || {
    create(anchor_spec, 1);
    create(agent_random, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.5);
    wait(10.0);
});
wait(1.0);

// ── Segment 8: Seed 10, Both ───────────────────────────────────
seed(12648500);
scene("s8_seed10_both", || {
    create(anchor_spec, 1);
    create(agent_hereditary, 8)
        .place(random_log(110.0, 880.0))
        .landscape_weight(0.5);
    wait(10.0);
});
