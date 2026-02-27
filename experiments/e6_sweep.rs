/// E6 parameter sweep: systematically explore conditions for hereditary adaptation.
/// Run with: cargo run --release --bin e6sweep
use crate::sim::{E6Condition, E6RunConfig, run_e6};

const ANCHOR_HZ: f32 = 220.0;
const STEPS_CAP: usize = 200_000;
const SNAPSHOT_INTERVAL: usize = 100;
const FIRST_K: usize = 20;

const SWEEP_SEEDS: [u64; 20] = [
    0xC0FFEE_u64 + 60,
    0xC0FFEE_u64 + 61,
    0xC0FFEE_u64 + 62,
    0xC0FFEE_u64 + 63,
    0xC0FFEE_u64 + 64,
    0xC0FFEE_u64 + 65,
    0xC0FFEE_u64 + 66,
    0xC0FFEE_u64 + 67,
    0xC0FFEE_u64 + 68,
    0xC0FFEE_u64 + 69,
    0xC0FFEE_u64 + 70,
    0xC0FFEE_u64 + 71,
    0xC0FFEE_u64 + 72,
    0xC0FFEE_u64 + 73,
    0xC0FFEE_u64 + 74,
    0xC0FFEE_u64 + 75,
    0xC0FFEE_u64 + 76,
    0xC0FFEE_u64 + 77,
    0xC0FFEE_u64 + 78,
    0xC0FFEE_u64 + 79,
];

struct SweepParams {
    label: &'static str,
    mutation_sigma: f32,
    min_deaths: usize,
    pop_size: usize,
}

pub fn run_sweep() {
    let params = vec![
        // Round 5: fine-tune around sig0.003_d2000_p32 (best from round 4, p=0.0085)
        SweepParams { label: "sig0.002_d2000_p32", mutation_sigma: 0.002, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.0025_d2000_p32", mutation_sigma: 0.0025, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.003_d2000_p32", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.0035_d2000_p32", mutation_sigma: 0.0035, min_deaths: 2000, pop_size: 32 },
        // Also try 0.003 with more deaths
        SweepParams { label: "sig0.003_d2500_p32", mutation_sigma: 0.003, min_deaths: 2500, pop_size: 32 },
        SweepParams { label: "sig0.003_d3000_p32", mutation_sigma: 0.003, min_deaths: 3000, pop_size: 32 },
        // And with pop=24
        SweepParams { label: "sig0.003_d2000_p24", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 24 },
    ];

    println!("config,condition,seed,total_deaths,n_snapshots,mean_c_start,mean_c_end,delta_mean_c,entropy_end");

    for p in &params {
        for condition in [E6Condition::Heredity, E6Condition::Random] {
            for &seed in &SWEEP_SEEDS {
                let cfg = E6RunConfig {
                    seed,
                    steps_cap: STEPS_CAP,
                    min_deaths: p.min_deaths,
                    pop_size: p.pop_size,
                    first_k: FIRST_K,
                    condition,
                    mutation_sigma: p.mutation_sigma,
                    snapshot_interval: SNAPSHOT_INTERVAL,
                };
                let result = run_e6(&cfg);

                // Compute summary stats from snapshots
                let anchor_log2 = (ANCHOR_HZ as f64).log2();
                let snaps = &result.snapshots;
                if snaps.len() < 2 {
                    continue;
                }

                let consonance_of_snap = |snap: &crate::sim::E6PitchSnapshot| -> f64 {
                    let landscape = crate::sim::e3_reference_landscape(ANCHOR_HZ);
                    let mut total = 0.0f64;
                    let mut count = 0;
                    for &freq in &snap.freqs_hz {
                        if freq <= 0.0 || !freq.is_finite() { continue; }
                        total += landscape.evaluate_pitch_level(freq) as f64;
                        count += 1;
                    }
                    if count > 0 { total / count as f64 } else { 0.0 }
                };

                let entropy_of_snap = |snap: &crate::sim::E6PitchSnapshot| -> f64 {
                    let bin_width = 0.25f64; // semitones
                    let mut bins: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
                    for &freq in &snap.freqs_hz {
                        if freq <= 0.0 || !freq.is_finite() { continue; }
                        let st = ((freq as f64).log2() - anchor_log2) * 12.0;
                        let bin = (st / bin_width).round() as i32;
                        *bins.entry(bin).or_insert(0) += 1;
                    }
                    let total: usize = bins.values().sum();
                    if total == 0 { return 0.0; }
                    let mut h = 0.0f64;
                    for &c in bins.values() {
                        let p = c as f64 / total as f64;
                        if p > 0.0 { h -= p * p.ln(); }
                    }
                    h
                };

                // Use first and last snapshots for start/end
                let c_start = consonance_of_snap(&snaps[0]);
                let c_end = consonance_of_snap(snaps.last().unwrap());
                let entropy_end = entropy_of_snap(snaps.last().unwrap());

                println!(
                    "{},{},{},{},{},{:.6},{:.6},{:.6},{:.4}",
                    p.label,
                    condition.label(),
                    seed,
                    result.total_deaths,
                    snaps.len(),
                    c_start,
                    c_end,
                    c_end - c_start,
                    entropy_end,
                );
            }
        }
        eprintln!("Done: {}", p.label);
    }
}
