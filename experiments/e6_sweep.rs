/// E6 parameter sweep: systematically explore conditions for hereditary adaptation.
/// Run with: cargo run --release --bin e6sweep
use crate::sim::{E6Condition, E6RunConfig, run_e6};
use std::collections::HashMap;

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
        // Round 6: C_score metric (no sigmoid). Broad sweep.
        // Vary sigma
        SweepParams { label: "sig0.001_d2000_p32", mutation_sigma: 0.001, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.002_d2000_p32", mutation_sigma: 0.002, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.003_d2000_p32", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.005_d2000_p32", mutation_sigma: 0.005, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.01_d2000_p32", mutation_sigma: 0.01, min_deaths: 2000, pop_size: 32 },
        SweepParams { label: "sig0.02_d2000_p32", mutation_sigma: 0.02, min_deaths: 2000, pop_size: 32 },
        // Vary deaths with sigma=0.003
        SweepParams { label: "sig0.003_d3000_p32", mutation_sigma: 0.003, min_deaths: 3000, pop_size: 32 },
        SweepParams { label: "sig0.003_d4000_p32", mutation_sigma: 0.003, min_deaths: 4000, pop_size: 32 },
        SweepParams { label: "sig0.003_d5000_p32", mutation_sigma: 0.003, min_deaths: 5000, pop_size: 32 },
        // Vary pop_size with sigma=0.003, deaths=2000
        SweepParams { label: "sig0.003_d2000_p16", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 16 },
        SweepParams { label: "sig0.003_d2000_p24", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 24 },
        SweepParams { label: "sig0.003_d2000_p48", mutation_sigma: 0.003, min_deaths: 2000, pop_size: 48 },
        // High-generation combos
        SweepParams { label: "sig0.005_d3000_p32", mutation_sigma: 0.005, min_deaths: 3000, pop_size: 32 },
        SweepParams { label: "sig0.002_d3000_p32", mutation_sigma: 0.002, min_deaths: 3000, pop_size: 32 },
        SweepParams { label: "sig0.001_d3000_p32", mutation_sigma: 0.001, min_deaths: 3000, pop_size: 32 },
    ];

    println!("config,condition,seed,total_deaths,n_snapshots,mean_c_start,mean_c_end,delta_mean_c,entropy_end");

    // Collect per-config, per-condition delta_c values for summary
    struct CondData {
        deltas: Vec<f64>,
        c_ends: Vec<f64>,
    }
    let mut summary: HashMap<String, HashMap<String, CondData>> = HashMap::new();

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
                        total += landscape.evaluate_pitch_score(freq) as f64;
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
                let delta_c = c_end - c_start;

                println!(
                    "{},{},{},{},{},{:.6},{:.6},{:.6},{:.4}",
                    p.label,
                    condition.label(),
                    seed,
                    result.total_deaths,
                    snaps.len(),
                    c_start,
                    c_end,
                    delta_c,
                    entropy_end,
                );

                let cond_label = condition.label().to_string();
                summary
                    .entry(p.label.to_string())
                    .or_insert_with(HashMap::new)
                    .entry(cond_label)
                    .or_insert_with(|| CondData {
                        deltas: Vec::new(),
                        c_ends: Vec::new(),
                    })
                    .deltas
                    .push(delta_c);
                summary
                    .get_mut(p.label)
                    .unwrap()
                    .get_mut(condition.label())
                    .unwrap()
                    .c_ends
                    .push(c_end);
            }
        }
        eprintln!("Done: {}", p.label);
    }

    // --- Print summary table ---
    eprintln!("\n=== Sweep Summary ===");
    eprintln!("config,heredity_delta_mean,heredity_delta_sd,random_delta_mean,random_delta_sd,separation,welch_t,welch_p,positive_sep,sig_p05");

    fn mean_sd(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        (mean, var.sqrt())
    }

    fn welch_t_test(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
        let (ma, sa) = mean_sd(a);
        let (mb, sb) = mean_sd(b);
        let na = a.len() as f64;
        let nb = b.len() as f64;
        let va = sa * sa / na;
        let vb = sb * sb / nb;
        let denom = (va + vb).sqrt();
        if denom < 1e-15 {
            return (0.0, 1.0, 1.0);
        }
        let t = (ma - mb) / denom;
        let df = (va + vb).powi(2) / (va * va / (na - 1.0) + vb * vb / (nb - 1.0));
        // Approximate p via betacf for incomplete beta I_x(a,b)
        let p = two_tailed_t_p(t.abs(), df);
        (t, df, p)
    }

    fn ln_gamma(x: f64) -> f64 {
        let coeffs = [
            76.18009172947146,
            -86.50532032941677,
            24.01409824083091,
            -1.231739572450155,
            0.1208650973866179e-2,
            -0.5395239384953e-5,
        ];
        let mut y = x;
        let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
        let mut ser = 1.000000000190015_f64;
        for &c in &coeffs {
            y += 1.0;
            ser += c / y;
        }
        -tmp + (2.5066282746310005 * ser / x).ln()
    }

    fn betacf(a: f64, b: f64, x: f64) -> f64 {
        let fpmin = 1.0e-30;
        let eps = 3.0e-12;
        let qab = a + b;
        let qap = a + 1.0;
        let qam = a - 1.0;
        let mut c = 1.0_f64;
        let mut d = (1.0 - qab * x / qap).recip();
        if d.abs() < fpmin { d = fpmin; }
        let mut h = d;
        for m in 1..=200 {
            let m_f = m as f64;
            let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
            d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
            c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
            d = d.recip(); h *= d * c;
            let aa = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
            d = 1.0 + aa * d; if d.abs() < fpmin { d = fpmin; }
            c = 1.0 + aa / c; if c.abs() < fpmin { c = fpmin; }
            d = d.recip(); let delta = d * c; h *= delta;
            if (delta - 1.0).abs() < eps { break; }
        }
        h
    }

    fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
        if x < (a + 1.0) / (a + b + 2.0) {
            bt * betacf(a, b, x) / a
        } else {
            1.0 - bt * betacf(b, a, 1.0 - x) / b
        }
    }

    fn two_tailed_t_p(t_abs: f64, df: f64) -> f64 {
        let x = df / (df + t_abs * t_abs);
        reg_inc_beta(df / 2.0, 0.5, x)
    }

    let mut n_positive = 0;
    let mut n_sig = 0;
    let n_total = params.len();

    // Sort by label for deterministic output
    let mut labels: Vec<String> = summary.keys().cloned().collect();
    labels.sort();

    for label in &labels {
        let conds = &summary[label];
        let h_data = conds.get("heredity");
        let r_data = conds.get("random");
        if let (Some(h), Some(r)) = (h_data, r_data) {
            let (h_mean, h_sd) = mean_sd(&h.deltas);
            let (r_mean, r_sd) = mean_sd(&r.deltas);
            let sep = h_mean - r_mean;
            let (t, _df, p) = welch_t_test(&h.deltas, &r.deltas);
            let positive = sep > 0.0;
            let sig = p < 0.05;
            if positive { n_positive += 1; }
            if sig { n_sig += 1; }
            eprintln!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.3},{:.6},{},{}",
                label, h_mean, h_sd, r_mean, r_sd, sep, t, p, positive, sig
            );
        }
    }
    eprintln!(
        "\nPositive separation: {}/{}, Significant (p<0.05): {}/{}",
        n_positive, n_total, n_sig, n_total
    );
}
