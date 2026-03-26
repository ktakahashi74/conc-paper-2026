use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::{create_dir_all, read_to_string, remove_dir_all};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use plotters::coord::types::RangedCoordf32;
use plotters::coord::{CoordTranslate, Shift};
use plotters::element::DashedPathElement;
use plotters::prelude::*;

use crate::sim::{
    E3_BINS_PER_OCT, E3_FMAX, E3_FMIN, E3Condition, E3DeathRecord, E3RunConfig, E4_ANCHOR_HZ,
    E4_ENV_PARTIAL_DECAY_DEFAULT, E4_ENV_PARTIALS_DEFAULT, E6AgentSnapshot, E6B_DEFAULT_POP_SIZE,
    E6Condition, E6DeathCause, E6ParentProposalKind, E6PitchSnapshot, E6StopReason, E6bAzimuthMode,
    E6bRandomBaselineMode, E6bRunConfig, E6bRunResult, configure_shared_hillclimb_core,
    e3_policy_params, e3_reference_landscape, e3_tessitura_bounds_for_range, run_e3_collect_deaths,
    run_e6b, shared_hill_move_cost_coeff,
};
use conchordal::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{
    Landscape, LandscapeParams, PitchObjectiveMode, RoughnessScalarMode,
};
use conchordal::core::log2space::Log2Space;
use conchordal::core::phase::wrap_pm_pi;
use conchordal::core::psycho_state;
use conchordal::core::roughness_kernel::{
    KernelParams, RoughnessKernel, crowding_runtime_delta_erb, erb_grid, eval_kernel_delta_erb,
};
use conchordal::life::articulation_core::kuramoto_phase_step;
use conchordal::life::individual::PitchHillClimbPitchCore;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::StdRng};

// ── Nocturnal palette ──────────────────────────────────────────────
const PAL_H: RGBColor = RGBColor(58, 106, 120); // #3A6A78 deep steel teal
const PAL_R: RGBColor = RGBColor(139, 34, 82); // #8B2252 dark rose
const PAL_C: RGBColor = RGBColor(93, 87, 107); // #5D576B purple grey
const PAL_CD: RGBColor = RGBColor(62, 111, 182); // #3E6FB6 cobalt blue

const SPACE_BINS_PER_OCT: u32 = 400;

const E2_SWEEPS: usize = 8;
const E2_BURN_IN: usize = 2;
const E2_ANCHOR_SHIFT_STEP: usize = usize::MAX;
const E2_ANCHOR_SHIFT_RATIO: f32 = 0.5;
const E2_STEP_SEMITONES: f32 = 0.25;
const E2_ANCHOR_BIN_ST: f32 = 0.5;
const E2_PAIRWISE_BIN_ST: f32 = 0.05;
const E2_PAIRWISE_DISPLAY_BIN_ST: f32 = 0.25;
const E2_N_AGENTS: usize = 24;
const E2_CROWDING_WEIGHT: f32 = 0.15;
const E2_INIT_CONSONANT_EXCLUSION_ST: f32 = 0.35;
const E2_INIT_MAX_TRIES: usize = 5000;
const E2_C_LEVEL_BETA: f32 = 2.0;
const E2_C_LEVEL_THETA: f32 = 0.0;
const E2_ACCEPT_ENABLED: bool = true;
const E2_ACCEPT_T0: f32 = 0.05;
const E2_ACCEPT_TAU_STEPS: f32 = 4.8;
const E2_ACCEPT_RESET_ON_PHASE: bool = true;
const E2_SCORE_IMPROVE_EPS: f32 = 1e-4;
const E2_ANTI_BACKTRACK_ENABLED: bool = true;
const E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY: bool = false;
const E2_BACKTRACK_ALLOW_EPS: f32 = 1e-4;
const E2_PHASE_SWITCH_STEP: usize = E2_SWEEPS / 2;
const E2_DIVERSITY_BIN_ST: f32 = 0.25;
const E2_STEP_SEMITONES_SWEEP: [f32; 4] = [0.125, 0.25, 0.5, 1.0];
const E2_LAZY_MOVE_PROB: f32 = 0.65;
const E2_SEMITONE_EPS: f32 = 1e-6;
const E2_EMIT_KBINS_SWEEP_SUMMARY: bool = false;
const E2_EMIT_CONSONANT_MASS_STATS: bool = false;
const E2_SEEDS: [u64; 20] = [
    0xC0FFEE_u64,
    0xA5A5A5A5_u64,
    0x1BADB002_u64,
    0xDEADBEEF_u64,
    0xFACEFEED_u64,
    0x1234ABCD_u64,
    0x31415926_u64,
    0x27182818_u64,
    0xCAFEBABE_u64,
    0x9E3779B9_u64,
    0x0F0F0F0F_u64,
    0x55AA55AA_u64,
    0x87654321_u64,
    0xABCDEF01_u64,
    0x0C0FFEE0_u64,
    0x13579BDF_u64,
    0x2468ACE0_u64,
    0xBADC0FFE_u64,
    0x10203040_u64,
    0x55667788_u64,
];
const E2_PAPER_N_AGENTS: usize = 24;
const E2_PAPER_RANGE_OCT: f32 = 4.0;
const E2_PAPER_REP_SEED_INDEX: usize = 0;
const E2_CLOSE_PAIR_WINDOW_ST: f32 = 0.5;
const E2_DENSE_SWEEP_RANGES_OCT: [f32; 3] = [2.0, 3.0, 4.0];
const E2_DENSE_SWEEP_VOICES_PER_OCT: [f32; 4] = [2.5, 4.0, 6.0, 8.0];
const E2_DENSE_SWEEP_2OCT_EXTREME_N: [usize; 2] = [24, 48];
const E2_DENSE_SWEEP_QUICK_SEEDS: usize = 8;
const E2_CANDIDATE_SEARCH_2OCT_N: [usize; 8] = [5, 6, 7, 8, 10, 12, 16, 24];
const E2_CANDIDATE_SEARCH_3OCT_N: [usize; 9] = [6, 7, 8, 9, 10, 12, 14, 16, 18];
const E2_CANDIDATE_SEARCH_4OCT_N: [usize; 9] = [8, 9, 10, 12, 14, 16, 18, 20, 24];
const E2_CANDIDATE_SEARCH_RENDER_PARTIALS: u32 = 4;
const E2_CANDIDATE_SEARCH_TOP_PER_RANGE: usize = 2;
const E2_SCENE_G_ENV_PARTIALS: u32 = 4;
const E2_SCENE_G_ENV_DECAY: f32 = 1.0;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum E2InitMode {
    Uniform,
    RejectConsonant,
}

impl E2InitMode {
    fn label(self) -> &'static str {
        match self {
            E2InitMode::Uniform => "uniform",
            E2InitMode::RejectConsonant => "reject_consonant",
        }
    }
}

const E2_INIT_MODE: E2InitMode = E2InitMode::RejectConsonant;
// Just-intonation positions: 12 * log2(ratio) semitones
const E2_CONSONANT_STEPS: [f32; 8] = [
    0.0,       // 1:1 unison
    3.1564128, // 6:5 minor third
    3.8631372, // 5:4 major third
    4.9804499, // 4:3 perfect fourth
    7.0195501, // 3:2 perfect fifth
    8.136_863, // 8:5 minor sixth
    8.843_587, // 5:3 major sixth
    12.0,      // 2:1 octave
];
const E2_CONSONANT_TARGETS_CORE: [f32; 3] = [3.1564128, 3.8631372, 7.0195501];
const E2_CONSONANT_TARGETS_EXTENDED: [f32; 6] = [
    3.1564128, // 6:5 minor third
    3.8631372, // 5:4 major third
    4.9804499, // 4:3 perfect fourth
    7.0195501, // 3:2 perfect fifth
    8.136_863, // 8:5 minor sixth
    8.843_587, // 5:3 major sixth
];
// Display-only guides for Fig. 3: omit 6:5, which is not a salient peak in the paper setting.
const E2_INTERVAL_GUIDE_STEPS: [f32; 7] = [
    0.0,       // 1:1 unison
    3.8631372, // 5:4 major third
    4.9804499, // 4:3 perfect fourth
    7.0195501, // 3:2 perfect fifth
    8.136_863, // 8:5 minor sixth
    8.843_587, // 5:3 major sixth
    12.0,      // 2:1 octave
];
const E2_INTERVAL_GUIDE_TARGETS_CORE: [f32; 3] = [3.8631372, 4.9804499, 7.0195501];
const E2_CONSONANT_WINDOW_ST: f32 = 0.25;
const E2_PERM_MAX_EXACT_COMBOS: u64 = 500_000;
const E2_PERM_MC_ITERS: usize = 50_000;
const E2_PERM_MC_SEED: u64 = 0xC0FFEE_u64 + 0xE2;

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum E2UpdateSchedule {
    Checkerboard,
    Lazy,
    RandomSingle,
    SequentialRotate,
}

const E2_UPDATE_SCHEDULE: E2UpdateSchedule = E2UpdateSchedule::SequentialRotate;

const FLOAT_KEY_SCALE: f32 = 1000.0;

const E3_FIRST_K: usize = 20;
const E3_POP_SIZE: usize = 32;
const E3_MIN_DEATHS: usize = 200;
const E3_STEPS_CAP: usize = 6000;
const E3_SEEDS: [u64; 20] = [
    0xC0FFEE_u64 + 30,
    0xC0FFEE_u64 + 31,
    0xC0FFEE_u64 + 32,
    0xC0FFEE_u64 + 33,
    0xC0FFEE_u64 + 34,
    0xC0FFEE_u64 + 35,
    0xC0FFEE_u64 + 36,
    0xC0FFEE_u64 + 37,
    0xC0FFEE_u64 + 38,
    0xC0FFEE_u64 + 39,
    0xC0FFEE_u64 + 40,
    0xC0FFEE_u64 + 41,
    0xC0FFEE_u64 + 42,
    0xC0FFEE_u64 + 43,
    0xC0FFEE_u64 + 44,
    0xC0FFEE_u64 + 45,
    0xC0FFEE_u64 + 46,
    0xC0FFEE_u64 + 47,
    0xC0FFEE_u64 + 48,
    0xC0FFEE_u64 + 49,
];

// ── E5: Vitality-Coupled Entrainment ─────────────────────────────
const E5_KICK_OMEGA: f32 = 2.0 * PI * 2.0;
const E5_AGENT_OMEGA_MEAN: f32 = 2.0 * PI * 1.8;
const E5_AGENT_JITTER: f32 = 0.02;
const E5_K_TIME: f32 = 3.0; // base coupling (= old K_BASE)
const E5_LAMBDA_V: f32 = 1.0; // vitality sensitivity in coupling_multiplier
const E5_V_FLOOR: f32 = 0.0; // vitality floor (0 = full range)
const E5_VITALITY_EXPONENT: f32 = 0.5; // = sqrt (matches main)
const E5_N_AGENTS: usize = 32;
const E5_DT: f32 = 0.02;
const E5_STEPS: usize = 2000;
const E5_TIME_PLV_WINDOW_STEPS: usize = 200;
const E5_ANCHOR_HZ: f32 = 220.0;
const E5_E_CAP: f32 = 1.0;
const E5_E_INIT: f32 = 0.0;
const E5_RECHARGE: f32 = 1.0;
const E5_DECAY: f32 = 0.5;
const E5_SEEDS: [u64; 20] = [
    0xE5C0FF_u64,
    0xE5C0FF_u64 + 1,
    0xE5C0FF_u64 + 2,
    0xE5C0FF_u64 + 3,
    0xE5C0FF_u64 + 4,
    0xE5C0FF_u64 + 5,
    0xE5C0FF_u64 + 6,
    0xE5C0FF_u64 + 7,
    0xE5C0FF_u64 + 8,
    0xE5C0FF_u64 + 9,
    0xE5C0FF_u64 + 10,
    0xE5C0FF_u64 + 11,
    0xE5C0FF_u64 + 12,
    0xE5C0FF_u64 + 13,
    0xE5C0FF_u64 + 14,
    0xE5C0FF_u64 + 15,
    0xE5C0FF_u64 + 16,
    0xE5C0FF_u64 + 17,
    0xE5C0FF_u64 + 18,
    0xE5C0FF_u64 + 19,
];
const E5_REPRESENTATIVE_SEED_IDX: usize = 2;

const PAL_E5_VITALITY: RGBColor = PAL_CD; // cobalt blue for vitality condition
const PAL_E5_UNIFORM: RGBColor = PAL_R; // dark rose for uniform condition
const PAL_E5_CONTROL: RGBColor = PAL_C; // purple grey for control

// ── E7: Temporal Scaffold Assay ──────────────────────────────────
const E7_KICK_HZ: f32 = 2.0;
const E7_KICK_OMEGA: f32 = 2.0 * PI * E7_KICK_HZ;
const E7_AGENT_OMEGA_MEAN: f32 = 2.0 * PI * 1.8;
const E7_AGENT_JITTER: f32 = 0.02;
const E7_K_TIME: f32 = 3.0;
const E7_N_AGENTS: usize = 32;
const E7_DT: f32 = 0.02;
const E7_STEPS: usize = 2000;
const E7_TIME_PLV_WINDOW_STEPS: usize = 200;
const E7_STEPS_PER_CYCLE: usize = 25;
const E7_PHASE_HIST_BINS: usize = 24;
const E7_ONSET_ANALYSIS_START_SEC: f32 = 5.0;
const E7_REPRESENTATIVE_SEED_IDX: usize = 10;
const E7_SEEDS: [u64; 20] = [
    0xE70000_u64,
    0xE70000_u64 + 1,
    0xE70000_u64 + 2,
    0xE70000_u64 + 3,
    0xE70000_u64 + 4,
    0xE70000_u64 + 5,
    0xE70000_u64 + 6,
    0xE70000_u64 + 7,
    0xE70000_u64 + 8,
    0xE70000_u64 + 9,
    0xE70000_u64 + 10,
    0xE70000_u64 + 11,
    0xE70000_u64 + 12,
    0xE70000_u64 + 13,
    0xE70000_u64 + 14,
    0xE70000_u64 + 15,
    0xE70000_u64 + 16,
    0xE70000_u64 + 17,
    0xE70000_u64 + 18,
    0xE70000_u64 + 19,
];

const PAL_E7_SHARED: RGBColor = PAL_CD;
const PAL_E7_SCRAMBLED: RGBColor = PAL_R;
const PAL_E7_OFF: RGBColor = PAL_C;

const E6_SNAPSHOT_INTERVAL: usize = 10;
const E6_FIG_X_MAX: f32 = 10_000.0;
const E6_INTERVAL_BIN_ST: f32 = 0.25;
const E6_MAIN_RANGE_OCT: f32 = 2.0;
const E6B_FIRST_K: usize = 20;
const E6B_MIN_DEATHS: usize = 2500;
const E6B_STEPS_CAP: usize = 12_000;
const E6B_SNAPSHOT_INTERVAL: usize = 10;
const E6B_SEEDS: [u64; 20] = [
    0xE6B0_0001_u64,
    0xE6B0_0002_u64,
    0xE6B0_0003_u64,
    0xE6B0_0004_u64,
    0xE6B0_0005_u64,
    0xE6B0_0006_u64,
    0xE6B0_0007_u64,
    0xE6B0_0008_u64,
    0xE6B0_0009_u64,
    0xE6B0_000A_u64,
    0xE6B0_000B_u64,
    0xE6B0_000C_u64,
    0xE6B0_000D_u64,
    0xE6B0_000E_u64,
    0xE6B0_000F_u64,
    0xE6B0_0010_u64,
    0xE6B0_0011_u64,
    0xE6B0_0012_u64,
    0xE6B0_0013_u64,
    0xE6B0_0014_u64,
];
const E6B_QUICK_SEED_COUNT: usize = 4;
const PAPER_PLOTS_LOCK_FILE: &str = "experiments/.paper_plots.lock";
const PAPER_PLOTS_BASE_DIR: &str = "experiments/plots";

#[derive(Clone, Copy, Debug)]
struct E6bCliOptions {
    quick: bool,
    seed_limit: Option<usize>,
    skip_benchmark: bool,
    crowding_weight: Option<f32>,
    capacity_weight: Option<f32>,
    capacity_radius_cents: Option<f32>,
    capacity_free_voices: Option<usize>,
    parent_share_weight: Option<f32>,
    parent_energy_weight: Option<f32>,
    juvenile_enabled: Option<bool>,
    juvenile_ticks: Option<u32>,
    survival_score_low: Option<f32>,
    survival_score_high: Option<f32>,
    survival_recharge_per_sec: Option<f32>,
    background_death_rate_per_sec: Option<f32>,
    parent_proposal_kind: Option<E6ParentProposalKind>,
    parent_proposal_sigma_st: Option<f32>,
    parent_proposal_candidate_count: Option<usize>,
    azimuth_mode: Option<E6bAzimuthMode>,
    respawn_parent_prior_mix: Option<f32>,
    respawn_same_band_discount: Option<f32>,
    respawn_octave_discount: Option<f32>,
}

fn parse_on_off_cli_opt(args: &[String], key: &str) -> Result<Option<bool>, String> {
    let mut value: Option<String> = None;
    let key_eq = format!("{key}=");
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == key {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix(&key_eq) {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" | "yes" => Ok(Some(true)),
        "off" | "false" | "0" | "no" => Ok(Some(false)),
        _ => Err(format!(
            "Invalid {key} value '{raw}'. Use on/off.\n{}",
            usage()
        )),
    }
}

impl E6bCliOptions {
    fn seed_slice(self) -> &'static [u64] {
        let limit = self
            .seed_limit
            .unwrap_or(if self.quick {
                E6B_QUICK_SEED_COUNT
            } else {
                E6B_SEEDS.len()
            })
            .clamp(1, E6B_SEEDS.len());
        &E6B_SEEDS[..limit]
    }
}

fn e6_effective_range_bounds_st() -> (f32, f32) {
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let (min_freq, max_freq) =
        e3_tessitura_bounds_for_range(E4_ANCHOR_HZ, &space, E6_MAIN_RANGE_OCT);
    (
        12.0 * (min_freq / E4_ANCHOR_HZ).log2(),
        12.0 * (max_freq / E4_ANCHOR_HZ).log2(),
    )
}

fn log_output_path(path: &Path) {
    println!("write {}", path.display());
}

fn write_with_log<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    let path = path.as_ref();
    log_output_path(path);
    std::fs::write(path, contents)
}

fn bitmap_root<'a>(out_path: &'a Path, size: (u32, u32)) -> SVGBackend<'a> {
    log_output_path(out_path);
    SVGBackend::new(out_path, size)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Experiment {
    E1,
    E2,
    E3,
    E5,
    E6b,
    E7,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum E2PhaseMode {
    DissonanceThenConsonance,
    ConsonanceOnly,
}

impl E2PhaseMode {
    fn label(self) -> &'static str {
        match self {
            Self::DissonanceThenConsonance => "dissonance_then_consonance",
            Self::ConsonanceOnly => "consonance_only",
        }
    }

    fn score_sign(self, step: usize) -> f32 {
        match self {
            Self::DissonanceThenConsonance => {
                if step < E2_PHASE_SWITCH_STEP {
                    -1.0
                } else {
                    1.0
                }
            }
            Self::ConsonanceOnly => 1.0,
        }
    }

    fn switch_step(self) -> Option<usize> {
        match self {
            Self::DissonanceThenConsonance => Some(E2_PHASE_SWITCH_STEP),
            Self::ConsonanceOnly => None,
        }
    }
}

impl Experiment {
    fn label(self) -> &'static str {
        match self {
            Experiment::E1 => "E1",
            Experiment::E2 => "E2",
            Experiment::E3 => "E3",
            Experiment::E5 => "E5",
            Experiment::E6b => "E6b",
            Experiment::E7 => "E7",
        }
    }

    fn dir_name(self) -> &'static str {
        match self {
            Experiment::E1 => "e1",
            Experiment::E2 => "e2",
            Experiment::E3 => "e3",
            Experiment::E5 => "e5",
            Experiment::E6b => "e6b",
            Experiment::E7 => "e7",
        }
    }

    fn all() -> Vec<Experiment> {
        vec![
            Experiment::E1,
            Experiment::E2,
            Experiment::E3,
            Experiment::E5,
            Experiment::E6b,
            Experiment::E7,
        ]
    }

    fn paper_default() -> Vec<Experiment> {
        vec![
            Experiment::E1,
            Experiment::E2,
            Experiment::E3,
            Experiment::E6b,
            Experiment::E7,
        ]
    }
}

struct PaperRunLock {
    _file: std::fs::File,
}

impl PaperRunLock {
    fn acquire(path: &Path) -> io::Result<Self> {
        use std::os::unix::io::AsRawFd;
        unsafe extern "C" {
            fn flock(fd: i32, operation: i32) -> i32;
        }
        const LOCK_EX: i32 = 2;
        const LOCK_NB: i32 = 4;

        if let Some(parent) = path.parent() {
            create_dir_all(parent)?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(path)?;
        let ret = unsafe { flock(file.as_raw_fd(), LOCK_EX | LOCK_NB) };
        if ret != 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::WouldBlock {
                return Err(io::Error::other(format!(
                    "paper plots already running (lock held: {})",
                    path.display()
                )));
            }
            return Err(err);
        }
        Ok(Self { _file: file })
    }
}

fn usage() -> String {
    [
        "Usage: paper [--exp E1,E2,...] [--clean[=on|off]]",
        "Examples:",
        "  paper --exp 2",
        "  paper --exp all",
        "  paper 1 3 5",
        "  paper --clean --exp e2,e6b",
        "  paper --e2-render-baseline 5 4 2",
        "  paper --e2-render-nohill 5 4 2",
        "  paper --e2-render-proposal 5 4 2",
        "  paper --exp e2 --e2-diagnostics",
        "  paper --exp e6b",
        "  paper --exp e6b --e6b-quick",
        "  paper --exp e6b --e6b-juvenile off",
        "  paper --exp e6b --e6b-parent-proposal contextual_c --e6b-parent-proposal-sigma 4.9",
        "  paper --exp e6b --e6b-azimuth search",
        "  paper --exp e6b --e6b-seeds 6 --e6b-skip-benchmark",
        "  paper --exp e6b --e6b-quick --e6b-crowding 0.005 --e6b-capacity-weight 0.04",
        "  paper --exp e6b --e6b-quick --e6b-parent-share-weight 0.75",
        "  paper --exp e6b --e6b-quick --e6b-survival-low 0.15 --e6b-survival-high 0.55 --e6b-survival-recharge 0.20",
        "  paper --exp e6b --e6b-quick --e6b-respawn-parent-prior-mix 0.35",
        "  paper --exp e2 --e2-quick --e2-dense-sweep",
        "  paper --exp e2 --e2-quick --e2-candidate-search",
        "If no experiment is specified, paper defaults (E1,E2,E3,E6b,E7; paper order Exp. 1,2,4,3) run.",
        "Use --exp e6b to run the current paper Exp. 4 hereditary adaptation assay.",
        "--e6b-quick uses a small seed subset and skips the Exp1 benchmark for faster iteration.",
        "--e6b-seeds N limits E6b to the first N fixed seeds.",
        "--e6b-skip-benchmark omits the Exp1 reference rerun and writes a skipped note instead.",
        "--e6b-crowding X overrides the E6b polyphonic crowding weight.",
        "--e6b-capacity-weight X overrides the E6b local-capacity penalty weight.",
        "--e6b-capacity-radius X overrides the E6b local-capacity radius in cents.",
        "--e6b-capacity-free-voices N overrides the free voices allowed per local band.",
        "--e6b-parent-share-weight X scales how strongly crowded bands lose parent-choice weight.",
        "--e6b-juvenile on|off toggles newborn local settlement in E6b.",
        "--e6b-juvenile-ticks N overrides the E6b juvenile settlement window.",
        "--e6b-parent-proposal contextual_c|h_only selects the E6b parent-owned proposal density.",
        "--e6b-parent-proposal-sigma X sets the E6b parent-relative Gaussian width in semitones.",
        "--e6b-parent-proposal-candidates N sets how many parent-proposed families are filtered by the scene.",
        "--e6b-azimuth inherit|search chooses whether E6b inherits parent azimuth or searches within the chosen family.",
        "--e6b-survival-low X sets the score where E6b survival recharge starts ramping up.",
        "--e6b-survival-high X sets the score where E6b survival recharge saturates.",
        "--e6b-survival-recharge X overrides the maximum E6b survival recharge per second.",
        "--e6b-background-death X overrides the E6b background death rate per second.",
        "--e6b-respawn-parent-prior-mix X blends parent-profile weight into E6b hereditary respawn.",
        "--e6b-respawn-same-band-discount X scales same-band hereditary respawn probability.",
        "--e6b-respawn-octave-discount X scales octave-near hereditary respawn probability.",
        "E2 uses dissonance_then_consonance phase schedule.",
        "--e2-render-baseline N P [OCT] writes seed-0 baseline replay WAVs for N agents, P render partials, and optional OCT total range.",
        "--e2-render-nohill N P [OCT] writes seed-0 no-hill replay WAVs for N agents, P render partials, and optional OCT total range.",
        "--e2-render-proposal N P [OCT] writes a proposal-based replay using step + local/global peak + ratio candidates.",
        "--e2-diagnostics enables the full Fig. 2 supplementary control/terrain dump set (S7/S8 outputs).",
        "--e2-dense-sweep writes dense-regime baseline vs no-crowding diagnostics into experiments/plots/e2/.",
        "--e2-candidate-search writes 2oct/3oct/4oct baseline-vs-nohill candidate summaries and shortlist renders.",
        "Outputs are written to experiments/plots/<exp>/ (e.g. experiments/plots/e2).",
        "By default only selected experiment dirs are overwritten.",
        "Use --clean to clear experiments/plots before running.",
    ]
    .join("\n")
}

fn parse_experiments(args: &[String]) -> Result<Vec<Experiment>, String> {
    let mut values: Vec<String> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--clean" {
            if i + 1 < args.len() {
                let next = args[i + 1].as_str();
                if !next.starts_with('-') {
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        if arg.starts_with("--clean=") {
            i += 1;
            continue;
        }
        if arg == "--e2-render-baseline" {
            if i + 2 >= args.len() {
                return Err(format!("Missing values after {arg}\n{}", usage()));
            }
            i += 3;
            if i < args.len() && !args[i].starts_with('-') {
                i += 1;
            }
            continue;
        }
        if arg == "--e2-render-nohill" {
            if i + 2 >= args.len() {
                return Err(format!("Missing values after {arg}\n{}", usage()));
            }
            i += 3;
            if i < args.len() && !args[i].starts_with('-') {
                i += 1;
            }
            continue;
        }
        if arg == "--e2-render-proposal" {
            if i + 2 >= args.len() {
                return Err(format!("Missing values after {arg}\n{}", usage()));
            }
            i += 3;
            if i < args.len() && !args[i].starts_with('-') {
                i += 1;
            }
            continue;
        }
        if arg == "--e2-replay"
            || arg == "--e2-quick"
            || arg == "--e2-diagnostics"
            || arg == "--e2-dense-sweep"
            || arg == "--e2-candidate-search"
            || arg == "--e6b-quick"
            || arg == "--e6b-skip-benchmark"
        {
            i += 1;
            continue;
        }
        if arg == "--e6b-seeds"
            || arg == "--e6b-capacity-free-voices"
            || arg == "--e6b-parent-share-weight"
            || arg == "--e6b-juvenile"
            || arg == "--e6b-juvenile-ticks"
            || arg == "--e6b-parent-proposal"
            || arg == "--e6b-parent-proposal-sigma"
            || arg == "--e6b-parent-proposal-candidates"
            || arg == "--e6b-azimuth"
            || arg == "--e6b-survival-low"
            || arg == "--e6b-survival-high"
            || arg == "--e6b-survival-recharge"
            || arg == "--e6b-background-death"
            || arg == "--e6b-respawn-parent-prior-mix"
            || arg == "--e6b-respawn-same-band-discount"
            || arg == "--e6b-respawn-octave-discount"
        {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg.starts_with("--e6b-seeds=")
            || arg.starts_with("--e6b-crowding=")
            || arg.starts_with("--e6b-capacity-weight=")
            || arg.starts_with("--e6b-capacity-radius=")
            || arg.starts_with("--e6b-capacity-free-voices=")
            || arg.starts_with("--e6b-parent-share-weight=")
            || arg.starts_with("--e6b-juvenile=")
            || arg.starts_with("--e6b-juvenile-ticks=")
            || arg.starts_with("--e6b-parent-proposal=")
            || arg.starts_with("--e6b-parent-proposal-sigma=")
            || arg.starts_with("--e6b-parent-proposal-candidates=")
            || arg.starts_with("--e6b-azimuth=")
            || arg.starts_with("--e6b-survival-low=")
            || arg.starts_with("--e6b-survival-high=")
            || arg.starts_with("--e6b-survival-recharge=")
            || arg.starts_with("--e6b-background-death=")
            || arg.starts_with("--e6b-respawn-parent-prior-mix=")
            || arg.starts_with("--e6b-respawn-same-band-discount=")
            || arg.starts_with("--e6b-respawn-octave-discount=")
        {
            i += 1;
            continue;
        }
        if arg == "--e6b-crowding"
            || arg == "--e6b-capacity-weight"
            || arg == "--e6b-capacity-radius"
            || arg == "--e6b-parent-share-weight"
        {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            i += 2;
            continue;
        }
        if arg == "-e" || arg == "--exp" || arg == "--experiment" {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            values.push(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--exp=") {
            values.push(rest.to_string());
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--experiment=") {
            values.push(rest.to_string());
            i += 1;
            continue;
        }
        values.push(arg.to_string());
        i += 1;
    }

    if values.is_empty() {
        return Ok(Vec::new());
    }

    let mut experiments: Vec<Experiment> = Vec::new();
    let mut saw_all = false;
    for value in values {
        for token in value.split(',') {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            if token.eq_ignore_ascii_case("all") {
                saw_all = true;
                continue;
            }
            let exp = match token {
                "1" | "e1" | "E1" => Experiment::E1,
                "2" | "e2" | "E2" => Experiment::E2,
                "3" | "e3" | "E3" => Experiment::E3,
                "5" | "e5" | "E5" => Experiment::E5,
                "e6b" | "E6B" => Experiment::E6b,
                "7" | "e7" | "E7" => Experiment::E7,
                _ => {
                    return Err(format!("Unknown experiment '{token}'.\n{}", usage()));
                }
            };
            if !experiments.contains(&exp) {
                experiments.push(exp);
            }
        }
    }

    if saw_all {
        return Ok(Experiment::all());
    }

    Ok(experiments)
}

fn parse_e2_render_baseline(args: &[String]) -> Result<Option<(usize, u32, f32)>, String> {
    let Some(pos) = args.iter().position(|arg| arg == "--e2-render-baseline") else {
        return Ok(None);
    };
    if pos + 2 >= args.len() {
        return Err(format!(
            "Missing values after --e2-render-baseline\n{}",
            usage()
        ));
    }
    let n_agents = args[pos + 1].parse::<usize>().map_err(|_| {
        format!(
            "Invalid N for --e2-render-baseline: {}\n{}",
            args[pos + 1],
            usage()
        )
    })?;
    let partials = args[pos + 2].parse::<u32>().map_err(|_| {
        format!(
            "Invalid P for --e2-render-baseline: {}\n{}",
            args[pos + 2],
            usage()
        )
    })?;
    let octaves = if pos + 3 < args.len() && !args[pos + 3].starts_with('-') {
        args[pos + 3].parse::<f32>().map_err(|_| {
            format!(
                "Invalid OCT for --e2-render-baseline: {}\n{}",
                args[pos + 3],
                usage()
            )
        })?
    } else {
        2.0
    };
    Ok(Some((
        n_agents.max(1),
        partials.clamp(1, 16),
        octaves.clamp(0.5, 6.0),
    )))
}

fn parse_e2_render_nohill(args: &[String]) -> Result<Option<(usize, u32, f32)>, String> {
    let Some(pos) = args.iter().position(|arg| arg == "--e2-render-nohill") else {
        return Ok(None);
    };
    if pos + 2 >= args.len() {
        return Err(format!(
            "Missing values after --e2-render-nohill\n{}",
            usage()
        ));
    }
    let n_agents = args[pos + 1].parse::<usize>().map_err(|_| {
        format!(
            "Invalid N for --e2-render-nohill: {}\n{}",
            args[pos + 1],
            usage()
        )
    })?;
    let partials = args[pos + 2].parse::<u32>().map_err(|_| {
        format!(
            "Invalid P for --e2-render-nohill: {}\n{}",
            args[pos + 2],
            usage()
        )
    })?;
    let octaves = if pos + 3 < args.len() && !args[pos + 3].starts_with('-') {
        args[pos + 3].parse::<f32>().map_err(|_| {
            format!(
                "Invalid OCT for --e2-render-nohill: {}\n{}",
                args[pos + 3],
                usage()
            )
        })?
    } else {
        2.0
    };
    Ok(Some((
        n_agents.max(1),
        partials.clamp(1, 16),
        octaves.clamp(0.5, 6.0),
    )))
}

fn parse_e2_render_proposal(args: &[String]) -> Result<Option<(usize, u32, f32)>, String> {
    let Some(pos) = args.iter().position(|arg| arg == "--e2-render-proposal") else {
        return Ok(None);
    };
    if pos + 2 >= args.len() {
        return Err(format!(
            "Missing values after --e2-render-proposal\n{}",
            usage()
        ));
    }
    let n_agents = args[pos + 1].parse::<usize>().map_err(|_| {
        format!(
            "Invalid N for --e2-render-proposal: {}\n{}",
            args[pos + 1],
            usage()
        )
    })?;
    let partials = args[pos + 2].parse::<u32>().map_err(|_| {
        format!(
            "Invalid P for --e2-render-proposal: {}\n{}",
            args[pos + 2],
            usage()
        )
    })?;
    let octaves = if pos + 3 < args.len() && !args[pos + 3].starts_with('-') {
        args[pos + 3].parse::<f32>().map_err(|_| {
            format!(
                "Invalid OCT for --e2-render-proposal: {}\n{}",
                args[pos + 3],
                usage()
            )
        })?
    } else {
        2.0
    };
    Ok(Some((
        n_agents.max(1),
        partials.clamp(1, 16),
        octaves.clamp(0.5, 6.0),
    )))
}

fn parse_u32_cli_opt(args: &[String], key: &str) -> Result<Option<u32>, String> {
    let mut value: Option<String> = None;
    let key_eq = format!("{key}=");
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == key {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix(&key_eq) {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    let parsed = raw
        .parse::<u32>()
        .map_err(|_| format!("Invalid value for {key}: '{raw}'\n{}", usage()))?;
    Ok(Some(parsed))
}

fn parse_f32_cli_opt(args: &[String], key: &str) -> Result<Option<f32>, String> {
    let mut value: Option<String> = None;
    let key_eq = format!("{key}=");
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == key {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix(&key_eq) {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    let parsed = raw
        .parse::<f32>()
        .map_err(|_| format!("Invalid value for {key}: '{raw}'\n{}", usage()))?;
    if !parsed.is_finite() {
        return Err(format!("Invalid value for {key}: '{raw}'\n{}", usage()));
    }
    Ok(Some(parsed))
}

fn parse_usize_cli_opt(args: &[String], key: &str) -> Result<Option<usize>, String> {
    let value = parse_u32_cli_opt(args, key)?;
    value
        .map(|v| {
            usize::try_from(v).map_err(|_| format!("Invalid value for {key}: {v}\n{}", usage()))
        })
        .transpose()
}

fn parse_e6b_seed_limit(args: &[String]) -> Result<Option<usize>, String> {
    parse_usize_cli_opt(args, "--e6b-seeds")
}

fn parse_e6b_crowding_weight(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-crowding")
}

fn parse_e6b_capacity_weight(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-capacity-weight")
}

fn parse_e6b_capacity_radius(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-capacity-radius")
}

fn parse_e6b_capacity_free_voices(args: &[String]) -> Result<Option<usize>, String> {
    parse_usize_cli_opt(args, "--e6b-capacity-free-voices")
}

fn parse_e6b_parent_share_weight(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-parent-share-weight")
}

fn parse_e6b_parent_energy_weight(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-parent-energy-weight")
}

fn parse_e6b_juvenile_enabled(args: &[String]) -> Result<Option<bool>, String> {
    parse_on_off_cli_opt(args, "--e6b-juvenile")
}

fn parse_e6b_juvenile_ticks(args: &[String]) -> Result<Option<u32>, String> {
    parse_u32_cli_opt(args, "--e6b-juvenile-ticks")
}

fn parse_e6b_parent_proposal_kind(args: &[String]) -> Result<Option<E6ParentProposalKind>, String> {
    let mut value: Option<String> = None;
    let key = "--e6b-parent-proposal";
    let key_eq = format!("{key}=");
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == key {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix(&key_eq) {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "contextual_c" | "contextualc" | "c" => Ok(Some(E6ParentProposalKind::ContextualC)),
        "h_only" | "honly" | "h" => Ok(Some(E6ParentProposalKind::HOnly)),
        _ => Err(format!(
            "Invalid --e6b-parent-proposal value '{raw}'. Use contextual_c|h_only.\n{}",
            usage()
        )),
    }
}

fn parse_e6b_parent_proposal_sigma(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-parent-proposal-sigma")
}

fn parse_e6b_parent_proposal_candidates(args: &[String]) -> Result<Option<usize>, String> {
    parse_usize_cli_opt(args, "--e6b-parent-proposal-candidates")
}

fn parse_e6b_azimuth_mode(args: &[String]) -> Result<Option<E6bAzimuthMode>, String> {
    let mut value: Option<String> = None;
    let key = "--e6b-azimuth";
    let key_eq = format!("{key}=");
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == key {
            if i + 1 >= args.len() {
                return Err(format!("Missing value after {arg}\n{}", usage()));
            }
            value = Some(args[i + 1].clone());
            i += 2;
            continue;
        }
        if let Some(rest) = arg.strip_prefix(&key_eq) {
            value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }
    let Some(raw) = value else {
        return Ok(None);
    };
    match raw.to_ascii_lowercase().as_str() {
        "inherit" | "inherited" => Ok(Some(E6bAzimuthMode::Inherit)),
        "search" | "local_search" | "local-search" => Ok(Some(E6bAzimuthMode::LocalSearch)),
        _ => Err(format!(
            "Invalid --e6b-azimuth value '{raw}'. Use inherit|search.\n{}",
            usage()
        )),
    }
}

fn parse_e6b_survival_low(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-survival-low")
}

fn parse_e6b_survival_high(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-survival-high")
}

fn parse_e6b_survival_recharge(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-survival-recharge")
}

fn parse_e6b_background_death(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-background-death")
}

fn parse_e6b_respawn_parent_prior_mix(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-respawn-parent-prior-mix")
}

fn parse_e6b_respawn_same_band_discount(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-respawn-same-band-discount")
}

fn parse_e6b_respawn_octave_discount(args: &[String]) -> Result<Option<f32>, String> {
    parse_f32_cli_opt(args, "--e6b-respawn-octave-discount")
}

fn parse_clean(args: &[String]) -> Result<bool, String> {
    let mut clean_flag = false;
    let mut clean_value: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let arg = args[i].as_str();
        if arg == "--clean" {
            clean_flag = true;
            if i + 1 < args.len() {
                let next = args[i + 1].as_str();
                if !next.starts_with('-') {
                    clean_value = Some(next.to_string());
                    i += 2;
                    continue;
                }
            }
            i += 1;
            continue;
        }
        if let Some(rest) = arg.strip_prefix("--clean=") {
            clean_value = Some(rest.to_string());
            i += 1;
            continue;
        }
        i += 1;
    }

    let Some(value) = clean_value else {
        return Ok(clean_flag);
    };
    let normalized = value.to_ascii_lowercase();
    match normalized.as_str() {
        "on" | "true" | "1" | "yes" => Ok(true),
        "off" | "false" | "0" | "no" => Ok(false),
        _ => Err(format!(
            "Invalid --clean value '{value}'. Use on/off.\n{}",
            usage()
        )),
    }
}

fn prepare_paper_output_dirs(
    base_dir: &Path,
    experiments: &[Experiment],
    clear_existing_selected: bool,
) -> io::Result<Vec<(Experiment, PathBuf)>> {
    if experiments.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(experiments.len());
    for &exp in experiments {
        let dir = base_dir.join(exp.dir_name());
        if clear_existing_selected && dir.exists() {
            remove_dir_all(&dir)?;
        }
        create_dir_all(&dir)?;
        out.push((exp, dir));
    }
    Ok(out)
}

pub(crate) fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!("{}", usage());
        return Ok(());
    }

    if let Some((n_agents, render_partials, range_oct)) =
        parse_e2_render_baseline(&args).map_err(io::Error::other)?
    {
        generate_e2_baseline_render(n_agents, render_partials, range_oct)?;
        return Ok(());
    }
    if let Some((n_agents, render_partials, range_oct)) =
        parse_e2_render_nohill(&args).map_err(io::Error::other)?
    {
        generate_e2_condition_render_named(
            E2Condition::NoHillClimb,
            "nohill",
            n_agents,
            render_partials,
            range_oct,
            "15_exp1_replay",
        )?;
        return Ok(());
    }
    if let Some((n_agents, render_partials, range_oct)) =
        parse_e2_render_proposal(&args).map_err(io::Error::other)?
    {
        generate_e2_proposal_render(n_agents, render_partials, range_oct)?;
        return Ok(());
    }
    // Handle --e2-replay: generate WAV with pure sine replay and exit
    if args.iter().any(|arg| arg == "--e2-replay") {
        let out_path = PathBuf::from("supplementary_audio/audio/15_exp1_replay.wav");
        generate_e2_replay_wav(&out_path)?;
        return Ok(());
    }

    // Handle --audio-rhai: generate Rhai replay scripts and exit
    if args.iter().any(|arg| arg == "--audio-rhai") {
        generate_audio_replay_rhai()?;
        return Ok(());
    }
    // Handle --e3-audio: generate temporal-scaffold monitor scenarios and
    // refresh the paper-facing WAVs via the internal renderer path.
    if args.iter().any(|arg| arg == "--e3-audio") {
        let cfg_shared = crate::sim::E3AudioConfig::default_shared();
        let cfg_scrambled = crate::sim::E3AudioConfig::default_scrambled();
        let cfg_off = crate::sim::E3AudioConfig::default_off();
        let scenario_dir = PathBuf::from("supplementary_audio/scenarios");
        let audio_dir = PathBuf::from("supplementary_audio/audio");
        crate::sim::generate_e3_rhai(
            &cfg_shared,
            &scenario_dir.join("temporal_scaffold_shared.rhai"),
        )?;
        crate::sim::generate_e3_rhai(
            &cfg_scrambled,
            &scenario_dir.join("temporal_scaffold_scrambled.rhai"),
        )?;
        crate::sim::generate_e3_rhai(&cfg_off, &scenario_dir.join("temporal_scaffold_off.rhai"))?;
        crate::sim::render_e3_audio(&cfg_shared, &audio_dir.join("temporal_scaffold_shared.wav"))?;
        crate::sim::render_e3_audio(
            &cfg_scrambled,
            &audio_dir.join("temporal_scaffold_scrambled.wav"),
        )?;
        crate::sim::render_e3_audio(&cfg_off, &audio_dir.join("temporal_scaffold_off.wav"))?;
        return Ok(());
    }
    if args
        .iter()
        .any(|arg| arg == "--postprocess-quicklisten-showcase")
    {
        postprocess_quicklisten_showcase_wav_default()?;
        return Ok(());
    }
    if args
        .iter()
        .any(|arg| arg == "--postprocess-quicklisten-controls")
    {
        postprocess_quicklisten_controls_wav_default()?;
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--postprocess-e6b") {
        postprocess_e6b_wav_default()?;
        return Ok(());
    }
    if args
        .iter()
        .any(|arg| arg == "--postprocess-hereditary-controls")
    {
        postprocess_hereditary_controls_wav_default()?;
        return Ok(());
    }
    if args.iter().any(|arg| arg == "--postprocess-polyphony") {
        postprocess_polyphony_wav_default()?;
        return Ok(());
    }
    if args
        .iter()
        .any(|arg| arg == "--postprocess-polyphony-no-hill")
    {
        postprocess_polyphony_no_hill_wav_default()?;
        return Ok(());
    }

    let e6b_seed_limit = parse_e6b_seed_limit(&args).map_err(io::Error::other)?;
    let e6b_crowding_weight = parse_e6b_crowding_weight(&args).map_err(io::Error::other)?;
    let e6b_capacity_weight = parse_e6b_capacity_weight(&args).map_err(io::Error::other)?;
    let e6b_capacity_radius = parse_e6b_capacity_radius(&args).map_err(io::Error::other)?;
    let e6b_capacity_free_voices =
        parse_e6b_capacity_free_voices(&args).map_err(io::Error::other)?;
    let e6b_parent_share_weight = parse_e6b_parent_share_weight(&args).map_err(io::Error::other)?;
    let e6b_parent_energy_weight =
        parse_e6b_parent_energy_weight(&args).map_err(io::Error::other)?;
    let e6b_juvenile_enabled = parse_e6b_juvenile_enabled(&args).map_err(io::Error::other)?;
    let e6b_juvenile_ticks = parse_e6b_juvenile_ticks(&args).map_err(io::Error::other)?;
    let e6b_parent_proposal_kind =
        parse_e6b_parent_proposal_kind(&args).map_err(io::Error::other)?;
    let e6b_parent_proposal_sigma =
        parse_e6b_parent_proposal_sigma(&args).map_err(io::Error::other)?;
    let e6b_parent_proposal_candidates =
        parse_e6b_parent_proposal_candidates(&args).map_err(io::Error::other)?;
    let e6b_azimuth_mode = parse_e6b_azimuth_mode(&args).map_err(io::Error::other)?;
    let e6b_survival_low = parse_e6b_survival_low(&args).map_err(io::Error::other)?;
    let e6b_survival_high = parse_e6b_survival_high(&args).map_err(io::Error::other)?;
    let e6b_survival_recharge = parse_e6b_survival_recharge(&args).map_err(io::Error::other)?;
    let e6b_background_death = parse_e6b_background_death(&args).map_err(io::Error::other)?;
    let e6b_respawn_parent_prior_mix =
        parse_e6b_respawn_parent_prior_mix(&args).map_err(io::Error::other)?;
    let e6b_respawn_same_band_discount =
        parse_e6b_respawn_same_band_discount(&args).map_err(io::Error::other)?;
    let e6b_respawn_octave_discount =
        parse_e6b_respawn_octave_discount(&args).map_err(io::Error::other)?;
    let e2_phase_mode = E2PhaseMode::DissonanceThenConsonance;
    let e2_quick = args.iter().any(|a| a == "--e2-quick");
    let e2_diagnostics = args.iter().any(|a| a == "--e2-diagnostics");
    let e2_dense_sweep = args.iter().any(|a| a == "--e2-dense-sweep");
    let e2_candidate_search = args.iter().any(|a| a == "--e2-candidate-search");
    let e6b_quick = args.iter().any(|a| a == "--e6b-quick");
    let e6b_skip_benchmark = args.iter().any(|a| a == "--e6b-skip-benchmark") || e6b_quick;
    let e6b_cli = E6bCliOptions {
        quick: e6b_quick,
        seed_limit: e6b_seed_limit,
        skip_benchmark: e6b_skip_benchmark,
        crowding_weight: e6b_crowding_weight,
        capacity_weight: e6b_capacity_weight,
        capacity_radius_cents: e6b_capacity_radius,
        capacity_free_voices: e6b_capacity_free_voices,
        parent_share_weight: e6b_parent_share_weight,
        parent_energy_weight: e6b_parent_energy_weight,
        juvenile_enabled: e6b_juvenile_enabled,
        juvenile_ticks: e6b_juvenile_ticks,
        parent_proposal_kind: e6b_parent_proposal_kind,
        parent_proposal_sigma_st: e6b_parent_proposal_sigma,
        parent_proposal_candidate_count: e6b_parent_proposal_candidates,
        azimuth_mode: e6b_azimuth_mode,
        survival_score_low: e6b_survival_low,
        survival_score_high: e6b_survival_high,
        survival_recharge_per_sec: e6b_survival_recharge,
        background_death_rate_per_sec: e6b_background_death,
        respawn_parent_prior_mix: e6b_respawn_parent_prior_mix,
        respawn_same_band_discount: e6b_respawn_same_band_discount,
        respawn_octave_discount: e6b_respawn_octave_discount,
    };
    let clean_all = parse_clean(&args).map_err(io::Error::other)?;
    let experiments = parse_experiments(&args).map_err(io::Error::other)?;
    let experiments = if experiments.is_empty() {
        Experiment::paper_default()
    } else {
        experiments
    };

    let lock_path = Path::new(PAPER_PLOTS_LOCK_FILE);
    let _run_lock = PaperRunLock::acquire(lock_path)?;

    let base_dir = Path::new(PAPER_PLOTS_BASE_DIR);
    debug_assert!(
        base_dir.ends_with(Path::new("experiments/plots")),
        "refusing to clear unexpected path: {}",
        base_dir.display()
    );
    if clean_all && base_dir.exists() {
        remove_dir_all(base_dir)?;
    }
    create_dir_all(base_dir)?;
    let experiment_dirs = prepare_paper_output_dirs(base_dir, &experiments, !clean_all)?;

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);

    let space_ref = &space;
    std::thread::scope(|s| -> Result<(), Box<dyn Error>> {
        let mut handles = Vec::new();
        for (exp, out_dir) in &experiment_dirs {
            let exp = *exp;
            let out_dir = out_dir.as_path();
            match exp {
                Experiment::E1 => {
                    let h = s.spawn(|| {
                        plot_e1_landscape_scan(out_dir, space_ref, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E2 => {
                    let h = s.spawn(move || {
                        plot_e2_emergent_harmony(
                            out_dir,
                            space_ref,
                            anchor_hz,
                            e2_phase_mode,
                            e2_quick,
                            e2_diagnostics,
                            e2_dense_sweep,
                            e2_candidate_search,
                        )
                        .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E3 => {
                    let h = s.spawn(|| {
                        plot_e3_metabolic_selection(out_dir, space_ref, anchor_hz)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E5 => {
                    let h = s.spawn(|| {
                        plot_e5_vitality_entrainment(out_dir)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E6b => {
                    let h = s.spawn(|| {
                        plot_e6b_hereditary_polyphony(out_dir, space_ref, anchor_hz, e6b_cli)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
                Experiment::E7 => {
                    let h = s.spawn(|| {
                        plot_e7_temporal_scaffold(out_dir)
                            .map_err(|err| io::Error::other(err.to_string()))
                    });
                    handles.push((exp.label(), h));
                }
            }
        }

        let mut first_err: Option<io::Error> = None;
        for (label, handle) in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => {
                    if first_err.is_none() {
                        first_err = Some(err);
                    }
                }
                Err(_) => {
                    if first_err.is_none() {
                        first_err = Some(io::Error::other(format!("{label} thread panicked")));
                    }
                }
            }
        }
        if let Some(err) = first_err {
            return Err(Box::new(err));
        }
        Ok(())
    })?;

    println!("Saved paper plots to {}", base_dir.display());
    Ok(())
}

fn plot_e1_landscape_scan(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let anchor_idx = space.nearest_index(anchor_hz);

    let (_erb_scan, du_scan) = erb_grid(space);
    let mut anchor_density_scan = vec![0.0f32; space.n_bins()];
    let denom = du_scan[anchor_idx].max(1e-12);
    anchor_density_scan[anchor_idx] = 1.0 / denom;

    let mut anchor_env_scan = vec![0.0f32; space.n_bins()];
    anchor_env_scan[anchor_idx] = 1.0;

    space.assert_scan_len_named(&anchor_density_scan, "anchor_density_scan");
    space.assert_scan_len_named(&anchor_env_scan, "anchor_env_scan");

    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let mut h_params_base = HarmonicityParams::default();
    h_params_base.gamma_root = 1.0;
    h_params_base.gamma_overtone = 1.0;
    h_params_base.rho_common_overtone = h_params_base.rho_common_root;
    h_params_base.mirror_weight = 0.5;
    let harmonicity_kernel = HarmonicityKernel::new(space, h_params_base);
    let mut h_params_m0 = h_params_base;
    h_params_m0.mirror_weight = 0.0;
    let mut h_params_m05 = h_params_base;
    h_params_m05.mirror_weight = 0.5;
    let mut h_params_m1 = h_params_base;
    h_params_m1.mirror_weight = 1.0;
    let harmonicity_kernel_m0 = HarmonicityKernel::new(space, h_params_m0);
    let harmonicity_kernel_m05 = HarmonicityKernel::new(space, h_params_m05);
    let harmonicity_kernel_m1 = HarmonicityKernel::new(space, h_params_m1);

    let (perc_h_pot_scan, _) =
        harmonicity_kernel.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m0, _) =
        harmonicity_kernel_m0.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m05, _) =
        harmonicity_kernel_m05.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_h_pot_scan_m1, _) =
        harmonicity_kernel_m1.potential_h_from_log2_spectrum(&anchor_env_scan, space);
    let (perc_r_pot_scan, _) =
        roughness_kernel.potential_r_from_log2_spectrum_density(&anchor_density_scan, space);

    space.assert_scan_len_named(&perc_h_pot_scan, "perc_h_pot_scan");
    space.assert_scan_len_named(&perc_h_pot_scan_m0, "perc_h_pot_scan_m0");
    space.assert_scan_len_named(&perc_h_pot_scan_m05, "perc_h_pot_scan_m05");
    space.assert_scan_len_named(&perc_h_pot_scan_m1, "perc_h_pot_scan_m1");
    space.assert_scan_len_named(&perc_r_pot_scan, "perc_r_pot_scan");

    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        roughness_kernel: roughness_kernel.clone(),
        harmonicity_kernel: harmonicity_kernel.clone(),
        consonance_kernel: ConsonanceKernel::default(),
        consonance_representation: ConsonanceRepresentationParams {
            beta: E2_C_LEVEL_BETA,
            theta: E2_C_LEVEL_THETA,
        },
        consonance_density_roughness_gain: 1.0,
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        loudness_exp: 1.0,
        ref_power: 1.0,
        tau_ms: 1.0,
        roughness_k: 1.0,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };

    let r_ref = psycho_state::compute_roughness_reference(&params, space);
    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        r_ref.peak,
        params.roughness_k,
        &mut perc_r_state01_scan,
    );

    let h_ref_max = 1.0f32;
    let mut perc_h_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::h_pot_scan_to_h_state01_scan(
        &perc_h_pot_scan,
        h_ref_max,
        &mut perc_h_state01_scan,
    );

    let mut perc_c_field_scan = vec![0.0f32; space.n_bins()];
    let mut perc_c_density_scan = vec![0.0f32; space.n_bins()];
    let density_kernel =
        ConsonanceKernel::density_with_rho(params.consonance_density_roughness_gain);
    for i in 0..space.n_bins() {
        let h01 = perc_h_state01_scan[i];
        let r01 = perc_r_state01_scan[i];
        let c_field = params.consonance_kernel.score(h01, r01);
        perc_c_field_scan[i] = if c_field.is_finite() { c_field } else { 0.0 };
        let c_density_raw = density_kernel.score(h01, r01).max(0.0);
        perc_c_density_scan[i] = if c_density_raw.is_finite() {
            c_density_raw
        } else {
            0.0
        };
    }

    let anchor_log2 = anchor_hz.log2();
    let log2_ratio_scan: Vec<f32> = space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect();

    space.assert_scan_len_named(&perc_r_state01_scan, "perc_r_state01_scan");
    space.assert_scan_len_named(&perc_h_state01_scan, "perc_h_state01_scan");
    space.assert_scan_len_named(&perc_c_field_scan, "perc_c_field_scan");
    space.assert_scan_len_named(&perc_c_density_scan, "perc_c_density_scan");
    space.assert_scan_len_named(&log2_ratio_scan, "log2_ratio_scan");

    let out_path = out_dir.join("paper_e1_landscape_scan_anchor220.svg");
    render_e1_plot(
        &out_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan,
        &perc_r_state01_scan,
        &perc_c_field_scan,
        &perc_c_density_scan,
    )?;
    let h_triplet_path = out_dir.join("paper_e1_h_mirror_m0_m05_m1.svg");
    render_e1_h_mirror_triplet_plot(
        &h_triplet_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan_m0,
        &perc_h_pot_scan_m05,
        &perc_h_pot_scan_m1,
    )?;
    let h_diff_path = out_dir.join("paper_e1_h_mirror_m0_minus_m1.svg");
    render_e1_h_mirror_diff_plot(
        &h_diff_path,
        anchor_hz,
        &log2_ratio_scan,
        &perc_h_pot_scan_m0,
        &perc_h_pot_scan_m1,
    )?;

    // --- B3: Anchor robustness (110 Hz, 440 Hz) ---
    // Run the same consonance scan at additional anchors and compare peak positions.
    {
        /// Find local-max bins of c_field within ±1 octave of anchor, return as (ratio, c_value).
        fn find_peaks(c_field: &[f32], log2_ratios: &[f32]) -> Vec<(f32, f32)> {
            let mut peaks = Vec::new();
            for i in 1..c_field.len() - 1 {
                let r = log2_ratios[i];
                if r.abs() > 1.0 {
                    continue; // within ±1 octave
                }
                if c_field[i] > c_field[i - 1] && c_field[i] > c_field[i + 1] && c_field[i] > 0.1 {
                    peaks.push((r, c_field[i]));
                }
            }
            peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            peaks.truncate(10); // top-10 peaks
            peaks
        }

        #[allow(clippy::too_many_arguments)]
        fn scan_at_anchor(
            space: &Log2Space,
            alt_hz: f32,
            roughness_kernel: &RoughnessKernel,
            harmonicity_kernel: &HarmonicityKernel,
            consonance_kernel: &ConsonanceKernel,
            r_ref_peak: f32,
            roughness_k: f32,
            h_ref_max: f32,
        ) -> Vec<(f32, f32)> {
            let alt_idx = space.nearest_index(alt_hz);
            let (_erb, du) = erb_grid(space);
            let mut density = vec![0.0f32; space.n_bins()];
            density[alt_idx] = 1.0 / du[alt_idx].max(1e-12);
            let mut env = vec![0.0f32; space.n_bins()];
            env[alt_idx] = 1.0;
            let (h_pot, _) = harmonicity_kernel.potential_h_from_log2_spectrum(&env, space);
            let (r_pot, _) =
                roughness_kernel.potential_r_from_log2_spectrum_density(&density, space);
            let mut r01 = vec![0.0f32; space.n_bins()];
            psycho_state::r_pot_scan_to_r_state01_scan(&r_pot, r_ref_peak, roughness_k, &mut r01);
            let mut h01 = vec![0.0f32; space.n_bins()];
            psycho_state::h_pot_scan_to_h_state01_scan(&h_pot, h_ref_max, &mut h01);
            let mut c_field = vec![0.0f32; space.n_bins()];
            for i in 0..space.n_bins() {
                let c = consonance_kernel.score(h01[i], r01[i]);
                c_field[i] = if c.is_finite() { c } else { 0.0 };
            }
            let alt_log2 = alt_hz.log2();
            let ratios: Vec<f32> = space.centers_log2.iter().map(|&l| l - alt_log2).collect();
            find_peaks(&c_field, &ratios)
        }

        let peaks_220 = find_peaks(&perc_c_field_scan, &log2_ratio_scan);

        let peaks_110 = scan_at_anchor(
            space,
            110.0,
            &roughness_kernel,
            &harmonicity_kernel,
            &params.consonance_kernel,
            r_ref.peak,
            params.roughness_k,
            h_ref_max,
        );
        let peaks_440 = scan_at_anchor(
            space,
            440.0,
            &roughness_kernel,
            &harmonicity_kernel,
            &params.consonance_kernel,
            r_ref.peak,
            params.roughness_k,
            h_ref_max,
        );

        // Compare: for each 220 Hz peak, find nearest peak in alt scan and report shift in cents
        fn compare_peaks(ref_peaks: &[(f32, f32)], alt_peaks: &[(f32, f32)]) -> Vec<f64> {
            let mut shifts = Vec::new();
            for &(r_ref, _) in ref_peaks {
                if let Some(&(r_alt, _)) = alt_peaks.iter().min_by(|a, b| {
                    (a.0 - r_ref)
                        .abs()
                        .partial_cmp(&(b.0 - r_ref).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    // shift in cents: (r_alt - r_ref) * 1200
                    shifts.push(((r_alt - r_ref) as f64) * 1200.0);
                }
            }
            shifts
        }

        let shifts_110 = compare_peaks(&peaks_220, &peaks_110);
        let shifts_440 = compare_peaks(&peaks_220, &peaks_440);

        let max_shift_110 = shifts_110.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let max_shift_440 = shifts_440.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        let overall_max = max_shift_110.max(max_shift_440);

        let mut report = String::new();
        report.push_str("=== E1 Anchor Robustness ===\n\n");
        report.push_str(&format!(
            "Reference: {} Hz ({} peaks found)\n",
            anchor_hz,
            peaks_220.len()
        ));
        report.push_str("Peaks (log2 ratio, c_field):\n");
        for (r, c) in &peaks_220 {
            report.push_str(&format!(
                "  ratio={:.4} ({:.1} cents), c={:.4}\n",
                r,
                r * 1200.0,
                c
            ));
        }
        report.push('\n');
        for (alt_hz_val, alt_peaks, shifts) in [
            (110.0, &peaks_110, &shifts_110),
            (440.0, &peaks_440, &shifts_440),
        ] {
            report.push_str(&format!(
                "Anchor {} Hz ({} peaks):\n",
                alt_hz_val,
                alt_peaks.len()
            ));
            for (r, c) in alt_peaks {
                report.push_str(&format!(
                    "  ratio={:.4} ({:.1} cents), c={:.4}\n",
                    r,
                    r * 1200.0,
                    c
                ));
            }
            let max_s = shifts.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
            report.push_str(&format!(
                "  max peak shift vs 220 Hz: {:.1} cents\n\n",
                max_s
            ));
        }
        report.push_str(&format!(
            "Overall max peak shift: {:.1} cents\n",
            overall_max
        ));

        write_with_log(out_dir.join("paper_e1_anchor_robustness.txt"), report)?;
    }

    Ok(())
}

fn plot_e2_emergent_harmony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
    quick: bool,
    diagnostics: bool,
    dense_sweep: bool,
    candidate_search: bool,
) -> Result<(), Box<dyn Error>> {
    let seeds = e2_dense_seed_slice(quick);
    let max_worker_threads = if quick { Some(4) } else { None };
    let (
        baseline_runs,
        baseline_stats,
        nohill_runs,
        nohill_stats,
        norep_runs,
        norep_stats,
        shuffled_runs,
    ) = std::thread::scope(|scope| {
        let baseline_handle = scope.spawn(|| {
            e2_seed_sweep_with_threads_for_seeds(
                space,
                anchor_hz,
                E2Condition::Baseline,
                E2_STEP_SEMITONES,
                phase_mode,
                None,
                0,
                E2_PAPER_N_AGENTS,
                E2_PAPER_RANGE_OCT,
                seeds,
                max_worker_threads,
            )
        });
        let nohill_handle = scope.spawn(|| {
            e2_seed_sweep_with_threads_for_seeds(
                space,
                anchor_hz,
                E2Condition::NoHillClimb,
                E2_STEP_SEMITONES,
                phase_mode,
                None,
                0,
                E2_PAPER_N_AGENTS,
                E2_PAPER_RANGE_OCT,
                seeds,
                max_worker_threads,
            )
        });
        let norep_handle = scope.spawn(|| {
            e2_seed_sweep_with_threads_for_seeds(
                space,
                anchor_hz,
                E2Condition::NoCrowding,
                E2_STEP_SEMITONES,
                phase_mode,
                None,
                0,
                E2_PAPER_N_AGENTS,
                E2_PAPER_RANGE_OCT,
                seeds,
                max_worker_threads,
            )
        });
        let shuffled_handle = diagnostics.then(|| {
            scope.spawn(|| {
                e2_seed_sweep_with_threads_for_seeds(
                    space,
                    anchor_hz,
                    E2Condition::ShuffledLandscape,
                    E2_STEP_SEMITONES,
                    phase_mode,
                    None,
                    0,
                    E2_PAPER_N_AGENTS,
                    E2_PAPER_RANGE_OCT,
                    seeds,
                    max_worker_threads,
                )
            })
        });
        let (baseline_runs, baseline_stats) = baseline_handle
            .join()
            .expect("baseline seed sweep thread panicked");
        let (nohill_runs, nohill_stats) = nohill_handle
            .join()
            .expect("nohill seed sweep thread panicked");
        let (norep_runs, norep_stats) = norep_handle
            .join()
            .expect("norep seed sweep thread panicked");
        let shuffled_runs = shuffled_handle.map(|handle| {
            handle
                .join()
                .expect("shuffled seed sweep thread panicked")
                .0
        });
        (
            baseline_runs,
            baseline_stats,
            nohill_runs,
            nohill_stats,
            norep_runs,
            norep_stats,
            shuffled_runs,
        )
    });
    let rep_index = baseline_runs
        .iter()
        .position(|run| run.seed == E2_SEEDS[E2_PAPER_REP_SEED_INDEX])
        .unwrap_or_else(|| pick_representative_run_index(&baseline_runs));
    let baseline_run = &baseline_runs[rep_index];
    let rep_seed = baseline_run.seed;
    let nohill_rep = &nohill_runs[nohill_runs
        .iter()
        .position(|run| run.seed == rep_seed)
        .unwrap_or_else(|| pick_representative_run_index(&nohill_runs))];
    let marker_steps = e2_marker_steps(phase_mode);
    let caption_suffix = e2_caption_suffix(phase_mode);
    let post_label = e2_post_label();
    let post_label_title = e2_post_label_title();
    let baseline_ci95_c_level = std_series_to_ci95(&baseline_stats.std_c_level, baseline_stats.n);
    let nohill_ci95_c_level = std_series_to_ci95(&nohill_stats.std_c_level, nohill_stats.n);
    let norep_ci95_c_level = std_series_to_ci95(&norep_stats.std_c_level, norep_stats.n);
    let baseline_ci95_c = std_series_to_ci95(&baseline_stats.std_c_score_loo, baseline_stats.n);
    let baseline_ci95_g_scene = std_series_to_ci95(&baseline_stats.std_g_scene, baseline_stats.n);
    let nohill_ci95_g_scene = std_series_to_ci95(&nohill_stats.std_g_scene, nohill_stats.n);

    write_with_log(
        out_dir.join("paper_e2_representative_seed.txt"),
        representative_seed_text(&baseline_runs, rep_index, phase_mode),
    )?;

    write_with_log(
        out_dir.join("paper_e2_meta.txt"),
        e2_meta_text(
            baseline_run.n_agents,
            baseline_run.fixed_drone_hz,
            baseline_run.k_bins,
            baseline_run.density_mass_mean,
            baseline_run.density_mass_min,
            baseline_run.density_mass_max,
            baseline_run.r_ref_peak,
            baseline_run.roughness_k,
            baseline_run.roughness_ref_eps,
            baseline_run.r_state01_min,
            baseline_run.r_state01_mean,
            baseline_run.r_state01_max,
            phase_mode,
        ),
    )?;

    if diagnostics {
        let c_score_csv = series_csv("step,mean_c_score", &baseline_run.mean_c_series);
        write_with_log(out_dir.join("paper_e2_c_score_timeseries.csv"), c_score_csv)?;
        write_with_log(
            out_dir.join("paper_e2_c_level_timeseries.csv"),
            series_csv("step,mean_c_level", &baseline_run.mean_c_level_series),
        )?;
        write_with_log(
            out_dir.join("paper_e2_mean_c_score_loo_over_time.csv"),
            series_csv(
                "step,mean_c_score_loo",
                &baseline_run.mean_c_score_loo_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.csv"),
            series_csv(
                "step,mean_c_score_chosen_loo",
                &baseline_run.mean_c_score_chosen_loo_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_score_timeseries.csv"),
            series_csv("step,mean_score", &baseline_run.mean_score_series),
        )?;
        write_with_log(
            out_dir.join("paper_e2_crowding_timeseries.csv"),
            series_csv("step,mean_crowding", &baseline_run.mean_crowding_series),
        )?;
        write_with_log(
            out_dir.join("paper_e2_moved_frac_timeseries.csv"),
            series_csv("step,moved_frac", &baseline_run.moved_frac_series),
        )?;
        write_with_log(
            out_dir.join("paper_e2_accepted_worse_frac_timeseries.csv"),
            series_csv(
                "step,accepted_worse_frac",
                &baseline_run.accepted_worse_frac_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_attempted_update_frac_timeseries.csv"),
            series_csv(
                "step,attempted_update_frac",
                &baseline_run.attempted_update_frac_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_moved_given_attempt_frac_timeseries.csv"),
            series_csv(
                "step,moved_given_attempt_frac",
                &baseline_run.moved_given_attempt_frac_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.csv"),
            series_csv(
                "step,mean_abs_delta_semitones",
                &baseline_run.mean_abs_delta_semitones_series,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.csv"),
            series_csv(
                "step,mean_abs_delta_semitones_moved",
                &baseline_run.mean_abs_delta_semitones_moved_series,
            ),
        )?;

        write_with_log(
            out_dir.join("paper_e2_agent_trajectories.csv"),
            trajectories_csv(baseline_run),
        )?;
        if e2_anchor_shift_enabled() {
            write_with_log(
                out_dir.join("paper_e2_anchor_shift_stats.csv"),
                anchor_shift_csv(baseline_run),
            )?;
        }
        write_with_log(
            out_dir.join("paper_e2_final_agents.csv"),
            final_agents_csv(baseline_run),
        )?;

        let baseline_g_scene = e2_scene_g_series(baseline_run, space);
        let nohill_g_scene = e2_scene_g_series(nohill_rep, space);
        write_with_log(
            out_dir.join("paper_e2_scene_g_seed0.csv"),
            e2_scene_g_pair_csv(&baseline_g_scene, &nohill_g_scene),
        )?;
        let g_scene_plot_path = out_dir.join("paper_e2_scene_g_seed0.svg");
        render_e2_scene_g_pair_plot(
            &g_scene_plot_path,
            &baseline_g_scene,
            &nohill_g_scene,
            phase_mode,
            baseline_run.n_agents,
        )?;

        let mean_plot_path = out_dir.join("paper_e2_mean_c_level_over_time.svg");
        render_series_plot_fixed_y(
            &mean_plot_path,
            &format!("Mean C_level01 Over Time ({caption_suffix})"),
            "mean C_level01",
            &series_pairs(&baseline_run.mean_c_level_series),
            &marker_steps,
            0.0,
            1.0,
        )?;

        let mean_c_score_path = out_dir.join("paper_e2_mean_c_score_over_time.svg");
        render_series_plot_with_markers(
            &mean_c_score_path,
            &format!("Mean C Score Over Time ({caption_suffix})"),
            "mean C score",
            &series_pairs(&baseline_run.mean_c_series),
            &marker_steps,
        )?;

        let mean_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time.svg");
        render_series_plot_with_markers(
            &mean_c_score_loo_path,
            &format!("Mean C Score (LOO Current) Over Time ({caption_suffix})"),
            "mean C score (LOO current)",
            &series_pairs(&baseline_run.mean_c_score_loo_series),
            &marker_steps,
        )?;

        let mean_c_score_chosen_loo_path =
            out_dir.join("paper_e2_mean_c_score_chosen_loo_over_time.svg");
        render_series_plot_with_markers(
            &mean_c_score_chosen_loo_path,
            &format!("Mean C Score (LOO Chosen) Over Time ({caption_suffix})"),
            "mean C score (LOO chosen)",
            &series_pairs(&baseline_run.mean_c_score_chosen_loo_series),
            &marker_steps,
        )?;

        let accept_worse_path = out_dir.join("paper_e2_accepted_worse_frac_over_time.svg");
        render_series_plot_fixed_y(
            &accept_worse_path,
            &format!("Accepted Worse Fraction ({caption_suffix})"),
            "accepted worse frac",
            &series_pairs(&baseline_run.accepted_worse_frac_series),
            &marker_steps,
            0.0,
            1.0,
        )?;

        let mean_score_path = out_dir.join("paper_e2_mean_score_over_time.svg");
        render_series_plot_with_markers(
            &mean_score_path,
            &format!("Mean Score Over Time ({caption_suffix})"),
            "mean score (C - λ_R·R - λ_C·K)",
            &series_pairs(&baseline_run.mean_score_series),
            &marker_steps,
        )?;

        let mean_crowding_path = out_dir.join("paper_e2_mean_crowding_over_time.svg");
        render_series_plot_with_markers(
            &mean_crowding_path,
            &format!("Mean Crowding Over Time ({caption_suffix})"),
            "mean crowding",
            &series_pairs(&baseline_run.mean_crowding_series),
            &marker_steps,
        )?;

        let moved_frac_path = out_dir.join("paper_e2_moved_frac_over_time.svg");
        render_series_plot_with_markers(
            &moved_frac_path,
            &format!("Moved Fraction Over Time ({caption_suffix})"),
            "moved fraction",
            &series_pairs(&baseline_run.moved_frac_series),
            &marker_steps,
        )?;

        let attempted_update_path = out_dir.join("paper_e2_attempted_update_frac_over_time.svg");
        render_series_plot_fixed_y(
            &attempted_update_path,
            &format!("Attempted Update Fraction ({caption_suffix})"),
            "attempted update frac",
            &series_pairs(&baseline_run.attempted_update_frac_series),
            &marker_steps,
            0.0,
            1.0,
        )?;

        let moved_given_attempt_path =
            out_dir.join("paper_e2_moved_given_attempt_frac_over_time.svg");
        render_series_plot_fixed_y(
            &moved_given_attempt_path,
            &format!("Moved Given Attempt ({caption_suffix})"),
            "moved given attempt frac",
            &series_pairs(&baseline_run.moved_given_attempt_frac_series),
            &marker_steps,
            0.0,
            1.0,
        )?;

        let abs_delta_path = out_dir.join("paper_e2_mean_abs_delta_semitones_over_time.svg");
        render_series_plot_with_markers(
            &abs_delta_path,
            &format!("Mean |Δ| Semitones Over Time ({caption_suffix})"),
            "mean |Δ| semitones",
            &series_pairs(&baseline_run.mean_abs_delta_semitones_series),
            &marker_steps,
        )?;

        let abs_delta_moved_path =
            out_dir.join("paper_e2_mean_abs_delta_semitones_moved_over_time.svg");
        render_series_plot_with_markers(
            &abs_delta_moved_path,
            &format!("Mean |Δ| Semitones (Moved) Over Time ({caption_suffix})"),
            "mean |Δ| semitones (moved only)",
            &series_pairs(&baseline_run.mean_abs_delta_semitones_moved_series),
            &marker_steps,
        )?;

        let trajectory_path = out_dir.join("paper_e2_agent_trajectories.svg");
        render_agent_trajectories_plot(&trajectory_path, &baseline_run.trajectory_semitones)?;

        let pairwise_intervals = pairwise_interval_samples(&baseline_run.final_semitones);
        write_with_log(
            out_dir.join("paper_e2_pairwise_intervals.csv"),
            pairwise_intervals_csv(&pairwise_intervals),
        )?;
        emit_pairwise_interval_dumps_for_condition(out_dir, "baseline", &baseline_runs)?;
        emit_pairwise_interval_dumps_for_condition(out_dir, "nohill", &nohill_runs)?;
        emit_pairwise_interval_dumps_for_condition(out_dir, "nocrowd", &norep_runs)?;
        let pairwise_hist_path = out_dir.join("paper_e2_pairwise_interval_histogram.svg");
        render_interval_histogram(
            &pairwise_hist_path,
            "Pairwise Interval Histogram (Semitones, 12=octave)",
            &pairwise_intervals,
            0.0,
            12.0,
            E2_PAIRWISE_BIN_ST,
            "semitones",
        )?;

        let hist_path = out_dir.join("paper_e2_interval_histogram.svg");
        let hist_caption = format!("Interval Histogram ({post_label}, bin=50 ct)");
        render_interval_histogram(
            &hist_path,
            &hist_caption,
            &baseline_run.semitone_samples_post,
            -12.0,
            12.0,
            E2_ANCHOR_BIN_ST,
            "semitones",
        )?;
    }

    write_with_log(
        out_dir.join("paper_e2_summary.csv"),
        e2_summary_csv(&baseline_runs),
    )?;

    if diagnostics {
        let flutter_segments = e2_flutter_segments(phase_mode, baseline_run.n_agents);
        let mut flutter_csv = String::from(
            "segment,start_step,end_step,pingpong_rate_moves,reversal_rate_moves,move_rate_stepwise,mean_abs_delta_moved,step_count,moved_step_count,move_count,pingpong_count_moves,reversal_count_moves\n",
        );
        for (label, start, end) in &flutter_segments {
            let metrics =
                flutter_metrics_for_trajectories(&baseline_run.trajectory_semitones, *start, *end);
            flutter_csv.push_str(&format!(
                "{label},{start},{end},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
                metrics.pingpong_rate_moves,
                metrics.reversal_rate_moves,
                metrics.move_rate_stepwise,
                metrics.mean_abs_delta_moved,
                metrics.step_count,
                metrics.moved_step_count,
                metrics.move_count,
                metrics.pingpong_count_moves,
                metrics.reversal_count_moves
            ));
        }
        write_with_log(out_dir.join("paper_e2_flutter_metrics.csv"), flutter_csv)?;

        render_e2_histogram_sweep(out_dir, baseline_run)?;

        let mut flutter_rows = Vec::new();
        for (cond, runs) in [
            ("baseline", &baseline_runs),
            ("nohill", &nohill_runs),
            ("nocrowd", &norep_runs),
        ] {
            for run in runs.iter() {
                for (segment, start, end) in &flutter_segments {
                    let metrics =
                        flutter_metrics_for_trajectories(&run.trajectory_semitones, *start, *end);
                    flutter_rows.push(FlutterRow {
                        condition: cond,
                        seed: run.seed,
                        segment,
                        metrics,
                    });
                }
            }
        }
        write_with_log(
            out_dir.join("paper_e2_flutter_by_seed.csv"),
            flutter_by_seed_csv(&flutter_rows),
        )?;
        write_with_log(
            out_dir.join("paper_e2_flutter_summary.csv"),
            flutter_summary_csv(&flutter_rows, &flutter_segments),
        )?;

        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_c.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_c,
                &baseline_stats.std_c,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_c_ci95.csv"),
            sweep_csv_with_ci95(
                "step,mean,std,ci95,n\n",
                &baseline_stats.mean_c,
                &baseline_stats.std_c,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_c_level.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_c_level,
                &baseline_stats.std_c_level,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_c_level_ci95.csv"),
            sweep_csv_with_ci95(
                "step,mean,std,ci95,n\n",
                &baseline_stats.mean_c_level,
                &baseline_stats.std_c_level,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_score.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_score,
                &baseline_stats.std_score,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_crowding.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_crowding,
                &baseline_stats.std_crowding,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_c_score_loo.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_c_score_loo,
                &baseline_stats.std_c_score_loo,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_g_scene.csv"),
            sweep_csv(
                "step,mean,std,n",
                &baseline_stats.mean_g_scene,
                &baseline_stats.std_g_scene,
                baseline_stats.n,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_seed_sweep_mean_g_scene_ci95.csv"),
            sweep_csv_with_ci95(
                "step,mean,std,ci95,n\n",
                &baseline_stats.mean_g_scene,
                &baseline_stats.std_g_scene,
                baseline_stats.n,
            ),
        )?;
        if E2_EMIT_KBINS_SWEEP_SUMMARY {
            write_with_log(
                out_dir.join("paper_e2_kbins_sweep_summary.csv"),
                e2_kbins_sweep_csv(space, anchor_hz, phase_mode),
            )?;
        }

        let sweep_mean_path = out_dir.join("paper_e2_mean_c_level_over_time_seeds.svg");
        render_series_plot_with_band(
            &sweep_mean_path,
            "Mean C_level01 (seed sweep)",
            "mean C_level01",
            &baseline_stats.mean_c_level,
            &baseline_stats.std_c_level,
            &marker_steps,
        )?;
        let sweep_mean_ci_path = out_dir.join("paper_e2_mean_c_level_over_time_seeds_ci95.svg");
        render_series_plot_with_band(
            &sweep_mean_ci_path,
            "Mean C_level01 (seed sweep, 95% CI)",
            "mean C_level01",
            &baseline_stats.mean_c_level,
            &baseline_ci95_c_level,
            &marker_steps,
        )?;

        let sweep_score_path = out_dir.join("paper_e2_mean_score_over_time_seeds.svg");
        render_series_plot_with_band(
            &sweep_score_path,
            "Mean Score (seed sweep)",
            "mean score",
            &baseline_stats.mean_score,
            &baseline_stats.std_score,
            &marker_steps,
        )?;

        let sweep_c_score_loo_path = out_dir.join("paper_e2_mean_c_score_loo_over_time_seeds.svg");
        render_series_plot_with_band(
            &sweep_c_score_loo_path,
            "Mean C Score (LOO current, seed sweep)",
            "mean C score (LOO current)",
            &baseline_stats.mean_c_score_loo,
            &baseline_stats.std_c_score_loo,
            &marker_steps,
        )?;
        let sweep_g_scene_path = out_dir.join("paper_e2_mean_g_scene_over_time_seeds.svg");
        render_series_plot_multi_with_band(
            &sweep_g_scene_path,
            "Mean scene consonance G(F) (seed sweep, 95% CI)",
            "mean G(F) (a.u.)",
            &[
                (
                    "baseline",
                    &baseline_stats.mean_g_scene,
                    &baseline_ci95_g_scene,
                    PAL_H,
                ),
                (
                    "no hill-climb",
                    &nohill_stats.mean_g_scene,
                    &nohill_ci95_g_scene,
                    PAL_R,
                ),
            ],
            &marker_steps,
        )?;

        let sweep_crowding_path = out_dir.join("paper_e2_mean_crowding_over_time_seeds.svg");
        render_series_plot_with_band(
            &sweep_crowding_path,
            "Mean Crowding (seed sweep)",
            "mean crowding",
            &baseline_stats.mean_crowding,
            &baseline_stats.std_crowding,
            &marker_steps,
        )?;

        write_with_log(
            out_dir.join("paper_e2_control_mean_c.csv"),
            e2_controls_csv_c(&baseline_stats, &nohill_stats, &norep_stats),
        )?;
        write_with_log(
            out_dir.join("paper_e2_control_mean_c_level.csv"),
            e2_controls_csv_c_level(&baseline_stats, &nohill_stats, &norep_stats),
        )?;

        let control_plot_path = out_dir.join("paper_e2_mean_c_level_over_time_controls.svg");
        render_series_plot_multi(
            &control_plot_path,
            "Mean C_level01 (controls)",
            "mean C_level01",
            &[
                ("baseline", &baseline_stats.mean_c_level, PAL_H),
                ("no hill-climb", &nohill_stats.mean_c_level, PAL_R),
                ("no crowding", &norep_stats.mean_c_level, PAL_CD),
            ],
            &marker_steps,
        )?;

        let control_c_path = out_dir.join("paper_e2_mean_c_over_time_controls_seeds.svg");
        render_series_plot_multi_with_band(
            &control_c_path,
            "Mean C score (controls, seed sweep)",
            "mean C score",
            &[
                (
                    "baseline",
                    &baseline_stats.mean_c,
                    &baseline_stats.std_c,
                    PAL_H,
                ),
                (
                    "no hill-climb",
                    &nohill_stats.mean_c,
                    &nohill_stats.std_c,
                    PAL_R,
                ),
                (
                    "no crowding",
                    &norep_stats.mean_c,
                    &norep_stats.std_c,
                    PAL_CD,
                ),
            ],
            &marker_steps,
        )?;

        let control_c_score_loo_path =
            out_dir.join("paper_e2_mean_c_score_loo_over_time_controls_seeds.svg");
        render_series_plot_multi_with_band(
            &control_c_score_loo_path,
            "Mean C score (LOO current, controls, seed sweep)",
            "mean C score (LOO current)",
            &[
                (
                    "baseline",
                    &baseline_stats.mean_c_score_loo,
                    &baseline_stats.std_c_score_loo,
                    PAL_H,
                ),
                (
                    "no hill-climb",
                    &nohill_stats.mean_c_score_loo,
                    &nohill_stats.std_c_score_loo,
                    PAL_R,
                ),
                (
                    "no crowding",
                    &norep_stats.mean_c_score_loo,
                    &norep_stats.std_c_score_loo,
                    PAL_CD,
                ),
            ],
            &marker_steps,
        )?;

        let annotated_mean_path =
            out_dir.join("paper_e2_mean_c_level_over_time_seeds_annotated.svg");
        render_e2_mean_c_level_annotated(
            &annotated_mean_path,
            &baseline_stats.mean_c_level,
            &baseline_ci95_c_level,
            &nohill_stats.mean_c_level,
            &nohill_ci95_c_level,
            &norep_stats.mean_c_level,
            &norep_ci95_c_level,
            E2_BURN_IN,
            phase_mode.switch_step(),
        )?;
    }

    let mut diversity_rows_vec = Vec::new();
    diversity_rows_vec.extend(diversity_rows("baseline", &baseline_runs));
    diversity_rows_vec.extend(diversity_rows("nohill", &nohill_runs));
    diversity_rows_vec.extend(diversity_rows("nocrowd", &norep_runs));
    if diagnostics {
        write_with_log(
            out_dir.join("paper_e2_diversity_by_seed.csv"),
            diversity_by_seed_csv(&diversity_rows_vec),
        )?;
        write_with_log(
            out_dir.join("paper_e2_diversity_summary.csv"),
            diversity_summary_csv(&diversity_rows_vec),
        )?;
        write_with_log(
            out_dir.join("paper_e2_diversity_summary_ci95.csv"),
            diversity_summary_ci95_csv(&diversity_rows_vec),
        )?;
        let diversity_plot_path = out_dir.join("paper_e2_diversity_summary.svg");
        render_diversity_summary_plot(&diversity_plot_path, &diversity_rows_vec)?;
        let diversity_ci95_plot_path = out_dir.join("paper_e2_diversity_summary_ci95.svg");
        render_diversity_summary_ci95_plot(&diversity_ci95_plot_path, &diversity_rows_vec)?;
    }
    eprintln!("  E2: computing trajectory LOO summaries...");
    let (baseline_traj_c_score_loo_mean, baseline_traj_c_score_loo_ci95) =
        e2_trajectory_mean_c_score_loo_stats(space, &baseline_runs);
    let (nohill_traj_c_score_loo_mean, nohill_traj_c_score_loo_ci95) =
        e2_trajectory_mean_c_score_loo_stats(space, &nohill_runs);

    let figure1_path = out_dir.join("paper_e2_figure_e2_1.svg");
    eprintln!("  E2: rendering figure panels...");
    render_e2_figure1(
        &figure1_path,
        &baseline_traj_c_score_loo_mean,
        &baseline_traj_c_score_loo_ci95,
        &nohill_traj_c_score_loo_mean,
        &nohill_traj_c_score_loo_ci95,
        &baseline_stats,
        &nohill_stats,
        &baseline_ci95_g_scene,
        &nohill_ci95_g_scene,
        &diversity_rows_vec,
        &baseline_run.trajectory_semitones,
        phase_mode,
    )?;

    let mut hist_rows = Vec::new();
    hist_rows.extend(hist_structure_rows("baseline", &baseline_runs));
    hist_rows.extend(hist_structure_rows("nohill", &nohill_runs));
    hist_rows.extend(hist_structure_rows("nocrowd", &norep_runs));
    if diagnostics {
        write_with_log(
            out_dir.join("paper_e2_hist_structure_by_seed.csv"),
            hist_structure_by_seed_csv(&hist_rows),
        )?;
        write_with_log(
            out_dir.join("paper_e2_hist_structure_summary.csv"),
            hist_structure_summary_csv(&hist_rows),
        )?;
        let hist_plot_path = out_dir.join("paper_e2_hist_structure_summary.svg");
        render_hist_structure_summary_plot(&hist_plot_path, &hist_rows)?;

        let rep_seed = E2_SEEDS[E2_PAPER_REP_SEED_INDEX];
        let norep_rep = &norep_runs[norep_runs
            .iter()
            .position(|run| run.seed == rep_seed)
            .unwrap_or_else(|| pick_representative_run_index(&norep_runs))];
        render_e2_control_histograms(out_dir, baseline_run, nohill_rep, norep_rep)?;
    }

    let hist_min = -12.0f32;
    let hist_max = 12.0f32;
    eprintln!("  E2: computing histogram sweeps...");
    let hist_stats_05 = e2_hist_seed_sweep(&baseline_runs, 0.5, hist_min, hist_max);
    if diagnostics {
        write_with_log(
            out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.csv"),
            e2_hist_seed_sweep_csv(&hist_stats_05),
        )?;
        let hist_plot_05 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p50.svg");
        render_hist_mean_std(
            &hist_plot_05,
            &format!("{post_label_title} Interval Histogram (seed sweep, mean frac, bin=50 ct)"),
            &hist_stats_05.centers,
            &hist_stats_05.mean_frac,
            &hist_stats_05.std_frac,
            0.5,
            "mean fraction",
        )?;

        let hist_stats_025 = e2_hist_seed_sweep(&baseline_runs, 0.25, hist_min, hist_max);
        write_with_log(
            out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.csv"),
            e2_hist_seed_sweep_csv(&hist_stats_025),
        )?;
        let hist_plot_025 = out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25.svg");
        render_hist_mean_std(
            &hist_plot_025,
            &format!("{post_label_title} Interval Histogram (seed sweep, mean frac, bin=25 ct)"),
            &hist_stats_025.centers,
            &hist_stats_025.mean_frac,
            &hist_stats_025.std_frac,
            0.25,
            "mean fraction",
        )?;
        let hist_plot_025_paper =
            out_dir.join("paper_e2_interval_hist_post_seed_sweep_bw0p25_paper.svg");
        render_hist_mean_std_fraction_auto_y(
            &hist_plot_025_paper,
            &format!("{post_label_title} Interval Histogram (paper, bin=25 ct)"),
            &hist_stats_025.centers,
            &hist_stats_025.mean_frac,
            &hist_stats_025.std_frac,
            0.25,
            "semitones",
            &[-7.0, -4.0, -3.0, 3.0, 4.0, 7.0],
        )?;
        let (folded_centers, folded_mean, folded_std) = fold_hist_abs_semitones(
            &hist_stats_025.centers,
            &hist_stats_025.mean_frac,
            &hist_stats_025.std_frac,
            0.25,
        );
        write_with_log(
            out_dir.join("paper_e2_anchor_interval_hist_post_folded.csv"),
            folded_hist_csv(&folded_centers, &folded_mean, &folded_std, hist_stats_025.n),
        )?;
        let folded_hist_plot = out_dir.join("paper_e2_anchor_interval_hist_post_folded.svg");
        render_anchor_hist_post_folded(
            &folded_hist_plot,
            &hist_stats_025.centers,
            &hist_stats_025.mean_frac,
            &hist_stats_025.std_frac,
            &folded_centers,
            &folded_mean,
            &folded_std,
            0.25,
        )?;
    }

    let (pairwise_hist_stats, pairwise_n_pairs) =
        e2_pairwise_hist_seed_sweep(&baseline_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let (pairwise_hist_nohill, pairwise_n_pairs_nohill) =
        e2_pairwise_hist_seed_sweep(&nohill_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let (pairwise_hist_norep, pairwise_n_pairs_norep) =
        e2_pairwise_hist_seed_sweep(&norep_runs, E2_PAIRWISE_BIN_ST, 0.0, 12.0);
    let pairwise_n_pairs_controls = pairwise_n_pairs
        .min(pairwise_n_pairs_nohill)
        .min(pairwise_n_pairs_norep);
    let pairwise_ci95_frac =
        std_series_to_ci95(&pairwise_hist_stats.std_frac, pairwise_hist_stats.n);
    if diagnostics {
        write_with_log(
            out_dir.join("paper_e2_pairwise_interval_histogram_seeds.csv"),
            e2_pairwise_hist_seed_sweep_csv(&pairwise_hist_stats, pairwise_n_pairs),
        )?;
        write_with_log(
            out_dir.join("paper_e2_pairwise_interval_histogram_seeds_ci95.csv"),
            e2_pairwise_hist_seed_sweep_ci95_csv(&pairwise_hist_stats, pairwise_n_pairs),
        )?;
        let pairwise_hist_plot = out_dir.join("paper_e2_pairwise_interval_histogram_seeds.svg");
        render_hist_mean_std(
            &pairwise_hist_plot,
            &format!(
                "Pairwise Interval Histogram (final snapshot, seed sweep, mean frac, bin={:.2} ct)",
                E2_PAIRWISE_BIN_ST
            ),
            &pairwise_hist_stats.centers,
            &pairwise_hist_stats.mean_frac,
            &pairwise_hist_stats.std_frac,
            E2_PAIRWISE_BIN_ST,
            "mean fraction",
        )?;
        let pairwise_hist_plot_paper =
            out_dir.join("paper_e2_pairwise_interval_histogram_seeds_paper.svg");
        render_pairwise_histogram_paper(
            &pairwise_hist_plot_paper,
            "Pairwise Interval Histogram (paper style, 95% CI)",
            &pairwise_hist_stats.centers,
            &pairwise_hist_stats.mean_frac,
            &pairwise_ci95_frac,
            E2_PAIRWISE_BIN_ST,
        )?;
        write_with_log(
            out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds.csv"),
            e2_pairwise_hist_controls_seed_sweep_csv(
                &pairwise_hist_stats,
                &pairwise_hist_nohill,
                &pairwise_hist_norep,
                pairwise_n_pairs_controls,
            ),
        )?;
        write_with_log(
            out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds_ci95.csv"),
            e2_pairwise_hist_controls_seed_sweep_ci95_csv(
                &pairwise_hist_stats,
                &pairwise_hist_nohill,
                &pairwise_hist_norep,
                pairwise_n_pairs_controls,
            ),
        )?;
        let pairwise_controls_plot =
            out_dir.join("paper_e2_pairwise_interval_histogram_controls_seeds.svg");
        render_pairwise_histogram_controls_overlay(
            &pairwise_controls_plot,
            "Pairwise Interval Histogram (controls overlay)",
            &pairwise_hist_stats.centers,
            &pairwise_hist_stats.mean_frac,
            &pairwise_hist_nohill.mean_frac,
            &pairwise_hist_norep.mean_frac,
        )?;
    }

    if diagnostics {
        let mut consonant_rows = Vec::new();
        consonant_rows.extend(consonant_mass_rows_for_condition(
            "baseline",
            &baseline_runs,
        ));
        consonant_rows.extend(consonant_mass_rows_for_condition("nohill", &nohill_runs));
        consonant_rows.extend(consonant_mass_rows_for_condition("nocrowd", &norep_runs));
        write_with_log(
            out_dir.join("paper_e2_consonant_mass_by_seed.csv"),
            consonant_mass_by_seed_csv(&consonant_rows),
        )?;
        write_with_log(
            out_dir.join("paper_e2_consonant_mass_summary.csv"),
            consonant_mass_summary_csv(&consonant_rows),
        )?;
        if E2_EMIT_CONSONANT_MASS_STATS {
            write_with_log(
                out_dir.join("paper_e2_consonant_mass_stats.csv"),
                consonant_mass_stats_csv(&consonant_rows),
            )?;
        }
        let consonant_mass_plot = out_dir.join("paper_e2_consonant_mass_summary.svg");
        render_consonant_mass_summary_plot(&consonant_mass_plot, &consonant_rows)?;
    }

    let figure2_path = out_dir.join("paper_e2_figure_e2_2.svg");
    render_e2_figure2(
        &figure2_path,
        &pairwise_hist_stats.centers,
        &pairwise_hist_stats.mean_frac,
        &pairwise_ci95_frac,
        &pairwise_hist_nohill.mean_frac,
        &pairwise_hist_norep.mean_frac,
        &hist_rows,
        &diversity_rows_vec,
    )?;
    if dense_sweep {
        plot_e2_dense_sweep(out_dir, space, anchor_hz, phase_mode, quick)?;
    }
    if candidate_search {
        plot_e2_candidate_search(out_dir, space, anchor_hz, phase_mode, quick)?;
    }

    if !diagnostics {
        return Ok(());
    }

    let nohill_hist_05 = e2_hist_seed_sweep(&nohill_runs, 0.5, hist_min, hist_max);
    let norep_hist_05 = e2_hist_seed_sweep(&norep_runs, 0.5, hist_min, hist_max);
    let mut controls_csv = String::from(
        "bin_center,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = hist_stats_05
        .centers
        .len()
        .min(nohill_hist_05.centers.len())
        .min(norep_hist_05.centers.len());
    for i in 0..len {
        controls_csv.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            hist_stats_05.centers[i],
            hist_stats_05.mean_frac[i],
            hist_stats_05.std_frac[i],
            nohill_hist_05.mean_frac[i],
            nohill_hist_05.std_frac[i],
            norep_hist_05.mean_frac[i],
            norep_hist_05.std_frac[i]
        ));
    }
    write_with_log(
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.csv"),
        controls_csv,
    )?;

    let control_hist_plot =
        out_dir.join("paper_e2_interval_hist_post_controls_seed_sweep_bw0p50.svg");
    render_hist_controls_fraction(
        &control_hist_plot,
        &format!("{post_label_title} Interval Histogram (controls, mean frac, bin=50 ct)"),
        &hist_stats_05.centers,
        &[
            ("baseline", &hist_stats_05.mean_frac, PAL_H),
            ("no hill-climb", &nohill_hist_05.mean_frac, PAL_R),
            ("no crowding", &norep_hist_05.mean_frac, PAL_CD),
        ],
    )?;

    let pre_label = e2_pre_label();
    let post_label = e2_post_label();
    let pre_step = e2_pre_step();
    let pre_meta = format!(
        "# pre_step={pre_step} pre_label={pre_label} shift_enabled={}\n",
        e2_anchor_shift_enabled()
    );
    let mut delta_csv = pre_meta.clone();
    delta_csv.push_str(&format!(
        "seed,cond,c_init,c_{pre_label},c_{post_label},delta_{pre_label},delta_{post_label}\n"
    ));
    let mut delta_summary = pre_meta;
    delta_summary.push_str(&format!(
        "cond,mean_init,std_init,mean_{pre_label},std_{pre_label},mean_{post_label},std_{post_label},mean_delta_{pre_label},std_delta_{pre_label},mean_delta_{post_label},std_delta_{post_label}\n"
    ));
    for (label, runs) in [
        ("baseline", &baseline_runs),
        ("nohill", &nohill_runs),
        ("nocrowd", &norep_runs),
    ] {
        let mut init_vals = Vec::new();
        let mut pre_vals = Vec::new();
        let mut post_vals = Vec::new();
        let mut delta_pre_vals = Vec::new();
        let mut delta_post_vals = Vec::new();
        for run in runs.iter() {
            let (init, pre, post) = e2_c_snapshot(run);
            let delta_pre = pre - init;
            let delta_post = post - init;
            init_vals.push(init);
            pre_vals.push(pre);
            post_vals.push(post);
            delta_pre_vals.push(delta_pre);
            delta_post_vals.push(delta_post);
            delta_csv.push_str(&format!(
                "{},{label},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                run.seed, init, pre, post, delta_pre, delta_post
            ));
        }
        let (mean_init, std_init) = mean_std_scalar(&init_vals);
        let (mean_pre, std_pre) = mean_std_scalar(&pre_vals);
        let (mean_post, std_post) = mean_std_scalar(&post_vals);
        let (mean_dpre, std_dpre) = mean_std_scalar(&delta_pre_vals);
        let (mean_dpost, std_dpost) = mean_std_scalar(&delta_post_vals);
        delta_summary.push_str(&format!(
            "{label},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            mean_init,
            std_init,
            mean_pre,
            std_pre,
            mean_post,
            std_post,
            mean_dpre,
            std_dpre,
            mean_dpost,
            std_dpost
        ));
    }
    write_with_log(out_dir.join("paper_e2_delta_c_by_seed.csv"), delta_csv)?;
    write_with_log(out_dir.join("paper_e2_delta_c_summary.csv"), delta_summary)?;

    // ── C1: Independent consonance evaluation (JI ratio-complexity metric) ──
    {
        let conditions: Vec<(&str, &[E2Run])> = vec![
            ("baseline", &baseline_runs),
            ("no_hill_climb", &nohill_runs),
            ("no_crowding", &norep_runs),
        ];

        let mut eval_text = String::from(
            "E2 Independent Consonance Evaluation (JI ratio-complexity metric)\n\
             ================================================================\n\n",
        );

        for (label, runs) in &conditions {
            let ji_scores: Vec<f32> = runs
                .iter()
                .map(|run| ji_population_score(&run.final_freqs_hz, anchor_hz))
                .collect();
            let n = ji_scores.len();
            let mean = if n > 0 {
                ji_scores.iter().sum::<f32>() / n as f32
            } else {
                0.0
            };
            let sd = if n > 1 {
                let var =
                    ji_scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
                var.sqrt()
            } else {
                0.0
            };
            eval_text.push_str(&format!(
                "{}: JI_score mean={:.4} sd={:.4} (n={})\n",
                label, mean, sd, n,
            ));
        }
        write_with_log(out_dir.join("paper_e2_independent_eval.txt"), eval_text)?;
    }

    // ── Consonance-only control (supplementary) ──
    if !quick && phase_mode == E2PhaseMode::DissonanceThenConsonance {
        eprintln!(
            "  Running consonance-only control (baseline × {} seeds)...",
            E2_SEEDS.len()
        );
        let (co_runs, co_stats) = e2_seed_sweep(
            space,
            anchor_hz,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            E2PhaseMode::ConsonanceOnly,
            None,
            0,
        );

        // Diversity metrics
        let co_diversity: Vec<DiversityRow> = diversity_rows("consonance_only", &co_runs);
        let cur_diversity: Vec<DiversityRow> = diversity_rows("curriculum", &baseline_runs);

        fn div_summary(rows: &[DiversityRow]) -> (f32, f32, f32, f32) {
            let bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
            let nns: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
            let (mb, sb) = mean_std_scalar(&bins);
            let (mn, sn) = mean_std_scalar(&nns);
            (mb, sb, mn, sn)
        }

        let (co_bins_m, co_bins_s, co_nn_m, co_nn_s) = div_summary(&co_diversity);
        let (cur_bins_m, cur_bins_s, cur_nn_m, cur_nn_s) = div_summary(&cur_diversity);

        // C_score (post burn-in mean for consonance-only; since no phase switch,
        // use last half of rounds as the "post" region to parallel curriculum analysis)
        let half = E2_SWEEPS / 2;
        let burn = E2_BURN_IN;
        fn post_mean_c(runs: &[E2Run], start: usize, burn: usize) -> Vec<f32> {
            runs.iter()
                .map(|r| {
                    let from = start + burn;
                    let slice = &r.mean_c_score_chosen_loo_series[from..];
                    if slice.is_empty() {
                        0.0
                    } else {
                        slice.iter().sum::<f32>() / slice.len() as f32
                    }
                })
                .collect()
        }
        let co_c_scores = post_mean_c(&co_runs, 0, burn);
        let cur_c_scores = post_mean_c(&baseline_runs, half, burn);
        let (co_c_m, co_c_s) = mean_std_scalar(&co_c_scores);
        let (cur_c_m, cur_c_s) = mean_std_scalar(&cur_c_scores);

        // Interval entropy (from final positions)
        fn entropy_from_runs(runs: &[E2Run]) -> Vec<f32> {
            runs.iter()
                .map(|r| {
                    let bin_st = 0.05f32;
                    let n_bins = (12.0 / bin_st).round() as usize;
                    let mut hist = vec![0u32; n_bins];
                    for &st in &r.final_semitones {
                        for &st2 in &r.final_semitones {
                            let d = (st - st2).abs();
                            if d < 1e-6 {
                                continue;
                            }
                            let d_mod = d % 12.0;
                            let idx = (d_mod / bin_st).round() as usize;
                            if idx < n_bins {
                                hist[idx] += 1;
                            }
                        }
                    }
                    let total: u32 = hist.iter().sum();
                    if total == 0 {
                        return 0.0;
                    }
                    let mut h = 0.0f32;
                    for &c in &hist {
                        if c > 0 {
                            let p = c as f32 / total as f32;
                            h -= p * p.ln();
                        }
                    }
                    h
                })
                .collect()
        }
        let co_ent = entropy_from_runs(&co_runs);
        let cur_ent = entropy_from_runs(&baseline_runs);
        let (co_ent_m, co_ent_s) = mean_std_scalar(&co_ent);
        let (cur_ent_m, cur_ent_s) = mean_std_scalar(&cur_ent);

        // Welch t-tests
        fn welch_t(a: &[f32], b: &[f32]) -> (f32, f32) {
            let na = a.len() as f32;
            let nb = b.len() as f32;
            let (ma, sa) = mean_std_scalar(a);
            let (mb, sb) = mean_std_scalar(b);
            let va = sa * sa / na;
            let vb = sb * sb / nb;
            let denom = (va + vb).sqrt();
            if denom < 1e-15 {
                return (0.0, 1.0);
            }
            let t = (ma - mb) / denom;
            // df via Welch-Satterthwaite
            let df = (va + vb).powi(2) / (va * va / (na - 1.0) + vb * vb / (nb - 1.0));
            // Approximate two-tailed p (conservative: use df as f64 for betacf)
            let t_abs = t.abs() as f64;
            let df64 = df as f64;
            let x = df64 / (df64 + t_abs * t_abs);
            // Regularized incomplete beta via continued fraction
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
                let (fpmin, eps) = (1e-30, 3e-12);
                let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
                let mut c = 1.0_f64;
                let mut d = (1.0 - qab * x / qap).recip();
                if d.abs() < fpmin {
                    d = fpmin;
                }
                let mut h = d;
                for m in 1..=200 {
                    let mf = m as f64;
                    let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
                    d = 1.0 + aa * d;
                    if d.abs() < fpmin {
                        d = fpmin;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < fpmin {
                        c = fpmin;
                    }
                    d = d.recip();
                    h *= d * c;
                    let aa = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
                    d = 1.0 + aa * d;
                    if d.abs() < fpmin {
                        d = fpmin;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < fpmin {
                        c = fpmin;
                    }
                    d = d.recip();
                    let delta = d * c;
                    h *= delta;
                    if (delta - 1.0).abs() < eps {
                        break;
                    }
                }
                h
            }
            fn rib(a: f64, b: f64, x: f64) -> f64 {
                if x <= 0.0 {
                    return 0.0;
                }
                if x >= 1.0 {
                    return 1.0;
                }
                let bt =
                    (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln())
                        .exp();
                if x < (a + 1.0) / (a + b + 2.0) {
                    bt * betacf(a, b, x) / a
                } else {
                    1.0 - bt * betacf(b, a, 1.0 - x) / b
                }
            }
            let p = rib(df64 / 2.0, 0.5, x) as f32;
            (t, p)
        }

        let bins_vec_co: Vec<f32> = co_diversity
            .iter()
            .map(|r| r.metrics.unique_bins as f32)
            .collect();
        let bins_vec_cur: Vec<f32> = cur_diversity
            .iter()
            .map(|r| r.metrics.unique_bins as f32)
            .collect();
        let nn_vec_co: Vec<f32> = co_diversity.iter().map(|r| r.metrics.nn_mean).collect();
        let nn_vec_cur: Vec<f32> = cur_diversity.iter().map(|r| r.metrics.nn_mean).collect();
        let (t_bins, p_bins) = welch_t(&bins_vec_co, &bins_vec_cur);
        let (t_nn, p_nn) = welch_t(&nn_vec_co, &nn_vec_cur);
        let (t_c, p_c) = welch_t(&co_c_scores, &cur_c_scores);
        let (t_ent, p_ent) = welch_t(&co_ent, &cur_ent);

        let report = format!(
            "Consonance-Only vs Curriculum Control (Consonance Search, baseline condition)\n\
             ======================================================================\n\
             n_seeds = {}\n\n\
             Metric                  Curriculum         Consonance-Only    Welch t     p\n\
             ---------------------   ----------------   ----------------   --------   ------\n\
             C_score (post)          {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n\
             Unique pitch bins       {:.1} ± {:.1}      {:.1} ± {:.1}      {:.2}      {:.4}\n\
             NN distance (ct)        {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n\
             Interval entropy        {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n",
            E2_SEEDS.len(),
            cur_c_m,
            cur_c_s,
            co_c_m,
            co_c_s,
            t_c,
            p_c,
            cur_bins_m,
            cur_bins_s,
            co_bins_m,
            co_bins_s,
            t_bins,
            p_bins,
            cur_nn_m,
            cur_nn_s,
            co_nn_m,
            co_nn_s,
            t_nn,
            p_nn,
            cur_ent_m,
            cur_ent_s,
            co_ent_m,
            co_ent_s,
            t_ent,
            p_ent,
        );

        // Also output time series for consonance-only
        let co_ci95_c = std_series_to_ci95(&co_stats.std_c_score_loo, co_stats.n);
        let mut ts_csv =
            String::from("step,curriculum_mean,curriculum_ci95,consonly_mean,consonly_ci95\n");
        let len = co_stats
            .mean_c_score_loo
            .len()
            .min(baseline_stats.mean_c_score_loo.len());
        for i in 0..len {
            ts_csv.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6}\n",
                i,
                baseline_stats.mean_c_score_loo[i],
                baseline_ci95_c[i],
                co_stats.mean_c_score_loo[i],
                co_ci95_c[i],
            ));
        }

        write_with_log(
            out_dir.join("paper_e2_consonance_only_comparison.txt"),
            report,
        )?;
        write_with_log(
            out_dir.join("paper_e2_consonance_only_timeseries.csv"),
            ts_csv,
        )?;
        eprintln!("  Consonance-only control written.");
    }

    // ── Shuffled-landscape control (supplementary) ──
    if let Some(shuffled_runs) = shuffled_runs.as_ref() {
        eprintln!("  Writing shuffled-landscape comparison...");
        let shuf_diversity: Vec<DiversityRow> = diversity_rows("shuffled", shuffled_runs);
        let base_diversity: Vec<DiversityRow> = diversity_rows("baseline", &baseline_runs);

        fn div_summary_ctrl(rows: &[DiversityRow]) -> (f32, f32, f32, f32) {
            let bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
            let nns: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
            let (mb, sb) = mean_std_scalar(&bins);
            let (mn, sn) = mean_std_scalar(&nns);
            (mb, sb, mn, sn)
        }

        let (shuf_bins_m, shuf_bins_s, shuf_nn_m, shuf_nn_s) = div_summary_ctrl(&shuf_diversity);
        let (base_bins_m, base_bins_s, base_nn_m, base_nn_s) = div_summary_ctrl(&base_diversity);

        let half = E2_SWEEPS / 2;
        let burn = E2_BURN_IN;
        fn post_mean_c_ctrl(runs: &[E2Run], start: usize, burn: usize) -> Vec<f32> {
            runs.iter()
                .map(|r| {
                    let from = start + burn;
                    let slice = &r.mean_c_score_chosen_loo_series[from..];
                    if slice.is_empty() {
                        0.0
                    } else {
                        slice.iter().sum::<f32>() / slice.len() as f32
                    }
                })
                .collect()
        }
        let shuf_c_scores = post_mean_c_ctrl(shuffled_runs, half, burn);
        let base_c_scores = post_mean_c_ctrl(&baseline_runs, half, burn);
        let (shuf_c_m, shuf_c_s) = mean_std_scalar(&shuf_c_scores);
        let (base_c_m, base_c_s) = mean_std_scalar(&base_c_scores);

        fn entropy_from_runs_ctrl(runs: &[E2Run]) -> Vec<f32> {
            runs.iter()
                .map(|r| {
                    let bin_st = 0.05f32;
                    let n_bins = (12.0 / bin_st).round() as usize;
                    let mut hist = vec![0u32; n_bins];
                    for &st in &r.final_semitones {
                        for &st2 in &r.final_semitones {
                            let d = (st - st2).abs();
                            if d < 1e-6 {
                                continue;
                            }
                            let d_mod = d % 12.0;
                            let idx = (d_mod / bin_st).round() as usize;
                            if idx < n_bins {
                                hist[idx] += 1;
                            }
                        }
                    }
                    let total: u32 = hist.iter().sum();
                    if total == 0 {
                        return 0.0;
                    }
                    let mut h = 0.0f32;
                    for &c in &hist {
                        if c > 0 {
                            let p = c as f32 / total as f32;
                            h -= p * p.ln();
                        }
                    }
                    h
                })
                .collect()
        }
        let shuf_ent = entropy_from_runs_ctrl(shuffled_runs);
        let base_ent = entropy_from_runs_ctrl(&baseline_runs);
        let (shuf_ent_m, shuf_ent_s) = mean_std_scalar(&shuf_ent);
        let (base_ent_m, base_ent_s) = mean_std_scalar(&base_ent);

        fn welch_t_ctrl(a: &[f32], b: &[f32]) -> (f32, f32) {
            let na = a.len() as f32;
            let nb = b.len() as f32;
            let (ma, sa) = mean_std_scalar(a);
            let (mb, sb) = mean_std_scalar(b);
            let va = sa * sa / na;
            let vb = sb * sb / nb;
            let denom = (va + vb).sqrt();
            if denom < 1e-15 {
                return (0.0, 1.0);
            }
            let t = (ma - mb) / denom;
            let df = (va + vb).powi(2) / (va * va / (na - 1.0) + vb * vb / (nb - 1.0));
            let t_abs = t.abs() as f64;
            let df64 = df as f64;
            let x = df64 / (df64 + t_abs * t_abs);
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
                let (fpmin, eps) = (1e-30, 3e-12);
                let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
                let mut c = 1.0_f64;
                let mut d = (1.0 - qab * x / qap).recip();
                if d.abs() < fpmin {
                    d = fpmin;
                }
                let mut h = d;
                for m in 1..=200 {
                    let mf = m as f64;
                    let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
                    d = 1.0 + aa * d;
                    if d.abs() < fpmin {
                        d = fpmin;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < fpmin {
                        c = fpmin;
                    }
                    d = d.recip();
                    h *= d * c;
                    let aa = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
                    d = 1.0 + aa * d;
                    if d.abs() < fpmin {
                        d = fpmin;
                    }
                    c = 1.0 + aa / c;
                    if c.abs() < fpmin {
                        c = fpmin;
                    }
                    d = d.recip();
                    let delta = d * c;
                    h *= delta;
                    if (delta - 1.0).abs() < eps {
                        break;
                    }
                }
                h
            }
            fn rib(a: f64, b: f64, x: f64) -> f64 {
                if x <= 0.0 {
                    return 0.0;
                }
                if x >= 1.0 {
                    return 1.0;
                }
                let bt =
                    (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln())
                        .exp();
                if x < (a + 1.0) / (a + b + 2.0) {
                    bt * betacf(a, b, x) / a
                } else {
                    1.0 - bt * betacf(b, a, 1.0 - x) / b
                }
            }
            let p = rib(df64 / 2.0, 0.5, x) as f32;
            (t, p)
        }

        let bins_vec_shuf: Vec<f32> = shuf_diversity
            .iter()
            .map(|r| r.metrics.unique_bins as f32)
            .collect();
        let bins_vec_base: Vec<f32> = base_diversity
            .iter()
            .map(|r| r.metrics.unique_bins as f32)
            .collect();
        let nn_vec_shuf: Vec<f32> = shuf_diversity.iter().map(|r| r.metrics.nn_mean).collect();
        let nn_vec_base: Vec<f32> = base_diversity.iter().map(|r| r.metrics.nn_mean).collect();
        let (t_bins, p_bins) = welch_t_ctrl(&bins_vec_shuf, &bins_vec_base);
        let (t_nn, p_nn) = welch_t_ctrl(&nn_vec_shuf, &nn_vec_base);
        let (t_c, p_c) = welch_t_ctrl(&shuf_c_scores, &base_c_scores);
        let (t_ent, p_ent) = welch_t_ctrl(&shuf_ent, &base_ent);

        let report = format!(
            "Shuffled-Landscape vs Baseline Control (Consonance Search)\n\
             ===================================================================\n\
             n_seeds = {}\n\n\
             Metric                  Baseline           Shuffled           Welch t     p\n\
             ---------------------   ----------------   ----------------   --------   ------\n\
             C_score (post)          {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n\
             Unique pitch bins       {:.1} ± {:.1}      {:.1} ± {:.1}      {:.2}      {:.4}\n\
             NN distance (ct)        {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n\
             Interval entropy        {:.3} ± {:.3}      {:.3} ± {:.3}      {:.2}      {:.4}\n",
            E2_SEEDS.len(),
            base_c_m,
            base_c_s,
            shuf_c_m,
            shuf_c_s,
            t_c,
            p_c,
            base_bins_m,
            base_bins_s,
            shuf_bins_m,
            shuf_bins_s,
            t_bins,
            p_bins,
            base_nn_m,
            base_nn_s,
            shuf_nn_m,
            shuf_nn_s,
            t_nn,
            p_nn,
            base_ent_m,
            base_ent_s,
            shuf_ent_m,
            shuf_ent_s,
            t_ent,
            p_ent,
        );

        write_with_log(out_dir.join("paper_e2_shuffled_comparison.txt"), report)?;
        eprintln!("  Shuffled-landscape comparison written.");
    }

    // ── Terrain controls + coefficient sweep (supplementary) ──
    if !quick {
        eprintln!("  Running terrain controls + coefficient sweep (flat parallel jobs)...");
        plot_e2_terrain_controls(out_dir, space, anchor_hz, &baseline_runs)?;
        plot_e2_coefficient_sweep(out_dir, space, anchor_hz)?;
    }

    Ok(())
}

fn plot_e2_terrain_controls(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    baseline_runs: &[E2Run],
) -> Result<(), Box<dyn Error>> {
    let phase_mode = E2PhaseMode::DissonanceThenConsonance;
    let bins_per_cent = SPACE_BINS_PER_OCT as f32 / 1200.0;
    let misalign_cents: [f32; 4] = [137.0, 271.0, 389.0, 523.0];
    let shifts: Vec<i32> = misalign_cents
        .iter()
        .map(|&c| (c * bins_per_cent).round() as i32)
        .collect();

    let base_diversity: Vec<DiversityRow> = diversity_rows("baseline", baseline_runs);
    let base_bins: Vec<f32> = base_diversity
        .iter()
        .map(|r| r.metrics.unique_bins as f32)
        .collect();
    let base_nns: Vec<f32> = base_diversity.iter().map(|r| r.metrics.nn_mean).collect();
    let (base_bins_m, base_bins_s) = mean_std_scalar(&base_bins);
    let (base_nn_m, base_nn_s) = mean_std_scalar(&base_nns);

    fn entropy_from_runs_tc(runs: &[E2Run]) -> Vec<f32> {
        runs.iter()
            .map(|r| {
                let bin_st = 0.05f32;
                let n_bins = (12.0 / bin_st).round() as usize;
                let mut hist = vec![0u32; n_bins];
                for &st in &r.final_semitones {
                    for &st2 in &r.final_semitones {
                        let d = (st - st2).abs();
                        if d < 1e-6 {
                            continue;
                        }
                        let d_mod = d % 12.0;
                        let idx = (d_mod / bin_st).round() as usize;
                        if idx < n_bins {
                            hist[idx] += 1;
                        }
                    }
                }
                let total: u32 = hist.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                let mut h = 0.0f32;
                for &c in &hist {
                    if c > 0 {
                        let p = c as f32 / total as f32;
                        h -= p * p.ln();
                    }
                }
                h
            })
            .collect()
    }
    let base_ent = entropy_from_runs_tc(baseline_runs);
    let (base_ent_m, base_ent_s) = mean_std_scalar(&base_ent);

    let half = E2_SWEEPS / 2;
    let burn = E2_BURN_IN;
    fn post_mean_c_tc(runs: &[E2Run], start: usize, burn: usize) -> Vec<f32> {
        runs.iter()
            .map(|r| {
                let from = start + burn;
                let slice = &r.mean_c_score_chosen_loo_series[from..];
                if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().sum::<f32>() / slice.len() as f32
                }
            })
            .collect()
    }
    let base_c = post_mean_c_tc(baseline_runs, half, burn);
    let (base_c_m, base_c_s) = mean_std_scalar(&base_c);

    fn welch_t_tc(a: &[f32], b: &[f32]) -> (f32, f32) {
        let na = a.len() as f32;
        let nb = b.len() as f32;
        let (ma, sa) = mean_std_scalar(a);
        let (mb, sb) = mean_std_scalar(b);
        let va = sa * sa / na;
        let vb = sb * sb / nb;
        let denom = (va + vb).sqrt();
        if denom < 1e-15 {
            return (0.0, 1.0);
        }
        let t = (ma - mb) / denom;
        let df = (va + vb).powi(2) / (va * va / (na - 1.0) + vb * vb / (nb - 1.0));
        let t_abs = t.abs() as f64;
        let df64 = df as f64;
        let x = df64 / (df64 + t_abs * t_abs);
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
            let (fpmin, eps) = (1e-30, 3e-12);
            let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
            let mut c = 1.0_f64;
            let mut d = (1.0 - qab * x / qap).recip();
            if d.abs() < fpmin {
                d = fpmin;
            }
            let mut h = d;
            for m in 1..=200 {
                let mf = m as f64;
                let aa = mf * (b - mf) * x / ((qam + 2.0 * mf) * (a + 2.0 * mf));
                d = 1.0 + aa * d;
                if d.abs() < fpmin {
                    d = fpmin;
                }
                c = 1.0 + aa / c;
                if c.abs() < fpmin {
                    c = fpmin;
                }
                d = d.recip();
                h *= d * c;
                let aa = -(a + mf) * (qab + mf) * x / ((a + 2.0 * mf) * (qap + 2.0 * mf));
                d = 1.0 + aa * d;
                if d.abs() < fpmin {
                    d = fpmin;
                }
                c = 1.0 + aa / c;
                if c.abs() < fpmin {
                    c = fpmin;
                }
                d = d.recip();
                let delta = d * c;
                h *= delta;
                if (delta - 1.0).abs() < eps {
                    break;
                }
            }
            h
        }
        fn rib(a: f64, b: f64, x: f64) -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            if x >= 1.0 {
                return 1.0;
            }
            let bt =
                (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln())
                    .exp();
            if x < (a + 1.0) / (a + b + 2.0) {
                bt * betacf(a, b, x) / a
            } else {
                1.0 - bt * betacf(b, a, 1.0 - x) / b
            }
        }
        let p = rib(df64 / 2.0, 0.5, x) as f32;
        (t, p)
    }

    let mut report = format!(
        "Terrain Controls: Baseline vs H/R Misalignment (Consonance Search)\n\
         =================================================================\n\
         n_seeds = {}\n\
         bins_per_cent = {:.4}\n\n\
         Condition        shift_cents  shift_bins  unique_bins       nn_mean           entropy           C_score(ref)\n\
         -----------      ----------   ----------  ----------------  ----------------  ----------------  ----------------\n\
         Baseline         0            0           {:.1} ± {:.1}     {:.3} ± {:.3}     {:.3} ± {:.3}     {:.3} ± {:.3}\n",
        E2_SEEDS.len(),
        bins_per_cent,
        base_bins_m,
        base_bins_s,
        base_nn_m,
        base_nn_s,
        base_ent_m,
        base_ent_s,
        base_c_m,
        base_c_s,
    );

    // Run all 4 misalignment shifts in a bounded flat job pool.
    struct MisResult {
        cents: f32,
        shift: i32,
        bins_v: Vec<f32>,
        nn_v: Vec<f32>,
        ent_v: Vec<f32>,
        c_v: Vec<f32>,
    }
    let misalignment_jobs: Vec<(f32, i32)> = misalign_cents
        .iter()
        .copied()
        .zip(shifts.iter().copied())
        .collect();
    let mis_results: Vec<MisResult> =
        parallel_map_ordered(&misalignment_jobs, None, |&(cents, shift)| {
            eprintln!(
                "    Misalignment Δ={} cents (shift={} bins)...",
                cents, shift
            );
            let (mis_runs, _) = e2_seed_sweep_with_threads(
                space,
                anchor_hz,
                E2Condition::Baseline,
                E2_STEP_SEMITONES,
                phase_mode,
                None,
                shift,
                E2_N_AGENTS,
                2.0,
                Some(1),
            );
            let mis_div: Vec<DiversityRow> = diversity_rows("misaligned", &mis_runs);
            let bins_v: Vec<f32> = mis_div
                .iter()
                .map(|r| r.metrics.unique_bins as f32)
                .collect();
            let nn_v: Vec<f32> = mis_div.iter().map(|r| r.metrics.nn_mean).collect();
            let ent_v = entropy_from_runs_tc(&mis_runs);
            let c_v = post_mean_c_tc(&mis_runs, half, burn);
            MisResult {
                cents,
                shift,
                bins_v,
                nn_v,
                ent_v,
                c_v,
            }
        });

    let mut all_mis_bins: Vec<f32> = Vec::new();
    let mut all_mis_nns: Vec<f32> = Vec::new();
    let mut all_mis_ent: Vec<f32> = Vec::new();
    let mut all_mis_c: Vec<f32> = Vec::new();

    for (i, mr) in mis_results.iter().enumerate() {
        let (mb, sb) = mean_std_scalar(&mr.bins_v);
        let (mn, sn) = mean_std_scalar(&mr.nn_v);
        let (me, se) = mean_std_scalar(&mr.ent_v);
        let (mc, sc) = mean_std_scalar(&mr.c_v);

        let (t_bins, p_bins) = welch_t_tc(&mr.bins_v, &base_bins);
        let (t_nn, p_nn) = welch_t_tc(&mr.nn_v, &base_nns);
        let (t_ent, p_ent) = welch_t_tc(&mr.ent_v, &base_ent);

        report.push_str(&format!(
            "Misaligned[{}]   {:<11.0}  {:<10}  {:.1} ± {:.1}     {:.3} ± {:.3}     {:.3} ± {:.3}     {:.3} ± {:.3}  t_bins={:.2} p={:.4} t_nn={:.2} p={:.4} t_ent={:.2} p={:.4}\n",
            i, mr.cents, mr.shift, mb, sb, mn, sn, me, se, mc, sc, t_bins, p_bins, t_nn, p_nn, t_ent, p_ent,
        ));

        all_mis_bins.extend_from_slice(&mr.bins_v);
        all_mis_nns.extend_from_slice(&mr.nn_v);
        all_mis_ent.extend_from_slice(&mr.ent_v);
        all_mis_c.extend_from_slice(&mr.c_v);
    }

    let (grand_bins_m, grand_bins_s) = mean_std_scalar(&all_mis_bins);
    let (grand_nn_m, grand_nn_s) = mean_std_scalar(&all_mis_nns);
    let (grand_ent_m, grand_ent_s) = mean_std_scalar(&all_mis_ent);
    let (grand_c_m, grand_c_s) = mean_std_scalar(&all_mis_c);

    let base_bins_rep: Vec<f32> = (0..shifts.len())
        .flat_map(|_| base_bins.iter().copied())
        .collect();
    let base_nns_rep: Vec<f32> = (0..shifts.len())
        .flat_map(|_| base_nns.iter().copied())
        .collect();
    let base_ent_rep: Vec<f32> = (0..shifts.len())
        .flat_map(|_| base_ent.iter().copied())
        .collect();
    let (t_bins, p_bins) = welch_t_tc(&all_mis_bins, &base_bins_rep);
    let (t_nn, p_nn) = welch_t_tc(&all_mis_nns, &base_nns_rep);
    let (t_ent, p_ent) = welch_t_tc(&all_mis_ent, &base_ent_rep);

    report.push_str(&format!(
        "Misaligned(avg)  --           --          {:.1} ± {:.1}     {:.3} ± {:.3}     {:.3} ± {:.3}     {:.3} ± {:.3}  t_bins={:.2} p={:.4} t_nn={:.2} p={:.4} t_ent={:.2} p={:.4}\n",
        grand_bins_m, grand_bins_s, grand_nn_m, grand_nn_s, grand_ent_m, grand_ent_s, grand_c_m, grand_c_s,
        t_bins, p_bins, t_nn, p_nn, t_ent, p_ent,
    ));

    write_with_log(out_dir.join("paper_e2_terrain_controls.txt"), report)?;
    Ok(())
}

fn plot_e2_coefficient_sweep(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let phase_mode = E2PhaseMode::DissonanceThenConsonance;
    let b_values: [f32; 4] = [-2.0, -1.35, -0.7, 0.0];
    let c_values: [f32; 4] = [0.0, 0.5, 1.0, 1.5];

    fn entropy_from_runs_cs(runs: &[E2Run]) -> Vec<f32> {
        runs.iter()
            .map(|r| {
                let bin_st = 0.05f32;
                let n_bins = (12.0 / bin_st).round() as usize;
                let mut hist = vec![0u32; n_bins];
                for &st in &r.final_semitones {
                    for &st2 in &r.final_semitones {
                        let d = (st - st2).abs();
                        if d < 1e-6 {
                            continue;
                        }
                        let d_mod = d % 12.0;
                        let idx = (d_mod / bin_st).round() as usize;
                        if idx < n_bins {
                            hist[idx] += 1;
                        }
                    }
                }
                let total: u32 = hist.iter().sum();
                if total == 0 {
                    return 0.0;
                }
                let mut h = 0.0f32;
                for &ct in &hist {
                    if ct > 0 {
                        let p = ct as f32 / total as f32;
                        h -= p * p.ln();
                    }
                }
                h
            })
            .collect()
    }

    let half = E2_SWEEPS / 2;
    let burn = E2_BURN_IN;

    let mut report = String::from(
        "Coefficient Sweep (b × c) — Consonance Search\n\
         =============================================\n\
         a=1.0, d=0.0 fixed. Default: b=-1.35, c=1.0\n\n\
         b\tc\tunique_bins_m\tunique_bins_s\tnn_mean_m\tnn_mean_s\tentropy_m\tentropy_s\tc_score_m\tc_score_s\n",
    );

    // Run all 16 (b,c) combos in a bounded flat job pool.
    let combos: Vec<(f32, f32)> = b_values
        .iter()
        .flat_map(|&b| c_values.iter().map(move |&c| (b, c)))
        .collect();
    struct CoeffResult {
        b: f32,
        c: f32,
        bins_m: f32,
        bins_s: f32,
        nn_m: f32,
        nn_s: f32,
        ent_m: f32,
        ent_s: f32,
        c_m: f32,
        c_s: f32,
    }
    let results: Vec<CoeffResult> = parallel_map_ordered(&combos, None, |&(b, c)| {
        eprintln!("    Coefficient sweep b={:.2}, c={:.2}...", b, c);
        let kernel = ConsonanceKernel {
            a: 1.0,
            b,
            c,
            d: 0.0,
        };
        let (runs, _) = e2_seed_sweep_with_threads(
            space,
            anchor_hz,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            phase_mode,
            Some(kernel),
            0,
            E2_N_AGENTS,
            2.0,
            Some(1),
        );
        let div: Vec<DiversityRow> = diversity_rows("sweep", &runs);
        let bins_v: Vec<f32> = div.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_v: Vec<f32> = div.iter().map(|r| r.metrics.nn_mean).collect();
        let ent_v = entropy_from_runs_cs(&runs);
        let c_scores: Vec<f32> = runs
            .iter()
            .map(|r| {
                let from = half + burn;
                let slice = &r.mean_c_score_chosen_loo_series[from..];
                if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().sum::<f32>() / slice.len() as f32
                }
            })
            .collect();
        let (bm, bs) = mean_std_scalar(&bins_v);
        let (nm, ns) = mean_std_scalar(&nn_v);
        let (em, es) = mean_std_scalar(&ent_v);
        let (cm, cs) = mean_std_scalar(&c_scores);
        CoeffResult {
            b,
            c,
            bins_m: bm,
            bins_s: bs,
            nn_m: nm,
            nn_s: ns,
            ent_m: em,
            ent_s: es,
            c_m: cm,
            c_s: cs,
        }
    });
    for cr in &results {
        report.push_str(&format!(
            "{:.2}\t{:.2}\t{:.1}\t{:.1}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\n",
            cr.b, cr.c, cr.bins_m, cr.bins_s, cr.nn_m, cr.nn_s, cr.ent_m, cr.ent_s, cr.c_m, cr.c_s,
        ));
    }

    write_with_log(out_dir.join("paper_e2_coefficient_sweep.txt"), report)?;
    Ok(())
}

fn e2_anchor_shift_enabled() -> bool {
    E2_ANCHOR_SHIFT_STEP != usize::MAX
}

fn e2_pre_step_for(anchor_shift_enabled: bool, anchor_shift_step: usize, burn_in: usize) -> usize {
    if anchor_shift_enabled {
        anchor_shift_step.saturating_sub(1)
    } else {
        burn_in.saturating_sub(1)
    }
}

fn e2_pre_step() -> usize {
    e2_pre_step_for(e2_anchor_shift_enabled(), E2_ANCHOR_SHIFT_STEP, E2_BURN_IN)
}

fn e2_post_step_for(steps: usize) -> usize {
    steps.saturating_sub(1)
}

fn e2_microsteps_per_sweep(n_agents: usize) -> usize {
    n_agents.max(1)
}

fn e2_trajectory_burn_in_step(n_agents: usize) -> usize {
    E2_BURN_IN.saturating_mul(e2_microsteps_per_sweep(n_agents))
}

fn e2_trajectory_phase_switch_step(phase_mode: E2PhaseMode, n_agents: usize) -> Option<usize> {
    phase_mode
        .switch_step()
        .map(|step| step.saturating_mul(e2_microsteps_per_sweep(n_agents)))
}

fn e2_caption_suffix(phase_mode: E2PhaseMode) -> String {
    let base = if e2_anchor_shift_enabled() {
        format!("burn-in={E2_BURN_IN}, shift@{E2_ANCHOR_SHIFT_STEP}")
    } else {
        format!("burn-in={E2_BURN_IN}, shift=off")
    };
    if let Some(step) = phase_mode.switch_step() {
        format!("{base}, phase_switch@{step}")
    } else {
        base
    }
}

fn e2_pre_label() -> &'static str {
    if e2_anchor_shift_enabled() {
        "pre"
    } else {
        "burnin_end"
    }
}

fn e2_post_label() -> &'static str {
    "post"
}

fn e2_post_label_title() -> &'static str {
    "Post"
}

fn e2_post_window_start_step() -> usize {
    if e2_anchor_shift_enabled() {
        E2_ANCHOR_SHIFT_STEP
    } else {
        E2_BURN_IN
    }
}

fn e2_post_window_end_step() -> usize {
    E2_SWEEPS.saturating_sub(1)
}

fn e2_accept_temperature(step: usize, phase_mode: E2PhaseMode) -> f32 {
    if !E2_ACCEPT_ENABLED {
        return 0.0;
    }
    if E2_ACCEPT_TAU_STEPS <= 0.0 {
        return E2_ACCEPT_T0.max(0.0);
    }
    let mut phase_step = step;
    if E2_ACCEPT_RESET_ON_PHASE
        && let Some(switch_step) = phase_mode.switch_step()
        && step >= switch_step
    {
        phase_step = step - switch_step;
    }
    E2_ACCEPT_T0.max(0.0) * (-(phase_step as f32) / E2_ACCEPT_TAU_STEPS).exp()
}

fn build_e2_update_order(
    schedule: E2UpdateSchedule,
    len: usize,
    step: usize,
    rng: &mut StdRng,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..len).collect();
    match schedule {
        E2UpdateSchedule::RandomSingle => order.shuffle(rng),
        E2UpdateSchedule::SequentialRotate => {
            if !order.is_empty() {
                let start = step % order.len();
                order.rotate_left(start);
            }
        }
        E2UpdateSchedule::Checkerboard | E2UpdateSchedule::Lazy => {}
    }
    order
}

fn e2_should_block_backtrack(phase_mode: E2PhaseMode, step: usize) -> bool {
    if !E2_ANTI_BACKTRACK_ENABLED {
        return false;
    }
    if E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY && let Some(switch_step) = phase_mode.switch_step() {
        return step < switch_step;
    }
    true
}

fn e2_update_backtrack_targets(targets: &mut [usize], before: &[usize], after: &[usize]) {
    debug_assert_eq!(
        targets.len(),
        before.len(),
        "backtrack_targets len mismatch"
    );
    debug_assert_eq!(before.len(), after.len(), "backtrack_targets len mismatch");
    for i in 0..targets.len() {
        if after[i] != before[i] {
            targets[i] = before[i];
        }
    }
}

fn is_consonant_near(semitone_abs: f32) -> bool {
    for target in E2_CONSONANT_STEPS {
        if (semitone_abs - target).abs() <= E2_INIT_CONSONANT_EXCLUSION_ST {
            return true;
        }
    }
    false
}

fn init_e2_agent_indices_uniform<R: Rng + ?Sized>(
    rng: &mut R,
    min_idx: usize,
    max_idx: usize,
    n_agents: usize,
) -> Vec<usize> {
    (0..n_agents)
        .map(|_| rng.random_range(min_idx..=max_idx))
        .collect()
}

fn init_e2_agent_indices_reject_consonant<R: Rng + ?Sized>(
    rng: &mut R,
    min_idx: usize,
    max_idx: usize,
    log2_ratio_scan: &[f32],
    n_agents: usize,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(n_agents);
    for _ in 0..n_agents {
        let mut last = min_idx;
        let mut chosen = None;
        for _ in 0..E2_INIT_MAX_TRIES {
            let idx = rng.random_range(min_idx..=max_idx);
            last = idx;
            let semitone_abs = (12.0 * log2_ratio_scan[idx]).abs();
            if !is_consonant_near(semitone_abs) {
                chosen = Some(idx);
                break;
            }
        }
        indices.push(chosen.unwrap_or(last));
    }
    indices
}

#[allow(clippy::too_many_arguments)]
fn run_e2_once(
    space: &Log2Space,
    anchor_hz: f32,
    seed: u64,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
) -> E2Run {
    run_e2_once_cfg(
        space,
        anchor_hz,
        seed,
        condition,
        step_semitones,
        phase_mode,
        kernel,
        r_shift_bins,
        E2_N_AGENTS,
        2.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_e2_once_cfg(
    space: &Log2Space,
    anchor_hz: f32,
    seed: u64,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
    n_agents: usize,
    range_oct: f32,
) -> E2Run {
    let mut rng = seeded_rng(seed);
    let fixed_drone = e2_fixed_drone(space, anchor_hz);
    let mut anchor_hz_current = anchor_hz;
    let mut log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
    let half_range_oct = 0.5 * range_oct.max(0.01);
    let (mut min_idx, mut max_idx) =
        log2_ratio_bounds(&log2_ratio_scan, -half_range_oct, half_range_oct);

    let mut agent_indices = match E2_INIT_MODE {
        E2InitMode::Uniform => init_e2_agent_indices_uniform(&mut rng, min_idx, max_idx, n_agents),
        E2InitMode::RejectConsonant => init_e2_agent_indices_reject_consonant(
            &mut rng,
            min_idx,
            max_idx,
            &log2_ratio_scan,
            n_agents,
        ),
    };

    let (erb_scan, du_scan) = erb_grid(space);
    let kernel_params = KernelParams::default();
    let mut workspace = build_consonance_workspace(space);
    if let Some(k) = kernel {
        workspace.params.consonance_kernel = k;
    }
    workspace.r_shift_bins = r_shift_bins;
    let k_bins = k_from_semitones(step_semitones);

    let mut mean_c_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_level_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_chosen_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_score_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_crowding_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut accepted_worse_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut attempted_update_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_given_attempt_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_moved_series = Vec::with_capacity(E2_SWEEPS);
    let mut semitone_samples_pre = Vec::new();
    let mut semitone_samples_post = Vec::new();
    let mut density_mass_sum = 0.0f32;
    let mut density_mass_min = f32::INFINITY;
    let mut density_mass_max = 0.0f32;
    let mut density_mass_count = 0u32;
    let mut r_state01_min = f32::INFINITY;
    let mut r_state01_max = f32::NEG_INFINITY;
    let mut r_state01_mean_sum = 0.0f32;
    let mut r_state01_mean_count = 0u32;

    let trajectory_steps_cap = 1 + E2_SWEEPS.saturating_mul(e2_microsteps_per_sweep(n_agents));
    let mut trajectory_semitones = (0..n_agents)
        .map(|_| Vec::with_capacity(trajectory_steps_cap))
        .collect::<Vec<_>>();
    let mut backtrack_targets = agent_indices.clone();
    let use_proposal = matches!(
        condition,
        E2Condition::Baseline | E2Condition::NoCrowding | E2Condition::ShuffledLandscape
    );
    let mut pitch_cores = if use_proposal {
        build_e2_pitch_cores(space, &agent_indices, step_semitones)
    } else {
        Vec::new()
    };

    // Fixed permutation for ShuffledLandscape (generated once per run).
    let shuffle_perm: Vec<usize> = if matches!(condition, E2Condition::ShuffledLandscape) {
        let mut perm: Vec<usize> = (0..space.n_bins()).collect();
        perm.shuffle(&mut rng);
        perm
    } else {
        Vec::new()
    };

    let mut anchor_shift = E2AnchorShiftStats {
        step: E2_ANCHOR_SHIFT_STEP,
        anchor_hz_before: anchor_hz_current,
        anchor_hz_after: anchor_hz_current * E2_ANCHOR_SHIFT_RATIO,
        count_min: 0,
        count_max: 0,
        respawned: 0,
    };

    let anchor_shift_enabled = e2_anchor_shift_enabled();
    let phase_switch_step = phase_mode.switch_step();
    for sweep in 0..E2_SWEEPS {
        if sweep == E2_ANCHOR_SHIFT_STEP {
            let before = anchor_hz_current;
            anchor_hz_current *= E2_ANCHOR_SHIFT_RATIO;
            log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
            let (new_min, new_max) =
                log2_ratio_bounds(&log2_ratio_scan, -half_range_oct, half_range_oct);
            let (count_min, count_max, respawned) = shift_indices_by_ratio(
                space,
                &mut agent_indices,
                E2_ANCHOR_SHIFT_RATIO,
                new_min,
                new_max,
                &mut rng,
            );
            anchor_shift = E2AnchorShiftStats {
                step: sweep,
                anchor_hz_before: before,
                anchor_hz_after: anchor_hz_current,
                count_min,
                count_max,
                respawned,
            };
            min_idx = new_min;
            max_idx = new_max;
        }
        if let Some(switch_step) = phase_switch_step
            && sweep == switch_step
        {
            backtrack_targets.clone_from_slice(&agent_indices);
        }

        let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
            space,
            std::slice::from_ref(&fixed_drone.idx),
            &agent_indices,
            &du_scan,
        );
        let (c_score_scan, c_level_scan, density_mass, r_state_stats) =
            compute_c_score_level_scans(space, &workspace, &env_scan, &density_scan, &du_scan);
        let mut landscape = if use_proposal {
            Some(build_pitch_core_landscape_from_scans(
                space,
                &c_score_scan,
                &c_level_scan,
                &env_scan,
            ))
        } else {
            None
        };
        if let Some(ref mut ls) = landscape {
            ls.pitch_objective_mode = if phase_mode.score_sign(sweep) < 0.0 {
                PitchObjectiveMode::NegativeConsonance
            } else {
                PitchObjectiveMode::Consonance
            };
        }
        if density_mass.is_finite() {
            density_mass_sum += density_mass;
            density_mass_min = density_mass_min.min(density_mass);
            density_mass_max = density_mass_max.max(density_mass);
            density_mass_count += 1;
        }
        if r_state_stats.mean.is_finite() {
            r_state01_min = r_state01_min.min(r_state_stats.min);
            r_state01_max = r_state01_max.max(r_state_stats.max);
            r_state01_mean_sum += r_state_stats.mean;
            r_state01_mean_count += 1;
        }

        let mean_c = mean_at_indices(&c_score_scan, &agent_indices);
        let mean_c_level = mean_at_indices(&c_level_scan, &agent_indices);
        mean_c_series.push(mean_c);
        mean_c_level_series.push(mean_c_level);

        if sweep == 0 {
            record_e2_trajectory_snapshot(
                &mut trajectory_semitones,
                &agent_indices,
                &log2_ratio_scan,
            );
        }

        if sweep >= E2_BURN_IN {
            let target = if anchor_shift_enabled && sweep < E2_ANCHOR_SHIFT_STEP {
                &mut semitone_samples_pre
            } else {
                &mut semitone_samples_post
            };
            target.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }

        let temperature = e2_accept_temperature(sweep, phase_mode);
        let score_sign = phase_mode.score_sign(sweep);
        let block_backtrack = e2_should_block_backtrack(phase_mode, sweep);
        let positions_before_update = agent_indices.clone();
        let stats = match condition {
            E2Condition::Baseline => update_e2_sweep_pitch_core_proposal(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
                &mut pitch_cores,
                space,
                landscape
                    .as_ref()
                    .expect("proposal landscape missing for baseline"),
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &erb_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                score_sign,
                E2_CROWDING_WEIGHT,
                &kernel_params,
                temperature,
                sweep,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                Some(trajectory_semitones.as_mut_slice()),
                &mut rng,
            ),
            E2Condition::NoCrowding => {
                if score_sign < 0.0 {
                    update_e2_sweep_scored_loo(
                        E2_UPDATE_SCHEDULE,
                        &mut agent_indices,
                        &positions_before_update,
                        space,
                        &workspace,
                        &env_scan,
                        &density_scan,
                        &du_scan,
                        &erb_scan,
                        &log2_ratio_scan,
                        min_idx,
                        max_idx,
                        k_bins,
                        score_sign,
                        0.0,
                        &kernel_params,
                        temperature,
                        sweep,
                        block_backtrack,
                        if block_backtrack {
                            Some(backtrack_targets.as_slice())
                        } else {
                            None
                        },
                        Some(trajectory_semitones.as_mut_slice()),
                        &mut rng,
                    )
                } else {
                    update_e2_sweep_pitch_core_proposal(
                        E2_UPDATE_SCHEDULE,
                        &mut agent_indices,
                        &positions_before_update,
                        &mut pitch_cores,
                        space,
                        landscape
                            .as_ref()
                            .expect("proposal landscape missing for no crowding"),
                        &workspace,
                        &env_scan,
                        &density_scan,
                        &du_scan,
                        &erb_scan,
                        &log2_ratio_scan,
                        min_idx,
                        max_idx,
                        score_sign,
                        0.0,
                        &kernel_params,
                        temperature,
                        sweep,
                        block_backtrack,
                        if block_backtrack {
                            Some(backtrack_targets.as_slice())
                        } else {
                            None
                        },
                        Some(trajectory_semitones.as_mut_slice()),
                        &mut rng,
                    )
                }
            }
            E2Condition::NoHillClimb => update_e2_sweep_nohill(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &erb_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                score_sign,
                E2_CROWDING_WEIGHT,
                &kernel_params,
                sweep,
                Some(trajectory_semitones.as_mut_slice()),
                &mut rng,
            ),
            E2Condition::ShuffledLandscape => {
                let shuffled_scores: Vec<f32> =
                    shuffle_perm.iter().map(|&i| c_score_scan[i]).collect();
                if score_sign < 0.0 {
                    update_e2_sweep_prescored(
                        E2_UPDATE_SCHEDULE,
                        &mut agent_indices,
                        &positions_before_update,
                        &shuffled_scores,
                        &erb_scan,
                        &log2_ratio_scan,
                        min_idx,
                        max_idx,
                        k_bins,
                        score_sign,
                        E2_CROWDING_WEIGHT,
                        &kernel_params,
                        temperature,
                        sweep,
                        block_backtrack,
                        if block_backtrack {
                            Some(backtrack_targets.as_slice())
                        } else {
                            None
                        },
                        Some(trajectory_semitones.as_mut_slice()),
                        &mut rng,
                    )
                } else {
                    update_e2_sweep_pitch_core_proposal_prescored(
                        E2_UPDATE_SCHEDULE,
                        &mut agent_indices,
                        &positions_before_update,
                        &mut pitch_cores,
                        space,
                        landscape
                            .as_ref()
                            .expect("proposal landscape missing for shuffled"),
                        &shuffled_scores,
                        &erb_scan,
                        &log2_ratio_scan,
                        min_idx,
                        max_idx,
                        score_sign,
                        E2_CROWDING_WEIGHT,
                        &kernel_params,
                        temperature,
                        sweep,
                        block_backtrack,
                        if block_backtrack {
                            Some(backtrack_targets.as_slice())
                        } else {
                            None
                        },
                        Some(trajectory_semitones.as_mut_slice()),
                        &mut rng,
                    )
                }
            }
        };
        e2_update_backtrack_targets(
            &mut backtrack_targets,
            &positions_before_update,
            &agent_indices,
        );
        let condition_label = match condition {
            E2Condition::Baseline => "baseline",
            E2Condition::NoCrowding => "nocrowd",
            E2Condition::NoHillClimb => "nohill",
            E2Condition::ShuffledLandscape => "shuffled",
        };
        debug_assert!(
            stats.mean_c_score_current_loo.is_finite(),
            "mean_c_score_current_loo not finite (cond={condition_label}, sweep={sweep}, value={})",
            stats.mean_c_score_current_loo
        );
        debug_assert!(
            stats.mean_c_score_chosen_loo.is_finite(),
            "mean_c_score_chosen_loo not finite (cond={condition_label}, sweep={sweep}, value={})",
            stats.mean_c_score_chosen_loo
        );
        mean_c_score_loo_series.push(stats.mean_c_score_current_loo);
        mean_c_score_chosen_loo_series.push(stats.mean_c_score_chosen_loo);
        mean_score_series.push(stats.mean_score);
        mean_crowding_series.push(stats.mean_crowding);
        moved_frac_series.push(stats.moved_frac);
        accepted_worse_frac_series.push(stats.accepted_worse_frac);
        attempted_update_frac_series.push(stats.attempted_update_frac);
        moved_given_attempt_frac_series.push(stats.moved_given_attempt_frac);
        mean_abs_delta_semitones_series.push(stats.mean_abs_delta_semitones);
        mean_abs_delta_semitones_moved_series.push(stats.mean_abs_delta_semitones_moved);
    }

    let mut final_semitones = Vec::with_capacity(n_agents);
    let mut final_log2_ratios = Vec::with_capacity(n_agents);
    let mut final_freqs_hz = Vec::with_capacity(n_agents);
    for &idx in &agent_indices {
        final_semitones.push(12.0 * log2_ratio_scan[idx]);
        final_log2_ratios.push(log2_ratio_scan[idx]);
        final_freqs_hz.push(space.centers_hz[idx]);
    }

    let density_mass_mean = if density_mass_count > 0 {
        density_mass_sum / density_mass_count as f32
    } else {
        0.0
    };
    let density_mass_min = if density_mass_min.is_finite() {
        density_mass_min
    } else {
        0.0
    };
    let density_mass_max = if density_mass_max.is_finite() {
        density_mass_max
    } else {
        0.0
    };
    let r_state01_mean = if r_state01_mean_count > 0 {
        r_state01_mean_sum / r_state01_mean_count as f32
    } else {
        0.0
    };
    let r_state01_min = if r_state01_min.is_finite() {
        r_state01_min
    } else {
        0.0
    };
    let r_state01_max = if r_state01_max.is_finite() {
        r_state01_max
    } else {
        0.0
    };
    let trajectory_c_level = compute_e2_trajectory_c_levels(
        space,
        &workspace,
        &du_scan,
        fixed_drone.idx,
        anchor_hz_current,
        &trajectory_semitones,
    );

    E2Run {
        seed,
        mean_c_series,
        mean_c_level_series,
        mean_c_score_loo_series,
        mean_c_score_chosen_loo_series,
        mean_score_series,
        mean_crowding_series,
        moved_frac_series,
        accepted_worse_frac_series,
        attempted_update_frac_series,
        moved_given_attempt_frac_series,
        mean_abs_delta_semitones_series,
        mean_abs_delta_semitones_moved_series,
        semitone_samples_pre,
        semitone_samples_post,
        final_semitones,
        final_freqs_hz,
        final_log2_ratios,
        trajectory_semitones,
        trajectory_c_level,
        anchor_shift,
        density_mass_mean,
        density_mass_min,
        density_mass_max,
        r_state01_min,
        r_state01_mean,
        r_state01_max,
        r_ref_peak: workspace.r_ref_peak,
        roughness_k: workspace.params.roughness_k,
        roughness_ref_eps: workspace.params.roughness_ref_eps,
        fixed_drone_hz: fixed_drone.hz,
        n_agents,
        k_bins,
    }
}

fn e2_seed_sweep(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
) -> (Vec<E2Run>, E2SweepStats) {
    e2_seed_sweep_cfg(
        space,
        anchor_hz,
        condition,
        step_semitones,
        phase_mode,
        kernel,
        r_shift_bins,
        E2_N_AGENTS,
        2.0,
    )
}

fn e2_seed_sweep_cfg(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
    n_agents: usize,
    range_oct: f32,
) -> (Vec<E2Run>, E2SweepStats) {
    e2_seed_sweep_with_threads(
        space,
        anchor_hz,
        condition,
        step_semitones,
        phase_mode,
        kernel,
        r_shift_bins,
        n_agents,
        range_oct,
        None,
    )
}

fn e2_seed_sweep_with_threads(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
    n_agents: usize,
    range_oct: f32,
    max_worker_threads: Option<usize>,
) -> (Vec<E2Run>, E2SweepStats) {
    e2_seed_sweep_with_threads_for_seeds(
        space,
        anchor_hz,
        condition,
        step_semitones,
        phase_mode,
        kernel,
        r_shift_bins,
        n_agents,
        range_oct,
        &E2_SEEDS,
        max_worker_threads,
    )
}

fn e2_seed_sweep_with_threads_for_seeds(
    space: &Log2Space,
    anchor_hz: f32,
    condition: E2Condition,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    kernel: Option<ConsonanceKernel>,
    r_shift_bins: i32,
    n_agents: usize,
    range_oct: f32,
    seeds: &[u64],
    max_worker_threads: Option<usize>,
) -> (Vec<E2Run>, E2SweepStats) {
    let max_threads = max_worker_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let worker_count = max_threads.min(seeds.len()).max(1);
    let runs = if worker_count <= 1 || seeds.len() <= 1 {
        let mut runs = Vec::with_capacity(seeds.len());
        for &seed in seeds {
            runs.push(run_e2_once_cfg(
                space,
                anchor_hz,
                seed,
                condition,
                step_semitones,
                phase_mode,
                kernel,
                r_shift_bins,
                n_agents,
                range_oct,
            ));
        }
        runs
    } else {
        let next = AtomicUsize::new(0);
        let runs = Mutex::new({
            let mut runs = Vec::with_capacity(seeds.len());
            runs.resize_with(seeds.len(), || None);
            runs
        });
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                scope.spawn(|| {
                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= seeds.len() {
                            break;
                        }
                        let seed = seeds[idx];
                        let run = run_e2_once_cfg(
                            space,
                            anchor_hz,
                            seed,
                            condition,
                            step_semitones,
                            phase_mode,
                            kernel,
                            r_shift_bins,
                            n_agents,
                            range_oct,
                        );
                        let mut guard = runs.lock().expect("runs lock poisoned");
                        guard[idx] = Some(run);
                    }
                });
            }
        });
        runs.into_inner()
            .expect("runs lock poisoned")
            .into_iter()
            .map(|run| run.expect("missing run"))
            .collect()
    };

    let n = runs.len();
    let mean_c = mean_std_series(runs.iter().map(|r| &r.mean_c_series).collect::<Vec<_>>());
    let mean_c_level = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_c_level_series)
            .collect::<Vec<_>>(),
    );
    let mean_c_score_loo = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_c_score_loo_series)
            .collect::<Vec<_>>(),
    );
    let g_scene_series: Vec<Vec<f32>> = runs
        .iter()
        .map(|r| {
            e2_scene_g_series(r, space)
                .into_iter()
                .map(|point| point.g_scene)
                .collect()
        })
        .collect();
    let mean_g_scene = mean_std_series(g_scene_series.iter().collect::<Vec<_>>());
    let mean_score = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_score_series)
            .collect::<Vec<_>>(),
    );
    let mean_crowding = mean_std_series(
        runs.iter()
            .map(|r| &r.mean_crowding_series)
            .collect::<Vec<_>>(),
    );

    (
        runs,
        E2SweepStats {
            mean_c: mean_c.0,
            std_c: mean_c.1,
            mean_c_level: mean_c_level.0,
            std_c_level: mean_c_level.1,
            mean_c_score_loo: mean_c_score_loo.0,
            std_c_score_loo: mean_c_score_loo.1,
            mean_g_scene: mean_g_scene.0,
            std_g_scene: mean_g_scene.1,
            mean_score: mean_score.0,
            std_score: mean_score.1,
            mean_crowding: mean_crowding.0,
            std_crowding: mean_crowding.1,
            n,
        },
    )
}

fn parallel_map_ordered<T, R, F>(
    items: &[T],
    max_worker_threads: Option<usize>,
    map_fn: F,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    if items.is_empty() {
        return Vec::new();
    }
    let max_threads = max_worker_threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let worker_count = max_threads.min(items.len()).max(1);
    if worker_count <= 1 || items.len() <= 1 {
        return items.iter().map(map_fn).collect();
    }

    let next = AtomicUsize::new(0);
    let results = Mutex::new({
        let mut results = Vec::with_capacity(items.len());
        results.resize_with(items.len(), || None);
        results
    });
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= items.len() {
                        break;
                    }
                    let result = map_fn(&items[idx]);
                    let mut guard = results.lock().expect("results lock poisoned");
                    guard[idx] = Some(result);
                }
            });
        }
    });
    results
        .into_inner()
        .expect("results lock poisoned")
        .into_iter()
        .map(|result| result.expect("missing parallel result"))
        .collect()
}

fn e2_kbins_sweep_csv(space: &Log2Space, anchor_hz: f32, phase_mode: E2PhaseMode) -> String {
    let mut out = String::from(
        "step_semitones,k_bins,mean_delta_c,mean_delta_c_level,mean_delta_c_score_loo\n",
    );
    for &step_semitones in &E2_STEP_SEMITONES_SWEEP {
        let (runs, _) = e2_seed_sweep(
            space,
            anchor_hz,
            E2Condition::Baseline,
            step_semitones,
            phase_mode,
            None,
            0,
        );
        let mut delta_c = Vec::with_capacity(runs.len());
        let mut delta_c_level = Vec::with_capacity(runs.len());
        let mut delta_c_score_loo = Vec::with_capacity(runs.len());
        for run in &runs {
            let start = run.mean_c_series.first().copied().unwrap_or(0.0);
            let end = run.mean_c_series.last().copied().unwrap_or(start);
            delta_c.push(end - start);
            let start_level = run.mean_c_level_series.first().copied().unwrap_or(0.0);
            let end_level = run
                .mean_c_level_series
                .last()
                .copied()
                .unwrap_or(start_level);
            delta_c_level.push(end_level - start_level);
            let start_loo = run.mean_c_score_loo_series.first().copied().unwrap_or(0.0);
            let end_loo = run
                .mean_c_score_loo_series
                .last()
                .copied()
                .unwrap_or(start_loo);
            delta_c_score_loo.push(end_loo - start_loo);
        }
        let (mean_delta_c, _) = mean_std_scalar(&delta_c);
        let (mean_delta_c_level, _) = mean_std_scalar(&delta_c_level);
        let (mean_delta_c_score_loo, _) = mean_std_scalar(&delta_c_score_loo);
        out.push_str(&format!(
            "{:.3},{},{:.6},{:.6},{:.6}\n",
            step_semitones,
            k_from_semitones(step_semitones),
            mean_delta_c,
            mean_delta_c_level,
            mean_delta_c_score_loo
        ));
    }
    out
}

fn plot_e3_metabolic_selection(
    out_dir: &Path,
    _space: &Log2Space,
    _anchor_hz: f32,
) -> Result<(), Box<dyn Error>> {
    let conditions = [E3Condition::Baseline, E3Condition::NoRecharge];

    let mut long_csv = String::from(
        "condition,seed,life_id,agent_id,birth_step,death_step,lifetime_steps,c_score_birth,c_score_firstk,avg_c_score_tick,c_score_std_over_life,avg_c_score_attack,c_level01_birth,c_level01_firstk,avg_c_level01_tick,c_level01_std_over_life,avg_c_level01_attack,attack_tick_count\n",
    );
    let mut summary_csv = String::from(
        "condition,seed,n_deaths,pearson_r_firstk,pearson_p_firstk,spearman_rho_firstk,spearman_p_firstk,logrank_p_firstk,logrank_p_firstk_q25q75,median_high_firstk,median_low_firstk,pearson_r_birth,pearson_p_birth,spearman_rho_birth,spearman_p_birth,pearson_r_attack,pearson_p_attack,spearman_rho_attack,spearman_p_attack,n_attack_lives\n",
    );
    let mut policy_csv = String::from(
        "condition,dt_sec,basal_cost_per_sec,action_cost_per_attack,recharge_per_attack,continuous_recharge_per_sec\n",
    );
    for condition in conditions {
        let params = e3_policy_params(condition);
        policy_csv.push_str(&format!(
            "{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            params.condition,
            params.dt_sec,
            params.basal_cost_per_sec,
            params.action_cost_per_attack,
            params.recharge_per_attack,
            params.continuous_recharge_per_sec
        ));
    }
    write_with_log(out_dir.join("paper_e3_policy_params.csv"), policy_csv)?;
    write_with_log(
        out_dir.join("paper_e3_metric_definition.txt"),
        e3_metric_definition_text(),
    )?;

    let mut seed_outputs: Vec<E3SeedOutput> = Vec::new();

    for condition in conditions {
        let cond_label = condition.label();
        for &seed in &E3_SEEDS {
            let cfg = E3RunConfig {
                seed,
                steps_cap: E3_STEPS_CAP,
                min_deaths: E3_MIN_DEATHS,
                pop_size: E3_POP_SIZE,
                first_k: E3_FIRST_K,
                condition,
            };
            let deaths = run_e3_collect_deaths(&cfg);

            for rec in &deaths {
                long_csv.push_str(&format!(
                    "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                    rec.condition,
                    rec.seed,
                    rec.life_id,
                    rec.agent_id,
                    rec.birth_step,
                    rec.death_step,
                    rec.lifetime_steps,
                    rec.c_score_birth,
                    rec.c_score_firstk,
                    rec.avg_c_score_tick,
                    rec.c_score_std_over_life,
                    rec.avg_c_score_attack,
                    rec.c_level_birth,
                    rec.c_level_firstk,
                    rec.avg_c_level_tick,
                    rec.c_level_std_over_life,
                    rec.avg_c_level_attack,
                    rec.attack_tick_count
                ));
            }

            let lifetimes_path = out_dir.join(format!(
                "paper_e3_lifetimes_seed{}_{}.csv",
                seed, cond_label
            ));
            write_with_log(lifetimes_path, e3_lifetimes_csv(&deaths))?;

            let arrays = e3_extract_arrays(&deaths);

            let scatter_firstk_path = out_dir.join(format!(
                "paper_e3_firstk_vs_lifetime_seed{}_{}.svg",
                seed, cond_label
            ));
            let corr_stats_firstk = render_e3_scatter_with_stats(
                &scatter_firstk_path,
                "C_level01_firstK vs Lifetime",
                "C_level01_firstK",
                &arrays.c_level_firstk,
                &arrays.lifetimes,
                seed ^ 0xE301_u64,
            )?;

            let scatter_birth_path = out_dir.join(format!(
                "paper_e3_birth_vs_lifetime_seed{}_{}.svg",
                seed, cond_label
            ));
            let corr_stats_birth = render_e3_scatter_with_stats(
                &scatter_birth_path,
                "C_level01_birth vs Lifetime",
                "C_level01_birth",
                &arrays.c_level_birth,
                &arrays.lifetimes,
                seed ^ 0xE302_u64,
            )?;

            let survival_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_seed{}_{}.svg",
                seed, cond_label
            ));
            let surv_firstk_stats = render_survival_split_plot(
                &survival_path,
                "Survival by C_level01_firstK (median split)",
                &arrays.lifetimes,
                &arrays.c_level_firstk,
                SplitKind::Median,
                seed ^ 0xE310_u64,
            )?;

            let survival_q_path = out_dir.join(format!(
                "paper_e3_survival_by_firstk_q25q75_seed{}_{}.svg",
                seed, cond_label
            ));
            let surv_firstk_q_stats = render_survival_split_plot(
                &survival_q_path,
                "Survival by C_level01_firstK (q25 vs q75)",
                &arrays.lifetimes,
                &arrays.c_level_firstk,
                SplitKind::Quartiles,
                seed ^ 0xE311_u64,
            )?;

            let (attack_lifetimes, attack_vals) = e3_attack_subset(&arrays);
            let attack_lives = attack_vals.len();
            let mut corr_stats_attack = None;
            if attack_lives >= 10 {
                let attack_scatter_path = out_dir.join(format!(
                    "paper_e3_attack_vs_lifetime_seed{}_{}.svg",
                    seed, cond_label
                ));
                corr_stats_attack = Some(render_e3_scatter_with_stats(
                    &attack_scatter_path,
                    "C_level01_attack vs Lifetime",
                    "C_level01_attack",
                    &attack_vals,
                    &attack_lifetimes,
                    seed ^ 0xE303_u64,
                )?);

                let attack_survival_path = out_dir.join(format!(
                    "paper_e3_survival_by_attack_seed{}_{}.svg",
                    seed, cond_label
                ));
                let _ = render_survival_split_plot(
                    &attack_survival_path,
                    "Survival by C_level01_attack (median split)",
                    &attack_lifetimes,
                    &attack_vals,
                    SplitKind::Median,
                    seed ^ 0xE313_u64,
                )?;
            }

            if cond_label == "baseline" && seed == E3_SEEDS[0] {
                let mut legacy_csv = String::from("life_id,lifetime_steps,c_level01_firstk\n");
                let mut legacy_deaths = Vec::with_capacity(deaths.len());
                for d in &deaths {
                    legacy_csv.push_str(&format!(
                        "{},{},{:.6}\n",
                        d.life_id, d.lifetime_steps, d.c_level_firstk
                    ));
                    legacy_deaths.push((d.life_id as usize, d.lifetime_steps, d.c_level_firstk));
                }
                write_with_log(out_dir.join("paper_e3_lifetimes.csv"), legacy_csv)?;
                let legacy_scatter = out_dir.join("paper_e3_firstk_vs_lifetime.svg");
                render_consonance_lifetime_scatter(&legacy_scatter, &legacy_deaths)?;
                let legacy_survival = out_dir.join("paper_e3_survival_curve.svg");
                render_survival_curve(&legacy_survival, &legacy_deaths)?;
                let legacy_survival_c_level = out_dir.join("paper_e3_survival_by_c_level.svg");
                render_survival_by_c_level(&legacy_survival_c_level, &legacy_deaths)?;
            }

            summary_csv.push_str(&format!(
                "{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                cond_label,
                seed,
                arrays.lifetimes.len(),
                corr_stats_firstk.pearson_r,
                corr_stats_firstk.pearson_p,
                corr_stats_firstk.spearman_rho,
                corr_stats_firstk.spearman_p,
                surv_firstk_stats.logrank_p,
                surv_firstk_q_stats.logrank_p,
                surv_firstk_stats.median_high,
                surv_firstk_stats.median_low,
                corr_stats_birth.pearson_r,
                corr_stats_birth.pearson_p,
                corr_stats_birth.spearman_rho,
                corr_stats_birth.spearman_p,
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.pearson_r)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.pearson_p)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.spearman_rho)
                    .unwrap_or(f32::NAN),
                corr_stats_attack
                    .as_ref()
                    .map(|s| s.spearman_p)
                    .unwrap_or(f32::NAN),
                attack_lives
            ));

            seed_outputs.push(E3SeedOutput {
                condition,
                seed,
                arrays,
                corr_firstk: corr_stats_firstk,
            });
        }
    }

    if let Some(rep_seed) = pick_e3_representative_seed(&seed_outputs) {
        let mut rep_note =
            String::from("Representative seed selection (baseline firstK Pearson r):\n");
        let mut baseline_stats: Vec<(u64, f32)> = seed_outputs
            .iter()
            .filter(|o| o.condition == E3Condition::Baseline)
            .map(|o| (o.seed, o.corr_firstk.pearson_r))
            .collect();
        baseline_stats.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (seed, r) in &baseline_stats {
            rep_note.push_str(&format!("{seed},{r:.6}\n"));
        }
        rep_note.push_str(&format!("chosen_seed={rep_seed}\n"));
        write_with_log(out_dir.join("paper_e3_representative_seed.txt"), rep_note)?;

        let base = seed_outputs
            .iter()
            .find(|o| o.condition == E3Condition::Baseline && o.seed == rep_seed);
        let norecharge = seed_outputs
            .iter()
            .find(|o| o.condition == E3Condition::NoRecharge && o.seed == rep_seed);
        if let (Some(base), Some(norecharge)) = (base, norecharge) {
            let base_scatter = build_scatter_data(
                &base.arrays.c_score_firstk,
                &base.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let norecharge_scatter = build_scatter_data(
                &norecharge.arrays.c_score_firstk,
                &norecharge.arrays.lifetimes,
                rep_seed ^ 0xE301_u64,
            );
            let compare_scatter = out_dir.join("paper_e3_firstk_scatter_compare.svg");
            render_scatter_compare(
                &compare_scatter,
                "C_score_firstK vs Lifetime",
                "C_score_firstK",
                "Baseline",
                &base_scatter,
                "NoRecharge",
                &norecharge_scatter,
            )?;

            let base_surv = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c_level_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let norecharge_surv = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c_level_firstk,
                SplitKind::Median,
                rep_seed ^ 0xE310_u64,
            );
            let compare_surv = out_dir.join("paper_e3_firstk_survival_compare.svg");
            render_survival_compare(
                &compare_surv,
                "Survival by C_level01_firstK (median split)",
                "Baseline",
                &base_surv,
                "NoRecharge",
                &norecharge_surv,
            )?;

            let base_surv_q = build_survival_data(
                &base.arrays.lifetimes,
                &base.arrays.c_level_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let norecharge_surv_q = build_survival_data(
                &norecharge.arrays.lifetimes,
                &norecharge.arrays.c_level_firstk,
                SplitKind::Quartiles,
                rep_seed ^ 0xE311_u64,
            );
            let compare_surv_q = out_dir.join("paper_e3_firstk_survival_compare_q25q75.svg");
            render_survival_compare(
                &compare_surv_q,
                "Survival by C_level01_firstK (q25 vs q75)",
                "Baseline",
                &base_surv_q,
                "NoRecharge",
                &norecharge_surv_q,
            )?;
        }
    }

    let pooled_baseline = e3_pooled_arrays(&seed_outputs, E3Condition::Baseline);
    let pooled_norecharge = e3_pooled_arrays(&seed_outputs, E3Condition::NoRecharge);
    let pooled_scatter_path = out_dir.join("paper_e3_firstk_scatter_compare_pooled.svg");
    let pooled_base_scatter = build_scatter_data(
        &pooled_baseline.c_score_firstk,
        &pooled_baseline.lifetimes,
        0xE3B0_u64,
    );
    let pooled_nore_scatter = build_scatter_data(
        &pooled_norecharge.c_score_firstk,
        &pooled_norecharge.lifetimes,
        0xE3B1_u64,
    );
    render_scatter_compare(
        &pooled_scatter_path,
        "C_score_firstK vs Lifetime (pooled)",
        "C_score_firstK",
        "Baseline",
        &pooled_base_scatter,
        "NoRecharge",
        &pooled_nore_scatter,
    )?;

    let pooled_surv_base = build_survival_data(
        &pooled_baseline.lifetimes,
        &pooled_baseline.c_level_firstk,
        SplitKind::Median,
        0xE3B2_u64,
    );
    let pooled_surv_nore = build_survival_data(
        &pooled_norecharge.lifetimes,
        &pooled_norecharge.c_level_firstk,
        SplitKind::Median,
        0xE3B3_u64,
    );
    let pooled_surv_path = out_dir.join("paper_e3_firstk_survival_compare_pooled.svg");
    render_survival_compare(
        &pooled_surv_path,
        "Survival by Early Consonance",
        "Baseline",
        &pooled_surv_base,
        "NoRecharge",
        &pooled_surv_nore,
    )?;

    let figure4_path = out_dir.join("paper_e3_figure4.svg");
    render_e3_figure4(
        &figure4_path,
        &pooled_surv_base,
        &pooled_surv_nore,
        &pooled_base_scatter,
        &pooled_nore_scatter,
    )?;

    let pooled_surv_base_q = build_survival_data(
        &pooled_baseline.lifetimes,
        &pooled_baseline.c_level_firstk,
        SplitKind::Quartiles,
        0xE3B4_u64,
    );
    let pooled_surv_nore_q = build_survival_data(
        &pooled_norecharge.lifetimes,
        &pooled_norecharge.c_level_firstk,
        SplitKind::Quartiles,
        0xE3B5_u64,
    );
    let pooled_surv_q_path = out_dir.join("paper_e3_firstk_survival_compare_pooled_q25q75.svg");
    render_survival_compare(
        &pooled_surv_q_path,
        "Survival by C_level01_firstK (q25 vs q75, pooled)",
        "Baseline",
        &pooled_surv_base_q,
        "NoRecharge",
        &pooled_surv_nore_q,
    )?;

    let mut pooled_summary_csv = String::from(
        "condition,n,pearson_r_firstk,pearson_p_firstk,spearman_rho_firstk,spearman_p_firstk,logrank_p_firstk,median_high_firstk,median_low_firstk,median_diff_firstk\n",
    );
    for (condition, scatter, survival) in [
        (
            E3Condition::Baseline,
            &pooled_base_scatter,
            &pooled_surv_base,
        ),
        (
            E3Condition::NoRecharge,
            &pooled_nore_scatter,
            &pooled_surv_nore,
        ),
    ] {
        let median_diff = survival.stats.median_high - survival.stats.median_low;
        pooled_summary_csv.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.3}\n",
            condition.label(),
            scatter.stats.n,
            scatter.stats.pearson_r,
            scatter.stats.pearson_p,
            scatter.stats.spearman_rho,
            scatter.stats.spearman_p,
            survival.stats.logrank_p,
            survival.stats.median_high,
            survival.stats.median_low,
            median_diff
        ));
    }
    write_with_log(
        out_dir.join("paper_e3_summary_pooled.csv"),
        pooled_summary_csv,
    )?;

    let pooled_hist_path = out_dir.join("paper_e3_firstk_hist.svg");
    render_e3_firstk_histogram(
        &pooled_hist_path,
        &pooled_baseline.c_level_firstk,
        &pooled_norecharge.c_level_firstk,
        0.02,
        0.5,
    )?;

    write_with_log(out_dir.join("paper_e3_lifetimes_long.csv"), long_csv)?;
    write_with_log(out_dir.join("paper_e3_summary_by_seed.csv"), summary_csv)?;

    // --- Seed-level summary statistics (B1) ---
    {
        let baseline_rs: Vec<f64> = seed_outputs
            .iter()
            .filter(|o| o.condition == E3Condition::Baseline)
            .map(|o| o.corr_firstk.pearson_r as f64)
            .collect();
        let norecharge_rs: Vec<f64> = seed_outputs
            .iter()
            .filter(|o| o.condition == E3Condition::NoRecharge)
            .map(|o| o.corr_firstk.pearson_r as f64)
            .collect();

        fn mean_sd(xs: &[f64]) -> (f64, f64) {
            let n = xs.len() as f64;
            let mean = xs.iter().sum::<f64>() / n;
            let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            (mean, var.sqrt())
        }

        /// Lanczos approximation of ln(Gamma(x))
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

        /// Continued-fraction evaluation for I_x(a,b) (Numerical Recipes betacf)
        fn betacf(a: f64, b: f64, x: f64) -> f64 {
            let max_iter = 200;
            let eps = 3.0e-12;
            let fpmin = 1.0e-30;
            let qab = a + b;
            let qap = a + 1.0;
            let qam = a - 1.0;
            let mut c = 1.0_f64;
            let mut d = (1.0 - qab * x / qap).recip();
            if d.abs() < fpmin {
                d = fpmin;
            }
            let mut h = d;
            for m in 1..=max_iter {
                let m_f = m as f64;
                // even step
                let aa = m_f * (b - m_f) * x / ((qam + 2.0 * m_f) * (a + 2.0 * m_f));
                d = 1.0 + aa * d;
                if d.abs() < fpmin {
                    d = fpmin;
                }
                c = 1.0 + aa / c;
                if c.abs() < fpmin {
                    c = fpmin;
                }
                d = d.recip();
                h *= d * c;
                // odd step
                let aa = -(a + m_f) * (qab + m_f) * x / ((a + 2.0 * m_f) * (qap + 2.0 * m_f));
                d = 1.0 + aa * d;
                if d.abs() < fpmin {
                    d = fpmin;
                }
                c = 1.0 + aa / c;
                if c.abs() < fpmin {
                    c = fpmin;
                }
                d = d.recip();
                let delta = d * c;
                h *= delta;
                if (delta - 1.0).abs() < eps {
                    break;
                }
            }
            h
        }

        /// Regularized incomplete beta I_x(a,b)
        fn reg_inc_beta(a: f64, b: f64, x: f64) -> f64 {
            if x <= 0.0 {
                return 0.0;
            }
            if x >= 1.0 {
                return 1.0;
            }
            let bt =
                (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln())
                    .exp();
            if x < (a + 1.0) / (a + b + 2.0) {
                bt * betacf(a, b, x) / a
            } else {
                1.0 - bt * betacf(b, a, 1.0 - x) / b
            }
        }

        /// Two-tailed p-value for Student's t with given df
        fn two_tailed_t_p(t_abs: f64, df: f64) -> f64 {
            let x = df / (df + t_abs * t_abs);
            reg_inc_beta(df / 2.0, 0.5, x)
        }

        fn one_sample_t(xs: &[f64], mu0: f64) -> (f64, usize, f64) {
            let n = xs.len();
            let (mean, sd) = mean_sd(xs);
            let t = (mean - mu0) / (sd / (n as f64).sqrt());
            let df = n - 1;
            let p = two_tailed_t_p(t.abs(), df as f64);
            (t, df, p)
        }

        fn welch_t(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
            let (ma, sa) = mean_sd(a);
            let (mb, sb) = mean_sd(b);
            let na = a.len() as f64;
            let nb = b.len() as f64;
            let va = sa * sa / na;
            let vb = sb * sb / nb;
            let t = (ma - mb) / (va + vb).sqrt();
            let df = (va + vb).powi(2) / (va * va / (na - 1.0) + vb * vb / (nb - 1.0));
            let p = two_tailed_t_p(t.abs(), df);
            (t, df, p)
        }

        // Fisher z transform for averaging correlation coefficients
        fn fisher_z(r: f64) -> f64 {
            // arctanh, clamped to avoid ±inf
            let r_clamped = r.clamp(-0.9999, 0.9999);
            0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln()
        }
        fn fisher_z_inv(z: f64) -> f64 {
            z.tanh()
        }

        let baseline_zs: Vec<f64> = baseline_rs.iter().map(|&r| fisher_z(r)).collect();
        let norecharge_zs: Vec<f64> = norecharge_rs.iter().map(|&r| fisher_z(r)).collect();

        let (base_z_mean, base_z_sd) = mean_sd(&baseline_zs);
        let (nore_z_mean, nore_z_sd) = mean_sd(&norecharge_zs);
        let base_r_mean = fisher_z_inv(base_z_mean);
        let nore_r_mean = fisher_z_inv(nore_z_mean);

        // Raw r stats for reference
        let (base_raw_mean, base_raw_sd) = mean_sd(&baseline_rs);
        let (nore_raw_mean, nore_raw_sd) = mean_sd(&norecharge_rs);

        // One-sample t on Fisher z (H0: z = 0, i.e., r = 0)
        let (t_one, df_one, p_one) = one_sample_t(&baseline_zs, 0.0);
        // Welch t on Fisher z (baseline vs norecharge)
        let (t_welch, df_welch, p_welch) = welch_t(&baseline_zs, &norecharge_zs);

        // Range of raw r
        let base_r_min = baseline_rs.iter().cloned().fold(f64::INFINITY, f64::min);
        let base_r_max = baseline_rs
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        // CI95 for Fisher z mean, back-transformed
        let base_z_ci95 = 1.96 * base_z_sd / (baseline_zs.len() as f64).sqrt();
        let base_r_ci_lo = fisher_z_inv(base_z_mean - base_z_ci95);
        let base_r_ci_hi = fisher_z_inv(base_z_mean + base_z_ci95);

        let mut stats_text = String::new();
        stats_text.push_str("=== E3 Seed-Level Statistics (Fisher z) ===\n\n");
        stats_text.push_str(&format!(
            "Baseline (n={}): mean r = {:.3} (Fisher z mean={:.4}, SD={:.4})\n",
            baseline_rs.len(),
            base_r_mean,
            base_z_mean,
            base_z_sd
        ));
        stats_text.push_str(&format!(
            "  raw r: mean={:.3} ± {:.3}, range [{:.2}, {:.2}]\n",
            base_raw_mean, base_raw_sd, base_r_min, base_r_max
        ));
        stats_text.push_str(&format!(
            "  Fisher z 95% CI for r: [{:.2}, {:.2}]\n",
            base_r_ci_lo, base_r_ci_hi
        ));
        stats_text.push_str(&format!(
            "  per-seed r: {}\n",
            baseline_rs
                .iter()
                .map(|r| format!("{:.3}", r))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        stats_text.push_str(&format!(
            "  one-sample t({}) = {:.3}, p = {:.6} (H0: z = 0)\n\n",
            df_one, t_one, p_one
        ));
        stats_text.push_str(&format!(
            "NoRecharge (n={}): mean r = {:.3} (Fisher z mean={:.4}, SD={:.4})\n",
            norecharge_rs.len(),
            nore_r_mean,
            nore_z_mean,
            nore_z_sd
        ));
        stats_text.push_str(&format!(
            "  raw r: mean={:.3} ± {:.3}\n",
            nore_raw_mean, nore_raw_sd
        ));
        stats_text.push_str(&format!(
            "  per-seed r: {}\n\n",
            norecharge_rs
                .iter()
                .map(|r| format!("{:.3}", r))
                .collect::<Vec<_>>()
                .join(", ")
        ));
        stats_text.push_str(&format!(
            "Welch two-sample t({:.1}) = {:.3}, p = {:.6}\n",
            df_welch, t_welch, p_welch
        ));
        stats_text.push_str("  (Baseline vs NoRecharge, Fisher z)\n");

        write_with_log(out_dir.join("paper_e3_seed_level_stats.txt"), stats_text)?;
    }

    Ok(())
}

fn e3_extract_arrays(deaths: &[E3DeathRecord]) -> E3Arrays {
    let mut lifetimes = Vec::with_capacity(deaths.len());
    let mut c_score_birth = Vec::with_capacity(deaths.len());
    let mut c_score_firstk = Vec::with_capacity(deaths.len());
    let mut c_level_birth = Vec::with_capacity(deaths.len());
    let mut c_level_firstk = Vec::with_capacity(deaths.len());
    let mut avg_score_attack = Vec::with_capacity(deaths.len());
    let mut avg_attack = Vec::with_capacity(deaths.len());
    let mut attack_tick_count = Vec::with_capacity(deaths.len());
    for d in deaths {
        lifetimes.push(d.lifetime_steps);
        c_score_birth.push(d.c_score_birth);
        c_score_firstk.push(d.c_score_firstk);
        c_level_birth.push(d.c_level_birth);
        c_level_firstk.push(d.c_level_firstk);
        avg_score_attack.push(d.avg_c_score_attack);
        avg_attack.push(d.avg_c_level_attack);
        attack_tick_count.push(d.attack_tick_count);
    }
    E3Arrays {
        lifetimes,
        c_score_birth,
        c_score_firstk,
        c_level_birth,
        c_level_firstk,
        avg_score_attack,
        avg_attack,
        attack_tick_count,
    }
}

fn e3_pooled_arrays(outputs: &[E3SeedOutput], condition: E3Condition) -> E3Arrays {
    let mut lifetimes = Vec::new();
    let mut c_score_birth = Vec::new();
    let mut c_score_firstk = Vec::new();
    let mut c_level_birth = Vec::new();
    let mut c_level_firstk = Vec::new();
    let mut avg_score_attack = Vec::new();
    let mut avg_attack = Vec::new();
    let mut attack_tick_count = Vec::new();
    for output in outputs.iter().filter(|o| o.condition == condition) {
        lifetimes.extend(output.arrays.lifetimes.iter().copied());
        c_score_birth.extend(output.arrays.c_score_birth.iter().copied());
        c_score_firstk.extend(output.arrays.c_score_firstk.iter().copied());
        c_level_birth.extend(output.arrays.c_level_birth.iter().copied());
        c_level_firstk.extend(output.arrays.c_level_firstk.iter().copied());
        avg_score_attack.extend(output.arrays.avg_score_attack.iter().copied());
        avg_attack.extend(output.arrays.avg_attack.iter().copied());
        attack_tick_count.extend(output.arrays.attack_tick_count.iter().copied());
    }
    E3Arrays {
        lifetimes,
        c_score_birth,
        c_score_firstk,
        c_level_birth,
        c_level_firstk,
        avg_score_attack,
        avg_attack,
        attack_tick_count,
    }
}

fn fractions_from_counts(counts: &[(f32, f32)]) -> Vec<f32> {
    let total: f32 = counts.iter().map(|(_, v)| *v).sum();
    let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
    counts.iter().map(|(_, v)| v * inv).collect()
}

fn render_e3_firstk_histogram(
    out_path: &Path,
    baseline: &[f32],
    norecharge: &[f32],
    bin_width: f32,
    threshold: f32,
) -> Result<(), Box<dyn Error>> {
    let min = 0.0f32;
    let max = 1.0f32;
    let counts_base = histogram_counts_fixed(baseline, min, max, bin_width);
    let counts_nore = histogram_counts_fixed(norecharge, min, max, bin_width);
    let len = counts_base.len().min(counts_nore.len());
    if len == 0 {
        return Ok(());
    }
    let centers: Vec<f32> = counts_base.iter().take(len).map(|(c, _)| *c).collect();
    let base_frac = fractions_from_counts(&counts_base[..len]);
    let nore_frac = fractions_from_counts(&counts_nore[..len]);

    let mut y_max = 0.0f32;
    for &v in base_frac.iter().chain(nore_frac.iter()) {
        y_max = y_max.max(v);
    }
    y_max = y_max.max(1e-3);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("C_level01_firstK Histogram (pooled)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("C_level01_firstK")
        .y_desc("fraction")
        .x_labels(10)
        .draw()?;

    let base_line = centers.iter().copied().zip(base_frac.iter().copied());
    chart
        .draw_series(LineSeries::new(base_line, PAL_H))?
        .label("baseline")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_H));
    let nore_line = centers.iter().copied().zip(nore_frac.iter().copied());
    chart
        .draw_series(LineSeries::new(nore_line, PAL_R))?
        .label("no recharge")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_R));

    let thresh = threshold.clamp(min, max);
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(thresh, 0.0), (thresh, y_max * 1.05)],
        BLACK.mix(0.6),
    )))?;

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn e3_lifetimes_csv(deaths: &[E3DeathRecord]) -> String {
    let mut out = String::from(
        "life_id,agent_id,birth_step,death_step,lifetime_steps,c_score_birth,c_score_firstk,avg_c_score_tick,c_score_std_over_life,avg_c_score_attack,c_level01_birth,c_level01_firstk,avg_c_level01_tick,c_level01_std_over_life,avg_c_level01_attack,attack_tick_count\n",
    );
    for d in deaths {
        out.push_str(&format!(
            "{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            d.life_id,
            d.agent_id,
            d.birth_step,
            d.death_step,
            d.lifetime_steps,
            d.c_score_birth,
            d.c_score_firstk,
            d.avg_c_score_tick,
            d.c_score_std_over_life,
            d.avg_c_score_attack,
            d.c_level_birth,
            d.c_level_firstk,
            d.avg_c_level_tick,
            d.c_level_std_over_life,
            d.avg_c_level_attack,
            d.attack_tick_count
        ));
    }
    out
}

fn e3_attack_subset(arrays: &E3Arrays) -> (Vec<u32>, Vec<f32>) {
    let mut lifetimes = Vec::new();
    let mut avg_attack = Vec::new();
    for ((&lt, &avg), &count) in arrays
        .lifetimes
        .iter()
        .zip(arrays.avg_attack.iter())
        .zip(arrays.attack_tick_count.iter())
    {
        if count > 0 && avg.is_finite() {
            lifetimes.push(lt);
            avg_attack.push(avg);
        }
    }
    (lifetimes, avg_attack)
}

fn pick_e3_representative_seed(outputs: &[E3SeedOutput]) -> Option<u64> {
    let mut baseline: Vec<(u64, f32)> = outputs
        .iter()
        .filter(|o| o.condition == E3Condition::Baseline)
        .map(|o| (o.seed, o.corr_firstk.pearson_r))
        .collect();
    if baseline.is_empty() {
        return None;
    }
    baseline.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Some(baseline[baseline.len() / 2].0)
}

#[allow(clippy::type_complexity)]
fn plot_e5_vitality_entrainment(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let landscape = e3_reference_landscape(E5_ANCHOR_HZ);

    let conditions = [
        E5Condition::Vitality,
        E5Condition::Uniform,
        E5Condition::Control,
    ];

    // Run all seeds × conditions
    let mut all_results: Vec<E5VitalityResult> = Vec::new();
    for &cond in &conditions {
        for &seed in &E5_SEEDS {
            all_results.push(simulate_e5_vitality(seed, cond, &landscape));
        }
    }

    // ── CSV: per-seed summary ────────────────────────────────────
    let mut summary_csv = String::from("condition,seed,pearson_r,group_plv_final\n");
    for res in &all_results {
        let final_plv = res
            .group_plv_series
            .last()
            .map(|(_, p)| *p)
            .unwrap_or(f32::NAN);
        summary_csv.push_str(&format!(
            "{},{},{:.6},{:.6}\n",
            res.condition.label(),
            res.seed,
            res.pearson_r,
            final_plv,
        ));
    }
    write_with_log(out_dir.join("paper_e5_summary.csv"), &summary_csv)?;

    // ── CSV: per-agent for representative seed ───────────────────
    let mut agent_csv = String::from("condition,seed,agent,consonance,plv_final\n");
    for res in &all_results {
        if res.seed != E5_SEEDS[E5_REPRESENTATIVE_SEED_IDX] {
            continue;
        }
        for (i, a) in res.agent_final.iter().enumerate() {
            agent_csv.push_str(&format!(
                "{},{},{},{:.6},{:.6}\n",
                res.condition.label(),
                res.seed,
                i,
                a.consonance,
                a.plv,
            ));
        }
    }
    write_with_log(out_dir.join("paper_e5_agent_detail.csv"), &agent_csv)?;

    // ── Compute per-condition group PLV series (mean ± 95% CI) ──
    // (scatter data written after stratified sims below)
    let mut cond_series: Vec<(E5Condition, Vec<(f32, f32, f32)>)> = Vec::new();
    for &cond in &conditions {
        let cond_results: Vec<&E5VitalityResult> =
            all_results.iter().filter(|r| r.condition == cond).collect();
        let n_steps = cond_results
            .first()
            .map(|r| r.group_plv_series.len())
            .unwrap_or(0);
        let n_seeds = cond_results.len() as f32;
        let mut series: Vec<(f32, f32, f32)> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let t = cond_results[0].group_plv_series[step].0;
            let vals: Vec<f32> = cond_results
                .iter()
                .filter_map(|r| {
                    let p = r.group_plv_series[step].1;
                    p.is_finite().then_some(p)
                })
                .collect();
            if vals.is_empty() {
                series.push((t, f32::NAN, 0.0));
                continue;
            }
            let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            let ci = 1.96 * var.sqrt() / n_seeds.sqrt();
            series.push((t, mean, ci));
        }
        cond_series.push((cond, series));
    }

    // ── Panel B: stratified-pitch scatter (separate simulation) ───
    let rep_seed = E5_SEEDS[E5_REPRESENTATIVE_SEED_IDX];
    let scatter_vitality = simulate_e5_stratified(rep_seed, E5Condition::Vitality, &landscape);
    let scatter_control = simulate_e5_stratified(rep_seed, E5Condition::Control, &landscape);

    // Write scatter CSV (stratified pitches, Panel B only)
    {
        let mut csv = String::from("condition,seed,agent,consonance,plv_final\n");
        for res in [&scatter_vitality, &scatter_control] {
            for (i, a) in res.agent_final.iter().enumerate() {
                csv.push_str(&format!(
                    "{},{},{},{:.6},{:.6}\n",
                    res.condition.label(),
                    res.seed,
                    i,
                    a.consonance,
                    a.plv,
                ));
            }
        }
        write_with_log(out_dir.join("paper_e5_scatter_detail.csv"), &csv)?;
    }

    // ── Per-condition Pearson r values ───────────────────────────
    let mut pearson_by_cond: Vec<(E5Condition, Vec<f32>)> = Vec::new();
    for &cond in &conditions {
        let rs: Vec<f32> = all_results
            .iter()
            .filter(|r| r.condition == cond && r.pearson_r.is_finite())
            .map(|r| r.pearson_r)
            .collect();
        pearson_by_cond.push((cond, rs));
    }

    // ── Build paired contrast series for Panel A (bottom) ───────
    let result_by_cond_seed: HashMap<(E5Condition, u64), &E5VitalityResult> = all_results
        .iter()
        .map(|r| ((r.condition, r.seed), r))
        .collect();
    let delta_series = vec![
        build_e5_delta_series(
            "vitality - uniform",
            PAL_E5_UNIFORM,
            E5Condition::Vitality,
            E5Condition::Uniform,
            &E5_SEEDS,
            &result_by_cond_seed,
        ),
        build_e5_delta_series(
            "vitality - control",
            PAL_E5_VITALITY,
            E5Condition::Vitality,
            E5Condition::Control,
            &E5_SEEDS,
            &result_by_cond_seed,
        ),
    ];

    // ── Render 3-panel figure ────────────────────────────────────
    let fig_path = out_dir.join("paper_e5_figure.svg");
    render_e5_combined_figure(
        &fig_path,
        &cond_series,
        &delta_series,
        Some(&scatter_vitality),
        Some(&scatter_control),
        &pearson_by_cond,
    )?;

    // ── Print summary stats ──────────────────────────────────────
    for (cond, rs) in &pearson_by_cond {
        if rs.is_empty() {
            continue;
        }
        let (m, s) = mean_std_scalar(rs);
        eprintln!(
            "  E5 {} : Pearson r(C_field, PLV) = {:.3} ± {:.3} (n={})",
            cond.label(),
            m,
            s,
            rs.len()
        );
    }

    Ok(())
}

// ── E7 Temporal Scaffold Assay ───────────────────────────────────
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum E7Condition {
    Shared,
    Scrambled,
    Off,
}

impl E7Condition {
    fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Scrambled => "scrambled",
            Self::Off => "off",
        }
    }

    fn color(self) -> RGBColor {
        match self {
            Self::Shared => PAL_E7_SHARED,
            Self::Scrambled => PAL_E7_SCRAMBLED,
            Self::Off => PAL_E7_OFF,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct E7Onset {
    agent: usize,
    time_sec: f32,
    phase_rad: f32,
}

struct E7Result {
    group_plv_series: Vec<(f32, f32)>,
    onset_events: Vec<E7Onset>,
    vector_strength: f32,
    condition: E7Condition,
    seed: u64,
}

fn e7_vector_strength(phases: &[f32]) -> f32 {
    if phases.is_empty() {
        return 0.0;
    }
    let inv = 1.0 / phases.len() as f32;
    let mean_cos = phases.iter().map(|p| p.cos()).sum::<f32>() * inv;
    let mean_sin = phases.iter().map(|p| p.sin()).sum::<f32>() * inv;
    (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
}

fn e7_onset_phase_histogram(phases: &[f32]) -> Vec<(f32, f32)> {
    let bin_width = (2.0 * PI) / E7_PHASE_HIST_BINS as f32;
    let counts = histogram_counts_fixed(phases, 0.0, 2.0 * PI, bin_width);
    let total: f32 = counts.iter().map(|(_, c)| *c).sum();
    if total > 0.0 {
        counts.iter().map(|(x, c)| (*x, *c / total)).collect()
    } else {
        counts.iter().map(|(x, _)| (*x, 0.0)).collect()
    }
}

fn pick_e7_representative_seed(results: &[E7Result]) -> Option<u64> {
    let mut shared: Vec<(u64, f32)> = results
        .iter()
        .filter(|r| r.condition == E7Condition::Shared)
        .map(|r| (r.seed, r.vector_strength))
        .collect();
    if shared.is_empty() {
        return None;
    }
    shared.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    Some(shared[shared.len() / 2].0)
}

fn simulate_e7_temporal_scaffold(seed: u64, condition: E7Condition) -> E7Result {
    let mut rng = seeded_rng(seed);
    let omegas: Vec<f32> = (0..E7_N_AGENTS)
        .map(|_| {
            let jitter = rng.random_range(-E7_AGENT_JITTER..E7_AGENT_JITTER);
            E7_AGENT_OMEGA_MEAN * (1.0 + jitter)
        })
        .collect();
    let mut phases: Vec<f32> = (0..E7_N_AGENTS)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let mut prev_phases = phases.clone();
    let mut canonical_phase = 0.0f32;
    let mut scramble_offset = 0.0f32;
    let mut plv_buffers: Vec<SlidingPlv> = (0..E7_N_AGENTS)
        .map(|_| SlidingPlv::new(E7_TIME_PLV_WINDOW_STEPS))
        .collect();
    let mut group_plv_series: Vec<(f32, f32)> = Vec::with_capacity(E7_STEPS);
    let mut onset_events: Vec<E7Onset> = Vec::new();

    for step in 0..E7_STEPS {
        let t = step as f32 * E7_DT;
        if condition == E7Condition::Scrambled && step % E7_STEPS_PER_CYCLE == 0 {
            scramble_offset = rng.random_range(0.0f32..(2.0 * PI));
        }

        canonical_phase = (canonical_phase + E7_KICK_OMEGA * E7_DT).rem_euclid(2.0 * PI);
        let drive_phase = match condition {
            E7Condition::Shared => canonical_phase,
            E7Condition::Scrambled => (canonical_phase + scramble_offset).rem_euclid(2.0 * PI),
            E7Condition::Off => canonical_phase,
        };
        let k_eff = match condition {
            E7Condition::Shared | E7Condition::Scrambled => E7_K_TIME,
            E7Condition::Off => 0.0,
        };

        for i in 0..E7_N_AGENTS {
            prev_phases[i] = phases[i];
            phases[i] = kuramoto_phase_step(phases[i], omegas[i], drive_phase, k_eff, 0.0, E7_DT);
        }

        let mut plv_sum = 0.0f32;
        let mut plv_count = 0usize;
        for i in 0..E7_N_AGENTS {
            let phase_diff = wrap_pm_pi(phases[i] - canonical_phase);
            plv_buffers[i].push(phase_diff);
            if plv_buffers[i].is_full() {
                let p = plv_buffers[i].plv();
                if p.is_finite() {
                    plv_sum += p;
                    plv_count += 1;
                }
            }

            let prev_wraps = (prev_phases[i] / (2.0 * PI)).floor() as i32;
            let next_wraps = (phases[i] / (2.0 * PI)).floor() as i32;
            if next_wraps > prev_wraps && t >= E7_ONSET_ANALYSIS_START_SEC {
                onset_events.push(E7Onset {
                    agent: i,
                    time_sec: t,
                    phase_rad: canonical_phase,
                });
            }
        }

        let group_plv = if plv_count > 0 {
            plv_sum / plv_count as f32
        } else {
            f32::NAN
        };
        group_plv_series.push((t, group_plv));
    }

    let onset_phases: Vec<f32> = onset_events.iter().map(|o| o.phase_rad).collect();
    E7Result {
        group_plv_series,
        onset_events,
        vector_strength: e7_vector_strength(&onset_phases),
        condition,
        seed,
    }
}

fn plot_e7_temporal_scaffold(out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let conditions = [
        E7Condition::Shared,
        E7Condition::Scrambled,
        E7Condition::Off,
    ];
    let mut all_results: Vec<E7Result> = Vec::new();
    for &cond in &conditions {
        for &seed in &E7_SEEDS {
            all_results.push(simulate_e7_temporal_scaffold(seed, cond));
        }
    }

    let mut summary_csv =
        String::from("condition,seed,vector_strength,group_plv_final,onset_count\n");
    for res in &all_results {
        let final_plv = res
            .group_plv_series
            .last()
            .map(|(_, p)| *p)
            .unwrap_or(f32::NAN);
        summary_csv.push_str(&format!(
            "{},{},{:.6},{:.6},{}\n",
            res.condition.label(),
            res.seed,
            res.vector_strength,
            final_plv,
            res.onset_events.len()
        ));
    }
    write_with_log(out_dir.join("paper_e7_summary.csv"), &summary_csv)?;

    let phase_hist_by_cond: Vec<(E7Condition, Vec<(f32, f32, f32)>)> = conditions
        .iter()
        .copied()
        .map(|cond| {
            let cond_results: Vec<&E7Result> =
                all_results.iter().filter(|r| r.condition == cond).collect();
            let seed_hists: Vec<Vec<(f32, f32)>> = cond_results
                .iter()
                .map(|res| {
                    let phases: Vec<f32> = res.onset_events.iter().map(|o| o.phase_rad).collect();
                    e7_onset_phase_histogram(&phases)
                })
                .collect();
            let mut bins = Vec::with_capacity(E7_PHASE_HIST_BINS);
            if let Some(first) = seed_hists.first() {
                for bin_idx in 0..first.len() {
                    let center = first[bin_idx].0;
                    let vals: Vec<f32> = seed_hists.iter().map(|hist| hist[bin_idx].1).collect();
                    let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
                    let var =
                        vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
                    let ci95 = 1.96 * var.sqrt() / (vals.len() as f32).sqrt();
                    bins.push((center, mean, ci95));
                }
            }
            (cond, bins)
        })
        .collect();
    let mut phase_hist_csv =
        String::from("condition,bin_index,phase_center_rad,probability_mean,probability_ci95\n");
    for (cond, bins) in &phase_hist_by_cond {
        for (idx, (center, mean, ci95)) in bins.iter().enumerate() {
            phase_hist_csv.push_str(&format!(
                "{},{},{:.6},{:.6},{:.6}\n",
                cond.label(),
                idx,
                center,
                mean,
                ci95
            ));
        }
    }
    write_with_log(
        out_dir.join("paper_e7_phase_hist_summary.csv"),
        &phase_hist_csv,
    )?;

    let rep_seed = pick_e7_representative_seed(&all_results)
        .unwrap_or(E7_SEEDS[E7_REPRESENTATIVE_SEED_IDX.min(E7_SEEDS.len() - 1)]);
    let mut rep_note =
        String::from("Diagnostic representative seed selection (not used in Fig. 5B):\n");
    let mut ranked: Vec<(u64, f32)> = all_results
        .iter()
        .filter(|r| r.condition == E7Condition::Shared)
        .map(|r| (r.seed, r.vector_strength))
        .collect();
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    for (seed, vs) in &ranked {
        rep_note.push_str(&format!("{seed},{vs:.6}\n"));
    }
    rep_note.push_str(&format!("chosen_seed={rep_seed}\n"));
    write_with_log(out_dir.join("paper_e7_representative_seed.txt"), rep_note)?;

    let mut onset_csv = String::from("condition,seed,agent,time_sec,phase_rad\n");
    for res in all_results.iter().filter(|r| r.seed == rep_seed) {
        for onset in &res.onset_events {
            onset_csv.push_str(&format!(
                "{},{},{},{:.6},{:.6}\n",
                res.condition.label(),
                res.seed,
                onset.agent,
                onset.time_sec,
                onset.phase_rad
            ));
        }
    }
    write_with_log(out_dir.join("paper_e7_onset_phase_detail.csv"), &onset_csv)?;

    let mut cond_series: Vec<(E7Condition, Vec<(f32, f32, f32)>)> = Vec::new();
    for &cond in &conditions {
        let cond_results: Vec<&E7Result> =
            all_results.iter().filter(|r| r.condition == cond).collect();
        let n_steps = cond_results
            .first()
            .map(|r| r.group_plv_series.len())
            .unwrap_or(0);
        let n_runs = cond_results.len() as f32;
        let mut series: Vec<(f32, f32, f32)> = Vec::with_capacity(n_steps);
        for step in 0..n_steps {
            let t = cond_results[0].group_plv_series[step].0;
            let vals: Vec<f32> = cond_results
                .iter()
                .filter_map(|r| {
                    let p = r.group_plv_series[step].1;
                    p.is_finite().then_some(p)
                })
                .collect();
            if vals.is_empty() {
                series.push((t, f32::NAN, 0.0));
                continue;
            }
            let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
            let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
            let ci = 1.96 * var.sqrt() / n_runs.sqrt();
            series.push((t, mean, ci));
        }
        cond_series.push((cond, series));
    }

    let vector_strength_by_cond: Vec<(E7Condition, Vec<f32>)> = conditions
        .iter()
        .copied()
        .map(|cond| {
            let values = all_results
                .iter()
                .filter(|r| r.condition == cond)
                .map(|r| r.vector_strength)
                .collect();
            (cond, values)
        })
        .collect();
    let mut stats_text = String::from("=== E7 Seed-Level Onset Vector Strength ===\n\n");
    for (cond, values) in &vector_strength_by_cond {
        if values.is_empty() {
            continue;
        }
        let (mean, std) = mean_std_scalar(values);
        stats_text.push_str(&format!(
            "{}: {:.3} ± {:.3} (n={})\n",
            cond.label(),
            mean,
            std,
            values.len()
        ));
    }
    write_with_log(out_dir.join("paper_e7_seed_level_stats.txt"), &stats_text)?;

    render_e7_combined_figure(
        &out_dir.join("paper_e7_figure.svg"),
        &cond_series,
        &phase_hist_by_cond,
        &vector_strength_by_cond,
    )?;

    for (cond, values) in &vector_strength_by_cond {
        let (mean, std) = mean_std_scalar(values);
        eprintln!(
            "  E7 {} : vector strength = {:.3} ± {:.3} (n={})",
            cond.label(),
            mean,
            std,
            values.len()
        );
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
fn render_e7_combined_figure(
    out_path: &Path,
    cond_series: &[(E7Condition, Vec<(f32, f32, f32)>)],
    phase_hist_by_cond: &[(E7Condition, Vec<(f32, f32, f32)>)],
    vector_strength_by_cond: &[(E7Condition, Vec<f32>)],
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (3600, 770)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 3));

    draw_e7_plv_panel(&panels[0], cond_series)?;
    draw_e7_phase_hist_panel(&panels[1], phase_hist_by_cond)?;
    draw_e7_vector_strength_panel(&panels[2], vector_strength_by_cond)?;

    root.present()?;
    Ok(())
}

#[allow(clippy::type_complexity)]
fn draw_e7_plv_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    cond_series: &[(E7Condition, Vec<(f32, f32, f32)>)],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let x_max = cond_series
        .iter()
        .flat_map(|(_, s)| s.iter().map(|(t, _, _)| *t))
        .fold(0.0f32, f32::max)
        .max(1.0);
    let mut chart = ChartBuilder::on(area)
        .caption("A. PLV to canonical beat", ("sans-serif", 72))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..x_max, 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("PLV")
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    for (cond, series) in cond_series {
        if series.is_empty() {
            continue;
        }
        let color = cond.color();
        let mut band: Vec<(f32, f32)> = Vec::with_capacity(series.len() * 2);
        for &(x, mean, ci) in series {
            if mean.is_finite() {
                band.push((x, (mean + ci).clamp(0.0, 1.05)));
            }
        }
        for &(x, mean, ci) in series.iter().rev() {
            if mean.is_finite() {
                band.push((x, (mean - ci).clamp(0.0, 1.05)));
            }
        }
        if band.len() >= 3 {
            chart.draw_series(std::iter::once(Polygon::new(
                band,
                color.mix(0.18).filled(),
            )))?;
        }

        let line_points: Vec<(f32, f32)> = series
            .iter()
            .filter(|(_, m, _)| m.is_finite())
            .map(|(x, m, _)| (*x, *m))
            .collect();
        chart
            .draw_series(LineSeries::new(
                line_points,
                ShapeStyle::from(&color).stroke_width(3),
            ))?
            .label(cond.label())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 40).into_font())
        .draw()?;

    Ok(())
}

fn draw_e7_phase_hist_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    phase_hist_by_cond: &[(E7Condition, Vec<(f32, f32, f32)>)],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let bin_width = (2.0 * PI) / E7_PHASE_HIST_BINS as f32;
    let y_min = -0.1f32;
    let mut y_max = phase_hist_by_cond
        .iter()
        .flat_map(|(_, bins)| bins.iter().map(|(_, mean, ci95)| mean + ci95))
        .fold(0.1f32, f32::max);
    y_max *= 1.15;

    let mut chart = ChartBuilder::on(area)
        .caption("B. Seed-averaged onset phase histogram", ("sans-serif", 72))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..2.0f32, y_min..y_max.max(0.1))?;

    chart
        .configure_mesh()
        .x_desc("phase vs canonical beat (\u{00d7}\u{03c0} rad)")
        .y_desc("probability")
        .x_labels(5)
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    for (cond, bins) in phase_hist_by_cond {
        let color = cond.color();
        let mut upper_pairs = Vec::with_capacity(bins.len());
        let mut lower_pairs = Vec::with_capacity(bins.len());
        let mut mean_pairs = Vec::with_capacity(bins.len());
        for &(center, mean, ci95) in bins {
            upper_pairs.push((center, (mean + ci95).max(0.0)));
            lower_pairs.push((center, (mean - ci95).max(0.0)));
            mean_pairs.push((center, mean));
        }
        let step_points = |pairs: &[(f32, f32)]| {
            let mut points = Vec::with_capacity(pairs.len() * 3);
            for (idx, &(_, y)) in pairs.iter().enumerate() {
                let x_left = idx as f32 * bin_width / PI;
                let x_right = (((idx + 1) as f32 * bin_width) / PI).min(2.0);
                if idx == 0 {
                    points.push((x_left, y));
                } else {
                    let prev_y = pairs[idx - 1].1;
                    points.push((x_left, prev_y));
                    points.push((x_left, y));
                }
                points.push((x_right, y));
            }
            points
        };
        let mut band = step_points(&upper_pairs);
        let mut lower_rev = step_points(&lower_pairs);
        lower_rev.reverse();
        band.extend(lower_rev);
        if band.len() >= 4 {
            chart.draw_series(std::iter::once(Polygon::new(
                band,
                color.mix(0.14).filled(),
            )))?;
        }
        chart
            .draw_series(std::iter::once(PathElement::new(
                step_points(&mean_pairs),
                ShapeStyle::from(&color).stroke_width(4),
            )))?
            .label(cond.label())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 40).into_font())
        .draw()?;

    Ok(())
}

fn draw_e7_vector_strength_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    vector_strength_by_cond: &[(E7Condition, Vec<f32>)],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let n_conds = vector_strength_by_cond.len() as f32;
    let mut chart = ChartBuilder::on(area)
        .caption("C. Onset vector strength", ("sans-serif", 72))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(-0.5f32..(n_conds - 0.5), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc("vector strength")
        .x_labels(vector_strength_by_cond.len())
        .x_label_formatter(&|x| {
            let idx = x.round() as usize;
            vector_strength_by_cond
                .get(idx)
                .map(|(cond, _)| cond.label().to_string())
                .unwrap_or_default()
        })
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    for (i, (cond, values)) in vector_strength_by_cond.iter().enumerate() {
        let color = cond.color();
        let center = i as f32;
        if values.is_empty() {
            continue;
        }

        let mut jitter_rng = seeded_rng(i as u64 + 0xE7D0);
        for &value in values {
            let jx = jitter_rng.random_range(-0.15f32..0.15);
            chart.draw_series(std::iter::once(Circle::new(
                (center + jx, value),
                5,
                color.mix(0.5).filled(),
            )))?;
        }

        let (mean, std) = mean_std_scalar(values);
        let x0 = center - 0.2;
        let x1 = center + 0.2;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, (mean - std).max(0.0)), (x1, (mean + std).min(1.05))],
            color.mix(0.25).filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, mean), (x1, mean)],
            ShapeStyle::from(&color).stroke_width(4),
        )))?;
    }

    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct E6bSeriesPoint {
    snapshot_step: usize,
    death_count: usize,
    normalized_step: f32,
    loo_c_score: f32,
    g_scene: f32,
    pairwise_entropy: f32,
    ji_score: f32,
    unique_bins: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct E6bConvergenceMetrics {
    auc_c: f32,
    step_to_c80: Option<f32>,
    stable_step_to_c80: Option<f32>,
    step_to_ji50: Option<f32>,
    step_to_entropy_1p25: Option<f32>,
}

#[derive(Clone, Copy, Debug)]
struct E6bEndpointMetrics {
    loo_c_score: f32,
    g_scene: f32,
    pairwise_entropy: f32,
    ji_score: f32,
    unique_bins: f32,
    mean_lifetime_steps: f32,
    same_pitch_replacement_rate: f32,
    same_family_respawn_rate: f32,
    same_octave_respawn_rate: f32,
    underfilled_respawn_rate: f32,
    mean_respawn_band_occupancy: f32,
    mean_proposal_rank: f32,
    mean_proposal_mass: f32,
    mean_proposal_filter_gain: f32,
    mean_local_opt_delta_st: f32,
    mean_local_opt_score_gap: f32,
    energy_exhaustion_deaths: usize,
    background_turnover_deaths: usize,
    juvenile_cull_deaths: usize,
    auc_c: f32,
    step_to_c80: Option<f32>,
    stable_step_to_c80: Option<f32>,
    step_to_ji50: Option<f32>,
    step_to_entropy_1p25: Option<f32>,
    stop_reason: E6StopReason,
}

#[derive(Clone, Debug)]
struct E6bProcessedRun {
    label: &'static str,
    baseline_type: &'static str,
    selection_enabled: bool,
    juvenile_enabled: bool,
    seed: u64,
    series: Vec<E6bSeriesPoint>,
    final_metrics: Option<E6bEndpointMetrics>,
    heat_counts: Vec<(usize, i32, f32)>,
}

#[allow(clippy::type_complexity)]
fn e6b_run_jobs(
    specs: &[(&'static str, E6Condition, bool, E6bRandomBaselineMode)],
    seeds: &[u64],
    cli: E6bCliOptions,
) -> Vec<(
    &'static str,
    E6Condition,
    bool,
    E6bRandomBaselineMode,
    bool,
    u64,
    E6bRunResult,
)> {
    let juvenile_enabled = cli.juvenile_enabled.unwrap_or(true);
    let mut jobs: Vec<(
        &'static str,
        E6Condition,
        bool,
        E6bRandomBaselineMode,
        bool,
        u64,
    )> = Vec::new();
    for &(label, condition, selection_enabled, random_baseline_mode) in specs {
        for &seed in seeds {
            jobs.push((
                label,
                condition,
                selection_enabled,
                random_baseline_mode,
                juvenile_enabled,
                seed,
            ));
        }
    }
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len())
        .max(1);
    eprintln!(
        "  E6b: running {} simulation jobs across {} workers...",
        jobs.len(),
        worker_count
    );
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let slots: Mutex<
        Vec<
            Option<(
                &'static str,
                E6Condition,
                bool,
                E6bRandomBaselineMode,
                bool,
                u64,
                E6bRunResult,
            )>,
        >,
    > = Mutex::new((0..jobs.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs.len() {
                        break;
                    }
                    let (
                        label,
                        condition,
                        selection_enabled,
                        random_baseline_mode,
                        juvenile_enabled,
                        seed,
                    ) = jobs[idx];
                    let cfg = E6bRunConfig {
                        seed,
                        steps_cap: E6B_STEPS_CAP,
                        min_deaths: E6B_MIN_DEATHS,
                        pop_size: E6B_DEFAULT_POP_SIZE,
                        first_k: E6B_FIRST_K,
                        condition,
                        snapshot_interval: E6B_SNAPSHOT_INTERVAL,
                        selection_enabled,
                        shuffle_landscape: false,
                        polyphonic_crowding_weight_override: cli.crowding_weight,
                        polyphonic_overcapacity_weight_override: cli.capacity_weight,
                        polyphonic_capacity_radius_cents_override: cli.capacity_radius_cents,
                        polyphonic_capacity_free_voices_override: cli.capacity_free_voices,
                        polyphonic_parent_share_weight_override: cli.parent_share_weight,
                        polyphonic_parent_energy_weight_override: cli.parent_energy_weight,
                        juvenile_contextual_settlement_enabled_override: Some(juvenile_enabled),
                        juvenile_contextual_tuning_ticks_override: cli.juvenile_ticks,
                        survival_score_low_override: cli.survival_score_low,
                        survival_score_high_override: cli.survival_score_high,
                        survival_recharge_per_sec_override: cli.survival_recharge_per_sec,
                        background_death_rate_per_sec_override: cli.background_death_rate_per_sec,
                        respawn_parent_prior_mix_override: cli.respawn_parent_prior_mix,
                        respawn_same_band_discount_override: cli.respawn_same_band_discount,
                        respawn_octave_discount_override: cli.respawn_octave_discount,
                        parent_proposal_kind_override: cli.parent_proposal_kind,
                        parent_proposal_sigma_st_override: cli.parent_proposal_sigma_st,
                        parent_proposal_unison_notch_gain_override: None,
                        parent_proposal_unison_notch_sigma_st_override: None,
                        parent_proposal_candidate_count_override: cli
                            .parent_proposal_candidate_count,
                        azimuth_mode_override: cli.azimuth_mode,
                        random_baseline_mode,
                    };
                    let result = run_e6b(&cfg);
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "    E6b jobs: {done}/{} complete ({label}, seed {seed})",
                        jobs.len()
                    );
                    slots.lock().unwrap()[idx] = Some((
                        label,
                        condition,
                        selection_enabled,
                        random_baseline_mode,
                        juvenile_enabled,
                        seed,
                        result,
                    ));
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|slot| slot.expect("missing E6b job output"))
        .collect()
}

fn e6b_process_run(
    label: &'static str,
    baseline_type: &'static str,
    selection_enabled: bool,
    juvenile_enabled: bool,
    seed: u64,
    result: &E6bRunResult,
    anchor_hz: f32,
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    heat_min_st: f32,
    heat_max_st: f32,
    free_voices: usize,
) -> E6bProcessedRun {
    let mut death_count = 0usize;
    let steps_cap = E6B_STEPS_CAP.max(1);
    let mut series: Vec<E6bSeriesPoint> = Vec::with_capacity(result.snapshots.len());
    let mut heat_counts: Vec<(usize, i32, f32)> = Vec::new();

    for snapshot in &result.snapshots {
        while death_count < result.deaths.len()
            && result.deaths[death_count].death_step as usize <= snapshot.step
        {
            death_count += 1;
        }
        let (loo_c_score, g_scene) = e6_contextual_endpoint_metrics(
            &snapshot.freqs_hz,
            anchor_hz,
            space,
            workspace,
            du_scan,
        );
        let semitones: Vec<f32> = snapshot
            .freqs_hz
            .iter()
            .copied()
            .filter(|f| f.is_finite() && *f > 0.0)
            .map(|freq_hz| 12.0 * (freq_hz / anchor_hz).log2())
            .collect();
        let pairwise = pairwise_interval_samples(&semitones);
        let pairwise_probs =
            histogram_probabilities_fixed(&pairwise, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        let pairwise_entropy = hist_structure_metrics_from_probs(&pairwise_probs).entropy;
        let diversity = diversity_metrics_for_semitones(&semitones);
        series.push(E6bSeriesPoint {
            snapshot_step: snapshot.step,
            death_count,
            normalized_step: snapshot.step as f32 / steps_cap as f32,
            loo_c_score,
            g_scene,
            pairwise_entropy,
            ji_score: ji_population_score(&snapshot.freqs_hz, anchor_hz),
            unique_bins: diversity.unique_bins as f32,
        });

        if label == "heredity" {
            for &freq_hz in &snapshot.freqs_hz {
                if !freq_hz.is_finite() || freq_hz <= 0.0 {
                    continue;
                }
                let semitone = 12.0 * (freq_hz / anchor_hz).log2();
                let clamped = semitone.clamp(heat_min_st, heat_max_st - f32::EPSILON);
                let bin = ((clamped - heat_min_st) / E6_INTERVAL_BIN_ST).floor() as i32;
                heat_counts.push((snapshot.step, bin, 1.0));
            }
        }
    }

    let final_metrics = e6b_final_metrics(result, &series, free_voices);
    E6bProcessedRun {
        label,
        baseline_type,
        selection_enabled,
        juvenile_enabled,
        seed,
        series,
        final_metrics,
        heat_counts,
    }
}

#[allow(clippy::type_complexity)]
fn e6b_process_results(
    main_results: &[(
        &'static str,
        E6Condition,
        bool,
        E6bRandomBaselineMode,
        bool,
        u64,
        E6bRunResult,
    )],
    anchor_hz: f32,
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    heat_min_st: f32,
    heat_max_st: f32,
    free_voices: usize,
) -> Vec<E6bProcessedRun> {
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(main_results.len())
        .max(1);
    eprintln!(
        "  E6b: processing {} completed runs across {} workers...",
        main_results.len(),
        worker_count
    );
    let next = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<E6bProcessedRun>>> =
        Mutex::new((0..main_results.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= main_results.len() {
                        break;
                    }
                    let (
                        label,
                        _condition,
                        selection_enabled,
                        random_baseline_mode,
                        juvenile_enabled,
                        seed,
                        result,
                    ) = &main_results[idx];
                    let processed = e6b_process_run(
                        *label,
                        random_baseline_mode.label(),
                        *selection_enabled,
                        *juvenile_enabled,
                        *seed,
                        result,
                        anchor_hz,
                        space,
                        workspace,
                        du_scan,
                        heat_min_st,
                        heat_max_st,
                        free_voices,
                    );
                    let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                    eprintln!(
                        "    E6b processed: {done}/{} complete ({}, seed {})",
                        main_results.len(),
                        *label,
                        *seed
                    );
                    slots.lock().unwrap()[idx] = Some(processed);
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|slot| slot.expect("missing E6b processed output"))
        .collect()
}

fn e6b_auc_c(series: &[E6bSeriesPoint]) -> f32 {
    if series.len() < 2 {
        return series.first().map(|p| p.loo_c_score).unwrap_or(0.0);
    }
    let mut area = 0.0f32;
    for window in series.windows(2) {
        let a = window[0];
        let b = window[1];
        let width = (b.normalized_step - a.normalized_step).max(0.0);
        area += width * 0.5 * (a.loo_c_score + b.loo_c_score);
    }
    area
}

fn e6b_first_threshold_step<F>(
    series: &[E6bSeriesPoint],
    value_of: F,
    predicate: impl Fn(f32) -> bool,
) -> Option<f32>
where
    F: Fn(&E6bSeriesPoint) -> f32,
{
    series
        .iter()
        .find(|point| predicate(value_of(point)))
        .map(|point| point.normalized_step)
}

fn e6b_stable_threshold_step<F>(
    series: &[E6bSeriesPoint],
    value_of: F,
    predicate: impl Fn(f32) -> bool,
) -> Option<f32>
where
    F: Fn(&E6bSeriesPoint) -> f32,
{
    for (idx, point) in series.iter().enumerate() {
        if !predicate(value_of(point)) {
            continue;
        }
        if series[idx..].iter().all(|later| predicate(value_of(later))) {
            return Some(point.normalized_step);
        }
    }
    None
}

fn e6b_convergence_metrics(series: &[E6bSeriesPoint]) -> E6bConvergenceMetrics {
    E6bConvergenceMetrics {
        auc_c: e6b_auc_c(series),
        step_to_c80: e6b_first_threshold_step(series, |point| point.loo_c_score, |v| v >= 0.80),
        stable_step_to_c80: e6b_stable_threshold_step(
            series,
            |point| point.loo_c_score,
            |v| v >= 0.80,
        ),
        step_to_ji50: e6b_first_threshold_step(series, |point| point.ji_score, |v| v >= 0.50),
        step_to_entropy_1p25: e6b_first_threshold_step(
            series,
            |point| point.pairwise_entropy,
            |v| v <= 1.25,
        ),
    }
}

fn e6b_optional_metric_summary(values: &[Option<f32>]) -> (usize, f32, f32) {
    let finite: Vec<f32> = values
        .iter()
        .flatten()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return (0, 0.0, 0.0);
    }
    let (mean, std) = mean_std_scalar(&finite);
    (finite.len(), mean, std)
}

fn e6b_same_pitch_replacement_rate(result: &E6bRunResult, cents_window: f32) -> f32 {
    let mut total = 0usize;
    let mut same_pitch = 0usize;
    for respawn in &result.respawns {
        let (Some(dead_freq_hz), spawn_freq_hz) = (respawn.dead_freq_hz, respawn.spawn_freq_hz)
        else {
            continue;
        };
        if !dead_freq_hz.is_finite()
            || dead_freq_hz <= 0.0
            || !spawn_freq_hz.is_finite()
            || spawn_freq_hz <= 0.0
        {
            continue;
        }
        total += 1;
        let delta_cents = (1200.0 * (spawn_freq_hz / dead_freq_hz).log2()).abs();
        if delta_cents <= cents_window.max(0.0) {
            same_pitch += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        same_pitch as f32 / total as f32
    }
}

fn e6b_same_family_respawn_rate(result: &E6bRunResult) -> f32 {
    let mut total = 0usize;
    let mut same_family = 0usize;
    for respawn in &result.respawns {
        let Some(family_inherited) = respawn.family_inherited else {
            continue;
        };
        total += 1;
        if family_inherited {
            same_family += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        same_family as f32 / total as f32
    }
}

fn e6b_same_octave_respawn_rate(result: &E6bRunResult, cents_window: f32) -> f32 {
    let mut total = 0usize;
    let mut same_octave = 0usize;
    for respawn in &result.respawns {
        let (Some(parent_freq_hz), spawn_freq_hz) = (respawn.parent_freq_hz, respawn.spawn_freq_hz)
        else {
            continue;
        };
        if !parent_freq_hz.is_finite()
            || parent_freq_hz <= 0.0
            || !spawn_freq_hz.is_finite()
            || spawn_freq_hz <= 0.0
        {
            continue;
        }
        let delta_cents = 1200.0 * (spawn_freq_hz / parent_freq_hz).log2();
        let nearest_octave = (delta_cents / 1200.0).round();
        if nearest_octave.abs() < 0.5 {
            continue;
        }
        total += 1;
        if (delta_cents - nearest_octave * 1200.0).abs() <= cents_window.max(0.0) + 1e-6 {
            same_octave += 1;
        }
    }
    if total == 0 {
        0.0
    } else {
        same_octave as f32 / total as f32
    }
}

fn e6b_underfilled_respawn_stats(result: &E6bRunResult, free_voices: usize) -> (f32, f32) {
    let mut occupancies = Vec::new();
    for respawn in &result.respawns {
        if let Some(occupancy) = respawn.chosen_band_occupancy {
            occupancies.push(occupancy as f32);
        }
    }
    if occupancies.is_empty() {
        return (0.0, 0.0);
    }
    let underfilled = occupancies
        .iter()
        .filter(|&&occupancy| occupancy <= free_voices.max(1) as f32 + 1e-6)
        .count() as f32
        / occupancies.len() as f32;
    (underfilled, mean_std_scalar(&occupancies).0)
}

fn e6b_death_cause_counts(result: &E6bRunResult) -> (usize, usize, usize) {
    let mut energy_exhaustion = 0usize;
    let mut background_turnover = 0usize;
    let mut juvenile_cull = 0usize;
    for respawn in &result.respawns {
        match respawn
            .death_cause
            .unwrap_or(E6DeathCause::EnergyExhaustion)
        {
            E6DeathCause::EnergyExhaustion => energy_exhaustion += 1,
            E6DeathCause::BackgroundTurnover => background_turnover += 1,
            E6DeathCause::JuvenileCull => juvenile_cull += 1,
        }
    }
    (energy_exhaustion, background_turnover, juvenile_cull)
}

fn e6b_respawn_proposal_stats(result: &E6bRunResult) -> (f32, f32, f32) {
    let ranks: Vec<f32> = result
        .respawns
        .iter()
        .filter_map(|respawn| respawn.proposal_rank.map(|rank| rank as f32))
        .collect();
    let masses: Vec<f32> = result
        .respawns
        .iter()
        .filter_map(|respawn| respawn.proposal_mass)
        .collect();
    let gains: Vec<f32> = result
        .respawns
        .iter()
        .filter_map(|respawn| respawn.proposal_filter_gain)
        .collect();
    (
        mean_std_scalar(&ranks).0,
        mean_std_scalar(&masses).0,
        mean_std_scalar(&gains).0,
    )
}

fn e6b_local_opt_stats(result: &E6bRunResult) -> (f32, f32) {
    let deltas: Vec<f32> = result
        .respawns
        .iter()
        .filter_map(|respawn| respawn.local_opt_delta_st)
        .collect();
    let score_gaps: Vec<f32> = result
        .respawns
        .iter()
        .filter_map(|respawn| respawn.local_opt_score_gap)
        .collect();
    (mean_std_scalar(&deltas).0, mean_std_scalar(&score_gaps).0)
}

fn e6b_final_metrics(
    result: &E6bRunResult,
    series: &[E6bSeriesPoint],
    free_voices: usize,
) -> Option<E6bEndpointMetrics> {
    let final_point = *series.last()?;
    let mean_lifetime_steps = mean_std_scalar(
        &result
            .deaths
            .iter()
            .map(|death| death.lifetime_steps as f32)
            .collect::<Vec<_>>(),
    )
    .0;
    let same_pitch_replacement_rate = e6b_same_pitch_replacement_rate(result, 15.0);
    let same_family_respawn_rate = e6b_same_family_respawn_rate(result);
    let same_octave_respawn_rate = e6b_same_octave_respawn_rate(result, 35.0);
    let (underfilled_respawn_rate, mean_respawn_band_occupancy) =
        e6b_underfilled_respawn_stats(result, free_voices);
    let (mean_proposal_rank, mean_proposal_mass, mean_proposal_filter_gain) =
        e6b_respawn_proposal_stats(result);
    let (mean_local_opt_delta_st, mean_local_opt_score_gap) = e6b_local_opt_stats(result);
    let (energy_exhaustion_deaths, background_turnover_deaths, juvenile_cull_deaths) =
        e6b_death_cause_counts(result);
    let convergence = e6b_convergence_metrics(series);
    Some(E6bEndpointMetrics {
        loo_c_score: final_point.loo_c_score,
        g_scene: final_point.g_scene,
        pairwise_entropy: final_point.pairwise_entropy,
        ji_score: final_point.ji_score,
        unique_bins: final_point.unique_bins,
        mean_lifetime_steps,
        same_pitch_replacement_rate,
        same_family_respawn_rate,
        same_octave_respawn_rate,
        underfilled_respawn_rate,
        mean_respawn_band_occupancy,
        mean_proposal_rank,
        mean_proposal_mass,
        mean_proposal_filter_gain,
        mean_local_opt_delta_st,
        mean_local_opt_score_gap,
        energy_exhaustion_deaths,
        background_turnover_deaths,
        juvenile_cull_deaths,
        auc_c: convergence.auc_c,
        step_to_c80: convergence.step_to_c80,
        stable_step_to_c80: convergence.stable_step_to_c80,
        step_to_ji50: convergence.step_to_ji50,
        step_to_entropy_1p25: convergence.step_to_entropy_1p25,
        stop_reason: result.stop_reason,
    })
}

fn e6b_gap_closed(value: f32, baseline: f32, control: f32, higher_is_better: bool) -> f32 {
    let denom = if higher_is_better {
        baseline - control
    } else {
        control - baseline
    };
    if denom.abs() <= 1e-6 {
        0.0
    } else if higher_is_better {
        (value - control) / denom
    } else {
        (control - value) / denom
    }
}

fn e6_contextual_endpoint_metrics(
    freqs_hz: &[f32],
    anchor_hz: f32,
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
) -> (f32, f32) {
    let clean_freqs: Vec<f32> = freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return (0.0, 0.0);
    }

    let fixed = e2_fixed_drone(space, anchor_hz);
    let agent_indices: Vec<usize> = clean_freqs
        .iter()
        .map(|&f| space.nearest_index(f))
        .collect();
    let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
        space,
        std::slice::from_ref(&fixed.idx),
        &agent_indices,
        du_scan,
    );
    let mut env_loo = Vec::new();
    let mut density_loo = Vec::new();
    let (mean_c_score_loo, _) = mean_c_score_loo_pair_at_indices_with_prev_reused(
        space,
        workspace,
        &env_scan,
        &density_scan,
        du_scan,
        &agent_indices,
        &agent_indices,
        &agent_indices,
        &mut env_loo,
        &mut density_loo,
    );

    let mut freqs_with_anchor = clean_freqs;
    freqs_with_anchor.push(fixed.hz);
    let g_scene = e2_scene_g_point_for_freqs(space, workspace, du_scan, &freqs_with_anchor).g_scene;
    (mean_c_score_loo, g_scene)
}

fn e6_series_stats(
    by_seed: Option<&HashMap<u64, Vec<(usize, f32)>>>,
    snapshot_interval: usize,
) -> Vec<(f32, f32, f32)> {
    let Some(by_seed) = by_seed else {
        return Vec::new();
    };
    if by_seed.is_empty() {
        return Vec::new();
    }

    // Seed-aware carry-forward:
    // each seed keeps its own last observation, then we aggregate by step.
    let mut min_step = usize::MAX;
    let mut max_step = 0usize;
    let mut has_any = false;
    for samples in by_seed.values() {
        for &(step, value) in samples {
            if !value.is_finite() {
                continue;
            }
            min_step = min_step.min(step);
            max_step = max_step.max(step);
            has_any = true;
        }
    }
    if !has_any {
        return Vec::new();
    }

    let step_interval = snapshot_interval.max(1);
    let mut by_step: HashMap<usize, Vec<f32>> = HashMap::new();
    for samples in by_seed.values() {
        let mut ordered: Vec<(usize, f32)> = samples
            .iter()
            .copied()
            .filter(|(_, v)| v.is_finite())
            .collect();
        if ordered.is_empty() {
            continue;
        }
        ordered.sort_unstable_by_key(|(step, _)| *step);

        let mut idx = 0usize;
        let mut carry: Option<f32> = None;
        let mut step = min_step;
        while step <= max_step {
            while idx < ordered.len() && ordered[idx].0 <= step {
                carry = Some(ordered[idx].1);
                idx += 1;
            }
            if let Some(v) = carry {
                by_step.entry(step).or_default().push(v);
            }
            let Some(next_step) = step.checked_add(step_interval) else {
                break;
            };
            step = next_step;
        }
    }

    let mut keys: Vec<usize> = by_step.keys().copied().collect();
    keys.sort_unstable();

    let mut out = Vec::with_capacity(keys.len());
    for step in keys {
        let Some(values) = by_step.get(&step) else {
            continue;
        };
        let finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
        if finite.is_empty() {
            continue;
        }
        let (mean, std) = mean_std_scalar(&finite);
        let ci95 = ci95_from_std(std, finite.len());
        out.push((step as f32, mean, ci95));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn render_e6b_figure(
    out_path: &Path,
    c_heredity_nosel: &[(f32, f32, f32)],
    c_random_nosel: &[(f32, f32, f32)],
    c_heredity: &[(f32, f32, f32)],
    c_random: &[(f32, f32, f32)],
    heat_counts: &HashMap<(usize, i32), f32>,
    final_loo_heredity_nosel: &[f32],
    final_loo_random_nosel: &[f32],
    final_loo_heredity: &[f32],
    final_loo_random: &[f32],
    final_ji_heredity_nosel: &[f32],
    final_ji_random_nosel: &[f32],
    final_ji_heredity: &[f32],
    final_ji_random: &[f32],
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (3720, 935)).into_drawing_area();
    root.fill(&WHITE)?;
    let (left, right) = root.split_horizontally(2400);
    let left_panels = left.split_evenly((1, 2));
    let right_panels = right.split_evenly((1, 2));

    let series_specs = [
        E6FigureSeries {
            label: "heredity",
            series: c_heredity,
            color: PAL_H.to_rgba(),
        },
        E6FigureSeries {
            label: "random",
            series: c_random,
            color: PAL_R.to_rgba(),
        },
        E6FigureSeries {
            label: "heredity (no sel)",
            series: c_heredity_nosel,
            color: PAL_H.mix(0.45),
        },
        E6FigureSeries {
            label: "random (no sel)",
            series: c_random_nosel,
            color: PAL_R.mix(0.45),
        },
    ];
    draw_e6_series_panel_with_x_desc(
        &left_panels[0],
        "A. Mean LOO C_score",
        "LOO C_score",
        "step",
        &series_specs,
        0.0,
        1.0,
    )?;
    draw_e6_heatmap_panel_with_caption(
        &left_panels[1],
        "B. H+S pitch heatmap",
        "step",
        heat_counts,
    )?;
    draw_e6_factorial_metric_panel(
        &right_panels[0],
        "C. Final LOO C_score",
        "LOO C_score",
        final_loo_heredity_nosel,
        final_loo_random_nosel,
        final_loo_heredity,
        final_loo_random,
        0.0,
        Some(1.0),
        64,
        48,
        50,
        126,
    )?;
    draw_e6_factorial_metric_panel(
        &right_panels[1],
        "D. Final JI Score",
        "JI score",
        final_ji_heredity_nosel,
        final_ji_random_nosel,
        final_ji_heredity,
        final_ji_random,
        0.0,
        Some(0.8),
        62,
        46,
        50,
        126,
    )?;

    root.present()?;
    Ok(())
}

struct E6FigureSeries<'a> {
    label: &'a str,
    series: &'a [(f32, f32, f32)],
    color: RGBAColor,
}

fn draw_e6_series_panel_with_x_desc<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    caption: &str,
    y_desc: &str,
    x_desc: &str,
    series_specs: &[E6FigureSeries<'_>],
    y_lo: f32,
    y_hi: f32,
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let x_max = series_specs
        .iter()
        .flat_map(|spec| spec.series.iter().map(|(x, _, _)| *x))
        .fold(0.0f32, f32::max)
        .min(E6_FIG_X_MAX)
        .max(1.0);
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 64))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc(y_desc)
        .x_labels(11)
        .x_label_formatter(&|v| format!("{}", *v as i64))
        .label_style(("sans-serif", 48).into_font())
        .axis_desc_style(("sans-serif", 50).into_font())
        .draw()?;

    for spec in series_specs {
        let label = spec.label;
        let series = spec.series;
        let color = spec.color;
        if series.is_empty() {
            continue;
        }
        let mut band: Vec<(f32, f32)> = Vec::with_capacity(series.len() * 2);
        for (x, mean, ci) in series.iter().copied() {
            band.push((x, (mean + ci).clamp(y_lo, y_hi)));
        }
        for (x, mean, ci) in series.iter().rev().copied() {
            band.push((x, (mean - ci).clamp(y_lo, y_hi)));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band,
            color.mix(0.20).filled(),
        )))?;

        chart
            .draw_series(LineSeries::new(
                series.iter().map(|(x, mean, _)| (*x, *mean)),
                color,
            ))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));

        // Dashed extension from last point to x_max
        if let Some(&(last_x, last_mean, _)) = series.last()
            && last_x < x_max - 1.0
        {
            let dash_len = (x_max - last_x) / 80.0;
            let mut segments: Vec<PathElement<(f32, f32)>> = Vec::new();
            let mut cx = last_x;
            while cx < x_max {
                let nx = (cx + dash_len).min(x_max);
                segments.push(PathElement::new(
                    vec![(cx, last_mean), (nx, last_mean)],
                    color.mix(0.5).stroke_width(1),
                ));
                cx = nx + dash_len; // gap
            }
            chart.draw_series(segments)?;
        }
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK)
        .label_font(("sans-serif", 48).into_font())
        .draw()?;

    chart.draw_series(std::iter::once(Text::new(
        "light: no selection; dark: selection",
        (x_max * 0.03, y_hi - (y_hi - y_lo) * 0.06),
        ("sans-serif", 32).into_font(),
    )))?;
    Ok(())
}

fn draw_e6_factorial_metric_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    caption: &str,
    y_desc: &str,
    heredity_nosel: &[f32],
    random_nosel: &[f32],
    heredity_sel: &[f32],
    random_sel: &[f32],
    y_lo: f32,
    y_hi_override: Option<f32>,
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
    y_label_area_size: u32,
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let mean_ci = |values: &[f32]| -> (f32, f32) {
        if values.is_empty() {
            (0.0, 0.0)
        } else {
            (mean_std_scalar(values).0, ci95_half_width(values))
        }
    };
    let (h0_mean, h0_ci) = mean_ci(heredity_nosel);
    let (r0_mean, r0_ci) = mean_ci(random_nosel);
    let (h1_mean, h1_ci) = mean_ci(heredity_sel);
    let (r1_mean, r1_ci) = mean_ci(random_sel);
    let y_hi = y_hi_override.unwrap_or_else(|| {
        [
            h0_mean + h0_ci,
            r0_mean + r0_ci,
            h1_mean + h1_ci,
            r1_mean + r1_ci,
        ]
        .into_iter()
        .fold(y_lo + 1e-6, f32::max)
            * 1.15
    });

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", caption_size))
        .margin(22)
        .x_label_area_size(90)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(-0.5f32..1.5f32, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc(y_desc)
        .x_labels(2)
        .x_label_formatter(&|v| match v.round() as i32 {
            0 => "selection".to_string(),
            1 => "no selection".to_string(),
            _ => String::new(),
        })
        .label_style(("sans-serif", label_size).into_font())
        .axis_desc_style(("sans-serif", axis_desc_size).into_font())
        .draw()?;

    if y_lo < 0.0 && y_hi > 0.0 {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(-0.5f32, 0.0), (1.5f32, 0.0)],
            BLACK.mix(0.25),
        )))?;
    }

    let bar_specs = [
        (0.0f32 - 0.16, h1_mean, h1_ci, PAL_H.to_rgba()),
        (0.0f32 + 0.16, r1_mean, r1_ci, PAL_R.to_rgba()),
        (1.0f32 - 0.16, h0_mean, h0_ci, PAL_H.mix(0.75)),
        (1.0f32 + 0.16, r0_mean, r0_ci, PAL_R.mix(0.75)),
    ];
    for (center, mean, ci, color) in bar_specs {
        let x0 = center - 0.12;
        let x1 = center + 0.12;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y_lo), (x1, mean)],
            color.filled(),
        )))?;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![
                (center, (mean - ci).max(y_lo)),
                (center, (mean + ci).min(y_hi)),
            ],
            BLACK.stroke_width(2),
        )))?;
    }

    chart.draw_series(std::iter::once(Text::new(
        "teal: heredity; rose: random",
        (-0.42f32, y_hi * 0.95),
        ("sans-serif", 32).into_font(),
    )))?;

    Ok(())
}

fn draw_e6_heatmap_panel_with_caption<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    caption: &str,
    x_desc: &str,
    heat_counts: &HashMap<(usize, i32), f32>,
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let (min_range_st, max_range_st) = e6_effective_range_bounds_st();
    let min_range_cents = min_range_st * 100.0;
    let max_range_cents = max_range_st * 100.0;
    let x_max = heat_counts
        .keys()
        .map(|(step, _)| *step as f32)
        .fold(0.0f32, f32::max)
        .min(E6_FIG_X_MAX - E6_SNAPSHOT_INTERVAL as f32)
        + E6_SNAPSHOT_INTERVAL as f32;
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 64))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), min_range_cents..max_range_cents)?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("cents")
        .x_labels(11)
        .x_label_formatter(&|v| format!("{}", *v as i64))
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    let mut sorted_counts: Vec<f32> = heat_counts.values().copied().collect();
    sorted_counts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let clip_count = if sorted_counts.is_empty() {
        1.0
    } else {
        let idx = ((sorted_counts.len().saturating_sub(1)) as f32 * 0.90).round() as usize;
        sorted_counts[idx].max(1.0)
    };
    for ((step, bin), count) in heat_counts {
        let x0 = *step as f32;
        let x1 = x0 + E6_SNAPSHOT_INTERVAL as f32;
        let y0 = (min_range_st + *bin as f32 * E6_INTERVAL_BIN_ST) * 100.0;
        let y1 = y0 + E6_INTERVAL_BIN_ST * 100.0;
        let t = (*count / clip_count).clamp(0.0, 1.0).powf(0.65);
        let color = if t < 0.55 {
            let u = t / 0.55;
            let lerp =
                |a: u8, b: u8| -> u8 { ((a as f32) + ((b as f32) - (a as f32)) * u).round() as u8 };
            RGBColor(lerp(242, 116), lerp(249, 196), lerp(250, 167)).to_rgba()
        } else {
            let u = (t - 0.55) / 0.45;
            let lerp =
                |a: u8, b: u8| -> u8 { ((a as f32) + ((b as f32) - (a as f32)) * u).round() as u8 };
            RGBColor(lerp(116, 27), lerp(196, 78), lerp(167, 119)).to_rgba()
        };
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, y0), (x1, y1)],
            ShapeStyle {
                color,
                filled: true,
                stroke_width: 1,
            },
        )))?;
    }

    for &target in &E2_CONSONANT_STEPS {
        let mut y_st = target;
        while y_st > min_range_st {
            y_st -= 12.0;
        }
        while y_st <= max_range_st {
            let y = y_st * 100.0;
            if (min_range_cents..=max_range_cents).contains(&y) {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![(0.0, y), (x_max.max(1.0), y)],
                    BLACK.mix(0.18),
                )))?;
            }
            y_st += 12.0;
        }
    }

    Ok(())
}

#[allow(clippy::type_complexity)]
fn plot_e6b_hereditary_polyphony(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    cli: E6bCliOptions,
) -> Result<(), Box<dyn Error>> {
    let seeds = cli.seed_slice();
    eprintln!(
        "E6b: starting hereditary assay for {} seeds{}",
        seeds.len(),
        if cli.skip_benchmark {
            " (Exp1 benchmark skipped)"
        } else {
            ""
        }
    );
    let main_results = e6b_run_jobs(
        &[
            (
                "heredity_nosel",
                E6Condition::Heredity,
                false,
                E6bRandomBaselineMode::LogRandomFiltered,
            ),
            (
                "random_nosel",
                E6Condition::Random,
                false,
                E6bRandomBaselineMode::LogRandomFiltered,
            ),
            (
                "heredity",
                E6Condition::Heredity,
                true,
                E6bRandomBaselineMode::LogRandomFiltered,
            ),
            (
                "random",
                E6Condition::Random,
                true,
                E6bRandomBaselineMode::LogRandomFiltered,
            ),
            (
                "scene_random_nosel",
                E6Condition::Random,
                false,
                E6bRandomBaselineMode::MatchedScenePeaks,
            ),
            (
                "scene_random",
                E6Condition::Random,
                true,
                E6bRandomBaselineMode::MatchedScenePeaks,
            ),
            (
                "hard_random_nosel",
                E6Condition::Random,
                false,
                E6bRandomBaselineMode::HardRandomLog,
            ),
            (
                "hard_random",
                E6Condition::Random,
                true,
                E6bRandomBaselineMode::HardRandomLog,
            ),
        ],
        seeds,
        cli,
    );
    eprintln!("  E6b: main simulations finished.");

    let reference_landscape = e3_reference_landscape(anchor_hz);
    let contextual_space = reference_landscape.space.clone();
    let (_contextual_erb_scan, contextual_du_scan) = erb_grid(&contextual_space);
    let contextual_workspace = build_consonance_workspace(&contextual_space);

    let mut series_by_label_seed: HashMap<&'static str, HashMap<u64, Vec<(usize, f32)>>> =
        HashMap::new();
    let mut final_loo_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_entropy_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_ji_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_unique_bins_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_g_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_lifetime_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_same_pitch_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_same_family_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_same_octave_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_underfilled_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_band_occupancy_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_proposal_rank_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_proposal_mass_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_proposal_filter_gain_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_local_opt_delta_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_local_opt_gap_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_energy_deaths_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_background_deaths_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_juvenile_cull_deaths_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_auc_c_by_label: HashMap<&'static str, Vec<f32>> = HashMap::new();
    let mut final_step_to_c80_by_label: HashMap<&'static str, Vec<Option<f32>>> = HashMap::new();
    let mut final_stable_step_to_c80_by_label: HashMap<&'static str, Vec<Option<f32>>> =
        HashMap::new();
    let mut final_step_to_ji50_by_label: HashMap<&'static str, Vec<Option<f32>>> = HashMap::new();
    let mut final_step_to_entropy_by_label: HashMap<&'static str, Vec<Option<f32>>> =
        HashMap::new();
    let mut stop_reason_counts_by_label: HashMap<&'static str, (usize, usize)> = HashMap::new();
    let mut endpoint_csv = String::from(
        "condition,baseline_type,seed,selection_enabled,juvenile_enabled,stop_reason,loo_c_score,g_scene,pairwise_entropy,ji_score,unique_bins,mean_lifetime_steps,same_pitch_replacement_rate,same_family_respawn_rate,same_octave_respawn_rate,underfilled_respawn_rate,mean_respawn_band_occupancy,mean_proposal_rank,mean_proposal_mass,mean_proposal_filter_gain,mean_local_opt_delta_st,mean_local_opt_score_gap,energy_exhaustion_deaths,background_turnover_deaths,juvenile_cull_deaths,auc_c,step_to_c80,stable_step_to_c80,step_to_ji50,step_to_entropy_1p25\n",
    );
    let mut timeseries_csv = String::from(
        "condition,baseline_type,seed,selection_enabled,juvenile_enabled,snapshot_step,death_count,normalized_step,loo_c_score,g_scene,pairwise_entropy,ji_score,unique_bins\n",
    );
    let mut heat_counts: HashMap<(usize, i32), f32> = HashMap::new();
    let (heat_min_st, heat_max_st) = e6_effective_range_bounds_st();

    let processed_runs = e6b_process_results(
        &main_results,
        anchor_hz,
        &contextual_space,
        &contextual_workspace,
        &contextual_du_scan,
        heat_min_st,
        heat_max_st,
        cli.capacity_free_voices.unwrap_or(3),
    );
    eprintln!("  E6b: processed run summaries finished.");

    for processed in processed_runs {
        let c_series: Vec<(usize, f32)> = processed
            .series
            .iter()
            .map(|point| (point.snapshot_step, point.loo_c_score))
            .collect();
        series_by_label_seed
            .entry(processed.label)
            .or_default()
            .insert(processed.seed, c_series);
        for point in &processed.series {
            timeseries_csv.push_str(&format!(
                "{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.3}\n",
                processed.label,
                processed.baseline_type,
                processed.seed,
                u8::from(processed.selection_enabled),
                u8::from(processed.juvenile_enabled),
                point.snapshot_step,
                point.death_count,
                point.normalized_step,
                point.loo_c_score,
                point.g_scene,
                point.pairwise_entropy,
                point.ji_score,
                point.unique_bins,
            ));
        }
        for (step, bin, mass) in processed.heat_counts {
            *heat_counts.entry((step, bin)).or_insert(0.0) += mass;
        }

        if let Some(metrics) = processed.final_metrics {
            final_loo_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.loo_c_score);
            final_entropy_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.pairwise_entropy);
            final_ji_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.ji_score);
            final_unique_bins_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.unique_bins);
            final_g_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.g_scene);
            final_lifetime_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_lifetime_steps);
            final_same_pitch_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.same_pitch_replacement_rate);
            final_same_family_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.same_family_respawn_rate);
            final_same_octave_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.same_octave_respawn_rate);
            final_underfilled_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.underfilled_respawn_rate);
            final_band_occupancy_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_respawn_band_occupancy);
            final_proposal_rank_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_proposal_rank);
            final_proposal_mass_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_proposal_mass);
            final_proposal_filter_gain_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_proposal_filter_gain);
            final_local_opt_delta_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_local_opt_delta_st);
            final_local_opt_gap_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.mean_local_opt_score_gap);
            final_energy_deaths_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.energy_exhaustion_deaths as f32);
            final_background_deaths_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.background_turnover_deaths as f32);
            final_juvenile_cull_deaths_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.juvenile_cull_deaths as f32);
            final_auc_c_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.auc_c);
            final_step_to_c80_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.step_to_c80);
            final_stable_step_to_c80_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.stable_step_to_c80);
            final_step_to_ji50_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.step_to_ji50);
            final_step_to_entropy_by_label
                .entry(processed.label)
                .or_default()
                .push(metrics.step_to_entropy_1p25);
            let counter = stop_reason_counts_by_label
                .entry(processed.label)
                .or_insert((0usize, 0usize));
            match metrics.stop_reason {
                E6StopReason::MinDeaths => counter.0 += 1,
                E6StopReason::StepsCap => counter.1 += 1,
            }
            endpoint_csv.push_str(&format!(
                "{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.6},{:.6},{:.6},{:.6},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                processed.label,
                processed.baseline_type,
                processed.seed,
                u8::from(processed.selection_enabled),
                u8::from(processed.juvenile_enabled),
                metrics.stop_reason.label(),
                metrics.loo_c_score,
                metrics.g_scene,
                metrics.pairwise_entropy,
                metrics.ji_score,
                metrics.unique_bins,
                metrics.mean_lifetime_steps,
                metrics.same_pitch_replacement_rate,
                metrics.same_family_respawn_rate,
                metrics.same_octave_respawn_rate,
                metrics.underfilled_respawn_rate,
                metrics.mean_respawn_band_occupancy,
                metrics.mean_proposal_rank,
                metrics.mean_proposal_mass,
                metrics.mean_proposal_filter_gain,
                metrics.mean_local_opt_delta_st,
                metrics.mean_local_opt_score_gap,
                metrics.energy_exhaustion_deaths,
                metrics.background_turnover_deaths,
                metrics.juvenile_cull_deaths,
                metrics.auc_c,
                metrics.step_to_c80.unwrap_or(f32::NAN),
                metrics.stable_step_to_c80.unwrap_or(f32::NAN),
                metrics.step_to_ji50.unwrap_or(f32::NAN),
                metrics.step_to_entropy_1p25.unwrap_or(f32::NAN),
            ));
        }
    }

    write_with_log(out_dir.join("paper_e6b_endpoint_metrics.csv"), endpoint_csv)?;
    write_with_log(out_dir.join("paper_e6b_timeseries.csv"), timeseries_csv)?;
    eprintln!("  E6b: wrote endpoint/timeseries CSVs.");

    let c_heredity_nosel = e6_series_stats(
        series_by_label_seed.get("heredity_nosel"),
        E6B_SNAPSHOT_INTERVAL,
    );
    let c_random_nosel = e6_series_stats(
        series_by_label_seed.get("random_nosel"),
        E6B_SNAPSHOT_INTERVAL,
    );
    let c_heredity = e6_series_stats(series_by_label_seed.get("heredity"), E6B_SNAPSHOT_INTERVAL);
    let c_random = e6_series_stats(series_by_label_seed.get("random"), E6B_SNAPSHOT_INTERVAL);

    let final_loo_heredity_nosel = final_loo_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_loo_random_nosel = final_loo_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_loo_heredity = final_loo_by_label.remove("heredity").unwrap_or_default();
    let final_loo_random = final_loo_by_label.remove("random").unwrap_or_default();

    let final_entropy_heredity_nosel = final_entropy_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_entropy_random_nosel = final_entropy_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_entropy_heredity = final_entropy_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_entropy_random = final_entropy_by_label.remove("random").unwrap_or_default();
    let final_ji_heredity_nosel = final_ji_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_ji_random_nosel = final_ji_by_label.remove("random_nosel").unwrap_or_default();
    let final_ji_heredity = final_ji_by_label.remove("heredity").unwrap_or_default();
    let final_ji_random = final_ji_by_label.remove("random").unwrap_or_default();
    let ji_mean_heredity_nosel = mean_std_scalar(&final_ji_heredity_nosel).0;
    let ji_mean_random_nosel = mean_std_scalar(&final_ji_random_nosel).0;
    let ji_mean_heredity = mean_std_scalar(&final_ji_heredity).0;
    let ji_mean_random = mean_std_scalar(&final_ji_random).0;
    let final_lifetime_heredity_nosel = final_lifetime_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_lifetime_random_nosel = final_lifetime_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_lifetime_heredity = final_lifetime_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_lifetime_random = final_lifetime_by_label.remove("random").unwrap_or_default();
    let final_same_pitch_heredity_nosel = final_same_pitch_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_same_pitch_random_nosel = final_same_pitch_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_same_pitch_heredity = final_same_pitch_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_same_pitch_random = final_same_pitch_by_label
        .remove("random")
        .unwrap_or_default();
    let final_same_family_heredity_nosel = final_same_family_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_same_family_random_nosel = final_same_family_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_same_family_heredity = final_same_family_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_same_family_random = final_same_family_by_label
        .remove("random")
        .unwrap_or_default();
    let final_same_octave_heredity_nosel = final_same_octave_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_same_octave_random_nosel = final_same_octave_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_same_octave_heredity = final_same_octave_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_same_octave_random = final_same_octave_by_label
        .remove("random")
        .unwrap_or_default();
    let final_underfilled_heredity_nosel = final_underfilled_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_underfilled_random_nosel = final_underfilled_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_underfilled_heredity = final_underfilled_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_underfilled_random = final_underfilled_by_label
        .remove("random")
        .unwrap_or_default();
    let final_auc_c_heredity = final_auc_c_by_label.remove("heredity").unwrap_or_default();
    let final_auc_c_random = final_auc_c_by_label.remove("random").unwrap_or_default();
    let final_step_to_c80_heredity = final_step_to_c80_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_step_to_c80_random = final_step_to_c80_by_label
        .remove("random")
        .unwrap_or_default();
    let final_stable_step_to_c80_heredity = final_stable_step_to_c80_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_stable_step_to_c80_random = final_stable_step_to_c80_by_label
        .remove("random")
        .unwrap_or_default();
    let final_step_to_ji50_heredity = final_step_to_ji50_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_step_to_ji50_random = final_step_to_ji50_by_label
        .remove("random")
        .unwrap_or_default();
    let final_step_to_entropy_heredity = final_step_to_entropy_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_step_to_entropy_random = final_step_to_entropy_by_label
        .remove("random")
        .unwrap_or_default();
    let final_band_occupancy_heredity_nosel = final_band_occupancy_by_label
        .remove("heredity_nosel")
        .unwrap_or_default();
    let final_band_occupancy_random_nosel = final_band_occupancy_by_label
        .remove("random_nosel")
        .unwrap_or_default();
    let final_band_occupancy_heredity = final_band_occupancy_by_label
        .remove("heredity")
        .unwrap_or_default();
    let final_band_occupancy_random = final_band_occupancy_by_label
        .remove("random")
        .unwrap_or_default();

    render_e6b_figure(
        &out_dir.join("paper_e6b_figure.svg"),
        &c_heredity_nosel,
        &c_random_nosel,
        &c_heredity,
        &c_random,
        &heat_counts,
        &final_loo_heredity_nosel,
        &final_loo_random_nosel,
        &final_loo_heredity,
        &final_loo_random,
        &final_ji_heredity_nosel,
        &final_ji_random_nosel,
        &final_ji_heredity,
        &final_ji_random,
    )?;
    eprintln!("  E6b: rendered figure.");

    let benchmark_text = if cli.skip_benchmark {
        format!(
            "Hereditary Adaptation benchmark against Consonance Search\n\
             --------------------------------\n\
             skipped: {}\n\
             seeds used in this run: {}\n",
            if cli.quick {
                "--e6b-quick"
            } else {
                "--e6b-skip-benchmark"
            },
            seeds.len()
        )
    } else {
        eprintln!("  E6b: running Exp1 benchmark reference...");
        let max_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let baseline_threads = (max_threads / 2).max(1);
        let nohill_threads = max_threads.saturating_sub(baseline_threads).max(1);
        let ((baseline_runs, _), (nohill_runs, _)) = std::thread::scope(|scope| {
            let baseline = scope.spawn(|| {
                e2_seed_sweep_with_threads(
                    space,
                    anchor_hz,
                    E2Condition::Baseline,
                    E2_STEP_SEMITONES,
                    E2PhaseMode::DissonanceThenConsonance,
                    None,
                    0,
                    E2_PAPER_N_AGENTS,
                    E2_PAPER_RANGE_OCT,
                    Some(baseline_threads),
                )
            });
            let nohill = scope.spawn(|| {
                e2_seed_sweep_with_threads(
                    space,
                    anchor_hz,
                    E2Condition::NoHillClimb,
                    E2_STEP_SEMITONES,
                    E2PhaseMode::DissonanceThenConsonance,
                    None,
                    0,
                    E2_PAPER_N_AGENTS,
                    E2_PAPER_RANGE_OCT,
                    Some(nohill_threads),
                )
            });
            (
                baseline.join().expect("baseline Exp1 benchmark failed"),
                nohill.join().expect("nohill Exp1 benchmark failed"),
            )
        });
        let hs_loo_mean = mean_std_scalar(&final_loo_heredity).0;
        let hs_entropy_mean = mean_std_scalar(&final_entropy_heredity).0;
        let hs_ji_mean = ji_mean_heredity;
        let hs_unique_bins_mean = mean_std_scalar(
            final_unique_bins_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice),
        )
        .0;

        let baseline_c = mean_last_c_score_loo(&baseline_runs);
        let nohill_c = mean_last_c_score_loo(&nohill_runs);
        let baseline_entropy = mean_entropy_end(&baseline_runs);
        let nohill_entropy = mean_entropy_end(&nohill_runs);
        let baseline_ji = mean_ji_scene_end(&baseline_runs, anchor_hz);
        let nohill_ji = mean_ji_scene_end(&nohill_runs, anchor_hz);
        let baseline_bins = mean_unique_bins_end(&baseline_runs);
        let nohill_bins = mean_unique_bins_end(&nohill_runs);

        format!(
            "Hereditary Adaptation benchmark against Consonance Search\n\
             --------------------------------\n\
             H+S endpoint means (E6b):\n\
             LOO C_score      = {:.4}\n\
             interval entropy = {:.4}\n\
             JI score         = {:.4}\n\
             unique bins      = {:.3}\n\
             \n\
             Exp1 reference means:\n\
             local-search: C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}\n\
             random-walk : C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}\n\
             \n\
             Gap closed toward Exp1 local-search (1.0 = match local-search, 0.0 = random-walk):\n\
             C_score      = {:.4}\n\
             entropy      = {:.4}\n\
             JI score     = {:.4}\n\
             unique bins  = {:.4}\n",
            hs_loo_mean,
            hs_entropy_mean,
            hs_ji_mean,
            hs_unique_bins_mean,
            baseline_c,
            baseline_entropy,
            baseline_ji,
            baseline_bins,
            nohill_c,
            nohill_entropy,
            nohill_ji,
            nohill_bins,
            e6b_gap_closed(hs_loo_mean, baseline_c, nohill_c, true),
            e6b_gap_closed(hs_entropy_mean, baseline_entropy, nohill_entropy, false),
            e6b_gap_closed(hs_ji_mean, baseline_ji, nohill_ji, true),
            e6b_gap_closed(hs_unique_bins_mean, baseline_bins, nohill_bins, false),
        )
    };
    write_with_log(out_dir.join("paper_e6b_exp1_benchmark.txt"), benchmark_text)?;
    eprintln!("  E6b: wrote benchmark summary.");

    let stop_counts = |label: &'static str| -> (usize, usize) {
        stop_reason_counts_by_label
            .get(label)
            .copied()
            .unwrap_or((0usize, 0usize))
    };
    let juvenile_mode = if cli.juvenile_enabled.unwrap_or(true) {
        "on"
    } else {
        "off"
    };
    let proposal_kind = cli
        .parent_proposal_kind
        .unwrap_or(E6ParentProposalKind::ContextualC)
        .label();
    let azimuth_mode = cli
        .azimuth_mode
        .unwrap_or(E6bAzimuthMode::LocalSearch)
        .label();
    let auc_c_heredity = mean_std_scalar(&final_auc_c_heredity).0;
    let auc_c_random = mean_std_scalar(&final_auc_c_random).0;
    let (step_to_c80_heredity_n, step_to_c80_heredity_mean, _) =
        e6b_optional_metric_summary(&final_step_to_c80_heredity);
    let (step_to_c80_random_n, step_to_c80_random_mean, _) =
        e6b_optional_metric_summary(&final_step_to_c80_random);
    let (stable_step_to_c80_heredity_n, stable_step_to_c80_heredity_mean, _) =
        e6b_optional_metric_summary(&final_stable_step_to_c80_heredity);
    let (stable_step_to_c80_random_n, stable_step_to_c80_random_mean, _) =
        e6b_optional_metric_summary(&final_stable_step_to_c80_random);
    let (step_to_ji50_heredity_n, step_to_ji50_heredity_mean, _) =
        e6b_optional_metric_summary(&final_step_to_ji50_heredity);
    let (step_to_ji50_random_n, step_to_ji50_random_mean, _) =
        e6b_optional_metric_summary(&final_step_to_ji50_random);
    let (step_to_entropy_heredity_n, step_to_entropy_heredity_mean, _) =
        e6b_optional_metric_summary(&final_step_to_entropy_heredity);
    let (step_to_entropy_random_n, step_to_entropy_random_mean, _) =
        e6b_optional_metric_summary(&final_step_to_entropy_random);
    let mut summary_text = format!(
        "E6b summary\n\
         ----------\n\
         Tuning: crowding={:.4}, capacity_weight={:.4}, capacity_radius_cents={:.1}, free_voices={}, parent_share_weight={:.2}, parent_energy_weight={:.2}, proposal_kind={}, proposal_sigma_st={:.2}, proposal_candidates={}, azimuth_mode={}, juvenile_mode={}, juvenile_ticks={}, survival_low={:.2}, survival_high={:.2}, survival_recharge={:.3}, background_death={:.4}, respawn_parent_prior_mix={:.2}, same_band_discount={:.2}, octave_discount={:.2}\n\
         \n\
         Final LOO C_score:\n\
         heredity, no selection: {:.4} +/- {:.4}\n\
         random,   no selection: {:.4} +/- {:.4}\n\
         heredity, selection:    {:.4} +/- {:.4}\n\
         random,   selection:    {:.4} +/- {:.4}\n\
         \n\
         Final interval entropy:\n\
         heredity, no selection: {:.4} +/- {:.4}\n\
         random,   no selection: {:.4} +/- {:.4}\n\
         heredity, selection:    {:.4} +/- {:.4}\n\
         random,   selection:    {:.4} +/- {:.4}\n\
         \n\
         Final JI score (means):\n\
         heredity, no selection: {:.4}\n\
         random,   no selection: {:.4}\n\
         heredity, selection:    {:.4}\n\
         random,   selection:    {:.4}\n\
         \n\
         Final unique bins (means):\n\
         heredity, no selection: {:.3}\n\
         random,   no selection: {:.3}\n\
         heredity, selection:    {:.3}\n\
         random,   selection:    {:.3}\n\
         \n\
         Mean lifetime steps:\n\
         heredity, no selection: {:.2}\n\
         random,   no selection: {:.2}\n\
         heredity, selection:    {:.2}\n\
         random,   selection:    {:.2}\n\
         \n\
         Respawn redundancy diagnostics:\n\
         same-pitch (<=15c): hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         same-family        : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         same-octave        : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         underfilled target : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         mean band occupancy: hered_nosel={:.2}  rand_nosel={:.2}  hered_sel={:.2}  rand_sel={:.2}\n\
         mean proposal rank : hered_nosel={:.2}  rand_nosel={:.2}  hered_sel={:.2}  rand_sel={:.2}\n\
         mean proposal mass : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         mean filter gain   : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         mean local-opt dSt : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         mean local-opt gap : hered_nosel={:.4}  rand_nosel={:.4}  hered_sel={:.4}  rand_sel={:.4}\n\
         \n\
         Stop reasons (min_deaths / steps_cap):\n\
         heredity, no selection: {}/{}\n\
         random,   no selection: {}/{}\n\
         heredity, selection:    {}/{}\n\
         random,   selection:    {}/{}\n\
         \n\
         Mean death cause counts:\n\
         heredity, no selection: energy={:.1}, background={:.1}, juvenile={:.1}\n\
         random,   no selection: energy={:.1}, background={:.1}, juvenile={:.1}\n\
         heredity, selection:    energy={:.1}, background={:.1}, juvenile={:.1}\n\
         random,   selection:    energy={:.1}, background={:.1}, juvenile={:.1}\n\
         \n\
         Convergence diagnostics (selection conditions, normalized step):\n\
         AUC_C                 : heredity={:.4}  random={:.4}\n\
         step_to_C>=0.80       : heredity={}/{} mean={:.4}  random={}/{} mean={:.4}\n\
         stable_step_to_C>=0.80: heredity={}/{} mean={:.4}  random={}/{} mean={:.4}\n\
         step_to_JI>=0.50      : heredity={}/{} mean={:.4}  random={}/{} mean={:.4}\n\
         step_to_entropy<=1.25 : heredity={}/{} mean={:.4}  random={}/{} mean={:.4}\n",
        cli.crowding_weight.unwrap_or(0.005),
        cli.capacity_weight.unwrap_or(0.07),
        cli.capacity_radius_cents.unwrap_or(35.0),
        cli.capacity_free_voices.unwrap_or(3),
        cli.parent_share_weight.unwrap_or(1.0),
        cli.parent_energy_weight.unwrap_or(0.25),
        proposal_kind,
        cli.parent_proposal_sigma_st.unwrap_or(9.0),
        cli.parent_proposal_candidate_count.unwrap_or(16),
        azimuth_mode,
        juvenile_mode,
        cli.juvenile_ticks.unwrap_or(2),
        cli.survival_score_low.unwrap_or(0.30),
        cli.survival_score_high.unwrap_or(0.80),
        cli.survival_recharge_per_sec.unwrap_or(0.20),
        cli.background_death_rate_per_sec.unwrap_or(0.0),
        cli.respawn_parent_prior_mix.unwrap_or(0.15),
        cli.respawn_same_band_discount.unwrap_or(0.08),
        cli.respawn_octave_discount.unwrap_or(0.20),
        mean_std_scalar(&final_loo_heredity_nosel).0,
        ci95_half_width(&final_loo_heredity_nosel),
        mean_std_scalar(&final_loo_random_nosel).0,
        ci95_half_width(&final_loo_random_nosel),
        mean_std_scalar(&final_loo_heredity).0,
        ci95_half_width(&final_loo_heredity),
        mean_std_scalar(&final_loo_random).0,
        ci95_half_width(&final_loo_random),
        mean_std_scalar(&final_entropy_heredity_nosel).0,
        ci95_half_width(&final_entropy_heredity_nosel),
        mean_std_scalar(&final_entropy_random_nosel).0,
        ci95_half_width(&final_entropy_random_nosel),
        mean_std_scalar(&final_entropy_heredity).0,
        ci95_half_width(&final_entropy_heredity),
        mean_std_scalar(&final_entropy_random).0,
        ci95_half_width(&final_entropy_random),
        ji_mean_heredity_nosel,
        ji_mean_random_nosel,
        ji_mean_heredity,
        ji_mean_random,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(&final_lifetime_heredity_nosel).0,
        mean_std_scalar(&final_lifetime_random_nosel).0,
        mean_std_scalar(&final_lifetime_heredity).0,
        mean_std_scalar(&final_lifetime_random).0,
        mean_std_scalar(&final_same_pitch_heredity_nosel).0,
        mean_std_scalar(&final_same_pitch_random_nosel).0,
        mean_std_scalar(&final_same_pitch_heredity).0,
        mean_std_scalar(&final_same_pitch_random).0,
        mean_std_scalar(&final_same_family_heredity_nosel).0,
        mean_std_scalar(&final_same_family_random_nosel).0,
        mean_std_scalar(&final_same_family_heredity).0,
        mean_std_scalar(&final_same_family_random).0,
        mean_std_scalar(&final_same_octave_heredity_nosel).0,
        mean_std_scalar(&final_same_octave_random_nosel).0,
        mean_std_scalar(&final_same_octave_heredity).0,
        mean_std_scalar(&final_same_octave_random).0,
        mean_std_scalar(&final_underfilled_heredity_nosel).0,
        mean_std_scalar(&final_underfilled_random_nosel).0,
        mean_std_scalar(&final_underfilled_heredity).0,
        mean_std_scalar(&final_underfilled_random).0,
        mean_std_scalar(&final_band_occupancy_heredity_nosel).0,
        mean_std_scalar(&final_band_occupancy_random_nosel).0,
        mean_std_scalar(&final_band_occupancy_heredity).0,
        mean_std_scalar(&final_band_occupancy_random).0,
        mean_std_scalar(
            final_proposal_rank_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_rank_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_rank_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_rank_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_mass_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_mass_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_mass_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_mass_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_filter_gain_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_filter_gain_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_filter_gain_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_proposal_filter_gain_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_delta_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_delta_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_delta_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_delta_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_gap_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_gap_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_gap_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_local_opt_gap_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        stop_counts("heredity_nosel").0,
        stop_counts("heredity_nosel").1,
        stop_counts("random_nosel").0,
        stop_counts("random_nosel").1,
        stop_counts("heredity").0,
        stop_counts("heredity").1,
        stop_counts("random").0,
        stop_counts("random").1,
        mean_std_scalar(
            final_energy_deaths_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_background_deaths_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_juvenile_cull_deaths_by_label
                .get("heredity_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_energy_deaths_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_background_deaths_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_juvenile_cull_deaths_by_label
                .get("random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_energy_deaths_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_background_deaths_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_juvenile_cull_deaths_by_label
                .get("heredity")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_energy_deaths_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_background_deaths_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_juvenile_cull_deaths_by_label
                .get("random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        auc_c_heredity,
        auc_c_random,
        step_to_c80_heredity_n,
        final_step_to_c80_heredity.len(),
        step_to_c80_heredity_mean,
        step_to_c80_random_n,
        final_step_to_c80_random.len(),
        step_to_c80_random_mean,
        stable_step_to_c80_heredity_n,
        final_stable_step_to_c80_heredity.len(),
        stable_step_to_c80_heredity_mean,
        stable_step_to_c80_random_n,
        final_stable_step_to_c80_random.len(),
        stable_step_to_c80_random_mean,
        step_to_ji50_heredity_n,
        final_step_to_ji50_heredity.len(),
        step_to_ji50_heredity_mean,
        step_to_ji50_random_n,
        final_step_to_ji50_random.len(),
        step_to_ji50_random_mean,
        step_to_entropy_heredity_n,
        final_step_to_entropy_heredity.len(),
        step_to_entropy_heredity_mean,
        step_to_entropy_random_n,
        final_step_to_entropy_random.len(),
        step_to_entropy_random_mean,
    );
    summary_text.push_str(&format!(
        "\n\
         Main random baseline: log_random_filtered\n\
         Scene-peak matched random baseline (supplementary):\n\
         random,   no selection: C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}  AUC_C={:.4}\n\
         random,   selection:    C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}  AUC_C={:.4}\n\
         \n\
         Hard random baseline (supplementary):\n\
         random,   no selection: C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}  AUC_C={:.4}\n\
         random,   selection:    C={:.4}  entropy={:.4}  JI={:.4}  bins={:.3}  AUC_C={:.4}\n",
        mean_std_scalar(
            final_loo_by_label
                .get("scene_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_entropy_by_label
                .get("scene_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_ji_by_label
                .get("scene_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("scene_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_auc_c_by_label
                .get("scene_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_loo_by_label
                .get("scene_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_entropy_by_label
                .get("scene_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_ji_by_label
                .get("scene_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("scene_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_auc_c_by_label
                .get("scene_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_loo_by_label
                .get("hard_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_entropy_by_label
                .get("hard_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_ji_by_label
                .get("hard_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("hard_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_auc_c_by_label
                .get("hard_random_nosel")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_loo_by_label
                .get("hard_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_entropy_by_label
                .get("hard_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_ji_by_label
                .get("hard_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_unique_bins_by_label
                .get("hard_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
        mean_std_scalar(
            final_auc_c_by_label
                .get("hard_random")
                .map_or(&[][..], Vec::as_slice)
        )
        .0,
    ));
    write_with_log(out_dir.join("paper_e6b_summary.txt"), summary_text)?;
    eprintln!("  E6b: wrote textual summary.");

    Ok(())
}

// ── E5 Vitality-Coupled Entrainment: condition enum ──────────────
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum E5Condition {
    Vitality,
    Uniform,
    Control,
}

impl E5Condition {
    fn label(self) -> &'static str {
        match self {
            E5Condition::Vitality => "vitality",
            E5Condition::Uniform => "uniform",
            E5Condition::Control => "control",
        }
    }
    fn color(self) -> RGBColor {
        match self {
            E5Condition::Vitality => PAL_E5_VITALITY,
            E5Condition::Uniform => PAL_E5_UNIFORM,
            E5Condition::Control => PAL_E5_CONTROL,
        }
    }
}

/// Coupling multiplier matching `coupling_multiplier_from_mode()` in articulation_core.rs.
/// Maps vitality → multiplicative factor on k_time via `TemporalTimesVitality` logic.
#[inline]
fn e5_coupling_multiplier(vitality: f32, lambda_v: f32, v_floor: f32) -> f32 {
    let denom = (1.0 - v_floor).max(1e-6);
    let g = ((vitality - v_floor) / denom).clamp(0.0, 1.0);
    (lambda_v * g).clamp(0.0, 2.0) // MAX_COUPLING_MULT = 2.0 in main
}

// ── E5 result structs ────────────────────────────────────────────
struct E5VitalityResult {
    /// Per-step group mean PLV
    group_plv_series: Vec<(f32, f32)>,
    /// Per-agent final PLV and consonance
    agent_final: Vec<E5AgentFinal>,
    /// Pearson r(consonance, final PLV) for this run
    pearson_r: f32,
    condition: E5Condition,
    seed: u64,
}

struct E5DeltaSeries {
    label: &'static str,
    color: RGBColor,
    /// (t, mean, ci95)
    series: Vec<(f32, f32, f32)>,
}

struct E5SigmoidFit {
    base: f32,      // y at x → -∞
    amplitude: f32, // y at x → +∞ is base + amplitude
    steepness: f32, // slope of sigmoid (c50 fixed at 0)
}

#[derive(Clone, Copy)]
struct E5AgentFinal {
    consonance: f32,
    plv: f32,
}

// ── E5 pitch selection: stratified by raw C_field ────────────────
/// Scan the landscape over ±1 octave and return N pitches whose
/// raw C_field values are approximately uniformly distributed in [0, scan_max].
/// This ensures the independent variable (C_field) spans the full range.
fn e5_stratified_pitches(
    n: usize,
    landscape: &conchordal::core::landscape::Landscape,
    rng: &mut impl rand::Rng,
) -> (Vec<f32>, Vec<f32>) {
    let log2_lo = E5_ANCHOR_HZ.log2() - 1.0;
    let log2_hi = E5_ANCHOR_HZ.log2() + 1.0;

    // Dense scan: 4000 candidate pitches
    let n_scan = 4000usize;
    let candidates: Vec<(f32, f32)> = (0..n_scan)
        .map(|i| {
            let log2_f = log2_lo + (log2_hi - log2_lo) * (i as f32 / (n_scan - 1) as f32);
            let hz = 2.0f32.powf(log2_f);
            let score = landscape.evaluate_pitch_score(hz);
            (hz, score)
        })
        .collect();

    // Use raw C_field range [scan_min, scan_max] for stratification
    let s_min = candidates.iter().map(|c| c.1).fold(f32::INFINITY, f32::min);
    let s_max = candidates
        .iter()
        .map(|c| c.1)
        .fold(f32::NEG_INFINITY, f32::max);
    let s_range = (s_max - s_min).max(1e-6);

    // Assign each candidate to one of N strata over [s_min, s_max]
    let mut strata: Vec<Vec<(f32, f32)>> = (0..n).map(|_| Vec::new()).collect();
    for &(hz, c) in &candidates {
        let frac = ((c - s_min) / s_range).clamp(0.0, 1.0);
        let bin = ((frac * n as f32) as usize).min(n - 1);
        strata[bin].push((hz, c));
    }

    // Pick one random candidate from each stratum.
    // If a stratum is empty, pick the nearest candidate to that stratum's midpoint.
    let mut pitches = Vec::with_capacity(n);
    let mut consonances = Vec::with_capacity(n);
    for (i, stratum) in strata.iter().enumerate().take(n) {
        let mid_frac = (i as f32 + 0.5) / n as f32;
        let mid_val = s_min + mid_frac * s_range;
        if !stratum.is_empty() {
            let idx = rng.random_range(0..stratum.len());
            let (hz, c) = stratum[idx];
            pitches.push(hz);
            consonances.push(c);
        } else {
            // Fallback: find candidate closest to stratum midpoint (in raw C_field)
            let (hz, c) = *candidates
                .iter()
                .min_by(|a, b| {
                    (a.1 - mid_val)
                        .abs()
                        .partial_cmp(&(b.1 - mid_val).abs())
                        .unwrap()
                })
                .unwrap();
            pitches.push(hz);
            consonances.push(c);
        }
    }
    (pitches, consonances)
}

/// Run E5 simulation with stratified pitches (for Panel B scatter only).
/// Same dynamics as `simulate_e5_vitality` but pitches span the full
/// consonance range so the scatter plot is informative.
fn simulate_e5_stratified(
    seed: u64,
    condition: E5Condition,
    landscape: &conchordal::core::landscape::Landscape,
) -> E5VitalityResult {
    let mut rng = seeded_rng(seed);
    let (_pitches_hz, consonances) = e5_stratified_pitches(E5_N_AGENTS, landscape, &mut rng);

    // --- rest is identical to simulate_e5_vitality ---
    let omegas: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| {
            let jitter = rng.random_range(-E5_AGENT_JITTER..E5_AGENT_JITTER);
            E5_AGENT_OMEGA_MEAN * (1.0 + jitter)
        })
        .collect();
    let mut phases: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();
    let mut energies: Vec<f32> = vec![E5_E_INIT; E5_N_AGENTS];
    let mut theta_kick = 0.0f32;
    let mut plv_buffers: Vec<SlidingPlv> = (0..E5_N_AGENTS)
        .map(|_| SlidingPlv::new(E5_TIME_PLV_WINDOW_STEPS))
        .collect();
    let mut group_plv_series: Vec<(f32, f32)> = Vec::with_capacity(E5_STEPS);

    for step in 0..E5_STEPS {
        let t = step as f32 * E5_DT;
        // Energy dynamics: dE/dt = R·max(C_field,0) − γ·E
        for i in 0..E5_N_AGENTS {
            let recharge = E5_RECHARGE * consonances[i].max(0.0);
            let de = (recharge - E5_DECAY * energies[i]) * E5_DT;
            energies[i] = (energies[i] + de).clamp(0.0, E5_E_CAP);
        }
        // Vitality
        let vitalities: Vec<f32> = energies
            .iter()
            .map(|&e| (e / E5_E_CAP).clamp(0.0, 1.0).powf(E5_VITALITY_EXPONENT))
            .collect();
        let mean_v: f32 = vitalities.iter().sum::<f32>() / E5_N_AGENTS as f32;
        // Coupling (matches articulation_core coupling_multiplier_from_mode)
        let k_effs: Vec<f32> = match condition {
            E5Condition::Vitality => vitalities
                .iter()
                .map(|&v| E5_K_TIME * e5_coupling_multiplier(v, E5_LAMBDA_V, E5_V_FLOOR))
                .collect(),
            E5Condition::Uniform => {
                let cm = e5_coupling_multiplier(mean_v, E5_LAMBDA_V, E5_V_FLOOR);
                vec![E5_K_TIME * cm; E5_N_AGENTS]
            }
            E5Condition::Control => vec![0.0; E5_N_AGENTS],
        };
        // Phase update (couple each agent to external kick only)
        for i in 0..E5_N_AGENTS {
            phases[i] =
                kuramoto_phase_step(phases[i], omegas[i], theta_kick, k_effs[i], 0.0, E5_DT);
        }
        theta_kick += E5_KICK_OMEGA * E5_DT;
        // PLV tracking
        let mut plv_sum = 0.0f32;
        let mut plv_count = 0usize;
        for i in 0..E5_N_AGENTS {
            let dp = wrap_pm_pi(phases[i] - theta_kick);
            plv_buffers[i].push(dp);
            if plv_buffers[i].is_full() {
                let p = plv_buffers[i].plv();
                if p.is_finite() {
                    plv_sum += p;
                    plv_count += 1;
                }
            }
        }
        let group_plv = if plv_count > 0 {
            plv_sum / plv_count as f32
        } else {
            f32::NAN
        };
        group_plv_series.push((t, group_plv));
    }
    let agent_final: Vec<E5AgentFinal> = (0..E5_N_AGENTS)
        .map(|i| E5AgentFinal {
            consonance: consonances[i],
            plv: if plv_buffers[i].is_full() {
                plv_buffers[i].plv()
            } else {
                f32::NAN
            },
        })
        .collect();
    let pr = pearson_r_e5(&agent_final);
    E5VitalityResult {
        group_plv_series,
        agent_final,
        pearson_r: pr,
        condition,
        seed,
    }
}

// ── E5 simulation ────────────────────────────────────────────────
fn simulate_e5_vitality(
    seed: u64,
    condition: E5Condition,
    landscape: &conchordal::core::landscape::Landscape,
) -> E5VitalityResult {
    let mut rng = seeded_rng(seed);

    // Random pitches within ±1 octave of anchor
    let log2_anchor = E5_ANCHOR_HZ.log2();
    let pitches_hz: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| {
            let log2_f = rng.random_range((log2_anchor - 1.0)..(log2_anchor + 1.0));
            2.0f32.powf(log2_f)
        })
        .collect();

    // Evaluate consonance (raw C_field — no normalization)
    let consonances: Vec<f32> = pitches_hz
        .iter()
        .map(|&f| landscape.evaluate_pitch_score(f))
        .collect();

    // Intrinsic frequencies with jitter
    let omegas: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| {
            let jitter = rng.random_range(-E5_AGENT_JITTER..E5_AGENT_JITTER);
            E5_AGENT_OMEGA_MEAN * (1.0 + jitter)
        })
        .collect();

    // Initial phases (random)
    let mut phases: Vec<f32> = (0..E5_N_AGENTS)
        .map(|_| rng.random_range(0.0f32..(2.0 * PI)))
        .collect();

    // Energy state
    let mut energies: Vec<f32> = vec![E5_E_INIT; E5_N_AGENTS];

    // External kick phase
    let mut theta_kick = 0.0f32;

    // PLV tracking
    let mut plv_buffers: Vec<SlidingPlv> = (0..E5_N_AGENTS)
        .map(|_| SlidingPlv::new(E5_TIME_PLV_WINDOW_STEPS))
        .collect();

    let mut group_plv_series: Vec<(f32, f32)> = Vec::with_capacity(E5_STEPS);

    for step in 0..E5_STEPS {
        let t = step as f32 * E5_DT;

        // Energy dynamics: dE/dt = R·max(C_field,0) − γ·E
        for i in 0..E5_N_AGENTS {
            let recharge = E5_RECHARGE * consonances[i].max(0.0);
            let de = (recharge - E5_DECAY * energies[i]) * E5_DT;
            energies[i] = (energies[i] + de).clamp(0.0, E5_E_CAP);
        }

        // Compute vitality per agent
        let vitalities: Vec<f32> = energies
            .iter()
            .map(|&e| (e / E5_E_CAP).clamp(0.0, 1.0).powf(E5_VITALITY_EXPONENT))
            .collect();

        // Compute per-agent coupling (matches articulation_core coupling_multiplier_from_mode)
        let mean_vitality: f32 = vitalities.iter().copied().sum::<f32>() / E5_N_AGENTS as f32;

        // Advance kick phase
        theta_kick += E5_KICK_OMEGA * E5_DT;

        // Phase update per agent
        for i in 0..E5_N_AGENTS {
            let k_eff_i = match condition {
                E5Condition::Vitality => {
                    let cm = e5_coupling_multiplier(vitalities[i], E5_LAMBDA_V, E5_V_FLOOR);
                    E5_K_TIME * cm
                }
                E5Condition::Uniform => {
                    let cm = e5_coupling_multiplier(mean_vitality, E5_LAMBDA_V, E5_V_FLOOR);
                    E5_K_TIME * cm
                }
                E5Condition::Control => 0.0,
            };
            phases[i] = kuramoto_phase_step(phases[i], omegas[i], theta_kick, k_eff_i, 0.0, E5_DT);
        }

        // Per-agent PLV to kick
        let mut plv_sum = 0.0f32;
        let mut plv_count = 0usize;
        for i in 0..E5_N_AGENTS {
            let d_i = wrap_pm_pi(phases[i] - theta_kick);
            plv_buffers[i].push(d_i);
            if plv_buffers[i].is_full() {
                let p = plv_buffers[i].plv();
                if p.is_finite() {
                    plv_sum += p;
                    plv_count += 1;
                }
            }
        }
        let group_plv = if plv_count > 0 {
            plv_sum / plv_count as f32
        } else {
            f32::NAN
        };
        group_plv_series.push((t, group_plv));
    }

    // Collect per-agent final PLV
    let agent_final: Vec<E5AgentFinal> = (0..E5_N_AGENTS)
        .map(|i| E5AgentFinal {
            consonance: consonances[i],
            plv: if plv_buffers[i].is_full() {
                plv_buffers[i].plv()
            } else {
                f32::NAN
            },
        })
        .collect();

    // Pearson r(consonance, final PLV)
    let pearson_r = pearson_r_e5(&agent_final);

    E5VitalityResult {
        group_plv_series,
        agent_final,
        pearson_r,
        condition,
        seed,
    }
}

fn pearson_r_e5(agents: &[E5AgentFinal]) -> f32 {
    let valid: Vec<(f32, f32)> = agents
        .iter()
        .filter(|a| a.plv.is_finite())
        .map(|a| (a.consonance, a.plv))
        .collect();
    if valid.len() < 3 {
        return f32::NAN;
    }
    let n = valid.len() as f32;
    let mean_x = valid.iter().map(|(x, _)| x).sum::<f32>() / n;
    let mean_y = valid.iter().map(|(_, y)| y).sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_x = 0.0f32;
    let mut var_y = 0.0f32;
    for &(x, y) in &valid {
        let dx = x - mean_x;
        let dy = y - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom < 1e-12 { 0.0 } else { cov / denom }
}

fn build_e5_delta_series(
    label: &'static str,
    color: RGBColor,
    lhs: E5Condition,
    rhs: E5Condition,
    seeds: &[u64],
    result_by_cond_seed: &HashMap<(E5Condition, u64), &E5VitalityResult>,
) -> E5DeltaSeries {
    let mut first_len: Option<usize> = None;
    let mut t_axis: Vec<f32> = Vec::new();
    for &seed in seeds {
        let (Some(lhs_res), Some(rhs_res)) = (
            result_by_cond_seed.get(&(lhs, seed)),
            result_by_cond_seed.get(&(rhs, seed)),
        ) else {
            continue;
        };
        let n = lhs_res
            .group_plv_series
            .len()
            .min(rhs_res.group_plv_series.len());
        if n == 0 {
            continue;
        }
        first_len = Some(n);
        t_axis = lhs_res
            .group_plv_series
            .iter()
            .take(n)
            .map(|(t, _)| *t)
            .collect();
        break;
    }

    let n_steps = first_len.unwrap_or(0);
    if n_steps == 0 {
        return E5DeltaSeries {
            label,
            color,
            series: Vec::new(),
        };
    }

    let mut diffs_by_step: Vec<Vec<f32>> = (0..n_steps).map(|_| Vec::new()).collect();
    for &seed in seeds {
        let (Some(lhs_res), Some(rhs_res)) = (
            result_by_cond_seed.get(&(lhs, seed)),
            result_by_cond_seed.get(&(rhs, seed)),
        ) else {
            continue;
        };
        let n = n_steps
            .min(lhs_res.group_plv_series.len())
            .min(rhs_res.group_plv_series.len());
        for (step, diffs) in diffs_by_step.iter_mut().enumerate().take(n) {
            let v_l = lhs_res.group_plv_series[step].1;
            let v_r = rhs_res.group_plv_series[step].1;
            if v_l.is_finite() && v_r.is_finite() {
                diffs.push(v_l - v_r);
            }
        }
    }

    let mut series: Vec<(f32, f32, f32)> = Vec::with_capacity(n_steps);
    for (step, vals) in diffs_by_step.iter().enumerate().take(n_steps) {
        let t = t_axis.get(step).copied().unwrap_or(step as f32 * E5_DT);
        if vals.is_empty() {
            series.push((t, f32::NAN, 0.0));
            continue;
        }
        let mean = vals.iter().copied().sum::<f32>() / vals.len() as f32;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32;
        let ci95 = 1.96 * var.sqrt() / (vals.len() as f32).sqrt();
        series.push((t, mean, ci95));
    }

    E5DeltaSeries {
        label,
        color,
        series,
    }
}

/// Sigmoid fit with midpoint fixed at C_field = 0:
///   y = base + amplitude · σ(steepness · x)
/// where σ(z) = 1 / (1 + exp(-z)).
/// Grid-search over steepness; OLS for (base, amplitude) at each grid point.
fn fit_e5_sigmoid(points: &[(f32, f32)]) -> Option<E5SigmoidFit> {
    if points.len() < 4 {
        return None;
    }
    let n = points.len() as f32;
    let mut best: Option<E5SigmoidFit> = None;
    let mut best_sse = f32::INFINITY;

    // Grid search over steepness (1..40)
    for i_s in 1..=80 {
        let s = i_s as f32 * 0.5; // 0.5 .. 40.0

        // Compute g_i = σ(s · x_i) for each point
        let mut sum_g = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_gg = 0.0f32;
        let mut sum_gy = 0.0f32;

        for &(x, y) in points {
            let g = 1.0 / (1.0 + (-s * x).exp());
            sum_g += g;
            sum_y += y;
            sum_gg += g * g;
            sum_gy += g * y;
        }

        // OLS: y = base + amplitude · g
        let det = n * sum_gg - sum_g * sum_g;
        if det.abs() < 1e-8 {
            continue;
        }
        let base = (sum_y * sum_gg - sum_g * sum_gy) / det;
        let amplitude = (n * sum_gy - sum_g * sum_y) / det;
        if !base.is_finite() || !amplitude.is_finite() {
            continue;
        }

        let mut sse = 0.0f32;
        for &(x, y) in points {
            let g = 1.0 / (1.0 + (-s * x).exp());
            let y_hat = base + amplitude * g;
            let e = y - y_hat;
            sse += e * e;
        }
        if sse < best_sse {
            best_sse = sse;
            best = Some(E5SigmoidFit {
                base,
                amplitude,
                steepness: s,
            });
        }
    }

    best
}

fn render_e1_plot(
    out_path: &Path,
    _anchor_hz: f32,
    log2_ratio_scan: &[f32],
    perc_h_pot_scan: &[f32],
    perc_r_state01_scan: &[f32],
    perc_c_field_scan: &[f32],
    perc_c_density_scan: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -1.5f32;
    let x_max = 1.5f32;

    let mut h_points = Vec::new();
    let mut r_points = Vec::new();
    let mut c_points = Vec::new();
    let mut d_points = Vec::new();
    for i in 0..log2_ratio_scan.len() {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        h_points.push((x, perc_h_pot_scan[i]));
        r_points.push((x, perc_r_state01_scan[i]));
        c_points.push((x, perc_c_field_scan[i]));
        d_points.push((x, perc_c_density_scan[i]));
    }

    let y01_max = 1.05f32;

    // Compute C and D y-ranges before layout so we can size panels proportionally
    let mut c_min = f32::INFINITY;
    let mut c_max = f32::NEG_INFINITY;
    for &(_, y) in &c_points {
        if y.is_finite() {
            c_min = c_min.min(y);
            c_max = c_max.max(y);
        }
    }
    if !c_min.is_finite() || !c_max.is_finite() || (c_max - c_min).abs() < 1e-6 {
        c_min = -1.0;
        c_max = 1.0;
    }
    let pad = 0.05f32;
    let c_y_min = c_min - pad;
    let c_y_max = c_max + pad;
    let c_span = c_y_max - c_y_min;

    let mut d_max_val = f32::NEG_INFINITY;
    for &(_, y) in &d_points {
        if y.is_finite() {
            d_max_val = d_max_val.max(y);
        }
    }
    if !d_max_val.is_finite() {
        d_max_val = 1.0;
    }
    let d_y_min = 0.0f32;
    let d_y_max = d_max_val + pad;
    let d_span = d_y_max - d_y_min;

    // Size C and D panels so their chart areas share the same pixels-per-unit scale.
    // Vertical overhead: top_margin(12) + caption(~48) + bottom_margin + x_label_area
    //   Panel C: 12 + 48 + 24 + 0  = 84
    //   Panel D: 12 + 48 + 12 + 90 = 162
    let total_h: u32 = 1020;
    let h_ab: u32 = 217; // A,B each (15% smaller than equal quarter)
    let remaining = total_h - 2 * h_ab; // 586 for C+D (15% larger)
    let ov_c = 84.0f32;
    let ov_d = 162.0f32;
    let h_d =
        ((d_span * (remaining as f32 - ov_c) + ov_d * c_span) / (c_span + d_span)).round() as u32;
    let h_c = remaining - h_d;

    let root = bitmap_root(out_path, (2200, total_h)).into_drawing_area();
    root.fill(&WHITE)?;
    let (top_half, bottom_half) = root.split_vertically(2 * h_ab);
    let (panel_a, panel_b) = top_half.split_vertically(h_ab);
    let (panel_c, panel_d) = bottom_half.split_vertically(h_c);

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();
    let x_grid = [-1.5f32, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
    let ratio_guide_style = RGBColor(50, 100, 180).mix(0.55);
    let y_guides = [-0.5f32, 0.0, 0.5, 1.0];

    let mut chart_h = ChartBuilder::on(&panel_a)
        .caption("A: Harmonicity H\u{2080}\u{2081}(f)", ("sans-serif", 48))
        .margin(12)
        .margin_right(25)
        .margin_bottom(24)
        .x_label_area_size(0)
        .y_label_area_size(110)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y01_max)?;

    chart_h
        .configure_mesh()
        .disable_mesh()
        .y_desc("H\u{2080}\u{2081}")
        .x_labels(0)
        .y_labels(6)
        .label_style(("sans-serif", 42).into_font())
        .axis_desc_style(("sans-serif", 46).into_font())
        .draw()?;

    for &x in &x_grid {
        chart_h.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y01_max)],
            BLACK.mix(0.10),
        )))?;
    }
    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_h.draw_series(std::iter::once(DashedPathElement::new(
                vec![(x, 0.0), (x, y01_max)].into_iter(),
                8,
                6,
                ratio_guide_style,
            )))?;
        }
    }
    for &y in &y_guides {
        if y >= 0.0 && y <= y01_max {
            chart_h.draw_series(std::iter::once(PathElement::new(
                vec![(x_min, y), (x_max, y)],
                BLACK.mix(0.12),
            )))?;
        }
    }

    chart_h.draw_series(LineSeries::new(h_points, &PAL_H))?;

    let mut chart_r = ChartBuilder::on(&panel_b)
        .caption("B: Roughness R\u{2080}\u{2081}(f)", ("sans-serif", 48))
        .margin(12)
        .margin_right(25)
        .margin_bottom(24)
        .x_label_area_size(0)
        .y_label_area_size(110)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y01_max)?;

    chart_r
        .configure_mesh()
        .disable_mesh()
        .y_desc("R\u{2080}\u{2081}")
        .x_labels(0)
        .y_labels(6)
        .label_style(("sans-serif", 42).into_font())
        .axis_desc_style(("sans-serif", 46).into_font())
        .draw()?;

    for &x in &x_grid {
        chart_r.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y01_max)],
            BLACK.mix(0.10),
        )))?;
    }
    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_r.draw_series(std::iter::once(DashedPathElement::new(
                vec![(x, 0.0), (x, y01_max)].into_iter(),
                8,
                6,
                ratio_guide_style,
            )))?;
        }
    }
    for &y in &y_guides {
        if y >= 0.0 && y <= y01_max {
            chart_r.draw_series(std::iter::once(PathElement::new(
                vec![(x_min, y), (x_max, y)],
                BLACK.mix(0.12),
            )))?;
        }
    }

    chart_r.draw_series(LineSeries::new(r_points, &PAL_R))?;

    let mut chart_c = ChartBuilder::on(&panel_c)
        .caption("C: Consonance field C_field(f)", ("sans-serif", 48))
        .margin(12)
        .margin_right(25)
        .margin_bottom(24)
        .x_label_area_size(0)
        .y_label_area_size(110)
        .build_cartesian_2d(x_min..x_max, c_y_min..c_y_max)?;

    chart_c
        .configure_mesh()
        .disable_mesh()
        .y_desc("C_field")
        .x_labels(0)
        .y_labels(6)
        .label_style(("sans-serif", 42).into_font())
        .axis_desc_style(("sans-serif", 46).into_font())
        .draw()?;

    for &x in &x_grid {
        chart_c.draw_series(std::iter::once(PathElement::new(
            vec![(x, c_y_min), (x, c_y_max)],
            BLACK.mix(0.10),
        )))?;
    }
    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_c.draw_series(std::iter::once(DashedPathElement::new(
                vec![(x, c_y_min), (x, c_y_max)].into_iter(),
                8,
                6,
                ratio_guide_style,
            )))?;
        }
    }
    for &y in &y_guides {
        if y >= c_y_min && y <= c_y_max {
            chart_c.draw_series(std::iter::once(PathElement::new(
                vec![(x_min, y), (x_max, y)],
                BLACK.mix(0.12),
            )))?;
        }
    }

    chart_c.draw_series(LineSeries::new(c_points, &PAL_C))?;

    // Panel D: C_density (non-negative). Same pixels-per-unit scale as Panel C.
    let mut chart_d = ChartBuilder::on(&panel_d)
        .caption("D: Consonance density C_density(f)", ("sans-serif", 48))
        .margin(12)
        .margin_right(25)
        .x_label_area_size(90)
        .y_label_area_size(110)
        .build_cartesian_2d(x_min..x_max, d_y_min..d_y_max)?;

    chart_d
        .configure_mesh()
        .disable_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("C_density")
        .y_labels(3)
        .label_style(("sans-serif", 42).into_font())
        .axis_desc_style(("sans-serif", 46).into_font())
        .draw()?;

    for &x in &x_grid {
        chart_d.draw_series(std::iter::once(PathElement::new(
            vec![(x, d_y_min), (x, d_y_max)],
            BLACK.mix(0.10),
        )))?;
    }
    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart_d.draw_series(std::iter::once(DashedPathElement::new(
                vec![(x, d_y_min), (x, d_y_max)].into_iter(),
                8,
                6,
                ratio_guide_style,
            )))?;
        }
    }
    for &y in &y_guides {
        if y >= d_y_min && y <= d_y_max {
            chart_d.draw_series(std::iter::once(PathElement::new(
                vec![(x_min, y), (x_max, y)],
                BLACK.mix(0.12),
            )))?;
        }
    }

    chart_d.draw_series(LineSeries::new(d_points, &PAL_CD))?;

    root.present()?;
    Ok(())
}

fn render_e1_h_mirror_triplet_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    h_m0: &[f32],
    h_m05: &[f32],
    h_m1: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -3.0f32;
    let x_max = 3.0f32;
    let n = log2_ratio_scan
        .len()
        .min(h_m0.len())
        .min(h_m05.len())
        .min(h_m1.len());

    let mut p_m0 = Vec::new();
    let mut p_m05 = Vec::new();
    let mut p_m1 = Vec::new();
    for i in 0..n {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        p_m0.push((x, h_m0[i]));
        p_m05.push((x, h_m05[i]));
        p_m1.push((x, h_m1[i]));
    }

    let y_max = h_m0
        .iter()
        .chain(h_m05.iter())
        .chain(h_m1.iter())
        .copied()
        .fold(0.0f32, f32::max)
        .max(1e-6)
        * 1.1;

    let root = bitmap_root(out_path, (1600, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Harmonicity H(f) by mirror_weight | anchor {} Hz",
                anchor_hz
            ),
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(64)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;

    chart
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H potential")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.15),
            )))?;
        }
    }

    chart
        .draw_series(LineSeries::new(p_m0, &PAL_H))?
        .label("m0 (mirror_weight=0.0)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], PAL_H));
    chart
        .draw_series(LineSeries::new(p_m05, &PAL_C))?
        .label("m05 (mirror_weight=0.5)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], PAL_C));
    chart
        .draw_series(LineSeries::new(p_m1, &PAL_R))?
        .label("m1 (mirror_weight=1.0)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], PAL_R));
    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.85))
        .border_style(BLACK.mix(0.25))
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_e1_h_mirror_diff_plot(
    out_path: &Path,
    anchor_hz: f32,
    log2_ratio_scan: &[f32],
    h_m0: &[f32],
    h_m1: &[f32],
) -> Result<(), Box<dyn Error>> {
    let x_min = -3.0f32;
    let x_max = 3.0f32;
    let n = log2_ratio_scan.len().min(h_m0.len()).min(h_m1.len());
    let mut diff_points = Vec::new();
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for i in 0..n {
        let x = log2_ratio_scan[i];
        if x < x_min || x > x_max {
            continue;
        }
        let d = h_m0[i] - h_m1[i];
        diff_points.push((x, d));
        if d.is_finite() {
            y_min = y_min.min(d);
            y_max = y_max.max(d);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    }
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-4);
    let y_lo = (y_min - pad).min(0.0);
    let y_hi = (y_max + pad).max(0.0);

    let root = bitmap_root(out_path, (1600, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let ratio_guides = [0.5f32, 6.0 / 5.0, 1.25, 4.0 / 3.0, 1.5, 5.0 / 3.0, 2.0];
    let ratio_guides_log2: Vec<f32> = ratio_guides.iter().map(|r| r.log2()).collect();

    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!(
                "Harmonicity Difference H_m0(f) - H_m1(f) | anchor {} Hz",
                anchor_hz
            ),
            ("sans-serif", 24),
        )
        .margin(12)
        .x_label_area_size(42)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("log2(f / f_anchor)")
        .y_desc("H(m0) - H(m1)")
        .draw()?;

    for &x in &ratio_guides_log2 {
        if x >= x_min && x <= x_max {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, y_lo), (x, y_hi)],
                BLACK.mix(0.12),
            )))?;
        }
    }
    chart.draw_series(std::iter::once(PathElement::new(
        vec![(x_min, 0.0), (x_max, 0.0)],
        BLACK.mix(0.35),
    )))?;
    chart.draw_series(LineSeries::new(diff_points, &PAL_C))?;

    root.present()?;
    Ok(())
}

struct ConsonanceWorkspace {
    params: LandscapeParams,
    r_ref_peak: f32,
    r_shift_bins: i32,
}

#[derive(Clone, Copy)]
enum E2Condition {
    Baseline,
    NoHillClimb,
    NoCrowding,
    ShuffledLandscape,
}

struct E2AnchorShiftStats {
    step: usize,
    anchor_hz_before: f32,
    anchor_hz_after: f32,
    count_min: usize,
    count_max: usize,
    respawned: usize,
}

struct E2Run {
    seed: u64,
    mean_c_series: Vec<f32>,
    mean_c_level_series: Vec<f32>,
    mean_c_score_loo_series: Vec<f32>,
    mean_c_score_chosen_loo_series: Vec<f32>,
    mean_score_series: Vec<f32>,
    mean_crowding_series: Vec<f32>,
    moved_frac_series: Vec<f32>,
    accepted_worse_frac_series: Vec<f32>,
    attempted_update_frac_series: Vec<f32>,
    moved_given_attempt_frac_series: Vec<f32>,
    mean_abs_delta_semitones_series: Vec<f32>,
    mean_abs_delta_semitones_moved_series: Vec<f32>,
    semitone_samples_pre: Vec<f32>,
    semitone_samples_post: Vec<f32>,
    final_semitones: Vec<f32>,
    final_freqs_hz: Vec<f32>,
    final_log2_ratios: Vec<f32>,
    trajectory_semitones: Vec<Vec<f32>>,
    trajectory_c_level: Vec<Vec<f32>>,
    anchor_shift: E2AnchorShiftStats,
    density_mass_mean: f32,
    density_mass_min: f32,
    density_mass_max: f32,
    r_state01_min: f32,
    r_state01_mean: f32,
    r_state01_max: f32,
    r_ref_peak: f32,
    roughness_k: f32,
    roughness_ref_eps: f32,
    fixed_drone_hz: f32,
    n_agents: usize,
    k_bins: i32,
}

#[derive(Clone, Copy, Debug, Default)]
struct E2SceneGPoint {
    h_scene: f32,
    h_single_mean: f32,
    h_social: f32,
    r_scene: f32,
    r_single_mean: f32,
    r_social: f32,
    g_scene: f32,
}

struct E2SweepStats {
    mean_c: Vec<f32>,
    std_c: Vec<f32>,
    mean_c_level: Vec<f32>,
    std_c_level: Vec<f32>,
    mean_c_score_loo: Vec<f32>,
    std_c_score_loo: Vec<f32>,
    mean_g_scene: Vec<f32>,
    std_g_scene: Vec<f32>,
    mean_score: Vec<f32>,
    std_score: Vec<f32>,
    mean_crowding: Vec<f32>,
    std_crowding: Vec<f32>,
    n: usize,
}

struct E3Arrays {
    lifetimes: Vec<u32>,
    c_score_birth: Vec<f32>,
    c_score_firstk: Vec<f32>,
    c_level_birth: Vec<f32>,
    c_level_firstk: Vec<f32>,
    avg_score_attack: Vec<f32>,
    avg_attack: Vec<f32>,
    attack_tick_count: Vec<u32>,
}

struct E3SeedOutput {
    condition: E3Condition,
    seed: u64,
    arrays: E3Arrays,
    corr_firstk: CorrStats,
}

struct HistSweepStats {
    centers: Vec<f32>,
    mean_count: Vec<f32>,
    std_count: Vec<f32>,
    mean_frac: Vec<f32>,
    std_frac: Vec<f32>,
    n: usize,
}

#[derive(Clone, Copy)]
struct HistStructureMetrics {
    entropy: f32,
    gini: f32,
    peakiness: f32,
    kl_uniform: f32,
}

struct HistStructureRow {
    condition: &'static str,
    seed: u64,
    metrics: HistStructureMetrics,
}

#[derive(Clone, Copy)]
struct DiversityMetrics {
    unique_bins: usize,
    nn_mean: f32,
    nn_std: f32,
    semitone_var: f32,
    semitone_mad: f32,
}

struct DiversityRow {
    condition: &'static str,
    seed: u64,
    metrics: DiversityMetrics,
}

struct E2DenseSweepRow {
    range_oct: f32,
    n_agents: usize,
    voices_per_oct: f32,
    condition: &'static str,
    seed: u64,
    c_score_loo_end: f32,
    ji_scene_end: f32,
    entropy_end: f32,
    unique_bins_end: usize,
    nn_mean_end: f32,
    close_pair_frac_50ct: f32,
}

struct E2DenseSweepSummaryRow {
    range_oct: f32,
    n_agents: usize,
    voices_per_oct: f32,
    condition: &'static str,
    c_score_loo_mean: f32,
    c_score_loo_ci95: f32,
    ji_scene_mean: f32,
    ji_scene_ci95: f32,
    entropy_mean: f32,
    entropy_ci95: f32,
    unique_bins_mean: f32,
    unique_bins_ci95: f32,
    nn_mean_mean: f32,
    nn_mean_ci95: f32,
    close_pair_mean: f32,
    close_pair_ci95: f32,
    n: usize,
}

struct E2CandidateSummaryRow {
    range_oct: f32,
    n_agents: usize,
    n_seeds: usize,
    baseline_c_score_loo_mean: f32,
    nohill_c_score_loo_mean: f32,
    delta_c_score_loo: f32,
    baseline_ji_scene_mean: f32,
    nohill_ji_scene_mean: f32,
    delta_ji_scene: f32,
    baseline_entropy_mean: f32,
    nohill_entropy_mean: f32,
    delta_entropy: f32,
    baseline_unique_bins_mean: f32,
    nohill_unique_bins_mean: f32,
    delta_unique_bins: f32,
    baseline_nn_mean: f32,
    nohill_nn_mean: f32,
    shortlist_score: f32,
}

#[derive(Clone, Copy)]
struct ConsonantMassRow {
    condition: &'static str,
    seed: u64,
    mass_core: f32,
    mass_extended: f32,
}

fn seeded_rng(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

fn build_log2_ratio_scan(space: &Log2Space, anchor_hz: f32) -> Vec<f32> {
    let anchor_log2 = anchor_hz.log2();
    space
        .centers_log2
        .iter()
        .map(|&l| l - anchor_log2)
        .collect()
}

fn log2_ratio_bounds(log2_ratio_scan: &[f32], min: f32, max: f32) -> (usize, usize) {
    let mut min_idx: Option<usize> = None;
    let mut max_idx: Option<usize> = None;
    for (i, &v) in log2_ratio_scan.iter().enumerate() {
        if v >= min && v <= max {
            if min_idx.is_none() {
                min_idx = Some(i);
            }
            max_idx = Some(i);
        }
    }
    let fallback_min = 0;
    let fallback_max = log2_ratio_scan.len().saturating_sub(1);
    (
        min_idx.unwrap_or(fallback_min),
        max_idx.unwrap_or(fallback_max),
    )
}

fn float_key(value: f32) -> i32 {
    (value * FLOAT_KEY_SCALE).round() as i32
}

fn float_from_key(key: i32) -> f32 {
    key as f32 / FLOAT_KEY_SCALE
}

fn format_float_token(value: f32) -> String {
    let s = format!("{value:.2}");
    s.replace('.', "p")
}

#[allow(dead_code)]
struct E2FixedDrone {
    hz: f32,
    idx: usize,
}

fn e2_fixed_drone(space: &Log2Space, hz: f32) -> E2FixedDrone {
    let hz = hz.max(1.0);
    E2FixedDrone {
        hz,
        idx: space.nearest_index(hz),
    }
}

fn build_env_scans_with_fixed_sources(
    space: &Log2Space,
    fixed_source_indices: &[usize],
    agent_indices: &[usize],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    let mut density_scan = vec![0.0f32; space.n_bins()];

    let mut add_source = |idx: usize| {
        env_scan[idx] += 1.0;
        let denom = du_scan[idx].max(1e-12);
        density_scan[idx] += 1.0 / denom;
    };

    for &idx in fixed_source_indices {
        add_source(idx);
    }
    for &idx in agent_indices {
        add_source(idx);
    }

    (env_scan, density_scan)
}

#[cfg(test)]
fn build_env_scans(
    space: &Log2Space,
    anchor_idx: usize,
    agent_indices: &[usize],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    build_env_scans_with_fixed_sources(
        space,
        std::slice::from_ref(&anchor_idx),
        agent_indices,
        du_scan,
    )
}

fn build_env_density_from_freqs(
    space: &Log2Space,
    freqs_hz: &[f32],
    du_scan: &[f32],
    env_partials: u32,
    env_partial_decay: f32,
) -> (Vec<f32>, Vec<f32>) {
    let partials = env_partials.clamp(1, 32);
    let decay = if env_partial_decay.is_finite() {
        env_partial_decay.clamp(0.0, 4.0)
    } else {
        1.0
    };
    let mut env_scan = vec![0.0f32; space.n_bins()];
    let mut density_scan = vec![0.0f32; space.n_bins()];

    for &f0_hz in freqs_hz {
        if !f0_hz.is_finite() || f0_hz <= 0.0 {
            continue;
        }
        for k in 1..=partials {
            let freq_hz = f0_hz * k as f32;
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                continue;
            }
            let Some(idx) = space.index_of_freq(freq_hz) else {
                continue;
            };
            let weight = 1.0 / (k as f32).powf(decay);
            env_scan[idx] += weight;
            density_scan[idx] += weight / du_scan[idx].max(1e-12);
        }
    }

    (env_scan, density_scan)
}

fn build_consonance_workspace(space: &Log2Space) -> ConsonanceWorkspace {
    let roughness_kernel = RoughnessKernel::new(KernelParams::default(), 0.005);
    let mut harmonicity_params = HarmonicityParams::default();
    harmonicity_params.rho_common_overtone = harmonicity_params.rho_common_root;
    harmonicity_params.mirror_weight = 0.5;
    let harmonicity_kernel = HarmonicityKernel::new(space, harmonicity_params);
    let params = LandscapeParams {
        fs: 48_000.0,
        max_hist_cols: 1,
        roughness_kernel,
        harmonicity_kernel,
        consonance_kernel: ConsonanceKernel::default(),
        consonance_representation: ConsonanceRepresentationParams {
            beta: E2_C_LEVEL_BETA,
            theta: E2_C_LEVEL_THETA,
        },
        consonance_density_roughness_gain: 1.0,
        roughness_scalar_mode: RoughnessScalarMode::Total,
        roughness_half: 0.1,
        loudness_exp: 1.0,
        ref_power: 1.0,
        tau_ms: 1.0,
        roughness_k: 1.0,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };
    let r_ref = psycho_state::compute_roughness_reference(&params, space);
    ConsonanceWorkspace {
        params,
        r_ref_peak: r_ref.peak,
        r_shift_bins: 0,
    }
}

fn r_state01_stats(scan: &[f32]) -> RState01Stats {
    if scan.is_empty() {
        return RState01Stats {
            min: 0.0,
            mean: 0.0,
            max: 0.0,
        };
    }
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for &value in scan {
        if !value.is_finite() {
            continue;
        }
        min = min.min(value);
        max = max.max(value);
        sum += value;
        count += 1;
    }
    if count == 0 {
        return RState01Stats {
            min: 0.0,
            mean: 0.0,
            max: 0.0,
        };
    }
    let mean = sum / count as f32;
    RState01Stats {
        min: min.clamp(0.0, 1.0),
        mean: mean.clamp(0.0, 1.0),
        max: max.clamp(0.0, 1.0),
    }
}

fn compute_c_score_level_scans(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_scan: &[f32],
    density_scan: &[f32],
    du_scan: &[f32],
) -> (Vec<f32>, Vec<f32>, f32, RState01Stats) {
    space.assert_scan_len_named(du_scan, "du_scan");
    // Use the same epsilon as the roughness reference normalization so density scaling stays aligned.
    let (density_norm, density_mass) =
        psycho_state::normalize_density(density_scan, du_scan, workspace.params.roughness_ref_eps);
    let (perc_h_pot_scan, _) = workspace
        .params
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(env_scan, space);
    let (perc_r_pot_scan, _) = workspace
        .params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&density_norm, space);

    let mut perc_r_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::r_pot_scan_to_r_state01_scan(
        &perc_r_pot_scan,
        workspace.r_ref_peak,
        workspace.params.roughness_k,
        &mut perc_r_state01_scan,
    );
    if workspace.r_shift_bins != 0 {
        let n = perc_r_state01_scan.len();
        let shift = workspace.r_shift_bins.rem_euclid(n as i32) as usize;
        perc_r_state01_scan.rotate_left(shift);
    }
    let r_state_stats = r_state01_stats(&perc_r_state01_scan);

    let h_ref_max = 1.0f32;
    let mut perc_h_state01_scan = vec![0.0f32; space.n_bins()];
    psycho_state::h_pot_scan_to_h_state01_scan(
        &perc_h_pot_scan,
        h_ref_max,
        &mut perc_h_state01_scan,
    );

    let mut c_score_scan = vec![0.0f32; space.n_bins()];
    let mut c_level_scan = vec![0.0f32; space.n_bins()];
    for i in 0..space.n_bins() {
        let c_score = workspace
            .params
            .consonance_kernel
            .score(perc_h_state01_scan[i], perc_r_state01_scan[i]);
        let c_level = workspace.params.consonance_representation.level(c_score);
        c_score_scan[i] = c_score;
        c_level_scan[i] = c_level.clamp(0.0, 1.0);
    }
    (c_score_scan, c_level_scan, density_mass, r_state_stats)
}

fn e2_scene_spectrum_features_for_freqs(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    freqs_hz: &[f32],
) -> (f32, f32) {
    let clean_freqs: Vec<f32> = freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return (0.0, 0.0);
    }
    let (env_scan, _density_scan) = build_env_density_from_freqs(
        space,
        &clean_freqs,
        du_scan,
        E2_SCENE_G_ENV_PARTIALS,
        E2_SCENE_G_ENV_DECAY,
    );
    let h_dual = workspace
        .params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(&env_scan, space);
    let h_scene = h_dual.metrics.binding_strength.max(0.0);
    let r_scene = e2_scene_direct_roughness_from_freqs(workspace, &clean_freqs);
    (h_scene, r_scene)
}

fn e2_scene_direct_roughness_from_freqs(workspace: &ConsonanceWorkspace, freqs_hz: &[f32]) -> f32 {
    let clean_freqs: Vec<f32> = freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return 0.0;
    }
    let partials = E2_SCENE_G_ENV_PARTIALS.clamp(1, 32) as usize;
    let decay = if E2_SCENE_G_ENV_DECAY.is_finite() {
        E2_SCENE_G_ENV_DECAY.clamp(0.0, 4.0)
    } else {
        1.0
    };
    let harmonic_norm = (1..=partials)
        .map(|k| 1.0 / (k as f32).powf(decay))
        .sum::<f32>()
        .max(1e-6);
    let mut partials_erb = Vec::with_capacity(clean_freqs.len() * partials);
    for &freq in &clean_freqs {
        for k in 1..=partials {
            let harmonic = k as f32;
            let partial_freq = freq * harmonic;
            if !partial_freq.is_finite() || partial_freq <= 0.0 {
                continue;
            }
            let gain = (1.0 / harmonic.powf(decay)) / harmonic_norm;
            partials_erb.push((hz_to_erb(partial_freq), gain));
        }
    }
    let kernel_params = &workspace.params.roughness_kernel.params;
    let mut total = 0.0f32;
    for i in 0..partials_erb.len() {
        let (erb_i, gain_i) = partials_erb[i];
        for &(erb_j, gain_j) in partials_erb.iter().skip(i + 1) {
            total += gain_i * gain_j * eval_kernel_delta_erb(kernel_params, erb_i - erb_j);
        }
    }
    total.max(0.0)
}

fn e2_scene_g_phi(workspace: &ConsonanceWorkspace, h_scene: f32, r_scene: f32) -> f32 {
    let kernel = workspace.params.consonance_kernel;
    let h = if h_scene.is_finite() {
        h_scene.max(0.0)
    } else {
        0.0
    };
    let r = if r_scene.is_finite() {
        r_scene.max(0.0)
    } else {
        0.0
    };
    kernel.a * h + kernel.b * r + kernel.c * h * r + kernel.d
}

fn e2_scene_g_point_for_freqs(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    freqs_hz: &[f32],
) -> E2SceneGPoint {
    let clean_freqs: Vec<f32> = freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return E2SceneGPoint::default();
    }
    let (h_scene, r_scene) =
        e2_scene_spectrum_features_for_freqs(space, workspace, du_scan, &clean_freqs);

    let mut singleton_cache: HashMap<i32, (f32, f32)> = HashMap::new();
    let mut h_single_sum = 0.0f32;
    let mut r_single_sum = 0.0f32;
    for &freq in &clean_freqs {
        let key = (freq * 1000.0).round() as i32;
        let (h_single, r_single) = *singleton_cache.entry(key).or_insert_with(|| {
            let singleton = [freq];
            e2_scene_spectrum_features_for_freqs(space, workspace, du_scan, &singleton)
        });
        h_single_sum += h_single;
        r_single_sum += r_single;
    }
    let inv_n = 1.0 / clean_freqs.len() as f32;
    let h_single_mean = h_single_sum * inv_n;
    let r_single_mean = r_single_sum * inv_n;
    let h_social = h_scene - h_single_mean;
    let r_social = r_scene - r_single_mean;
    let g_scene = e2_scene_g_phi(workspace, h_social, r_social);

    E2SceneGPoint {
        h_scene,
        h_single_mean,
        h_social,
        r_scene,
        r_single_mean,
        r_social,
        g_scene,
    }
}

struct UpdateStats {
    mean_c_score_current_loo: f32,
    mean_c_score_chosen_loo: f32,
    mean_score: f32,
    mean_crowding: f32,
    moved_frac: f32,
    accepted_worse_frac: f32,
    attempted_update_frac: f32,
    moved_given_attempt_frac: f32,
    mean_abs_delta_semitones: f32,
    mean_abs_delta_semitones_moved: f32,
}

struct OneUpdateStats {
    moved: bool,
    accepted_worse: bool,
    abs_delta_semitones: f32,
    abs_delta_semitones_moved: f32,
    c_score_current: f32,
    c_score_chosen: f32,
    chosen_score: f32,
    chosen_crowding: f32,
}

#[derive(Clone, Copy, Debug)]
struct RState01Stats {
    min: f32,
    mean: f32,
    max: f32,
}

fn k_from_semitones(step_semitones: f32) -> i32 {
    let bins_per_semitone = SPACE_BINS_PER_OCT as f32 / 12.0;
    let k = (bins_per_semitone * step_semitones).round() as i32;
    k.max(1)
}

fn shift_indices_by_ratio(
    space: &Log2Space,
    indices: &mut [usize],
    ratio: f32,
    min_idx: usize,
    max_idx: usize,
    rng: &mut StdRng,
) -> (usize, usize, usize) {
    let mut count_min = 0usize;
    let mut count_max = 0usize;
    let mut respawned = 0usize;
    for idx in indices.iter_mut() {
        let target_hz = space.centers_hz[*idx] * ratio;
        let mut new_idx = space.nearest_index(target_hz);
        if new_idx < min_idx || new_idx > max_idx {
            let pick = rng.random_range(min_idx..(max_idx + 1));
            new_idx = pick;
            respawned += 1;
        }
        if new_idx == min_idx {
            count_min += 1;
        }
        if new_idx == max_idx {
            count_max += 1;
        }
        *idx = new_idx;
    }
    (count_min, count_max, respawned)
}

fn metropolis_accept(delta: f32, temperature: f32, u01: f32) -> (bool, bool) {
    if !delta.is_finite() {
        return (false, false);
    }
    if delta >= 0.0 {
        return (true, false);
    }
    if temperature <= 0.0 {
        return (false, false);
    }
    let prob = (delta / temperature).exp();
    if u01 < prob {
        (true, true)
    } else {
        (false, false)
    }
}

#[allow(clippy::too_many_arguments)]
fn update_one_agent_scored_loo(
    agent_i: usize,
    indices: &mut [usize],
    env_current: &mut [f32],
    density_current: &mut [f32],
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    temperature: f32,
    update_allowed: bool,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    u01: f32,
    env_loo: &mut [f32],
    density_loo: &mut [f32],
) -> OneUpdateStats {
    let agent_idx = indices[agent_i];
    env_loo.copy_from_slice(env_current);
    density_loo.copy_from_slice(density_current);
    env_loo[agent_idx] = (env_loo[agent_idx] - 1.0).max(0.0);
    let denom = du_scan[agent_idx].max(1e-12);
    density_loo[agent_idx] = (density_loo[agent_idx] - 1.0 / denom).max(0.0);
    let (c_score_scan, _, _, _) =
        compute_c_score_level_scans(space, workspace, env_loo, density_loo, du_scan);

    let current_erb_vals: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
    let current_erb = erb_scan[agent_idx];
    let skip_penalty = crowding_weight <= 0.0;
    let mut current_crowding = 0.0f32;
    if !skip_penalty {
        for (j, &other_erb) in current_erb_vals.iter().enumerate() {
            if j == agent_i {
                continue;
            }
            let d_erb = current_erb - other_erb;
            current_crowding += crowding_runtime_delta_erb(kernel_params, d_erb);
        }
    }
    let c_score_current = c_score_scan[agent_idx];
    let current_score = score_sign * c_score_current - crowding_weight * current_crowding;

    let backtrack_target = if block_backtrack {
        backtrack_targets.and_then(|prev| prev.get(agent_i).copied())
    } else {
        None
    };

    let (chosen_idx, chosen_score, chosen_crowding, accepted_worse) = if update_allowed {
        let start = (agent_idx as isize - k as isize).max(min_idx as isize) as usize;
        let end = (agent_idx as isize + k as isize).min(max_idx as isize) as usize;
        let mut best_idx = agent_idx;
        let mut best_score = f32::NEG_INFINITY;
        let mut best_crowding = 0.0f32;
        let mut found_candidate = false;
        for cand in start..=end {
            if cand == agent_idx {
                continue;
            }
            let cand_erb = erb_scan[cand];
            let mut crowding = 0.0f32;
            if !skip_penalty {
                for (j, &other_erb) in current_erb_vals.iter().enumerate() {
                    if j == agent_i {
                        continue;
                    }
                    let d_erb = cand_erb - other_erb;

                    crowding += crowding_runtime_delta_erb(kernel_params, d_erb);
                }
            }
            let c_score = c_score_scan[cand];
            let score = score_sign * c_score - crowding_weight * crowding;
            if let Some(prev_idx) = backtrack_target
                && cand == prev_idx
                && (score - current_score) <= E2_BACKTRACK_ALLOW_EPS
            {
                continue;
            }
            if score > best_score {
                best_score = score;
                best_idx = cand;
                best_crowding = crowding;
                found_candidate = true;
            }
        }
        if found_candidate {
            let delta = best_score - current_score;
            if delta > E2_SCORE_IMPROVE_EPS {
                (best_idx, best_score, best_crowding, false)
            } else if delta < 0.0 {
                let (accept, accepted_worse) = metropolis_accept(delta, temperature, u01);
                if accept {
                    (best_idx, best_score, best_crowding, accepted_worse)
                } else {
                    (agent_idx, current_score, current_crowding, false)
                }
            } else {
                (agent_idx, current_score, current_crowding, false)
            }
        } else {
            (agent_idx, current_score, current_crowding, false)
        }
    } else {
        (agent_idx, current_score, current_crowding, false)
    };

    if chosen_idx != agent_idx {
        e2_scene_apply_move(env_current, density_current, du_scan, agent_idx, chosen_idx);
        indices[agent_i] = chosen_idx;
    }
    let moved = chosen_idx != agent_idx;
    let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[agent_idx]);
    let abs_delta = delta_semitones.abs();
    let abs_delta_moved = if moved { abs_delta } else { 0.0 };
    let c_score_chosen = c_score_scan[chosen_idx];

    OneUpdateStats {
        moved,
        accepted_worse,
        abs_delta_semitones: abs_delta,
        abs_delta_semitones_moved: abs_delta_moved,
        c_score_current,
        c_score_chosen,
        chosen_score,
        chosen_crowding,
    }
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_scored_loo(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    _prev_indices: &[usize],
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    temperature: f32,
    sweep: usize,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    mut trajectory_semitones: Option<&mut [Vec<f32>]>,
    rng: &mut StdRng,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    space.assert_scan_len_named(env_total, "env_total");
    space.assert_scan_len_named(density_total, "density_total");
    space.assert_scan_len_named(du_scan, "du_scan");
    let order = build_e2_update_order(schedule, indices.len(), sweep, rng);
    let u01_by_agent: Vec<f32> = (0..indices.len()).map(|_| rng.random::<f32>()).collect();
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let mut env_current = env_total.to_vec();
    let mut density_current = density_total.to_vec();
    let mut env_loo = vec![0.0f32; env_total.len()];
    let mut density_loo = vec![0.0f32; density_total.len()];
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut accepted_worse_count = 0usize;

    for &agent_i in &order {
        let update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle | E2UpdateSchedule::SequentialRotate => true,
        };
        if update_allowed {
            attempt_count += 1;
        }
        let stats = update_one_agent_scored_loo(
            agent_i,
            indices,
            &mut env_current,
            &mut density_current,
            space,
            workspace,
            du_scan,
            erb_scan,
            log2_ratio_scan,
            min_idx,
            max_idx,
            k,
            score_sign,
            crowding_weight,
            kernel_params,
            temperature,
            update_allowed,
            block_backtrack,
            backtrack_targets,
            u01_by_agent[agent_i],
            &mut env_loo,
            &mut density_loo,
        );
        if stats.moved {
            moved_count += 1;
        }
        if stats.accepted_worse {
            accepted_worse_count += 1;
        }
        if let Some(trace) = trajectory_semitones.as_deref_mut() {
            record_e2_trajectory_snapshot(trace, indices, log2_ratio_scan);
        }
        if stats.abs_delta_semitones.is_finite() {
            abs_delta_sum += stats.abs_delta_semitones;
            abs_delta_moved_sum += stats.abs_delta_semitones_moved;
        }
        if stats.c_score_current.is_finite() {
            c_score_current_sum += stats.c_score_current;
            c_score_current_count += 1;
        }
        if stats.c_score_chosen.is_finite() {
            c_score_chosen_sum += stats.c_score_chosen;
            c_score_chosen_count += 1;
        }
        if stats.chosen_score.is_finite() {
            score_sum += stats.chosen_score;
            score_count += 1;
        }
        if stats.chosen_crowding.is_finite() {
            crowding_sum += stats.chosen_crowding;
            crowding_count += 1;
        }
    }

    let n = indices.len() as f32;
    let mean_c_score_current_loo = if c_score_current_count > 0 {
        c_score_current_sum / c_score_current_count as f32
    } else {
        0.0
    };
    let mean_c_score_chosen_loo = if c_score_chosen_count > 0 {
        c_score_chosen_sum / c_score_chosen_count as f32
    } else {
        0.0
    };
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_crowding = if crowding_count > 0 {
        crowding_sum / crowding_count as f32
    } else {
        0.0
    };
    let mean_abs_delta_semitones = abs_delta_sum / n;
    let mean_abs_delta_semitones_moved = if moved_count > 0 {
        abs_delta_moved_sum / moved_count as f32
    } else {
        0.0
    };
    UpdateStats {
        mean_c_score_current_loo,
        mean_c_score_chosen_loo,
        mean_score,
        mean_crowding,
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: accepted_worse_count as f32 / n,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones,
        mean_abs_delta_semitones_moved,
    }
}

#[allow(clippy::too_many_arguments, dead_code)]
fn update_e2_sweep_prescored(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    _prev_indices: &[usize],
    prescored_c_score_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    temperature: f32,
    sweep: usize,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    mut trajectory_semitones: Option<&mut [Vec<f32>]>,
    rng: &mut StdRng,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let skip_penalty = crowding_weight <= 0.0;
    let order = build_e2_update_order(schedule, indices.len(), sweep, rng);
    let u01_by_agent: Vec<f32> = (0..indices.len()).map(|_| rng.random::<f32>()).collect();
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut accepted_worse_count = 0usize;

    for &agent_i in &order {
        let update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle | E2UpdateSchedule::SequentialRotate => true,
        };
        if update_allowed {
            attempt_count += 1;
        }
        let agent_idx = indices[agent_i];
        let current_erb_vals: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
        let current_erb = erb_scan[agent_idx];
        let mut current_crowding = 0.0f32;
        if !skip_penalty {
            for (j, &other_erb) in current_erb_vals.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                let d_erb = current_erb - other_erb;

                current_crowding += crowding_runtime_delta_erb(kernel_params, d_erb);
            }
        }
        let c_score_current = prescored_c_score_scan[agent_idx];
        let current_score = score_sign * c_score_current - crowding_weight * current_crowding;

        let backtrack_target = if block_backtrack {
            backtrack_targets.and_then(|prev| prev.get(agent_i).copied())
        } else {
            None
        };

        let (chosen_idx, chosen_score_val, chosen_crowding_val, accepted_worse) = if update_allowed
        {
            let start = (agent_idx as isize - k as isize).max(min_idx as isize) as usize;
            let end = (agent_idx as isize + k as isize).min(max_idx as isize) as usize;
            let mut best_idx = agent_idx;
            let mut best_score = f32::NEG_INFINITY;
            let mut best_crowding = 0.0f32;
            let mut found_candidate = false;
            for cand in start..=end {
                if cand == agent_idx {
                    continue;
                }
                let cand_erb = erb_scan[cand];
                let mut crowding = 0.0f32;
                if !skip_penalty {
                    for (j, &other_erb) in current_erb_vals.iter().enumerate() {
                        if j == agent_i {
                            continue;
                        }
                        let d_erb = cand_erb - other_erb;

                        crowding += crowding_runtime_delta_erb(kernel_params, d_erb);
                    }
                }
                let c_score = prescored_c_score_scan[cand];
                let score = score_sign * c_score - crowding_weight * crowding;
                if let Some(prev_idx) = backtrack_target
                    && cand == prev_idx
                    && (score - current_score) <= E2_BACKTRACK_ALLOW_EPS
                {
                    continue;
                }
                if score > best_score {
                    best_score = score;
                    best_idx = cand;
                    best_crowding = crowding;
                    found_candidate = true;
                }
            }
            if found_candidate {
                let delta = best_score - current_score;
                if delta > E2_SCORE_IMPROVE_EPS {
                    (best_idx, best_score, best_crowding, false)
                } else if delta < 0.0 {
                    let (accept, acc_worse) =
                        metropolis_accept(delta, temperature, u01_by_agent[agent_i]);
                    if accept {
                        (best_idx, best_score, best_crowding, acc_worse)
                    } else {
                        (agent_idx, current_score, current_crowding, false)
                    }
                } else {
                    (agent_idx, current_score, current_crowding, false)
                }
            } else {
                (agent_idx, current_score, current_crowding, false)
            }
        } else {
            (agent_idx, current_score, current_crowding, false)
        };

        indices[agent_i] = chosen_idx;
        let moved = chosen_idx != agent_idx;
        let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[agent_idx]);
        let abs_delta = delta_semitones.abs();
        let abs_delta_moved = if moved { abs_delta } else { 0.0 };
        let c_score_chosen = prescored_c_score_scan[chosen_idx];

        if let Some(trace) = trajectory_semitones.as_deref_mut() {
            record_e2_trajectory_snapshot(trace, indices, log2_ratio_scan);
        }

        if moved {
            moved_count += 1;
        }
        if accepted_worse {
            accepted_worse_count += 1;
        }
        abs_delta_sum += abs_delta;
        abs_delta_moved_sum += abs_delta_moved;
        if c_score_current.is_finite() {
            c_score_current_sum += c_score_current;
            c_score_current_count += 1;
        }
        if c_score_chosen.is_finite() {
            c_score_chosen_sum += c_score_chosen;
            c_score_chosen_count += 1;
        }
        if chosen_score_val.is_finite() {
            score_sum += chosen_score_val;
            score_count += 1;
        }
        if chosen_crowding_val.is_finite() {
            crowding_sum += chosen_crowding_val;
            crowding_count += 1;
        }
    }

    let n = indices.len() as f32;
    let mean_c_score_current_loo = if c_score_current_count > 0 {
        c_score_current_sum / c_score_current_count as f32
    } else {
        0.0
    };
    let mean_c_score_chosen_loo = if c_score_chosen_count > 0 {
        c_score_chosen_sum / c_score_chosen_count as f32
    } else {
        0.0
    };
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_crowding = if crowding_count > 0 {
        crowding_sum / crowding_count as f32
    } else {
        0.0
    };
    UpdateStats {
        mean_c_score_current_loo,
        mean_c_score_chosen_loo,
        mean_score,
        mean_crowding,
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: accepted_worse_count as f32 / n,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones: abs_delta_sum / n,
        mean_abs_delta_semitones_moved: if moved_count > 0 {
            abs_delta_moved_sum / moved_count as f32
        } else {
            0.0
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_nohill(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    _prev_indices: &[usize],
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    k: i32,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    sweep: usize,
    mut trajectory_semitones: Option<&mut [Vec<f32>]>,
    rng: &mut StdRng,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let order = build_e2_update_order(schedule, indices.len(), sweep, rng);
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let mut env_current = env_total.to_vec();
    let mut density_current = density_total.to_vec();
    let mut env_loo = vec![0.0f32; env_total.len()];
    let mut density_loo = vec![0.0f32; density_total.len()];
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    for &agent_i in &order {
        let update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle | E2UpdateSchedule::SequentialRotate => true,
        };
        let current_idx = indices[agent_i];
        env_loo.copy_from_slice(&env_current);
        density_loo.copy_from_slice(&density_current);
        env_loo[current_idx] = (env_loo[current_idx] - 1.0).max(0.0);
        let denom = du_scan[current_idx].max(1e-12);
        density_loo[current_idx] = (density_loo[current_idx] - 1.0 / denom).max(0.0);
        let (c_score_scan, _, _, _) =
            compute_c_score_level_scans(space, workspace, &env_loo, &density_loo, du_scan);
        let current_erb_vals: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
        let current_erb = erb_scan[current_idx];
        let mut current_crowding = 0.0f32;
        if crowding_weight > 0.0 {
            for (j, &other_erb) in current_erb_vals.iter().enumerate() {
                if j == agent_i {
                    continue;
                }
                current_crowding +=
                    crowding_runtime_delta_erb(kernel_params, current_erb - other_erb);
            }
        }
        let c_score_current = c_score_scan[current_idx];
        let current_score = score_sign * c_score_current - crowding_weight * current_crowding;
        let (next_idx, chosen_score, chosen_crowding) = if update_allowed {
            attempt_count += 1;
            let step = rng.random_range(-k..=k);
            let next_idx =
                (current_idx as i32 + step).clamp(min_idx as i32, max_idx as i32) as usize;
            let next_erb = erb_scan[next_idx];
            let mut next_crowding = 0.0f32;
            if crowding_weight > 0.0 {
                for (j, &other_erb) in current_erb_vals.iter().enumerate() {
                    if j == agent_i {
                        continue;
                    }
                    next_crowding +=
                        crowding_runtime_delta_erb(kernel_params, next_erb - other_erb);
                }
            }
            let next_score = score_sign * c_score_scan[next_idx] - crowding_weight * next_crowding;
            (next_idx, next_score, next_crowding)
        } else {
            (current_idx, current_score, current_crowding)
        };
        if next_idx != current_idx {
            e2_scene_apply_move(
                &mut env_current,
                &mut density_current,
                du_scan,
                current_idx,
                next_idx,
            );
            indices[agent_i] = next_idx;
        }
        if next_idx != current_idx {
            moved_count += 1;
        }
        if let Some(trace) = trajectory_semitones.as_deref_mut() {
            record_e2_trajectory_snapshot(trace, indices, log2_ratio_scan);
        }
        if c_score_current.is_finite() {
            c_score_current_sum += c_score_current;
            c_score_current_count += 1;
        }
        let c_score_chosen = c_score_scan[next_idx];
        if c_score_chosen.is_finite() {
            c_score_chosen_sum += c_score_chosen;
            c_score_chosen_count += 1;
        }
        if chosen_score.is_finite() {
            score_sum += chosen_score;
            score_count += 1;
        }
        if chosen_crowding.is_finite() {
            crowding_sum += chosen_crowding;
            crowding_count += 1;
        }
        let delta_semitones = 12.0 * (log2_ratio_scan[next_idx] - log2_ratio_scan[current_idx]);
        let abs_delta = delta_semitones.abs();
        if abs_delta.is_finite() {
            abs_delta_sum += abs_delta;
            if next_idx != current_idx {
                abs_delta_moved_sum += abs_delta;
            }
        }
    }
    let n = indices.len() as f32;
    UpdateStats {
        mean_c_score_current_loo: if c_score_current_count > 0 {
            c_score_current_sum / c_score_current_count as f32
        } else {
            0.0
        },
        mean_c_score_chosen_loo: if c_score_chosen_count > 0 {
            c_score_chosen_sum / c_score_chosen_count as f32
        } else {
            0.0
        },
        mean_score: if score_count > 0 {
            score_sum / score_count as f32
        } else {
            0.0
        },
        mean_crowding: if crowding_count > 0 {
            crowding_sum / crowding_count as f32
        } else {
            0.0
        },
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: 0.0,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones: abs_delta_sum / n,
        mean_abs_delta_semitones_moved: if moved_count > 0 {
            abs_delta_moved_sum / moved_count as f32
        } else {
            0.0
        },
    }
}

#[allow(dead_code)]
fn score_stats_at_indices(
    indices: &[usize],
    c_score_scan: &[f32],
    erb_scan: &[f32],
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    let erb_vals: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    for (i, &idx) in indices.iter().enumerate() {
        let cand_erb = erb_scan[idx];
        let mut crowding = 0.0f32;
        for (j, &other_erb) in erb_vals.iter().enumerate() {
            if i == j {
                continue;
            }
            let d_erb = cand_erb - other_erb;
            crowding += crowding_runtime_delta_erb(kernel_params, d_erb);
        }
        let score = score_sign * c_score_scan[idx] - crowding_weight * crowding;
        if score.is_finite() {
            score_sum += score;
            score_count += 1;
        }
        if crowding.is_finite() {
            crowding_sum += crowding;
            crowding_count += 1;
        }
    }
    let mean_score = if score_count > 0 {
        score_sum / score_count as f32
    } else {
        0.0
    };
    let mean_crowding = if crowding_count > 0 {
        crowding_sum / crowding_count as f32
    } else {
        0.0
    };
    UpdateStats {
        mean_c_score_current_loo: f32::NAN,
        mean_c_score_chosen_loo: f32::NAN,
        mean_score,
        mean_crowding,
        moved_frac: 0.0,
        accepted_worse_frac: 0.0,
        attempted_update_frac: 0.0,
        moved_given_attempt_frac: 0.0,
        mean_abs_delta_semitones: 0.0,
        mean_abs_delta_semitones_moved: 0.0,
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
fn score_stats_at_indices_loo_reused(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    erb_scan: &[f32],
    prev_indices: &[usize],
    current_eval_indices: &[usize],
    chosen_eval_indices: &[usize],
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    env_loo: &mut Vec<f32>,
    density_loo: &mut Vec<f32>,
) -> UpdateStats {
    if prev_indices.is_empty() || current_eval_indices.is_empty() || chosen_eval_indices.is_empty()
    {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }
    debug_assert_eq!(
        prev_indices.len(),
        current_eval_indices.len(),
        "prev/current eval length mismatch"
    );
    debug_assert_eq!(
        prev_indices.len(),
        chosen_eval_indices.len(),
        "prev/chosen eval length mismatch"
    );

    let prev_erb: Vec<f32> = prev_indices.iter().map(|&idx| erb_scan[idx]).collect();
    let mut current_sum = 0.0f32;
    let mut current_count = 0u32;
    let mut chosen_sum = 0.0f32;
    let mut chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;

    for i in 0..prev_indices.len() {
        let prev_idx = prev_indices[i];
        let current_eval_idx = current_eval_indices[i];
        let chosen_eval_idx = chosen_eval_indices[i];
        let loo_c_score_scan = e2_loo_c_score_scan_for_agent_reused(
            space,
            workspace,
            env_total,
            density_total,
            du_scan,
            prev_idx,
            env_loo,
            density_loo,
        );
        let current_value = loo_c_score_scan[current_eval_idx];
        let chosen_value = loo_c_score_scan[chosen_eval_idx];
        if current_value.is_finite() {
            current_sum += current_value;
            current_count += 1;
        }
        if chosen_value.is_finite() {
            chosen_sum += chosen_value;
            chosen_count += 1;
        }

        let chosen_erb = erb_scan[chosen_eval_idx];
        let mut crowding = 0.0f32;
        for (j, &other_erb) in prev_erb.iter().enumerate() {
            if i == j {
                continue;
            }
            crowding += crowding_runtime_delta_erb(kernel_params, chosen_erb - other_erb);
        }
        let score = score_sign * chosen_value - crowding_weight * crowding;
        if score.is_finite() {
            score_sum += score;
            score_count += 1;
        }
        if crowding.is_finite() {
            crowding_sum += crowding;
            crowding_count += 1;
        }
    }

    UpdateStats {
        mean_c_score_current_loo: if current_count > 0 {
            current_sum / current_count as f32
        } else {
            0.0
        },
        mean_c_score_chosen_loo: if chosen_count > 0 {
            chosen_sum / chosen_count as f32
        } else {
            0.0
        },
        mean_score: if score_count > 0 {
            score_sum / score_count as f32
        } else {
            0.0
        },
        mean_crowding: if crowding_count > 0 {
            crowding_sum / crowding_count as f32
        } else {
            0.0
        },
        moved_frac: 0.0,
        accepted_worse_frac: 0.0,
        attempted_update_frac: 0.0,
        moved_given_attempt_frac: 0.0,
        mean_abs_delta_semitones: 0.0,
        mean_abs_delta_semitones_moved: 0.0,
    }
}

fn mean_at_indices(values: &[f32], indices: &[usize]) -> f32 {
    if indices.is_empty() {
        return 0.0;
    }
    let sum: f32 = indices.iter().map(|&idx| values[idx]).sum();
    sum / indices.len() as f32
}

#[allow(clippy::too_many_arguments, dead_code)]
fn mean_c_score_loo_pair_at_indices_with_prev_reused(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    prev_indices: &[usize],
    current_eval_indices: &[usize],
    chosen_eval_indices: &[usize],
    env_loo: &mut Vec<f32>,
    density_loo: &mut Vec<f32>,
) -> (f32, f32) {
    if prev_indices.is_empty() || current_eval_indices.is_empty() || chosen_eval_indices.is_empty()
    {
        return (0.0, 0.0);
    }
    debug_assert_eq!(
        prev_indices.len(),
        current_eval_indices.len(),
        "prev/current eval length mismatch"
    );
    debug_assert_eq!(
        prev_indices.len(),
        chosen_eval_indices.len(),
        "prev/chosen eval length mismatch"
    );
    space.assert_scan_len_named(env_total, "env_total");
    space.assert_scan_len_named(density_total, "density_total");
    space.assert_scan_len_named(du_scan, "du_scan");

    env_loo.resize(env_total.len(), 0.0);
    density_loo.resize(density_total.len(), 0.0);
    let mut current_sum = 0.0f32;
    let mut current_count = 0u32;
    let mut chosen_sum = 0.0f32;
    let mut chosen_count = 0u32;
    for i in 0..prev_indices.len() {
        let prev_idx = prev_indices[i];
        let current_eval_idx = current_eval_indices[i];
        let chosen_eval_idx = chosen_eval_indices[i];
        env_loo.copy_from_slice(env_total);
        density_loo.copy_from_slice(density_total);
        env_loo[prev_idx] = (env_loo[prev_idx] - 1.0).max(0.0);
        let denom = du_scan[prev_idx].max(1e-12);
        density_loo[prev_idx] = (density_loo[prev_idx] - 1.0 / denom).max(0.0);
        let (c_score_scan, _, _, _) =
            compute_c_score_level_scans(space, workspace, env_loo, density_loo, du_scan);
        let current_value = c_score_scan[current_eval_idx];
        if current_value.is_finite() {
            current_sum += current_value;
            current_count += 1;
        }
        let chosen_value = c_score_scan[chosen_eval_idx];
        if chosen_value.is_finite() {
            chosen_sum += chosen_value;
            chosen_count += 1;
        }
    }
    let current_mean = if current_count == 0 {
        0.0
    } else {
        current_sum / current_count as f32
    };
    let chosen_mean = if chosen_count == 0 {
        0.0
    } else {
        chosen_sum / chosen_count as f32
    };
    (current_mean, chosen_mean)
}

fn e2_loo_c_score_scan_for_agent_reused(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    prev_idx: usize,
    env_loo: &mut Vec<f32>,
    density_loo: &mut Vec<f32>,
) -> Vec<f32> {
    env_loo.resize(env_total.len(), 0.0);
    density_loo.resize(density_total.len(), 0.0);
    env_loo.copy_from_slice(env_total);
    density_loo.copy_from_slice(density_total);
    env_loo[prev_idx] = (env_loo[prev_idx] - 1.0).max(0.0);
    let denom = du_scan[prev_idx].max(1e-12);
    density_loo[prev_idx] = (density_loo[prev_idx] - 1.0 / denom).max(0.0);
    let (c_score_scan, _, _, _) =
        compute_c_score_level_scans(space, workspace, env_loo, density_loo, du_scan);
    c_score_scan
}

fn mean_std_series(series_list: Vec<&Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
    if series_list.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = series_list[0].len();
    let mut sum = vec![0.0f32; len];
    let mut sum_sq = vec![0.0f32; len];
    for series in &series_list {
        debug_assert_eq!(series.len(), len, "series length mismatch");
        for (i, &val) in series.iter().enumerate() {
            sum[i] += val;
            sum_sq[i] += val * val;
        }
    }
    let n = series_list.len() as f32;
    let mut mean = vec![0.0f32; len];
    let mut std = vec![0.0f32; len];
    for i in 0..len {
        mean[i] = sum[i] / n;
        let var = (sum_sq[i] / n) - mean[i] * mean[i];
        std[i] = var.max(0.0).sqrt();
    }
    (mean, std)
}

fn mean_ci95_series(series_list: Vec<&Vec<f32>>) -> (Vec<f32>, Vec<f32>) {
    let n = series_list.len();
    let (mean, std) = mean_std_series(series_list);
    let ci95 = std_series_to_ci95(&std, n);
    (mean, ci95)
}

fn e2_trajectory_mean_c_score_loo_stats(space: &Log2Space, runs: &[E2Run]) -> (Vec<f32>, Vec<f32>) {
    if runs.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let workspace = build_consonance_workspace(space);
    let (_erb_scan, du_scan) = erb_grid(space);
    let per_run: Vec<Vec<f32>> = parallel_map_ordered(runs, None, |run| {
        let fixed_drone_idx = e2_fixed_drone(space, run.fixed_drone_hz).idx;
        compute_e2_trajectory_mean_c_score_loo(
            space,
            &workspace,
            &du_scan,
            fixed_drone_idx,
            run.fixed_drone_hz,
            &run.trajectory_semitones,
        )
    });
    let refs: Vec<&Vec<f32>> = per_run.iter().collect();
    mean_ci95_series(refs)
}

fn series_pairs(series: &[f32]) -> Vec<(f32, f32)> {
    series
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f32, y))
        .collect()
}

fn series_csv(header: &str, series: &[f32]) -> String {
    let mut out = String::from(header);
    out.push('\n');
    for (i, value) in series.iter().enumerate() {
        out.push_str(&format!("{i},{value:.6}\n"));
    }
    out
}

fn sweep_csv(header: &str, mean: &[f32], std: &[f32], n: usize) -> String {
    let mut out = String::from(header);
    out.push('\n');
    let len = mean.len().min(std.len());
    for i in 0..len {
        out.push_str(&format!("{i},{:.6},{:.6},{}\n", mean[i], std[i], n));
    }
    out
}

fn e2_controls_csv_c_level(
    baseline: &E2SweepStats,
    nohill: &E2SweepStats,
    norep: &E2SweepStats,
) -> String {
    let mut out = String::from(
        "step,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = baseline
        .mean_c_level
        .len()
        .min(nohill.mean_c_level.len())
        .min(norep.mean_c_level.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            baseline.mean_c_level[i],
            baseline.std_c_level[i],
            nohill.mean_c_level[i],
            nohill.std_c_level[i],
            norep.mean_c_level[i],
            norep.std_c_level[i]
        ));
    }
    out
}

fn e2_controls_csv_c(
    baseline: &E2SweepStats,
    nohill: &E2SweepStats,
    norep: &E2SweepStats,
) -> String {
    let mut out = String::from(
        "step,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std\n",
    );
    let len = baseline
        .mean_c
        .len()
        .min(nohill.mean_c.len())
        .min(norep.mean_c.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            baseline.mean_c[i],
            baseline.std_c[i],
            nohill.mean_c[i],
            nohill.std_c[i],
            norep.mean_c[i],
            norep.std_c[i]
        ));
    }
    out
}

fn trajectories_csv(run: &E2Run) -> String {
    let mut out = String::from("step,agent_id,semitones,c_level\n");
    for (agent_id, semis) in run.trajectory_semitones.iter().enumerate() {
        let c_level = &run.trajectory_c_level[agent_id];
        let len = semis.len().min(c_level.len());
        for step in 0..len {
            out.push_str(&format!(
                "{step},{agent_id},{:.6},{:.6}\n",
                semis[step], c_level[step]
            ));
        }
    }
    out
}

fn anchor_shift_csv(run: &E2Run) -> String {
    let s = &run.anchor_shift;
    let mut out =
        String::from("step,anchor_hz_before,anchor_hz_after,count_min,count_max,respawned\n");
    out.push_str(&format!(
        "{},{:.3},{:.3},{},{},{}\n",
        s.step, s.anchor_hz_before, s.anchor_hz_after, s.count_min, s.count_max, s.respawned
    ));
    out
}

fn e2_summary_csv(runs: &[E2Run]) -> String {
    let mut out = String::from(
        "seed,init_mode,steps,burn_in,mean_c_step0,mean_c_step_end,delta_c,mean_c_level_step0,mean_c_level_step_end,delta_c_level,mean_c_score_loo_step0,mean_c_score_loo_step_end,delta_c_score_loo\n",
    );
    for run in runs {
        let start = run.mean_c_series.first().copied().unwrap_or(0.0);
        let end = run.mean_c_series.last().copied().unwrap_or(start);
        let delta = end - start;
        let start_level = run.mean_c_level_series.first().copied().unwrap_or(0.0);
        let end_level = run
            .mean_c_level_series
            .last()
            .copied()
            .unwrap_or(start_level);
        let delta_level = end_level - start_level;
        let start_loo = run.mean_c_score_loo_series.first().copied().unwrap_or(0.0);
        let end_loo = run
            .mean_c_score_loo_series
            .last()
            .copied()
            .unwrap_or(start_loo);
        let delta_loo = end_loo - start_loo;
        out.push_str(&format!(
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            run.seed,
            E2_INIT_MODE.label(),
            E2_SWEEPS,
            E2_BURN_IN,
            start,
            end,
            delta,
            start_level,
            end_level,
            delta_level,
            start_loo,
            end_loo,
            delta_loo
        ));
    }
    out
}

#[derive(Clone, Copy, Debug, Default)]
struct FlutterMetrics {
    pingpong_rate_moves: f32,
    reversal_rate_moves: f32,
    move_rate_stepwise: f32,
    mean_abs_delta_moved: f32,
    step_count: usize,
    moved_step_count: usize,
    move_count: usize,
    pingpong_count_moves: usize,
    reversal_count_moves: usize,
}

fn flutter_metrics_for_trajectories(
    trajectories: &[Vec<f32>],
    start_step: usize,
    end_step: usize,
) -> FlutterMetrics {
    if trajectories.is_empty() || start_step >= end_step {
        return FlutterMetrics::default();
    }
    let mut step_count = 0usize;
    let mut moved_step_count = 0usize;
    let mut move_count = 0usize;
    let mut pingpong_count_moves = 0usize;
    let mut reversal_count_moves = 0usize;
    let mut pingpong_den_moves = 0usize;
    let mut reversal_den_moves = 0usize;
    let mut abs_delta_sum = 0.0f32;

    for traj in trajectories {
        if traj.len() <= start_step + 1 {
            continue;
        }
        let end = end_step.min(traj.len().saturating_sub(1));
        let trimmed = &traj[(start_step + 1)..=end];
        for pair in trimmed.windows(2) {
            let delta = pair[1] - pair[0];
            let moved = delta.abs() > E2_SEMITONE_EPS;
            step_count += 1;
            if moved {
                moved_step_count += 1;
            }
        }

        let mut compressed: Vec<f32> = Vec::new();
        for &v in &traj[start_step..=end] {
            if compressed
                .last()
                .is_some_and(|last| (v - last).abs() <= E2_SEMITONE_EPS)
            {
                continue;
            }
            compressed.push(v);
        }
        let comp_len = compressed.len();
        if comp_len >= 2 {
            move_count += comp_len - 1;
            for i in 1..comp_len {
                abs_delta_sum += (compressed[i] - compressed[i - 1]).abs();
            }
        }
        if comp_len >= 3 {
            pingpong_den_moves += comp_len - 2;
            reversal_den_moves += comp_len - 2;
            for i in 2..comp_len {
                if (compressed[i] - compressed[i - 2]).abs() <= E2_SEMITONE_EPS {
                    pingpong_count_moves += 1;
                }
                let delta = compressed[i] - compressed[i - 1];
                let prev_delta = compressed[i - 1] - compressed[i - 2];
                if delta * prev_delta < 0.0 {
                    reversal_count_moves += 1;
                }
            }
        }
    }

    let move_rate_stepwise = if step_count > 0 {
        moved_step_count as f32 / step_count as f32
    } else {
        0.0
    };
    let pingpong_rate_moves = if pingpong_den_moves > 0 {
        pingpong_count_moves as f32 / pingpong_den_moves as f32
    } else {
        0.0
    };
    let reversal_rate_moves = if reversal_den_moves > 0 {
        reversal_count_moves as f32 / reversal_den_moves as f32
    } else {
        0.0
    };
    let mean_abs_delta_moved = if move_count > 0 {
        abs_delta_sum / move_count as f32
    } else {
        0.0
    };

    FlutterMetrics {
        pingpong_rate_moves,
        reversal_rate_moves,
        move_rate_stepwise,
        mean_abs_delta_moved,
        step_count,
        moved_step_count,
        move_count,
        pingpong_count_moves,
        reversal_count_moves,
    }
}

fn e2_flutter_segments(
    phase_mode: E2PhaseMode,
    n_agents: usize,
) -> Vec<(&'static str, usize, usize)> {
    let burn_in = e2_trajectory_burn_in_step(n_agents);
    let total_steps = E2_SWEEPS.saturating_mul(e2_microsteps_per_sweep(n_agents));
    if let Some(switch_step) = e2_trajectory_phase_switch_step(phase_mode, n_agents) {
        let pre_end = switch_step.saturating_sub(1);
        let post_start = switch_step;
        let mut segments = Vec::new();
        if pre_end >= burn_in {
            segments.push(("pre", burn_in, pre_end));
        }
        if post_start <= total_steps {
            segments.push(("post", post_start, total_steps));
        }
        segments
    } else {
        vec![("all", burn_in, total_steps)]
    }
}

#[derive(Clone)]
struct FlutterRow {
    condition: &'static str,
    seed: u64,
    segment: &'static str,
    metrics: FlutterMetrics,
}

fn flutter_by_seed_csv(rows: &[FlutterRow]) -> String {
    let mut out = String::from(
        "cond,seed,segment,pingpong_rate_moves,reversal_rate_moves,move_rate_stepwise,mean_abs_delta_moved,step_count,moved_step_count,move_count,pingpong_count_moves,reversal_count_moves\n",
    );
    for row in rows {
        let m = row.metrics;
        out.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{},{},{}\n",
            row.condition,
            row.seed,
            row.segment,
            m.pingpong_rate_moves,
            m.reversal_rate_moves,
            m.move_rate_stepwise,
            m.mean_abs_delta_moved,
            m.step_count,
            m.moved_step_count,
            m.move_count,
            m.pingpong_count_moves,
            m.reversal_count_moves
        ));
    }
    out
}

fn flutter_summary_csv(rows: &[FlutterRow], segments: &[(&'static str, usize, usize)]) -> String {
    let mut out = String::from(
        "cond,segment,mean_pingpong_rate_moves,std_pingpong_rate_moves,mean_reversal_rate_moves,std_reversal_rate_moves,mean_move_rate_stepwise,std_move_rate_stepwise,mean_abs_delta_moved,std_abs_delta_moved,n\n",
    );
    for &cond in ["baseline", "nohill", "nocrowd"].iter() {
        for (segment, _, _) in segments {
            let mut pingpong_vals = Vec::new();
            let mut reversal_vals = Vec::new();
            let mut move_vals = Vec::new();
            let mut abs_delta_vals = Vec::new();
            for row in rows
                .iter()
                .filter(|r| r.condition == cond && r.segment == *segment)
            {
                pingpong_vals.push(row.metrics.pingpong_rate_moves);
                reversal_vals.push(row.metrics.reversal_rate_moves);
                move_vals.push(row.metrics.move_rate_stepwise);
                abs_delta_vals.push(row.metrics.mean_abs_delta_moved);
            }
            let n = pingpong_vals.len();
            let (mean_ping, std_ping) = mean_std_scalar(&pingpong_vals);
            let (mean_rev, std_rev) = mean_std_scalar(&reversal_vals);
            let (mean_move, std_move) = mean_std_scalar(&move_vals);
            let (mean_abs, std_abs) = mean_std_scalar(&abs_delta_vals);
            out.push_str(&format!(
                "{cond},{segment},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                mean_ping, std_ping, mean_rev, std_rev, mean_move, std_move, mean_abs, std_abs, n
            ));
        }
    }
    out
}

fn final_agents_csv(run: &E2Run) -> String {
    let mut out = String::from("agent_id,freq_hz,log2_ratio,semitones\n");
    let len = run
        .final_freqs_hz
        .len()
        .min(run.final_log2_ratios.len())
        .min(run.final_semitones.len());
    for i in 0..len {
        out.push_str(&format!(
            "{},{:.4},{:.6},{:.6}\n",
            i, run.final_freqs_hz[i], run.final_log2_ratios[i], run.final_semitones[i]
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn e2_meta_text(
    n_agents: usize,
    fixed_drone_hz: f32,
    k_bins: i32,
    density_mass_mean: f32,
    density_mass_min: f32,
    density_mass_max: f32,
    r_ref_peak: f32,
    roughness_k: f32,
    roughness_ref_eps: f32,
    r_state01_min: f32,
    r_state01_mean: f32,
    r_state01_max: f32,
    phase_mode: E2PhaseMode,
) -> String {
    let mut out = String::new();
    out.push_str(&format!("SPACE_BINS_PER_OCT={}\n", SPACE_BINS_PER_OCT));
    out.push_str(&format!("E2_SWEEPS={}\n", E2_SWEEPS));
    out.push_str(&format!("E2_BURN_IN={}\n", E2_BURN_IN));
    out.push_str(&format!("E2_ANCHOR_SHIFT_STEP={}\n", E2_ANCHOR_SHIFT_STEP));
    out.push_str(&format!(
        "E2_ANCHOR_SHIFT_ENABLED={}\n",
        e2_anchor_shift_enabled()
    ));
    out.push_str(&format!(
        "E2_ANCHOR_SHIFT_RATIO={:.3}\n",
        E2_ANCHOR_SHIFT_RATIO
    ));
    out.push_str(&format!("E2_STEP_SEMITONES={:.3}\n", E2_STEP_SEMITONES));
    out.push_str("E2_REFERENCE_MODE=fixed_reference_condition\n");
    out.push_str(&format!(
        "E2_FIXED_DRONE_DESC=one immobile drone voice fixed at {:.3} Hz throughout the run\n",
        fixed_drone_hz
    ));
    out.push_str(&format!("E2_FIXED_DRONE_HZ={:.3}\n", fixed_drone_hz));
    out.push_str(&format!("E2_ADAPTIVE_VOICES={}\n", n_agents));
    out.push_str(&format!("E2_TOTAL_FIELD_VOICES={}\n", n_agents + 1));
    out.push_str(&format!("E2_N_AGENTS={}\n", n_agents));
    out.push_str(&format!("E2_K_BINS={}\n", k_bins));
    out.push_str(&format!("E2_CROWDING_WEIGHT={:.3}\n", E2_CROWDING_WEIGHT));
    out.push_str(&format!("E2_ACCEPT_ENABLED={}\n", E2_ACCEPT_ENABLED));
    out.push_str(&format!("E2_ACCEPT_T0={:.3}\n", E2_ACCEPT_T0));
    out.push_str(&format!("E2_ACCEPT_TAU_STEPS={:.3}\n", E2_ACCEPT_TAU_STEPS));
    out.push_str(&format!(
        "E2_ACCEPT_RESET_ON_PHASE={}\n",
        E2_ACCEPT_RESET_ON_PHASE
    ));
    out.push_str(&format!(
        "E2_SCORE_IMPROVE_EPS={:.6}\n",
        E2_SCORE_IMPROVE_EPS
    ));
    let update_schedule = match E2_UPDATE_SCHEDULE {
        E2UpdateSchedule::Checkerboard => "checkerboard",
        E2UpdateSchedule::Lazy => "lazy",
        E2UpdateSchedule::RandomSingle => "random_single",
        E2UpdateSchedule::SequentialRotate => "sequential_rotate",
    };
    out.push_str(&format!("E2_UPDATE_SCHEDULE={}\n", update_schedule));
    out.push_str(&format!("E2_LAZY_MOVE_PROB={:.3}\n", E2_LAZY_MOVE_PROB));
    out.push_str(&format!(
        "E2_ANTI_BACKTRACK_ENABLED={}\n",
        E2_ANTI_BACKTRACK_ENABLED
    ));
    out.push_str(&format!(
        "E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY={}\n",
        E2_ANTI_BACKTRACK_PRE_SWITCH_ONLY
    ));
    out.push_str(&format!(
        "E2_BACKTRACK_ALLOW_EPS={:.6}\n",
        E2_BACKTRACK_ALLOW_EPS
    ));
    out.push_str(&format!("E2_SEMITONE_EPS={:.6}\n", E2_SEMITONE_EPS));
    out.push_str(&format!("C_LEVEL_BETA={:.3}\n", E2_C_LEVEL_BETA));
    out.push_str(&format!("C_LEVEL_THETA={:.3}\n", E2_C_LEVEL_THETA));
    out.push_str(&format!("ROUGHNESS_REF_EPS={:.3e}\n", roughness_ref_eps));
    out.push_str(&format!("ROUGHNESS_K={:.3}\n", roughness_k));
    out.push_str(&format!("R_REF_PEAK={:.6}\n", r_ref_peak));
    out.push_str(&format!("R_STATE01_MIN={:.6}\n", r_state01_min));
    out.push_str(&format!("R_STATE01_MEAN={:.6}\n", r_state01_mean));
    out.push_str(&format!("R_STATE01_MAX={:.6}\n", r_state01_max));
    out.push_str(&format!("E2_DENSITY_MASS_MEAN={:.6}\n", density_mass_mean));
    out.push_str(&format!("E2_DENSITY_MASS_MIN={:.6}\n", density_mass_min));
    out.push_str(&format!("E2_DENSITY_MASS_MAX={:.6}\n", density_mass_max));
    out.push_str(&format!("E2_INIT_MODE={}\n", E2_INIT_MODE.label()));
    out.push_str(&format!(
        "E2_INIT_CONSONANT_EXCLUSION_ST={:.3}\n",
        E2_INIT_CONSONANT_EXCLUSION_ST
    ));
    out.push_str(&format!("E2_INIT_MAX_TRIES={}\n", E2_INIT_MAX_TRIES));
    out.push_str(&format!("E2_ANCHOR_BIN_ST={:.3}\n", E2_ANCHOR_BIN_ST));
    out.push_str(&format!("E2_PAIRWISE_BIN_ST={:.3}\n", E2_PAIRWISE_BIN_ST));
    out.push_str(&format!("E2_DIVERSITY_BIN_ST={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str(&format!("E2_SEEDS={:?}\n", E2_SEEDS));
    out.push_str(&format!("E2_PHASE_MODE={}\n", phase_mode.label()));
    if let Some(step) = phase_mode.switch_step() {
        out.push_str(&format!("E2_PHASE_SWITCH_STEP={step}\n"));
    }
    out.push_str("E2_DISTRIBUTION_MODE=window_aggregated\n");
    out.push_str(&format!(
        "E2_POST_WINDOW_START_STEP={}\n",
        e2_post_window_start_step()
    ));
    out.push_str(&format!(
        "E2_POST_WINDOW_END_STEP={}\n",
        e2_post_window_end_step()
    ));
    let pairwise_n_pairs = if n_agents < 2 {
        0usize
    } else {
        n_agents * (n_agents - 1) / 2
    };
    out.push_str("E2_PAIRWISE_INTERVAL_SOURCE=final_snapshot\n");
    out.push_str(&format!(
        "E2_PAIRWISE_INTERVAL_N_PAIRS={}\n",
        pairwise_n_pairs
    ));
    out.push_str(
        "E2_ADAPTIVE_STATS_SCOPE=adaptive voices only; fixed drone excluded from move/update, LOO means, pairwise intervals, diversity, and scene summaries\n",
    );
    out.push_str(
        "E2_MEAN_C_SCORE_CURRENT_LOO_DESC=mean C score at current positions using LOO env (env_total-1 at current bin, density_total-1/du)\n",
    );
    out.push_str(
        "E2_MEAN_C_SCORE_CHOSEN_LOO_DESC=mean C score at chosen positions using LOO env (removal at current bin)\n",
    );
    out.push_str("E2_MEAN_ABS_DELTA_SEMITONES_DESC=mean |Δ| over all agents\n");
    out.push_str("E2_MEAN_ABS_DELTA_SEMITONES_MOVED_DESC=mean |Δ| over moved agents\n");
    out.push_str("E2_ATTEMPTED_UPDATE_FRAC_DESC=attempted update / agents\n");
    out.push_str("E2_MOVED_GIVEN_ATTEMPT_FRAC_DESC=moved / attempted update\n");
    out.push_str(
        "E2_PINGPONG_RATE_DESC=move-compressed pingpong rate (count / max(0, compressed_len-2))\n",
    );
    out.push_str(
        "E2_REVERSAL_RATE_DESC=move-compressed reversal rate (count / max(0, move_count-1))\n",
    );
    out.push_str("E2_MOVE_RATE_STEPWISE_DESC=stepwise moved / steps\n");
    out
}

fn e2_marker_steps(phase_mode: E2PhaseMode) -> Vec<f32> {
    let mut steps = vec![E2_BURN_IN as f32];
    if e2_anchor_shift_enabled() {
        steps.push(E2_ANCHOR_SHIFT_STEP as f32);
    }
    if let Some(step) = phase_mode.switch_step() {
        steps.push(step as f32);
    }
    steps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    steps.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    steps
}

fn e3_metric_definition_text() -> String {
    let mut out = String::new();
    out.push_str("E3 metric definitions\n");
    out.push_str(
        "Plots report C_level01 (agent.last_consonance_level01()), while the continuous recharge term uses positive raw C_score from the reference landscape.\n",
    );
    out.push_str("C_firstK definition: mean over first K=20 ticks after birth (0..1).\n");
    out.push_str("Metabolism update (conceptual):\n");
    out.push_str(
        "  E <- E - basal_cost_per_sec * dt + continuous_recharge_per_sec * max(0, C_score) * dt\n",
    );
    out.push_str(
        "Only positive raw C_score contributes to recharge. NoRecharge sets continuous_recharge_per_sec=0, so the C-dependent recharge term is removed.\n",
    );
    out.push_str(
        "Representative seed is chosen by the median Pearson r of baseline C_firstK vs lifetime; pooled plots concatenate all seeds.\n",
    );
    out.push_str(
        "c_level01_birth=first tick value; c_level01_firstk=mean of first K ticks; avg_c_level01_tick=mean over life; c_level01_std_over_life=std over life; avg_c_level01_attack=mean over attack ticks (reported descriptively only; recharge is no longer attack-coupled).\n",
    );
    out
}

fn pick_representative_run_index(runs: &[E2Run]) -> usize {
    if runs.is_empty() {
        return 0;
    }
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c_level_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    scored[scored.len() / 2].0
}

fn representative_seed_text(runs: &[E2Run], rep_index: usize, phase_mode: E2PhaseMode) -> String {
    let mut scored: Vec<(usize, f32)> = runs
        .iter()
        .enumerate()
        .map(|(i, r)| (i, r.mean_c_level_series.last().copied().unwrap_or(0.0)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let metric_label = e2_post_label();
    let pre_label = e2_pre_label();
    let pre_step = e2_pre_step();
    let mut out = format!("metric={metric_label}_mean_c_level\n");
    out.push_str(&format!("phase_mode={}\n", phase_mode.label()));
    if let Some(step) = phase_mode.switch_step() {
        out.push_str(&format!("phase_switch_step={step}\n"));
    }
    out.push_str("rank,seed,metric\n");
    for (rank, (idx, metric)) in scored.iter().enumerate() {
        out.push_str(&format!("{rank},{},{}\n", runs[*idx].seed, metric));
    }
    let rep_metric = runs
        .get(rep_index)
        .and_then(|r| r.mean_c_level_series.last().copied())
        .unwrap_or(0.0);
    let rep_pre = runs
        .get(rep_index)
        .and_then(|r| r.mean_c_level_series.get(pre_step).copied())
        .unwrap_or(0.0);
    let rep_seed = runs.get(rep_index).map(|r| r.seed).unwrap_or(0);
    let rep_rank = scored
        .iter()
        .position(|(idx, _)| *idx == rep_index)
        .unwrap_or(0);
    out.push_str(&format!(
        "representative_seed={rep_seed}\nrepresentative_rank={rep_rank}\nrepresentative_metric={rep_metric}\nrepresentative_{pre_label}={rep_pre}\n"
    ));
    out
}

fn histogram_counts_fixed(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<(f32, f32)> {
    if bin_width <= 0.0 {
        return Vec::new();
    }
    let bins = ((max - min) / bin_width).ceil().max(1.0) as usize;
    let mut counts = vec![0.0f32; bins];
    for &value in values {
        if !value.is_finite() {
            continue;
        }
        if value < min || value > max {
            continue;
        }
        let mut idx = ((value - min) / bin_width).floor() as isize;
        if idx as usize >= bins {
            idx = (bins - 1) as isize;
        }
        if idx >= 0 {
            counts[idx as usize] += 1.0;
        }
    }
    (0..bins)
        .map(|i| (min + (i as f32 + 0.5) * bin_width, counts[i]))
        .collect()
}

fn histogram_probabilities_fixed(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<f32> {
    let counts = histogram_counts_fixed(values, min, max, bin_width);
    let total: f32 = counts.iter().map(|(_, c)| *c).sum();
    if total <= 0.0 {
        return vec![0.0; counts.len()];
    }
    counts.into_iter().map(|(_, c)| c / total).collect()
}

fn hist_structure_metrics_from_probs(probs: &[f32]) -> HistStructureMetrics {
    if probs.is_empty() {
        return HistStructureMetrics {
            entropy: 0.0,
            gini: 0.0,
            peakiness: 0.0,
            kl_uniform: 0.0,
        };
    }
    let sum: f32 = probs.iter().copied().sum();
    if sum <= 0.0 {
        return HistStructureMetrics {
            entropy: 0.0,
            gini: 0.0,
            peakiness: 0.0,
            kl_uniform: 0.0,
        };
    }
    let inv_sum = 1.0 / sum;
    let mut norm: Vec<f32> = probs.iter().map(|p| (p * inv_sum).max(0.0)).collect();

    let mut entropy = 0.0f32;
    let mut kl_uniform = 0.0f32;
    let n = norm.len() as f32;
    let uniform = 1.0 / n;
    for p in &norm {
        if *p > 0.0 {
            entropy -= *p * p.ln();
            kl_uniform += *p * (p / uniform).ln();
        }
    }

    norm.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mut weighted = 0.0f32;
    for (i, p) in norm.iter().enumerate() {
        weighted += (i as f32 + 1.0) * p;
    }
    let gini = ((2.0 * weighted) / (n * 1.0) - (n + 1.0) / n).clamp(0.0, 1.0);
    let peakiness = norm.iter().copied().fold(0.0f32, f32::max);

    HistStructureMetrics {
        entropy,
        gini,
        peakiness,
        kl_uniform,
    }
}

fn hist_structure_metrics_for_run(run: &E2Run) -> HistStructureMetrics {
    let samples = pairwise_interval_samples(&run.final_semitones);
    let probs = histogram_probabilities_fixed(&samples, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
    hist_structure_metrics_from_probs(&probs)
}

fn hist_structure_rows(condition: &'static str, runs: &[E2Run]) -> Vec<HistStructureRow> {
    runs.iter()
        .map(|run| HistStructureRow {
            condition,
            seed: run.seed,
            metrics: hist_structure_metrics_for_run(run),
        })
        .collect()
}

fn hist_structure_by_seed_csv(rows: &[HistStructureRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# source=pairwise_intervals bin_width={:.3}\n",
        E2_PAIRWISE_BIN_ST
    ));
    out.push_str("seed,cond,entropy,gini,peakiness,kl_uniform\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.seed,
            row.condition,
            row.metrics.entropy,
            row.metrics.gini,
            row.metrics.peakiness,
            row.metrics.kl_uniform
        ));
    }
    out
}

fn hist_structure_summary_csv(rows: &[HistStructureRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&HistStructureRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::new();
    out.push_str(&format!(
        "# source=pairwise_intervals bin_width={:.3}\n",
        E2_PAIRWISE_BIN_ST
    ));
    out.push_str(
        "cond,mean_entropy,std_entropy,mean_gini,std_gini,mean_peakiness,std_peakiness,mean_kl_uniform,std_kl_uniform,n\n",
    );
    for cond in ["baseline", "nohill", "nocrowd"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let entropy: Vec<f32> = rows.iter().map(|r| r.metrics.entropy).collect();
        let gini: Vec<f32> = rows.iter().map(|r| r.metrics.gini).collect();
        let peakiness: Vec<f32> = rows.iter().map(|r| r.metrics.peakiness).collect();
        let kl: Vec<f32> = rows.iter().map(|r| r.metrics.kl_uniform).collect();
        let (mean_entropy, std_entropy) = mean_std_scalar(&entropy);
        let (mean_gini, std_gini) = mean_std_scalar(&gini);
        let (mean_peak, std_peak) = mean_std_scalar(&peakiness);
        let (mean_kl, std_kl) = mean_std_scalar(&kl);
        out.push_str(&format!(
            "{cond},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            mean_entropy,
            std_entropy,
            mean_gini,
            std_gini,
            mean_peak,
            std_peak,
            mean_kl,
            std_kl,
            rows.len()
        ));
    }
    out
}

fn render_hist_structure_summary_plot(
    out_path: &Path,
    rows: &[HistStructureRow],
) -> Result<(), Box<dyn Error>> {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&HistStructureRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let conds = ["baseline", "nohill", "nocrowd"];
    let colors = [&PAL_H, &PAL_R, &PAL_CD];

    let mut means_entropy = [0.0f32; 3];
    let mut means_gini = [0.0f32; 3];
    let mut means_peak = [0.0f32; 3];
    let mut means_kl = [0.0f32; 3];
    for (i, cond) in conds.iter().enumerate() {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let entropy: Vec<f32> = rows.iter().map(|r| r.metrics.entropy).collect();
        let gini: Vec<f32> = rows.iter().map(|r| r.metrics.gini).collect();
        let peakiness: Vec<f32> = rows.iter().map(|r| r.metrics.peakiness).collect();
        let kl: Vec<f32> = rows.iter().map(|r| r.metrics.kl_uniform).collect();
        means_entropy[i] = mean_std_scalar(&entropy).0;
        means_gini[i] = mean_std_scalar(&gini).0;
        means_peak[i] = mean_std_scalar(&peakiness).0;
        means_kl[i] = mean_std_scalar(&kl).0;
    }

    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    draw_hist_structure_panel(
        &panels[0],
        "Histogram Structure: Entropy",
        "entropy",
        &means_entropy,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[1],
        "Histogram Structure: Gini",
        "gini",
        &means_gini,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[2],
        "Histogram Structure: Peakiness",
        "peakiness",
        &means_peak,
        &conds,
        &colors,
    )?;
    draw_hist_structure_panel(
        &panels[3],
        "Histogram Structure: KL vs Uniform",
        "KL",
        &means_kl,
        &conds,
        &colors,
    )?;

    root.present()?;
    Ok(())
}

fn draw_hist_structure_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    values: &[f32; 3],
    labels: &[&str; 3],
    colors: &[&RGBColor; 3],
) -> Result<(), Box<dyn Error>> {
    let mut y_max = values.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    y_max *= 1.1;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0f32..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc(y_desc)
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                labels[idx as usize].to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, value) in values.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *value)],
            colors[i].filled(),
        )))?;
    }
    Ok(())
}

fn diversity_metrics_for_run(run: &E2Run) -> DiversityMetrics {
    diversity_metrics_for_semitones(&run.final_semitones)
}

fn diversity_metrics_for_semitones(semitones: &[f32]) -> DiversityMetrics {
    let mut values: Vec<f32> = semitones
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return DiversityMetrics {
            unique_bins: 0,
            nn_mean: 0.0,
            nn_std: 0.0,
            semitone_var: 0.0,
            semitone_mad: 0.0,
        };
    }

    let mut unique_bins = std::collections::HashSet::new();
    for &v in &values {
        let bin = (v / E2_DIVERSITY_BIN_ST).round() as i32;
        unique_bins.insert(bin);
    }

    let mut nn_dists = Vec::with_capacity(values.len());
    for (i, &v) in values.iter().enumerate() {
        let mut best = f32::INFINITY;
        for (j, &u) in values.iter().enumerate() {
            if i == j {
                continue;
            }
            let dist = (v - u).abs();
            if dist < best {
                best = dist;
            }
        }
        if best.is_finite() {
            nn_dists.push(best);
        }
    }

    let (nn_mean, nn_std) = if nn_dists.is_empty() {
        (0.0, 0.0)
    } else {
        let (mean, std) = mean_std_scalar(&nn_dists);
        (mean, std)
    };

    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / values.len() as f32;

    let median = median_of_values(&mut values);
    let mut abs_dev: Vec<f32> = values.iter().map(|v| (v - median).abs()).collect();
    let mad = median_of_values(&mut abs_dev);

    DiversityMetrics {
        unique_bins: unique_bins.len(),
        nn_mean,
        nn_std,
        semitone_var: var,
        semitone_mad: mad,
    }
}

fn diversity_rows(condition: &'static str, runs: &[E2Run]) -> Vec<DiversityRow> {
    runs.iter()
        .map(|run| DiversityRow {
            condition,
            seed: run.seed,
            metrics: diversity_metrics_for_run(run),
        })
        .collect()
}

fn diversity_by_seed_csv(rows: &[DiversityRow]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# bin_width={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str("seed,cond,unique_bins,nn_mean,nn_std,semitone_var,semitone_mad\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{},{:.6},{:.6},{:.6},{:.6}\n",
            row.seed,
            row.condition,
            row.metrics.unique_bins,
            row.metrics.nn_mean,
            row.metrics.nn_std,
            row.metrics.semitone_var,
            row.metrics.semitone_mad
        ));
    }
    out
}

fn diversity_summary_csv(rows: &[DiversityRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::new();
    out.push_str(&format!("# bin_width={:.3}\n", E2_DIVERSITY_BIN_ST));
    out.push_str(
        "cond,mean_unique_bins,std_unique_bins,mean_nn,stdev_nn,mean_semitone_var,std_semitone_var,mean_semitone_mad,std_semitone_mad,n\n",
    );
    for cond in ["baseline", "nohill", "nocrowd"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        let (mean_bins, std_bins) = mean_std_scalar(&unique_bins);
        let (mean_nn, std_nn) = mean_std_scalar(&nn_mean);
        let (mean_var, std_var) = mean_std_scalar(&semitone_var);
        let (mean_mad, std_mad) = mean_std_scalar(&semitone_mad);
        out.push_str(&format!(
            "{cond},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            mean_bins,
            std_bins,
            mean_nn,
            std_nn,
            mean_var,
            std_var,
            mean_mad,
            std_mad,
            rows.len()
        ));
    }
    out
}

fn close_pair_fraction_for_semitones(semitones: &[f32], window_st: f32) -> f32 {
    if semitones.len() < 2 || window_st <= 0.0 {
        return 0.0;
    }
    let mut close = 0usize;
    let mut total = 0usize;
    for i in 0..semitones.len() {
        let a = semitones[i];
        if !a.is_finite() {
            continue;
        }
        for &b in &semitones[..i] {
            if !b.is_finite() {
                continue;
            }
            total += 1;
            if (a - b).abs() <= window_st + 1e-6 {
                close += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        close as f32 / total as f32
    }
}

fn e2_dense_seed_slice(quick: bool) -> &'static [u64] {
    if quick {
        &E2_SEEDS[..E2_DENSE_SWEEP_QUICK_SEEDS.min(E2_SEEDS.len())]
    } else {
        &E2_SEEDS
    }
}

fn e2_dense_n_agents(range_oct: f32, voices_per_oct: f32) -> usize {
    (range_oct * voices_per_oct).round().max(2.0) as usize
}

fn e2_dense_row_from_run(
    run: &E2Run,
    anchor_hz: f32,
    range_oct: f32,
    voices_per_oct: f32,
    condition: &'static str,
) -> E2DenseSweepRow {
    let diversity = diversity_metrics_for_run(run);
    E2DenseSweepRow {
        range_oct,
        n_agents: run.n_agents,
        voices_per_oct,
        condition,
        seed: run.seed,
        c_score_loo_end: run.mean_c_score_loo_series.last().copied().unwrap_or(0.0),
        ji_scene_end: ji_population_score(&run.final_freqs_hz, anchor_hz),
        entropy_end: hist_structure_metrics_for_run(run).entropy,
        unique_bins_end: diversity.unique_bins,
        nn_mean_end: diversity.nn_mean,
        close_pair_frac_50ct: close_pair_fraction_for_semitones(
            &run.final_semitones,
            E2_CLOSE_PAIR_WINDOW_ST,
        ),
    }
}

fn e2_dense_sweep_by_seed_csv(rows: &[E2DenseSweepRow]) -> String {
    let mut out = String::from(
        "range_oct,n_agents,voices_per_oct,condition,seed,c_score_loo_end,ji_scene_end,entropy_end,unique_bins_end,nn_mean_end,close_pair_frac_50ct\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.2},{},{:.2},{},{},{:.6},{:.6},{:.6},{},{:.6},{:.6}\n",
            row.range_oct,
            row.n_agents,
            row.voices_per_oct,
            row.condition,
            row.seed,
            row.c_score_loo_end,
            row.ji_scene_end,
            row.entropy_end,
            row.unique_bins_end,
            row.nn_mean_end,
            row.close_pair_frac_50ct
        ));
    }
    out
}

fn e2_dense_sweep_summary_rows(rows: &[E2DenseSweepRow]) -> Vec<E2DenseSweepSummaryRow> {
    let mut grouped: std::collections::BTreeMap<
        (i32, usize, i32, &'static str),
        Vec<&E2DenseSweepRow>,
    > = std::collections::BTreeMap::new();
    for row in rows {
        grouped
            .entry((
                float_key(row.range_oct),
                row.n_agents,
                float_key(row.voices_per_oct),
                row.condition,
            ))
            .or_default()
            .push(row);
    }
    let mut out = Vec::with_capacity(grouped.len());
    for ((range_key, n_agents, vpo_key, condition), group) in grouped {
        let c_vals: Vec<f32> = group.iter().map(|r| r.c_score_loo_end).collect();
        let ji_vals: Vec<f32> = group.iter().map(|r| r.ji_scene_end).collect();
        let ent_vals: Vec<f32> = group.iter().map(|r| r.entropy_end).collect();
        let bins_vals: Vec<f32> = group.iter().map(|r| r.unique_bins_end as f32).collect();
        let nn_vals: Vec<f32> = group.iter().map(|r| r.nn_mean_end).collect();
        let close_vals: Vec<f32> = group.iter().map(|r| r.close_pair_frac_50ct).collect();
        let (c_mean, _) = mean_std_scalar(&c_vals);
        let (ji_mean, _) = mean_std_scalar(&ji_vals);
        let (ent_mean, _) = mean_std_scalar(&ent_vals);
        let (bins_mean, _) = mean_std_scalar(&bins_vals);
        let (nn_mean, _) = mean_std_scalar(&nn_vals);
        let (close_mean, _) = mean_std_scalar(&close_vals);
        out.push(E2DenseSweepSummaryRow {
            range_oct: float_from_key(range_key),
            n_agents,
            voices_per_oct: float_from_key(vpo_key),
            condition,
            c_score_loo_mean: c_mean,
            c_score_loo_ci95: ci95_half_width(&c_vals),
            ji_scene_mean: ji_mean,
            ji_scene_ci95: ci95_half_width(&ji_vals),
            entropy_mean: ent_mean,
            entropy_ci95: ci95_half_width(&ent_vals),
            unique_bins_mean: bins_mean,
            unique_bins_ci95: ci95_half_width(&bins_vals),
            nn_mean_mean: nn_mean,
            nn_mean_ci95: ci95_half_width(&nn_vals),
            close_pair_mean: close_mean,
            close_pair_ci95: ci95_half_width(&close_vals),
            n: group.len(),
        });
    }
    out
}

fn e2_dense_sweep_summary_csv(rows: &[E2DenseSweepSummaryRow]) -> String {
    let mut out = String::from(
        "range_oct,n_agents,voices_per_oct,condition,c_score_loo_mean,c_score_loo_ci95,ji_scene_mean,ji_scene_ci95,entropy_mean,entropy_ci95,unique_bins_mean,unique_bins_ci95,nn_mean_mean,nn_mean_ci95,close_pair_mean,close_pair_ci95,n\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.2},{},{:.2},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            row.range_oct,
            row.n_agents,
            row.voices_per_oct,
            row.condition,
            row.c_score_loo_mean,
            row.c_score_loo_ci95,
            row.ji_scene_mean,
            row.ji_scene_ci95,
            row.entropy_mean,
            row.entropy_ci95,
            row.unique_bins_mean,
            row.unique_bins_ci95,
            row.nn_mean_mean,
            row.nn_mean_ci95,
            row.close_pair_mean,
            row.close_pair_ci95,
            row.n
        ));
    }
    out
}

fn e2_dense_delta_csv(rows: &[E2DenseSweepSummaryRow]) -> String {
    let mut out = String::from(
        "range_oct,n_agents,voices_per_oct,delta_c_score_loo_end,delta_ji_scene_end,delta_entropy_end,delta_unique_bins_end,delta_nn_mean_end,delta_close_pair_frac_50ct\n",
    );
    let mut grouped: std::collections::BTreeMap<
        (i32, usize, i32),
        (&E2DenseSweepSummaryRow, &E2DenseSweepSummaryRow),
    > = std::collections::BTreeMap::new();
    for row in rows {
        let key = (
            float_key(row.range_oct),
            row.n_agents,
            float_key(row.voices_per_oct),
        );
        let entry = grouped.entry(key).or_insert((row, row));
        match row.condition {
            "baseline" => entry.0 = row,
            "nocrowd" => entry.1 = row,
            _ => {}
        }
    }
    for ((range_key, n_agents, vpo_key), (baseline, nocrowd)) in grouped {
        if baseline.condition != "baseline" || nocrowd.condition != "nocrowd" {
            continue;
        }
        out.push_str(&format!(
            "{:.2},{},{:.2},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            float_from_key(range_key),
            n_agents,
            float_from_key(vpo_key),
            nocrowd.c_score_loo_mean - baseline.c_score_loo_mean,
            nocrowd.ji_scene_mean - baseline.ji_scene_mean,
            nocrowd.entropy_mean - baseline.entropy_mean,
            nocrowd.unique_bins_mean - baseline.unique_bins_mean,
            nocrowd.nn_mean_mean - baseline.nn_mean_mean,
            nocrowd.close_pair_mean - baseline.close_pair_mean
        ));
    }
    out
}

fn render_e2_dense_tradeoff_plot(
    out_path: &Path,
    rows: &[E2DenseSweepSummaryRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let ranges = E2_DENSE_SWEEP_RANGES_OCT;
    let root = bitmap_root(out_path, (1650, 520)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, ranges.len()));

    let mut x_max = 0.0f32;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for row in rows {
        x_max = x_max.max(row.close_pair_mean + row.close_pair_ci95);
        y_min = y_min.min(row.c_score_loo_mean - row.c_score_loo_ci95);
        y_max = y_max.max(row.c_score_loo_mean + row.c_score_loo_ci95);
    }
    let x_hi = (x_max * 1.15).max(0.05);
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let y_pad = ((y_max - y_min).abs() * 0.12).max(0.05);
    let y_lo = (y_min - y_pad).min(0.0);
    let y_hi = y_max + y_pad;

    for (panel_idx, range_oct) in ranges.iter().enumerate() {
        let panel_rows: Vec<&E2DenseSweepSummaryRow> = rows
            .iter()
            .filter(|row| (row.range_oct - *range_oct).abs() < 1e-6)
            .collect();
        if panel_rows.is_empty() {
            continue;
        }
        let mut chart = ChartBuilder::on(&panels[panel_idx])
            .caption(
                format!("range={range_oct:.0} oct | close-pairs vs final C_score"),
                ("sans-serif", 18),
            )
            .margin(10)
            .x_label_area_size(46)
            .y_label_area_size(58)
            .build_cartesian_2d(0.0f32..x_hi, y_lo..y_hi)?;
        chart
            .configure_mesh()
            .x_desc("close-pair fraction (< 50 ct)")
            .y_desc("final mean LOO C_score")
            .label_style(("sans-serif", 14).into_font())
            .axis_desc_style(("sans-serif", 16).into_font())
            .draw()?;

        let mut by_n: std::collections::BTreeMap<usize, Vec<&E2DenseSweepSummaryRow>> =
            std::collections::BTreeMap::new();
        for row in panel_rows {
            by_n.entry(row.n_agents).or_default().push(row);
        }
        for (n_agents, pair) in by_n {
            let baseline = pair.iter().find(|row| row.condition == "baseline").copied();
            let nocrowd = pair.iter().find(|row| row.condition == "nocrowd").copied();
            if let (Some(base), Some(no_rep)) = (baseline, nocrowd) {
                chart.draw_series(std::iter::once(PathElement::new(
                    vec![
                        (base.close_pair_mean, base.c_score_loo_mean),
                        (no_rep.close_pair_mean, no_rep.c_score_loo_mean),
                    ],
                    BLACK.mix(0.25),
                )))?;
            }
            for row in pair {
                let color = e2_condition_color(row.condition);
                chart.draw_series(std::iter::once(Circle::new(
                    (row.close_pair_mean, row.c_score_loo_mean),
                    5,
                    color.filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("N={n_agents}"),
                    (row.close_pair_mean + 0.01, row.c_score_loo_mean),
                    ("sans-serif", 12).into_font().color(&color),
                )))?;
            }
        }
    }
    root.present()?;
    Ok(())
}

fn e2_candidate_cases() -> Vec<(f32, usize)> {
    let mut cases = Vec::new();
    for &n_agents in &E2_CANDIDATE_SEARCH_2OCT_N {
        cases.push((2.0, n_agents));
    }
    for &n_agents in &E2_CANDIDATE_SEARCH_3OCT_N {
        cases.push((3.0, n_agents));
    }
    for &n_agents in &E2_CANDIDATE_SEARCH_4OCT_N {
        cases.push((4.0, n_agents));
    }
    cases
}

fn e2_candidate_summary_csv(rows: &[E2CandidateSummaryRow]) -> String {
    let mut out = String::from(
        "range_oct,n_agents,n_seeds,baseline_c_score_loo_mean,nohill_c_score_loo_mean,delta_c_score_loo,baseline_ji_scene_mean,nohill_ji_scene_mean,delta_ji_scene,baseline_entropy_mean,nohill_entropy_mean,delta_entropy,baseline_unique_bins_mean,nohill_unique_bins_mean,delta_unique_bins,baseline_nn_mean,nohill_nn_mean,shortlist_score\n",
    );
    for row in rows {
        out.push_str(&format!(
            "{:.2},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            row.range_oct,
            row.n_agents,
            row.n_seeds,
            row.baseline_c_score_loo_mean,
            row.nohill_c_score_loo_mean,
            row.delta_c_score_loo,
            row.baseline_ji_scene_mean,
            row.nohill_ji_scene_mean,
            row.delta_ji_scene,
            row.baseline_entropy_mean,
            row.nohill_entropy_mean,
            row.delta_entropy,
            row.baseline_unique_bins_mean,
            row.nohill_unique_bins_mean,
            row.delta_unique_bins,
            row.baseline_nn_mean,
            row.nohill_nn_mean,
            row.shortlist_score
        ));
    }
    out
}

fn e2_candidate_report(rows: &[E2CandidateSummaryRow]) -> String {
    let mut out = String::from(
        "Fig2A candidate search (baseline vs no-hill)\n=========================================\n\n",
    );
    for &range_oct in &[2.0f32, 3.0, 4.0] {
        out.push_str(&format!("range={range_oct:.0} oct\n"));
        out.push_str(
            "N    score   dEntropy  baseJI   dCscore  baseBins  nohillBins  baseNN  nohillNN\n",
        );
        let mut range_rows: Vec<&E2CandidateSummaryRow> = rows
            .iter()
            .filter(|row| (row.range_oct - range_oct).abs() < 1e-6)
            .collect();
        range_rows.sort_by(|a, b| {
            b.shortlist_score
                .partial_cmp(&a.shortlist_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for row in range_rows {
            out.push_str(&format!(
                "{:<4} {:>6.3} {:>9.3} {:>7.3} {:>8.3} {:>9.3} {:>11.3} {:>7.3} {:>9.3}\n",
                row.n_agents,
                row.shortlist_score,
                row.delta_entropy,
                row.baseline_ji_scene_mean,
                row.delta_c_score_loo,
                row.baseline_unique_bins_mean,
                row.nohill_unique_bins_mean,
                row.baseline_nn_mean,
                row.nohill_nn_mean,
            ));
        }
        out.push('\n');
    }
    out
}

fn e2_candidate_top_rows(rows: &[E2CandidateSummaryRow]) -> Vec<&E2CandidateSummaryRow> {
    let mut selected = Vec::new();
    for &range_oct in &[2.0f32, 3.0, 4.0] {
        let mut range_rows: Vec<&E2CandidateSummaryRow> = rows
            .iter()
            .filter(|row| (row.range_oct - range_oct).abs() < 1e-6)
            .collect();
        range_rows.sort_by(|a, b| {
            b.shortlist_score
                .partial_cmp(&a.shortlist_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        selected.extend(
            range_rows
                .into_iter()
                .take(E2_CANDIDATE_SEARCH_TOP_PER_RANGE),
        );
    }
    selected
}

fn e2_shortlist_score(
    baseline_c: f32,
    nohill_c: f32,
    baseline_ji: f32,
    baseline_entropy: f32,
    nohill_entropy: f32,
) -> f32 {
    let delta_entropy = nohill_entropy - baseline_entropy;
    let delta_c = baseline_c - nohill_c;
    delta_entropy + 0.5 * baseline_ji + 0.25 * delta_c
}

fn mean_last_c_score_loo(runs: &[E2Run]) -> f32 {
    let values: Vec<f32> = runs
        .iter()
        .map(|run| run.mean_c_score_loo_series.last().copied().unwrap_or(0.0))
        .collect();
    mean_std_scalar(&values).0
}

fn mean_ji_scene_end(runs: &[E2Run], _anchor_hz: f32) -> f32 {
    let values: Vec<f32> = runs
        .iter()
        .map(|run| ji_population_score(&run.final_freqs_hz, run.fixed_drone_hz))
        .collect();
    mean_std_scalar(&values).0
}

fn mean_entropy_end(runs: &[E2Run]) -> f32 {
    let values: Vec<f32> = runs
        .iter()
        .map(|run| hist_structure_metrics_for_run(run).entropy)
        .collect();
    mean_std_scalar(&values).0
}

fn mean_unique_bins_end(runs: &[E2Run]) -> f32 {
    let values: Vec<f32> = runs
        .iter()
        .map(|run| diversity_metrics_for_run(run).unique_bins as f32)
        .collect();
    mean_std_scalar(&values).0
}

fn mean_nn_end(runs: &[E2Run]) -> f32 {
    let values: Vec<f32> = runs
        .iter()
        .map(|run| diversity_metrics_for_run(run).nn_mean)
        .collect();
    mean_std_scalar(&values).0
}

fn e2_render_summary_csv(
    stem: &str,
    mode: &str,
    condition_label: &str,
    run: &E2Run,
    _anchor_hz: f32,
    render_partials: u32,
    range_oct: f32,
) -> String {
    let init_c = run.mean_c_series.first().copied().unwrap_or(0.0);
    let mid_c = run
        .mean_c_series
        .get(E2_PHASE_SWITCH_STEP.min(run.mean_c_series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(init_c);
    let end_c = run.mean_c_series.last().copied().unwrap_or(init_c);
    let init_level = run.mean_c_level_series.first().copied().unwrap_or(0.0);
    let mid_level = run
        .mean_c_level_series
        .get(E2_PHASE_SWITCH_STEP.min(run.mean_c_level_series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(init_level);
    let end_level = run
        .mean_c_level_series
        .last()
        .copied()
        .unwrap_or(init_level);
    let init_loo = run.mean_c_score_loo_series.first().copied().unwrap_or(0.0);
    let mid_loo = run
        .mean_c_score_loo_series
        .get(E2_PHASE_SWITCH_STEP.min(run.mean_c_score_loo_series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(init_loo);
    let end_loo = run
        .mean_c_score_loo_series
        .last()
        .copied()
        .unwrap_or(init_loo);
    let ji_series = e2_scene_ji_series(run);
    let init_ji = ji_series.first().copied().unwrap_or(0.0);
    let mid_ji = ji_series
        .get(E2_PHASE_SWITCH_STEP.min(ji_series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(init_ji);
    let end_ji = ji_series.last().copied().unwrap_or(init_ji);
    let moved_pre = run
        .moved_frac_series
        .iter()
        .take(E2_PHASE_SWITCH_STEP)
        .copied()
        .sum::<f32>()
        / E2_PHASE_SWITCH_STEP.max(1) as f32;
    let moved_post_len = run
        .moved_frac_series
        .len()
        .saturating_sub(E2_PHASE_SWITCH_STEP);
    let moved_post = if moved_post_len > 0 {
        run.moved_frac_series
            .iter()
            .skip(E2_PHASE_SWITCH_STEP)
            .copied()
            .sum::<f32>()
            / moved_post_len as f32
    } else {
        0.0
    };
    let mut out = String::from(
        "file,mode,seed,condition,n_agents,fixed_drone_hz,render_partials,range_octaves,steps,mean_c_init,mean_c_mid,mean_c_end,delta_mean_c,mean_c_level_init,mean_c_level_mid,mean_c_level_end,delta_mean_c_level,mean_c_score_loo_init,mean_c_score_loo_mid,mean_c_score_loo_end,delta_mean_c_score_loo,ji_scene_init,ji_scene_mid,ji_scene_end,delta_ji_scene,moved_pre,moved_post\n",
    );
    out.push_str(&format!(
        "{stem},{mode},{},{condition_label},{},{:.3},{},{:.1},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
        run.seed,
        run.n_agents,
        run.fixed_drone_hz,
        render_partials,
        range_oct,
        run.mean_c_series.len(),
        init_c,
        mid_c,
        end_c,
        end_c - init_c,
        init_level,
        mid_level,
        end_level,
        end_level - init_level,
        init_loo,
        mid_loo,
        end_loo,
        end_loo - init_loo,
        init_ji,
        mid_ji,
        end_ji,
        end_ji - init_ji,
        moved_pre,
        moved_post,
    ));
    out
}

fn generate_e2_condition_render_named(
    condition: E2Condition,
    condition_label: &str,
    n_agents: usize,
    render_partials: u32,
    range_oct: f32,
    stem_prefix: &str,
) -> io::Result<()> {
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
    let anchor_hz = E4_ANCHOR_HZ;
    let seed = E2_SEEDS[0];
    let run = run_e2_once_cfg(
        &space,
        anchor_hz,
        seed,
        condition,
        E2_STEP_SEMITONES,
        E2PhaseMode::DissonanceThenConsonance,
        None,
        0,
        n_agents,
        range_oct,
    );

    let out_dir = Path::new("supplementary_audio/audio");
    create_dir_all(out_dir)?;
    let stem = format!(
        "{stem_prefix}_seed0_{}_n{}_p{}_{}",
        condition_label,
        n_agents,
        render_partials,
        e2_range_oct_tag(range_oct)
    );
    let wav_path = out_dir.join(format!("{stem}.wav"));
    let csv_path = out_dir.join(format!("{stem}.csv"));
    let moves_path = out_dir.join(format!("{stem}_moves.csv"));
    let scene_metrics_path = out_dir.join(format!("{stem}_scene_metrics.csv"));

    render_e2_trajectory_wav(
        &wav_path,
        &run.trajectory_semitones,
        anchor_hz,
        render_partials,
        Some(run.fixed_drone_hz),
        true,
    )?;
    write_with_log(
        &csv_path,
        e2_render_summary_csv(
            &format!("{stem}.wav"),
            "continuous",
            condition_label,
            &run,
            anchor_hz,
            render_partials,
            range_oct,
        ),
    )?;
    write_with_log(&moves_path, e2_moves_csv(&run))?;
    write_with_log(&scene_metrics_path, e2_scene_metrics_csv(&run, anchor_hz))?;
    Ok(())
}

fn plot_e2_candidate_search(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
    quick: bool,
) -> Result<(), Box<dyn Error>> {
    let seeds = e2_dense_seed_slice(quick);
    let max_worker_threads = if quick { Some(4) } else { None };
    let mut rows = Vec::new();

    for (range_oct, n_agents) in e2_candidate_cases() {
        eprintln!(
            "  Candidate search: range={range_oct:.1}oct N={n_agents} seeds={}",
            seeds.len()
        );
        let (baseline_runs, _) = e2_seed_sweep_with_threads_for_seeds(
            space,
            anchor_hz,
            E2Condition::Baseline,
            E2_STEP_SEMITONES,
            phase_mode,
            None,
            0,
            n_agents,
            range_oct,
            seeds,
            max_worker_threads,
        );
        let (nohill_runs, _) = e2_seed_sweep_with_threads_for_seeds(
            space,
            anchor_hz,
            E2Condition::NoHillClimb,
            E2_STEP_SEMITONES,
            phase_mode,
            None,
            0,
            n_agents,
            range_oct,
            seeds,
            max_worker_threads,
        );
        let baseline_c = mean_last_c_score_loo(&baseline_runs);
        let nohill_c = mean_last_c_score_loo(&nohill_runs);
        let baseline_ji = mean_ji_scene_end(&baseline_runs, anchor_hz);
        let nohill_ji = mean_ji_scene_end(&nohill_runs, anchor_hz);
        let baseline_entropy = mean_entropy_end(&baseline_runs);
        let nohill_entropy = mean_entropy_end(&nohill_runs);
        let baseline_bins = mean_unique_bins_end(&baseline_runs);
        let nohill_bins = mean_unique_bins_end(&nohill_runs);
        let baseline_nn = mean_nn_end(&baseline_runs);
        let nohill_nn = mean_nn_end(&nohill_runs);
        rows.push(E2CandidateSummaryRow {
            range_oct,
            n_agents,
            n_seeds: baseline_runs.len(),
            baseline_c_score_loo_mean: baseline_c,
            nohill_c_score_loo_mean: nohill_c,
            delta_c_score_loo: baseline_c - nohill_c,
            baseline_ji_scene_mean: baseline_ji,
            nohill_ji_scene_mean: nohill_ji,
            delta_ji_scene: baseline_ji - nohill_ji,
            baseline_entropy_mean: baseline_entropy,
            nohill_entropy_mean: nohill_entropy,
            delta_entropy: nohill_entropy - baseline_entropy,
            baseline_unique_bins_mean: baseline_bins,
            nohill_unique_bins_mean: nohill_bins,
            delta_unique_bins: baseline_bins - nohill_bins,
            baseline_nn_mean: baseline_nn,
            nohill_nn_mean: nohill_nn,
            shortlist_score: e2_shortlist_score(
                baseline_c,
                nohill_c,
                baseline_ji,
                baseline_entropy,
                nohill_entropy,
            ),
        });
    }

    rows.sort_by(|a, b| {
        a.range_oct
            .partial_cmp(&b.range_oct)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.n_agents.cmp(&b.n_agents))
    });
    write_with_log(
        out_dir.join("paper_e2_candidate_search_summary.csv"),
        e2_candidate_summary_csv(&rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_candidate_search_report.txt"),
        e2_candidate_report(&rows),
    )?;

    let shortlist = e2_candidate_top_rows(&rows);
    let mut shortlist_text =
        String::from("range_oct,n_agents,score,baseline_render,nohill_render\n");
    for row in shortlist {
        generate_e2_condition_render_named(
            E2Condition::Baseline,
            "baseline",
            row.n_agents,
            E2_CANDIDATE_SEARCH_RENDER_PARTIALS,
            row.range_oct,
            "15_exp1_candidate",
        )?;
        generate_e2_condition_render_named(
            E2Condition::NoHillClimb,
            "nohill",
            row.n_agents,
            E2_CANDIDATE_SEARCH_RENDER_PARTIALS,
            row.range_oct,
            "15_exp1_candidate",
        )?;
        let range_tag = e2_range_oct_tag(row.range_oct);
        shortlist_text.push_str(&format!(
            "{:.1},{},{:.6},15_exp1_candidate_seed0_baseline_n{}_p{}_{}.wav,15_exp1_candidate_seed0_nohill_n{}_p{}_{}.wav\n",
            row.range_oct,
            row.n_agents,
            row.shortlist_score,
            row.n_agents,
            E2_CANDIDATE_SEARCH_RENDER_PARTIALS,
            range_tag,
            row.n_agents,
            E2_CANDIDATE_SEARCH_RENDER_PARTIALS,
            range_tag,
        ));
    }
    write_with_log(
        out_dir.join("paper_e2_candidate_search_shortlist.csv"),
        shortlist_text,
    )?;

    Ok(())
}

fn render_diversity_summary_plot(
    out_path: &Path,
    rows: &[DiversityRow],
) -> Result<(), Box<dyn Error>> {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let conds = ["baseline", "nohill", "nocrowd"];
    let colors = [&PAL_H, &PAL_R, &PAL_CD];

    let mut mean_bins = [0.0f32; 3];
    let mut mean_nn = [0.0f32; 3];
    let mut mean_var = [0.0f32; 3];
    let mut mean_mad = [0.0f32; 3];
    for (i, cond) in conds.iter().enumerate() {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        mean_bins[i] = mean_std_scalar(&unique_bins).0;
        mean_nn[i] = mean_std_scalar(&nn_mean).0;
        mean_var[i] = mean_std_scalar(&semitone_var).0;
        mean_mad[i] = mean_std_scalar(&semitone_mad).0;
    }

    let root = bitmap_root(out_path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 2));

    draw_diversity_panel(
        &panels[0],
        "Diversity: Unique Bins",
        "unique bins",
        &mean_bins,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[1],
        "Diversity: NN Distance",
        "nn mean (ct)",
        &mean_nn,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[2],
        "Diversity: Variance",
        "var (ct^2)",
        &mean_var,
        &conds,
        &colors,
    )?;
    draw_diversity_panel(
        &panels[3],
        "Diversity: MAD",
        "MAD (ct)",
        &mean_mad,
        &conds,
        &colors,
    )?;

    root.present()?;
    Ok(())
}

fn render_diversity_summary_ci95_plot(
    out_path: &Path,
    rows: &[DiversityRow],
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (500, 280)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));
    draw_diversity_metric_panel(&panels[0], "Unique Bins", "unique bins", rows, |metrics| {
        metrics.unique_bins as f32
    })?;
    draw_diversity_metric_panel(
        &panels[1],
        "NN Distance (ct)",
        "nn mean (ct)",
        rows,
        |metrics| metrics.nn_mean,
    )?;
    root.present()?;
    Ok(())
}

fn draw_diversity_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    values: &[f32; 3],
    labels: &[&str; 3],
    colors: &[&RGBColor; 3],
) -> Result<(), Box<dyn Error>> {
    let mut y_max = values.iter().copied().fold(0.0f32, f32::max).max(1e-6);
    y_max *= 1.1;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(50)
        .build_cartesian_2d(-0.5f32..2.5f32, 0f32..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc(y_desc)
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                labels[idx as usize].to_string()
            } else {
                String::new()
            }
        })
        .draw()?;

    for (i, value) in values.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, *value)],
            colors[i].filled(),
        )))?;
    }
    Ok(())
}

fn median_of_values(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn mean_std_values(values: &[Vec<f32>]) -> (Vec<f32>, Vec<f32>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = values[0].len();
    let mut sum = vec![0.0f32; len];
    let mut sum_sq = vec![0.0f32; len];
    for row in values {
        debug_assert_eq!(row.len(), len, "hist length mismatch");
        for (i, &val) in row.iter().enumerate() {
            sum[i] += val;
            sum_sq[i] += val * val;
        }
    }
    let n = values.len() as f32;
    let mut mean = vec![0.0f32; len];
    let mut std = vec![0.0f32; len];
    for i in 0..len {
        mean[i] = sum[i] / n;
        let var = (sum_sq[i] / n) - mean[i] * mean[i];
        std[i] = var.max(0.0).sqrt();
    }
    (mean, std)
}

fn mean_std_histograms(hists: &[Vec<(f32, f32)>]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    if hists.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let len = hists[0].len();
    let centers: Vec<f32> = hists[0].iter().map(|(c, _)| *c).collect();
    let mut values: Vec<Vec<f32>> = Vec::with_capacity(hists.len());
    for hist in hists {
        debug_assert_eq!(hist.len(), len, "hist length mismatch");
        values.push(hist.iter().map(|(_, v)| *v).collect());
    }
    let (mean, std) = mean_std_values(&values);
    (centers, mean, std)
}

fn mean_std_histogram_fractions(hists: &[Vec<(f32, f32)>]) -> (Vec<f32>, Vec<f32>) {
    if hists.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let len = hists[0].len();
    let mut values: Vec<Vec<f32>> = Vec::with_capacity(hists.len());
    for hist in hists {
        debug_assert_eq!(hist.len(), len, "hist length mismatch");
        let total: f32 = hist.iter().map(|(_, v)| *v).sum();
        let inv = if total > 0.0 { 1.0 / total } else { 0.0 };
        values.push(hist.iter().map(|(_, v)| v * inv).collect());
    }
    mean_std_values(&values)
}

fn e2_hist_seed_sweep(runs: &[E2Run], bin_width: f32, min: f32, max: f32) -> HistSweepStats {
    let hists: Vec<Vec<(f32, f32)>> = parallel_map_ordered(runs, None, |run| {
        histogram_counts_fixed(&run.semitone_samples_post, min, max, bin_width)
    });
    let (centers, mean_count, std_count) = mean_std_histograms(&hists);
    let (mean_frac, std_frac) = mean_std_histogram_fractions(&hists);
    if !mean_frac.is_empty() {
        let sum: f32 = mean_frac.iter().copied().sum();
        debug_assert!((sum - 1.0).abs() < 1e-3, "mean_frac sum not ~1 (sum={sum})");
    }
    HistSweepStats {
        centers,
        mean_count,
        std_count,
        mean_frac,
        std_frac,
        n: hists.len(),
    }
}

fn e2_hist_seed_sweep_csv(stats: &HistSweepStats) -> String {
    let mut out = String::from("bin_center,mean_frac,std_frac,n_seeds,mean_count,std_count\n");
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_frac[i],
            stats.std_frac[i],
            stats.n,
            stats.mean_count[i],
            stats.std_count[i]
        ));
    }
    out
}

fn e2_pairwise_hist_seed_sweep(
    runs: &[E2Run],
    bin_width: f32,
    min: f32,
    max: f32,
) -> (HistSweepStats, usize) {
    let pairwise: Vec<(Vec<(f32, f32)>, usize)> = parallel_map_ordered(runs, None, |run| {
        let samples = pairwise_interval_samples(&run.final_semitones);
        let expected = if run.final_semitones.len() < 2 {
            0usize
        } else {
            run.final_semitones.len() * (run.final_semitones.len() - 1) / 2
        };
        debug_assert_eq!(samples.len(), expected, "pairwise interval count mismatch");
        for &value in &samples {
            debug_assert!(
                value >= min - 1e-6 && value <= max + 1e-6,
                "pairwise interval out of range: {value}"
            );
        }
        (
            histogram_counts_fixed(&samples, min, max, bin_width),
            expected,
        )
    });
    let n_pairs = pairwise.first().map(|(_, n_pairs)| *n_pairs).unwrap_or(0);
    let hists: Vec<Vec<(f32, f32)>> = pairwise.into_iter().map(|(hist, _)| hist).collect();
    let (centers, mean_count, std_count) = mean_std_histograms(&hists);
    let (mean_frac, std_frac) = mean_std_histogram_fractions(&hists);
    if !mean_frac.is_empty() {
        let sum: f32 = mean_frac.iter().copied().sum();
        debug_assert!(
            (sum - 1.0).abs() < 1e-3,
            "pairwise mean_frac sum not ~1 (sum={sum})"
        );
    }
    (
        HistSweepStats {
            centers,
            mean_count,
            std_count,
            mean_frac,
            std_frac,
            n: hists.len(),
        },
        n_pairs,
    )
}

fn e2_pairwise_hist_seed_sweep_csv(stats: &HistSweepStats, n_pairs: usize) -> String {
    let mut out =
        String::from("bin_center,mean_frac,std_frac,n_seeds,n_pairs,mean_count,std_count\n");
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_frac[i],
            stats.std_frac[i],
            stats.n,
            n_pairs,
            stats.mean_count[i],
            stats.std_count[i]
        ));
    }
    out
}

fn pairwise_intervals_csv(intervals: &[f32]) -> String {
    let mut out = String::from("interval_semitones\n");
    for &interval in intervals {
        out.push_str(&format!("{interval:.6}\n"));
    }
    out
}

fn pairwise_interval_histogram_csv(
    hist: &[(f32, f32)],
    intervals_total: usize,
    bin_width: f32,
) -> String {
    let total = intervals_total as f32;
    let inv_total = if total > 0.0 { 1.0 / total } else { 0.0 };
    let mut out =
        format!("# source=pairwise_intervals bin_width={bin_width:.3} n_pairs={intervals_total}\n");
    out.push_str("bin_center,count,frac\n");
    for &(center, count) in hist {
        let frac = count * inv_total;
        out.push_str(&format!("{center:.4},{count:.6},{frac:.6}\n"));
    }
    out
}

fn emit_pairwise_interval_dumps_for_condition(
    out_dir: &Path,
    condition: &str,
    runs: &[E2Run],
) -> Result<(), Box<dyn Error>> {
    for run in runs {
        let intervals = pairwise_interval_samples(&run.final_semitones);
        let raw_path = out_dir.join(format!(
            "pairwise_intervals_{condition}_seed{}.csv",
            run.seed
        ));
        write_with_log(raw_path, pairwise_intervals_csv(&intervals))?;

        let hist = histogram_counts_fixed(&intervals, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        let hist_path = out_dir.join(format!(
            "pairwise_interval_hist_{condition}_seed{}.csv",
            run.seed
        ));
        write_with_log(
            hist_path,
            pairwise_interval_histogram_csv(&hist, intervals.len(), E2_PAIRWISE_BIN_ST),
        )?;
    }
    Ok(())
}

fn e2_pairwise_hist_controls_seed_sweep_csv(
    baseline: &HistSweepStats,
    nohill: &HistSweepStats,
    norep: &HistSweepStats,
    n_pairs: usize,
) -> String {
    let mut out = String::from(
        "bin_center,baseline_mean,baseline_std,nohill_mean,nohill_std,norep_mean,norep_std,n_seeds,n_pairs\n",
    );
    let len = baseline
        .centers
        .len()
        .min(baseline.mean_frac.len())
        .min(baseline.std_frac.len())
        .min(nohill.mean_frac.len())
        .min(nohill.std_frac.len())
        .min(norep.mean_frac.len())
        .min(norep.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            baseline.centers[i],
            baseline.mean_frac[i],
            baseline.std_frac[i],
            nohill.mean_frac[i],
            nohill.std_frac[i],
            norep.mean_frac[i],
            norep.std_frac[i],
            baseline.n,
            n_pairs
        ));
    }
    out
}

fn e2_pairwise_hist_controls_seed_sweep_ci95_csv(
    baseline: &HistSweepStats,
    nohill: &HistSweepStats,
    norep: &HistSweepStats,
    n_pairs: usize,
) -> String {
    let mut out = String::from(
        "bin_center,baseline_mean,baseline_ci95,nohill_mean,nohill_ci95,norep_mean,norep_ci95,n_seeds,n_pairs\n",
    );
    let len = baseline
        .centers
        .len()
        .min(baseline.mean_frac.len())
        .min(baseline.std_frac.len())
        .min(nohill.mean_frac.len())
        .min(nohill.std_frac.len())
        .min(norep.mean_frac.len())
        .min(norep.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}\n",
            baseline.centers[i],
            baseline.mean_frac[i],
            ci95_from_std(baseline.std_frac[i], baseline.n),
            nohill.mean_frac[i],
            ci95_from_std(nohill.std_frac[i], nohill.n),
            norep.mean_frac[i],
            ci95_from_std(norep.std_frac[i], norep.n),
            baseline.n,
            n_pairs
        ));
    }
    out
}

fn ci95_half_width(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let std = mean_std_scalar(values).1;
    1.96 * std / (values.len() as f32).sqrt()
}

fn ci95_from_std(std: f32, n: usize) -> f32 {
    if n == 0 {
        return 0.0;
    }
    1.96 * std / (n as f32).sqrt()
}

fn std_series_to_ci95(std: &[f32], n: usize) -> Vec<f32> {
    std.iter().map(|&s| ci95_from_std(s, n)).collect()
}

fn sweep_csv_with_ci95(header: &str, mean: &[f32], std: &[f32], n: usize) -> String {
    let mut out = String::from(header);
    let len = mean.len().min(std.len());
    for i in 0..len {
        out.push_str(&format!(
            "{i},{:.6},{:.6},{:.6},{}\n",
            mean[i],
            std[i],
            ci95_from_std(std[i], n),
            n
        ));
    }
    out
}

fn e2_pairwise_hist_seed_sweep_ci95_csv(stats: &HistSweepStats, n_pairs: usize) -> String {
    let mut out = String::from(
        "bin_center,mean_frac,std_frac,ci95_frac,n_seeds,n_pairs,mean_count,std_count\n",
    );
    let len = stats
        .centers
        .len()
        .min(stats.mean_count.len())
        .min(stats.std_count.len())
        .min(stats.mean_frac.len())
        .min(stats.std_frac.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{:.6},{},{},{:.6},{:.6}\n",
            stats.centers[i],
            stats.mean_frac[i],
            stats.std_frac[i],
            ci95_from_std(stats.std_frac[i], stats.n),
            stats.n,
            n_pairs,
            stats.mean_count[i],
            stats.std_count[i]
        ));
    }
    out
}

fn diversity_summary_ci95_csv(rows: &[DiversityRow]) -> String {
    let mut by_cond: std::collections::HashMap<&'static str, Vec<&DiversityRow>> =
        std::collections::HashMap::new();
    for row in rows {
        by_cond.entry(row.condition).or_default().push(row);
    }
    let mut out = String::from("cond,metric,mean,std,ci95,n\n");
    for cond in ["baseline", "nohill", "nocrowd"] {
        let rows = by_cond.get(cond).cloned().unwrap_or_default();
        let unique_bins: Vec<f32> = rows.iter().map(|r| r.metrics.unique_bins as f32).collect();
        let nn_mean: Vec<f32> = rows.iter().map(|r| r.metrics.nn_mean).collect();
        let semitone_var: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_var).collect();
        let semitone_mad: Vec<f32> = rows.iter().map(|r| r.metrics.semitone_mad).collect();
        for (metric, values) in [
            ("unique_bins", &unique_bins),
            ("nn_mean", &nn_mean),
            ("semitone_var", &semitone_var),
            ("semitone_mad", &semitone_mad),
        ] {
            let (mean, std) = mean_std_scalar(values);
            out.push_str(&format!(
                "{cond},{metric},{mean:.6},{std:.6},{:.6},{}\n",
                ci95_half_width(values),
                values.len()
            ));
        }
    }
    out
}

fn interval_distance_mod_12(value: f32, target: f32) -> f32 {
    (value - target)
        .abs()
        .min((value - (target + 12.0)).abs())
        .min((value - (target - 12.0)).abs())
}

fn consonant_mass_for_intervals(intervals: &[f32], targets: &[f32], window_st: f32) -> f32 {
    if intervals.is_empty() {
        return 0.0;
    }
    let hits = intervals
        .iter()
        .filter(|&&interval| {
            targets
                .iter()
                .any(|&target| interval_distance_mod_12(interval, target) <= window_st + 1e-6)
        })
        .count();
    hits as f32 / intervals.len() as f32
}

fn consonant_mass_rows_for_condition(
    condition: &'static str,
    runs: &[E2Run],
) -> Vec<ConsonantMassRow> {
    runs.iter()
        .map(|run| {
            let intervals = pairwise_interval_samples(&run.final_semitones);
            ConsonantMassRow {
                condition,
                seed: run.seed,
                mass_core: consonant_mass_for_intervals(
                    &intervals,
                    &E2_CONSONANT_TARGETS_CORE,
                    E2_CONSONANT_WINDOW_ST,
                ),
                mass_extended: consonant_mass_for_intervals(
                    &intervals,
                    &E2_CONSONANT_TARGETS_EXTENDED,
                    E2_CONSONANT_WINDOW_ST,
                ),
            }
        })
        .collect()
}

fn consonant_mass_values(
    rows: &[ConsonantMassRow],
    condition: &str,
    select: fn(&ConsonantMassRow) -> f32,
) -> Vec<f32> {
    rows.iter()
        .filter(|row| row.condition == condition)
        .map(select)
        .collect()
}

fn consonant_mass_core(row: &ConsonantMassRow) -> f32 {
    row.mass_core
}

fn consonant_mass_extended(row: &ConsonantMassRow) -> f32 {
    row.mass_extended
}

fn n_choose_k_capped(n: usize, k: usize, cap: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    if k == 0 {
        return 1;
    }
    let cap_u128 = cap as u128;
    let capped = cap.saturating_add(1);
    let mut value = 1u128;
    for i in 1..=k {
        value = value * (n - k + i) as u128 / i as u128;
        if value > cap_u128 {
            return capped;
        }
    }
    value as u64
}

fn exact_permutation_pvalue_mean_diff(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 1.0;
    }
    let mean_a = a.iter().copied().sum::<f32>() / a.len() as f32;
    let mean_b = b.iter().copied().sum::<f32>() / b.len() as f32;
    let obs_abs = (mean_a - mean_b).abs();
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let n_total = pooled.len();
    let n_a = a.len();
    if n_a == 0 || n_a >= n_total {
        return 1.0;
    }
    let pooled_sum = pooled.iter().copied().sum::<f32>();
    #[allow(clippy::too_many_arguments)]
    fn recurse(
        values: &[f32],
        n_a: usize,
        start: usize,
        picked: usize,
        sum_a: f32,
        pooled_sum: f32,
        obs_abs: f32,
        extreme: &mut usize,
        total: &mut usize,
    ) {
        if picked == n_a {
            let n_b = values.len() - n_a;
            if n_b == 0 {
                return;
            }
            let mean_a = sum_a / n_a as f32;
            let mean_b = (pooled_sum - sum_a) / n_b as f32;
            if (mean_a - mean_b).abs() + 1e-8 >= obs_abs {
                *extreme += 1;
            }
            *total += 1;
            return;
        }
        let remaining_needed = n_a - picked;
        let upper = values.len().saturating_sub(remaining_needed);
        for i in start..=upper {
            recurse(
                values,
                n_a,
                i + 1,
                picked + 1,
                sum_a + values[i],
                pooled_sum,
                obs_abs,
                extreme,
                total,
            );
        }
    }
    let mut extreme = 0usize;
    let mut total = 0usize;
    recurse(
        &pooled,
        n_a,
        0,
        0,
        0.0,
        pooled_sum,
        obs_abs,
        &mut extreme,
        &mut total,
    );
    if total == 0 {
        1.0
    } else {
        (extreme as f32 + 1.0) / (total as f32 + 1.0)
    }
}

fn permutation_pvalue_mean_diff(
    a: &[f32],
    b: &[f32],
    max_exact: u64,
    mc_iters: usize,
    rng_seed: u64,
) -> (f32, &'static str, u64) {
    if a.is_empty() || b.is_empty() {
        return (1.0, "exact", 0);
    }
    let n_total = a.len() + b.len();
    let n_a = a.len();
    let n_b = b.len();
    if n_a == 0 || n_b == 0 {
        return (1.0, "exact", 0);
    }

    let n_combos = n_choose_k_capped(n_total, n_a.min(n_b), max_exact);
    if n_combos <= max_exact {
        return (exact_permutation_pvalue_mean_diff(a, b), "exact", n_combos);
    }

    let mean_a = a.iter().copied().sum::<f32>() / n_a as f32;
    let mean_b = b.iter().copied().sum::<f32>() / n_b as f32;
    let obs_abs = (mean_a - mean_b).abs();
    let mut pooled = Vec::with_capacity(n_total);
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let pooled_sum = pooled.iter().copied().sum::<f32>();

    let iters = mc_iters.max(1) as u64;
    let mut rng = StdRng::seed_from_u64(rng_seed);
    let mut picks: Vec<usize> = (0..n_total).collect();
    let mut extreme = 0u64;
    for _ in 0..iters {
        picks.shuffle(&mut rng);
        let sum_a: f32 = picks[..n_a].iter().map(|&idx| pooled[idx]).sum();
        let perm_mean_a = sum_a / n_a as f32;
        let perm_mean_b = (pooled_sum - sum_a) / n_b as f32;
        if (perm_mean_a - perm_mean_b).abs() + 1e-8 >= obs_abs {
            extreme += 1;
        }
    }
    let p = (extreme as f32 + 1.0) / (iters as f32 + 1.0);
    (p, "mc", iters)
}

/// Sign-flip permutation test for H0: mean(x) = 0 (one-sample).
/// For n ≤ 20, uses exact enumeration (2^n sign patterns).
/// For n > 20, falls back to Monte Carlo with `mc_iters` iterations.
/// Returns (p_value, method_label).
#[cfg(test)]
fn permutation_pvalue_one_sample(x: &[f32], mc_iters: usize, seed: u64) -> (f32, &'static str) {
    if x.is_empty() {
        return (1.0, "exact");
    }
    let n = x.len();
    let obs_mean = x.iter().copied().sum::<f32>() / n as f32;
    let obs_abs = obs_mean.abs();

    if n <= 20 {
        // Exact enumeration: iterate over all 2^n sign patterns
        let total = 1u64 << n;
        let mut extreme = 0u64;
        for mask in 0..total {
            let mut sum = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                if mask & (1u64 << i) != 0 {
                    sum += xi;
                } else {
                    sum -= xi;
                }
            }
            if (sum / n as f32).abs() + 1e-8 >= obs_abs {
                extreme += 1;
            }
        }
        let p = extreme as f32 / total as f32;
        (p, "exact")
    } else {
        // Monte Carlo sign-flip
        let iters = mc_iters.max(1) as u64;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut extreme = 0u64;
        for _ in 0..iters {
            let mut sum = 0.0f32;
            for &xi in x {
                if rng.random_bool(0.5) {
                    sum += xi;
                } else {
                    sum -= xi;
                }
            }
            if (sum / n as f32).abs() + 1e-8 >= obs_abs {
                extreme += 1;
            }
        }
        let p = (extreme as f32 + 1.0) / (iters as f32 + 1.0);
        (p, "mc")
    }
}

fn cliffs_delta(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let mut greater = 0usize;
    let mut less = 0usize;
    for &x in a {
        for &y in b {
            if x > y + 1e-8 {
                greater += 1;
            } else if x + 1e-8 < y {
                less += 1;
            }
        }
    }
    (greater as f32 - less as f32) / (a.len() * b.len()) as f32
}

fn consonant_mass_by_seed_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = format!(
        "# window_st={:.3} targets_core=6:5|5:4|3:2 targets_extended=6:5|5:4|4:3|3:2|8:5|5:3\n",
        E2_CONSONANT_WINDOW_ST
    );
    out.push_str("seed,cond,mass_core_347,mass_extended_345789\n");
    for row in rows {
        out.push_str(&format!(
            "{},{},{:.6},{:.6}\n",
            row.seed, row.condition, row.mass_core, row.mass_extended
        ));
    }
    out
}

fn consonant_mass_summary_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = String::from("metric,cond,mean,std,ci95,n\n");
    for (metric, select) in [
        (
            "core_347",
            consonant_mass_core as fn(&ConsonantMassRow) -> f32,
        ),
        (
            "extended_345789",
            consonant_mass_extended as fn(&ConsonantMassRow) -> f32,
        ),
    ] {
        for cond in ["baseline", "nohill", "nocrowd"] {
            let values = consonant_mass_values(rows, cond, select);
            let (mean, std) = mean_std_scalar(&values);
            out.push_str(&format!(
                "{metric},{cond},{mean:.6},{std:.6},{:.6},{}\n",
                ci95_half_width(&values),
                values.len()
            ));
        }
    }
    out
}

fn consonant_mass_stats_csv(rows: &[ConsonantMassRow]) -> String {
    let mut out = String::from(
        "metric,comparison,mean_diff,p_perm,method,n_perm,cliffs_delta,n_baseline,n_control\n",
    );
    for (metric_idx, (metric, select)) in [
        (
            "core_347",
            consonant_mass_core as fn(&ConsonantMassRow) -> f32,
        ),
        (
            "extended_345789",
            consonant_mass_extended as fn(&ConsonantMassRow) -> f32,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let baseline = consonant_mass_values(rows, "baseline", select);
        for (comp_idx, (comp_label, comp_cond)) in [
            ("baseline_vs_nohill", "nohill"),
            ("baseline_vs_norep", "nocrowd"),
        ]
        .into_iter()
        .enumerate()
        {
            let control = consonant_mass_values(rows, comp_cond, select);
            let mean_diff = mean_std_scalar(&baseline).0 - mean_std_scalar(&control).0;
            let rng_seed = E2_PERM_MC_SEED ^ ((metric_idx as u64) << 32) ^ comp_idx as u64;
            let (p_perm, method, n_perm) = permutation_pvalue_mean_diff(
                &baseline,
                &control,
                E2_PERM_MAX_EXACT_COMBOS,
                E2_PERM_MC_ITERS,
                rng_seed,
            );
            let delta = cliffs_delta(&baseline, &control);
            out.push_str(&format!(
                "{metric},{comp_label},{mean_diff:.6},{p_perm:.6},{method},{n_perm},{delta:.6},{},{}\n",
                baseline.len(),
                control.len()
            ));
        }
    }
    out
}

fn fold_hist_abs_semitones(
    centers: &[f32],
    mean_frac: &[f32],
    std_frac: &[f32],
    bin_width: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = centers.len().min(mean_frac.len()).min(std_frac.len());
    let mut by_abs: std::collections::BTreeMap<i32, (f32, f32)> = std::collections::BTreeMap::new();
    for i in 0..len {
        let key = (centers[i].abs() / bin_width).round() as i32;
        let entry = by_abs.entry(key).or_insert((0.0, 0.0));
        entry.0 += mean_frac[i];
        entry.1 += std_frac[i] * std_frac[i];
    }
    let mut out_centers = Vec::with_capacity(by_abs.len());
    let mut out_mean = Vec::with_capacity(by_abs.len());
    let mut out_std = Vec::with_capacity(by_abs.len());
    for (key, (mean_sum, std_sq_sum)) in by_abs {
        out_centers.push(key as f32 * bin_width);
        out_mean.push(mean_sum);
        out_std.push(std_sq_sum.max(0.0).sqrt());
    }
    (out_centers, out_mean, out_std)
}

fn rebin_histogram_series(
    centers: &[f32],
    mean_frac: &[f32],
    std_frac: &[f32],
    from_bin_width: f32,
    to_bin_width: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let len = centers.len().min(mean_frac.len()).min(std_frac.len());
    if len == 0 || to_bin_width <= from_bin_width + 1e-6 {
        return (
            centers[..len].to_vec(),
            mean_frac[..len].to_vec(),
            std_frac[..len].to_vec(),
        );
    }
    let group = (to_bin_width / from_bin_width).round().max(1.0) as usize;
    if group <= 1 {
        return (
            centers[..len].to_vec(),
            mean_frac[..len].to_vec(),
            std_frac[..len].to_vec(),
        );
    }

    let mut out_centers = Vec::new();
    let mut out_mean = Vec::new();
    let mut out_std = Vec::new();
    let half_from = 0.5 * from_bin_width;
    let half_to = 0.5 * to_bin_width;
    for start in (0..len).step_by(group) {
        let end = (start + group).min(len);
        if end - start < group {
            break;
        }
        let bin_start = centers[start] - half_from;
        let center = bin_start + half_to;
        let mean_sum: f32 = mean_frac[start..end].iter().sum();
        let std_sum_sq: f32 = std_frac[start..end].iter().map(|v| v * v).sum();
        out_centers.push(center);
        out_mean.push(mean_sum);
        out_std.push(std_sum_sq.sqrt());
    }
    (out_centers, out_mean, out_std)
}

fn snap_to_hist_bin_center(value: f32, bin_width: f32) -> f32 {
    if !value.is_finite() || !bin_width.is_finite() || bin_width <= 0.0 {
        return value;
    }
    let idx = (value / bin_width - 0.5).round();
    (idx + 0.5) * bin_width
}

fn folded_hist_csv(centers: &[f32], mean: &[f32], std: &[f32], n_seeds: usize) -> String {
    let mut out = String::from("abs_semitones,mean_frac,std_frac,n_seeds\n");
    let len = centers.len().min(mean.len()).min(std.len());
    for i in 0..len {
        out.push_str(&format!(
            "{:.4},{:.6},{:.6},{}\n",
            centers[i], mean[i], std[i], n_seeds
        ));
    }
    out
}

fn render_hist_mean_std(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    bin_width: f32,
    y_desc: &str,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let max = centers.last().copied().unwrap_or(0.0) + 0.5 * bin_width;
    let mut y_max = 0.0f32;
    for i in 0..mean.len().min(std.len()) {
        y_max = y_max.max(mean[i] + std[i]);
    }
    y_max = y_max.max(1.0);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc(y_desc)
        .x_labels(25)
        .draw()?;

    let half = bin_width * 0.45;
    for i in 0..centers.len().min(mean.len()).min(std.len()) {
        let center = centers[i];
        let mean_val = mean[i];
        let std_val = std[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(center - half, 0.0), (center + half, mean_val)],
            PAL_H.mix(0.6).filled(),
        )))?;
        let y0 = (mean_val - std_val).max(0.0);
        let y1 = mean_val + std_val;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.6),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn e2_condition_display(condition: &str) -> &'static str {
    match condition {
        "baseline" => "local-search",
        "nohill" => "random-walk",
        "nocrowd" => "no-crowding",
        _ => "unknown",
    }
}

fn e2_condition_display_short(condition: &str) -> &'static str {
    match condition {
        "baseline" => "search",
        "nohill" => "random",
        "nocrowd" => "no-crowd",
        _ => "unknown",
    }
}

fn e2_condition_color(condition: &str) -> RGBColor {
    match condition {
        "baseline" => PAL_H,
        "nohill" => PAL_R,
        "nocrowd" => PAL_CD,
        _ => PAL_C,
    }
}

fn draw_e2_interval_guides_with_windows<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    y_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    for &target in &E2_INTERVAL_GUIDE_TARGETS_CORE {
        chart.draw_series(std::iter::once(Rectangle::new(
            [
                (target - E2_CONSONANT_WINDOW_ST, 0.0),
                (target + E2_CONSONANT_WINDOW_ST, y_max),
            ],
            RGBColor(240, 170, 60).mix(0.12).filled(),
        )))?;
    }
    for &x in &E2_INTERVAL_GUIDE_STEPS {
        let is_core = E2_INTERVAL_GUIDE_TARGETS_CORE
            .iter()
            .any(|&core| (core - x).abs() < 1e-6);
        let style = if is_core {
            ShapeStyle::from(&BLACK.mix(0.6)).stroke_width(2)
        } else {
            ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(1)
        };
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y_max)],
            style,
        )))?;
    }
    Ok(())
}

fn draw_e2_interval_guides_cents<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    y_max: f32,
    snap_bin_cents: Option<f32>,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let st2c = 100.0f32;
    for &target in &E2_INTERVAL_GUIDE_TARGETS_CORE {
        let tc_raw = target * st2c;
        let tc = snap_bin_cents
            .map(|bw| snap_to_hist_bin_center(tc_raw, bw))
            .unwrap_or(tc_raw);
        let wc = E2_CONSONANT_WINDOW_ST * st2c;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(tc - wc, 0.0), (tc + wc, y_max)],
            RGBColor(240, 170, 60).mix(0.12).filled(),
        )))?;
    }
    for &x in &E2_INTERVAL_GUIDE_STEPS {
        let xc_raw = x * st2c;
        let xc = snap_bin_cents
            .map(|bw| snap_to_hist_bin_center(xc_raw, bw))
            .unwrap_or(xc_raw);
        let is_core = E2_INTERVAL_GUIDE_TARGETS_CORE
            .iter()
            .any(|&core| (core - x).abs() < 1e-6);
        let style = if is_core {
            ShapeStyle::from(&BLACK.mix(0.6)).stroke_width(2)
        } else {
            ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(1)
        };
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(xc, 0.0), (xc, y_max)],
            style,
        )))?;
    }
    Ok(())
}

fn y_max_from_mean_err(mean: &[f32], err: &[f32]) -> f32 {
    let len = mean.len().min(err.len());
    let mut y_peak = 0.0f32;
    for i in 0..len {
        y_peak = y_peak.max(mean[i] + err[i].max(0.0));
    }
    (1.15 * y_peak.max(1e-4)).max(1e-4)
}

fn e2_burn_band_x(
    burn_in: usize,
    phase_switch_step: Option<usize>,
    x_min: usize,
    x_hi: usize,
) -> Option<(f32, f32)> {
    if burn_in == 0 {
        return None;
    }
    let start = phase_switch_step.unwrap_or(x_min).max(x_min);
    let end = start.saturating_add(burn_in).min(x_hi);
    if end > start {
        Some((start as f32, end as f32))
    } else {
        None
    }
}

fn render_pairwise_histogram_paper(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean_frac: &[f32],
    std_frac: &[f32],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers.len().min(mean_frac.len()).min(std_frac.len());
    if len == 0 {
        return Ok(());
    }
    let x_min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let x_max = centers.get(len - 1).copied().unwrap_or(12.0) + 0.5 * bin_width;
    let y_max = y_max_from_mean_err(&mean_frac[..len], &std_frac[..len]);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    draw_e2_interval_guides_with_windows(&mut chart, y_max)?;

    let half = bin_width * 0.45;
    for i in 0..len {
        let x = centers[i];
        let m = mean_frac[i];
        let s = std_frac[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - half, 0.0), (x + half, m)],
            PAL_H.mix(0.65).filled(),
        )))?;
        let y0 = (m - s).max(0.0);
        let y1 = (m + s).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y0), (x, y1)],
            BLACK.mix(0.6),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_pairwise_histogram_controls_overlay(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    baseline: &[f32],
    nohill: &[f32],
    norep: &[f32],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers
        .len()
        .min(baseline.len())
        .min(nohill.len())
        .min(norep.len());
    if len == 0 {
        return Ok(());
    }
    let bin_width = if len > 1 {
        (centers[1] - centers[0]).abs().max(1e-6)
    } else {
        E2_PAIRWISE_BIN_ST
    };
    let x_min = centers[0] - 0.5 * bin_width;
    let x_max = centers[len - 1] + 0.5 * bin_width;
    let mut y_peak = 0.0f32;
    for i in 0..len {
        y_peak = y_peak.max(baseline[i]).max(nohill[i]).max(norep[i]);
    }
    let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    draw_e2_interval_guides_with_windows(&mut chart, y_max)?;

    for (condition, values, color) in [
        ("baseline", baseline, PAL_H),
        ("nohill", nohill, PAL_R),
        ("nocrowd", norep, PAL_CD),
    ] {
        let line = centers
            .iter()
            .take(len)
            .copied()
            .zip(values.iter().take(len).copied());
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(e2_condition_display(condition))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_hist_mean_std_fraction_auto_y(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    bin_width: f32,
    x_desc: &str,
    guide_lines: &[f32],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() {
        return Ok(());
    }
    let len = centers.len().min(mean.len()).min(std.len());
    if len == 0 {
        return Ok(());
    }
    let x_min = centers[0] - 0.5 * bin_width;
    let x_max = centers[len - 1] + 0.5 * bin_width;
    let y_max = y_max_from_mean_err(&mean[..len], &std[..len]);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    for &x in guide_lines {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, 0.0), (x, y_max)],
            BLACK.mix(0.25),
        )))?;
    }

    let half = bin_width * 0.45;
    for i in 0..len {
        let x = centers[i];
        let m = mean[i];
        let s = std[i];
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x - half, 0.0), (x + half, m)],
            PAL_H.mix(0.65).filled(),
        )))?;
        let y0 = (m - s).max(0.0);
        let y1 = (m + s).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y0), (x, y1)],
            BLACK.mix(0.6),
        )))?;
    }
    root.present()?;
    Ok(())
}

fn render_hist_controls_fraction(
    out_path: &Path,
    caption: &str,
    centers: &[f32],
    series: &[(&str, &[f32], RGBColor)],
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() || series.is_empty() {
        return Ok(());
    }
    let bin_width = if centers.len() > 1 {
        (centers[1] - centers[0]).abs()
    } else {
        0.5
    };
    let min = centers.first().copied().unwrap_or(0.0) - 0.5 * bin_width;
    let max = centers.last().copied().unwrap_or(0.0) + 0.5 * bin_width;
    let mut y_max = 0.0f32;
    for (_, values, _) in series {
        for &v in *values {
            y_max = y_max.max(v);
        }
    }
    y_max = y_max.max(1e-3);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("mean fraction")
        .x_labels(25)
        .draw()?;

    for &(label, values, color) in series {
        let line = centers.iter().copied().zip(values.iter().copied());
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn diversity_values_for_condition(
    rows: &[DiversityRow],
    condition: &str,
    select: fn(&DiversityMetrics) -> f32,
) -> Vec<f32> {
    rows.iter()
        .filter(|row| row.condition == condition)
        .map(|row| select(&row.metrics))
        .collect()
}

fn draw_diversity_metric_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
) -> Result<(), Box<dyn Error>> {
    draw_diversity_metric_panel_impl(area, caption, y_desc, rows, select, 22, 16, 18)
}

#[allow(dead_code)]
fn draw_diversity_metric_panel_large(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
) -> Result<(), Box<dyn Error>> {
    draw_diversity_metric_panel_impl(area, caption, y_desc, rows, select, 32, 20, 24)
}

#[allow(dead_code)]
fn draw_diversity_metric_panel_large_for_conditions(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
    conditions: &[&str],
) -> Result<(), Box<dyn Error>> {
    draw_diversity_metric_panel_impl_for_conditions(
        area, caption, y_desc, rows, select, 32, 20, 24, conditions,
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_diversity_metric_panel_impl(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    draw_diversity_metric_panel_impl_for_conditions(
        area,
        caption,
        y_desc,
        rows,
        select,
        caption_size,
        label_size,
        axis_desc_size,
        &["baseline", "nohill", "nocrowd"],
    )
}

#[allow(clippy::too_many_arguments)]
fn draw_diversity_metric_panel_impl_for_conditions(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    rows: &[DiversityRow],
    select: fn(&DiversityMetrics) -> f32,
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
    conditions: &[&str],
) -> Result<(), Box<dyn Error>> {
    if conditions.is_empty() {
        return Ok(());
    }
    let mut means = Vec::with_capacity(conditions.len());
    let mut ci95 = Vec::with_capacity(conditions.len());
    let mut y_max = 0.0f32;
    for &cond in conditions {
        let values = diversity_values_for_condition(rows, cond, select);
        let mean = mean_std_scalar(&values).0;
        let ci = ci95_half_width(&values);
        y_max = y_max.max(mean + ci);
        means.push(mean);
        ci95.push(ci);
    }
    y_max = (1.15 * y_max.max(1e-4)).max(1e-4);
    let x_hi = (conditions.len().saturating_sub(1)) as f32 + 0.5;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", caption_size))
        .margin(8)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(-0.5f32..x_hi, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc(y_desc)
        .x_labels(conditions.len())
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if idx >= 0 && (idx as usize) < conditions.len() {
                e2_condition_display_short(conditions[idx as usize]).to_string()
            } else {
                String::new()
            }
        })
        .label_style(("sans-serif", label_size).into_font())
        .axis_desc_style(("sans-serif", axis_desc_size).into_font())
        .draw()?;

    for (i, &cond) in conditions.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        let color = e2_condition_color(cond);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, means[i])],
            color.mix(0.7).filled(),
        )))?;
        let y0 = (means[i] - ci95[i]).max(0.0);
        let y1 = (means[i] + ci95[i]).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.7),
        )))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_e2_timeseries_controls_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    baseline_mean: &[f32],
    baseline_std: &[f32],
    nohill_mean: &[f32],
    nohill_std: &[f32],
    norep_mean: &[f32],
    norep_std: &[f32],
    burn_in: usize,
    phase_switch_step: Option<usize>,
    x_min: usize,
    x_max: usize,
    draw_legend: bool,
) -> Result<(), Box<dyn Error>> {
    let x_hi = x_max.max(x_min + 1);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (mean, std) in [
        (baseline_mean, baseline_std),
        (nohill_mean, nohill_std),
        (norep_mean, norep_std),
    ] {
        let len = mean.len().min(std.len());
        for i in x_min..len.min(x_hi + 1) {
            let lo = mean[i] - std[i];
            let hi = mean[i] + std[i];
            if lo.is_finite() && hi.is_finite() {
                y_min = y_min.min(lo);
                y_max = y_max.max(hi);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-3);
    y_min = (y_min - pad).max(0.0);
    y_max += pad;

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(75)
        .build_cartesian_2d(x_min as f32..x_hi as f32, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("mean LOO C_score")
        .x_labels(9)
        .x_label_formatter(&|v| format!("{}", v.round() as i64))
        .label_style(("sans-serif", 28).into_font())
        .axis_desc_style(("sans-serif", 28).into_font())
        .draw()?;

    if let Some((burn_x0, burn_x1)) = e2_burn_band_x(burn_in, phase_switch_step, x_min, x_hi) {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(burn_x0, y_min), (burn_x1, y_max)],
            RGBColor(180, 180, 180).mix(0.15).filled(),
        )))?;
    }
    if let Some(step) = phase_switch_step
        && step >= x_min
        && step <= x_hi
    {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(step as f32, y_min), (step as f32, y_max)],
            ShapeStyle::from(&BLACK.mix(0.55)).stroke_width(2),
        )))?;
        let y_text = y_max - 0.05 * (y_max - y_min);
        chart.draw_series(std::iter::once(Text::new(
            "phase switch".to_string(),
            (step as f32, y_text),
            ("sans-serif", 22).into_font().color(&BLACK),
        )))?;
    }

    for (condition, mean, std, color) in [
        (
            "baseline",
            baseline_mean,
            baseline_std,
            e2_condition_color("baseline"),
        ),
        (
            "nohill",
            nohill_mean,
            nohill_std,
            e2_condition_color("nohill"),
        ),
        (
            "nocrowd",
            norep_mean,
            norep_std,
            e2_condition_color("nocrowd"),
        ),
    ] {
        let len = mean.len().min(std.len());
        if len <= x_min {
            continue;
        }
        let end = len.min(x_hi + 1);
        let mut band: Vec<(f32, f32)> = Vec::with_capacity((end - x_min) * 2);
        for i in x_min..end {
            band.push((i as f32, mean[i] + std[i]));
        }
        for i in (x_min..end).rev() {
            band.push((i as f32, mean[i] - std[i]));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band,
            color.mix(0.15).filled(),
        )))?;
        let line = (x_min..end).map(|i| (i as f32, mean[i]));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(e2_condition_display(condition))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
    }
    if draw_legend {
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 28).into_font())
            .draw()?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_e2_timeseries_pair_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    baseline_mean: &[f32],
    baseline_std: &[f32],
    nohill_mean: &[f32],
    nohill_std: &[f32],
    burn_in: usize,
    phase_switch_step: Option<usize>,
    x_min: usize,
    x_max: usize,
    draw_legend: bool,
    y_override: Option<(f32, f32)>,
) -> Result<(), Box<dyn Error>> {
    let x_hi = x_max.max(x_min + 1);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (mean, std) in [(baseline_mean, baseline_std), (nohill_mean, nohill_std)] {
        let len = mean.len().min(std.len());
        for i in x_min..len.min(x_hi + 1) {
            let lo = mean[i] - std[i];
            let hi = mean[i] + std[i];
            if lo.is_finite() && hi.is_finite() {
                y_min = y_min.min(lo);
                y_max = y_max.max(hi);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    if let Some((fixed_lo, fixed_hi)) = y_override {
        y_min = fixed_lo;
        y_max = fixed_hi;
    } else {
        let pad = ((y_max - y_min).abs() * 0.1).max(1e-3);
        y_min -= pad;
        y_max += pad;
    }

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(x_min as f32..x_hi as f32, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .x_labels(9)
        .x_label_formatter(&|v| format!("{}", v.round() as i64))
        .label_style(("sans-serif", 20).into_font())
        .axis_desc_style(("sans-serif", 24).into_font())
        .draw()?;

    if let Some((burn_x0, burn_x1)) = e2_burn_band_x(burn_in, phase_switch_step, x_min, x_hi) {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(burn_x0, y_min), (burn_x1, y_max)],
            RGBColor(180, 180, 180).mix(0.15).filled(),
        )))?;
    }
    if let Some(step) = phase_switch_step
        && step >= x_min
        && step <= x_hi
    {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(step as f32, y_min), (step as f32, y_max)],
            ShapeStyle::from(&BLACK.mix(0.55)).stroke_width(2),
        )))?;
        let y_text = y_max - 0.05 * (y_max - y_min);
        chart.draw_series(std::iter::once(Text::new(
            "phase switch".to_string(),
            (step as f32, y_text),
            ("sans-serif", 22).into_font().color(&BLACK),
        )))?;
    }

    for (condition, mean, std, color) in [
        (
            "baseline",
            baseline_mean,
            baseline_std,
            e2_condition_color("baseline"),
        ),
        (
            "nohill",
            nohill_mean,
            nohill_std,
            e2_condition_color("nohill"),
        ),
    ] {
        let len = mean.len().min(std.len());
        if len <= x_min {
            continue;
        }
        let end = len.min(x_hi + 1);
        let mut band: Vec<(f32, f32)> = Vec::with_capacity((end - x_min) * 2);
        for i in x_min..end {
            band.push((i as f32, mean[i] + std[i]));
        }
        for i in (x_min..end).rev() {
            band.push((i as f32, mean[i] - std[i]));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band,
            color.mix(0.15).filled(),
        )))?;
        let line = (x_min..end).map(|i| (i as f32, mean[i]));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(e2_condition_display(condition))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
    }
    if draw_legend {
        chart
            .configure_series_labels()
            .position(SeriesLabelPosition::UpperLeft)
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 22).into_font())
            .draw()?;
    }
    Ok(())
}

fn draw_trajectory_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    trajectories: &[Vec<f32>],
    burn_in: usize,
    phase_switch_step: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    if trajectories.is_empty() {
        return Ok(());
    }
    let steps = trajectories.iter().map(|tr| tr.len()).max().unwrap_or(0);
    if steps == 0 {
        return Ok(());
    }
    let y_min = -2500.0f32;
    let y_max = 2500.0f32;
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(
            0.0f32..(steps.saturating_sub(1) as f32).max(1.0),
            y_min..y_max,
        )?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("cents")
        .x_labels(9)
        .x_label_formatter(&|v| format!("{}", v.round() as i64))
        .y_labels(5)
        .y_label_formatter(&|v| format!("{}", *v as i32))
        .label_style(("sans-serif", 20).into_font())
        .axis_desc_style(("sans-serif", 24).into_font())
        .draw()?;
    if let Some((burn_x0, burn_x1)) =
        e2_burn_band_x(burn_in, phase_switch_step, 0, steps.saturating_sub(1))
    {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(burn_x0, y_min), (burn_x1, y_max)],
            RGBColor(180, 180, 180).mix(0.15).filled(),
        )))?;
    }
    if let Some(step) = phase_switch_step
        && step <= steps.saturating_sub(1)
    {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(step as f32, y_min), (step as f32, y_max)],
            ShapeStyle::from(&BLACK.mix(0.55)).stroke_width(2),
        )))?;
        let y_text = y_max - 0.05 * (y_max - y_min);
        chart.draw_series(std::iter::once(Text::new(
            "phase switch".to_string(),
            (step as f32, y_text),
            ("sans-serif", 22).into_font().color(&BLACK),
        )))?;
    }
    // Muted palette cycling through the paper's base colors
    let traj_colors: &[RGBColor] = &[PAL_H, PAL_R, PAL_C, PAL_CD];
    let n_colors = traj_colors.len();
    for (i, trace) in trajectories.iter().enumerate() {
        if trace.is_empty() {
            continue;
        }
        let base = traj_colors[i % n_colors];
        // Vary opacity to distinguish overlapping agents
        let alpha = 0.4
            + 0.4
                * ((i / n_colors) as f32 / (trajectories.len() as f32 / n_colors as f32).max(1.0))
                    .min(1.0);
        let color = base.mix(alpha as f64);
        let line = trace
            .iter()
            .enumerate()
            .map(|(step, &v)| (step as f32, v * 100.0));
        chart.draw_series(LineSeries::new(
            line,
            ShapeStyle::from(&color).stroke_width(2),
        ))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_e2_mean_c_level_annotated(
    out_path: &Path,
    baseline_mean: &[f32],
    baseline_std: &[f32],
    nohill_mean: &[f32],
    nohill_std: &[f32],
    norep_mean: &[f32],
    norep_std: &[f32],
    burn_in: usize,
    phase_switch_step: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let len = baseline_mean
        .len()
        .min(baseline_std.len())
        .min(nohill_mean.len())
        .min(nohill_std.len())
        .min(norep_mean.len())
        .min(norep_std.len());
    if len == 0 {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1200, 1000)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    draw_e2_timeseries_controls_panel(
        &panels[0],
        "Mean C_level01 (annotated controls, 95% CI)",
        baseline_mean,
        baseline_std,
        nohill_mean,
        nohill_std,
        norep_mean,
        norep_std,
        burn_in,
        phase_switch_step,
        0,
        len.saturating_sub(1),
        true,
    )?;
    let zoom_start = phase_switch_step
        .unwrap_or(burn_in)
        .min(len.saturating_sub(1));
    draw_e2_timeseries_controls_panel(
        &panels[1],
        "Post-switch zoom (95% CI)",
        baseline_mean,
        baseline_std,
        nohill_mean,
        nohill_std,
        norep_mean,
        norep_std,
        burn_in,
        phase_switch_step,
        zoom_start,
        len.saturating_sub(1),
        false,
    )?;
    root.present()?;
    Ok(())
}

fn draw_consonant_mass_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    subtitle: &str,
    rows: &[ConsonantMassRow],
    select: fn(&ConsonantMassRow) -> f32,
) -> Result<(), Box<dyn Error>> {
    draw_consonant_mass_panel_impl(area, caption, subtitle, rows, select, 22, 16, 18)
}

#[allow(clippy::too_many_arguments)]
fn draw_consonant_mass_panel_impl(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    subtitle: &str,
    rows: &[ConsonantMassRow],
    select: fn(&ConsonantMassRow) -> f32,
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    let conditions = ["baseline", "nohill", "nocrowd"];
    let mut means = [0.0f32; 3];
    let mut ci95 = [0.0f32; 3];
    for (i, cond) in conditions.iter().enumerate() {
        let values = consonant_mass_values(rows, cond, select);
        means[i] = mean_std_scalar(&values).0;
        ci95[i] = ci95_half_width(&values);
    }
    let mut y_max = 0.0f32;
    for i in 0..3 {
        y_max = y_max.max(means[i] + ci95[i]);
    }
    y_max = (1.15 * y_max.max(1e-4)).max(1e-4);

    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", caption_size))
        .margin(8)
        .x_label_area_size(40)
        .y_label_area_size(55)
        .build_cartesian_2d(-0.5f32..2.5f32, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc("consonant mass")
        .x_labels(3)
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if (0..=2).contains(&idx) {
                e2_condition_display_short(conditions[idx as usize]).to_string()
            } else {
                String::new()
            }
        })
        .label_style(("sans-serif", label_size).into_font())
        .axis_desc_style(("sans-serif", axis_desc_size).into_font())
        .draw()?;

    if !subtitle.is_empty() {
        let sub_color = BLACK.mix(0.55);
        let sub_style = TextStyle::from(("sans-serif", label_size).into_font()).color(&sub_color);
        area.draw_text(subtitle, &sub_style, (60i32, (caption_size + 6) as i32))?;
    }

    for (i, cond) in conditions.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        let color = desaturate_rgb(e2_condition_color(cond), 0.2);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, means[i])],
            color.mix(0.7).filled(),
        )))?;
        let y0 = (means[i] - ci95[i]).max(0.0);
        let y1 = (means[i] + ci95[i]).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.7),
        )))?;
    }
    Ok(())
}

#[allow(dead_code)]
fn draw_entropy_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    hist_rows: &[HistStructureRow],
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    draw_entropy_panel_for_conditions(
        area,
        caption,
        hist_rows,
        &["baseline", "nohill", "nocrowd"],
        caption_size,
        label_size,
        axis_desc_size,
    )
}

fn draw_condition_bar_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    y_desc: &str,
    conditions: &[&str],
    means: &[f32],
    ci95: &[f32],
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
    y_label_area_size: u32,
) -> Result<(), Box<dyn Error>> {
    if conditions.is_empty() {
        return Ok(());
    }
    let mut y_max = 0.0f32;
    for i in 0..conditions.len().min(means.len()).min(ci95.len()) {
        y_max = y_max.max(means[i] + ci95[i]);
    }
    y_max = (1.15 * y_max.max(1e-6)).max(1e-6);

    let x_hi = (conditions.len() as f32 - 0.5).max(0.5);
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", caption_size))
        .margin(8)
        .x_label_area_size(40)
        .y_label_area_size(y_label_area_size)
        .build_cartesian_2d(-0.5f32..x_hi, 0.0f32..y_max)?;
    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc(y_desc)
        .x_labels(conditions.len())
        .x_label_formatter(&|x| {
            let idx = x.round() as isize;
            if idx >= 0 && (idx as usize) < conditions.len() {
                e2_condition_display_short(conditions[idx as usize]).to_string()
            } else {
                String::new()
            }
        })
        .label_style(("sans-serif", label_size).into_font())
        .axis_desc_style(("sans-serif", axis_desc_size).into_font())
        .draw()?;

    for (i, cond) in conditions.iter().enumerate() {
        let center = i as f32;
        let x0 = center - 0.3;
        let x1 = center + 0.3;
        let color = desaturate_rgb(e2_condition_color(cond), 0.2);
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, means[i])],
            color.mix(0.7).filled(),
        )))?;
        let y0 = (means[i] - ci95[i]).max(0.0);
        let y1 = (means[i] + ci95[i]).min(y_max);
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(center, y0), (center, y1)],
            BLACK.mix(0.7),
        )))?;
    }
    Ok(())
}

fn draw_entropy_panel_for_conditions(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    hist_rows: &[HistStructureRow],
    conditions: &[&str],
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    let mut means = vec![0.0f32; conditions.len()];
    let mut ci95 = vec![0.0f32; conditions.len()];
    for (i, cond) in conditions.iter().enumerate() {
        let values: Vec<f32> = hist_rows
            .iter()
            .filter(|r| r.condition == *cond)
            .map(|r| r.metrics.entropy)
            .collect();
        means[i] = mean_std_scalar(&values).0;
        ci95[i] = ci95_half_width(&values);
    }
    draw_condition_bar_panel(
        area,
        caption,
        "entropy (nats)",
        conditions,
        &means,
        &ci95,
        caption_size,
        label_size,
        axis_desc_size,
        55,
    )
}

#[allow(dead_code)]
fn draw_polyphony_panel(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    diversity_rows: &[DiversityRow],
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    draw_polyphony_panel_for_conditions(
        area,
        caption,
        diversity_rows,
        &["baseline", "nohill", "nocrowd"],
        caption_size,
        label_size,
        axis_desc_size,
    )
}

fn draw_polyphony_panel_for_conditions(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    diversity_rows: &[DiversityRow],
    conditions: &[&str],
    caption_size: u32,
    label_size: u32,
    axis_desc_size: u32,
) -> Result<(), Box<dyn Error>> {
    let mut means = vec![0.0f32; conditions.len()];
    let mut ci95 = vec![0.0f32; conditions.len()];
    for (i, cond) in conditions.iter().enumerate() {
        let values: Vec<f32> = diversity_rows
            .iter()
            .filter(|r| r.condition == *cond)
            .map(|r| r.metrics.unique_bins as f32)
            .collect();
        means[i] = mean_std_scalar(&values).0;
        ci95[i] = ci95_half_width(&values);
    }
    draw_condition_bar_panel(
        area,
        caption,
        "distinct pitch pos. (25 ct)",
        conditions,
        &means,
        &ci95,
        caption_size,
        label_size,
        axis_desc_size,
        66,
    )
}

fn desaturate_rgb(color: RGBColor, amount: f32) -> RGBColor {
    let t = amount.clamp(0.0, 1.0);
    let RGBColor(r, g, b) = color;
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let gray = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let blend = |c: f32| ((1.0 - t) * c + t * gray).round().clamp(0.0, 255.0) as u8;
    RGBColor(blend(rf), blend(gf), blend(bf))
}

fn render_consonant_mass_summary_plot(
    out_path: &Path,
    rows: &[ConsonantMassRow],
) -> Result<(), Box<dyn Error>> {
    if rows.is_empty() {
        return Ok(());
    }
    let root = bitmap_root(out_path, (500, 560)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((2, 1));
    draw_consonant_mass_panel(
        &panels[0],
        "Consonant interval mass (95% CI)",
        "T = {6:5, 5:4, 3:2}",
        rows,
        consonant_mass_core,
    )?;
    draw_consonant_mass_panel(
        &panels[1],
        "Consonant interval mass (95% CI)",
        "T = {6:5, 5:4, 4:3, 3:2, 8:5, 5:3}",
        rows,
        consonant_mass_extended,
    )?;
    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_anchor_hist_post_folded(
    out_path: &Path,
    centers: &[f32],
    mean: &[f32],
    std: &[f32],
    folded_centers: &[f32],
    folded_mean: &[f32],
    folded_std: &[f32],
    bin_width: f32,
) -> Result<(), Box<dyn Error>> {
    if centers.is_empty() || folded_centers.is_empty() {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 2));

    {
        let len = centers.len().min(mean.len()).min(std.len());
        let x_min = centers[0] - 0.5 * bin_width;
        let x_max = centers[len - 1] + 0.5 * bin_width;
        let mut y_peak = 0.0f32;
        for &v in &mean[..len] {
            y_peak = y_peak.max(v);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

        let mut chart = ChartBuilder::on(&panels[0])
            .caption("Anchor intervals (post, mean frac)", ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("semitones")
            .y_desc("mean fraction")
            .draw()?;

        for &x in &[-7.0f32, -4.0, -3.0, 3.0, 4.0, 7.0] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.25),
            )))?;
        }
        let half = bin_width * 0.45;
        for i in 0..len {
            chart.draw_series(std::iter::once(Rectangle::new(
                [(centers[i] - half, 0.0), (centers[i] + half, mean[i])],
                PAL_H.mix(0.65).filled(),
            )))?;
            let y0 = (mean[i] - std[i]).max(0.0);
            let y1 = (mean[i] + std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(centers[i], y0), (centers[i], y1)],
                BLACK.mix(0.6),
            )))?;
        }
    }

    {
        let len = folded_centers
            .len()
            .min(folded_mean.len())
            .min(folded_std.len());
        let x_min = folded_centers[0] - 0.5 * bin_width;
        let x_max = folded_centers[len - 1] + 0.5 * bin_width;
        let mut y_peak = 0.0f32;
        for &v in &folded_mean[..len] {
            y_peak = y_peak.max(v);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);

        let mut chart = ChartBuilder::on(&panels[1])
            .caption("|Anchor intervals| folded to [0,12]", ("sans-serif", 16))
            .margin(10)
            .x_label_area_size(35)
            .y_label_area_size(45)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("|semitones|")
            .y_desc("mean fraction")
            .draw()?;
        for &x in &[3.0f32, 4.0, 7.0] {
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(x, 0.0), (x, y_max)],
                BLACK.mix(0.25),
            )))?;
        }
        let half = bin_width * 0.45;
        for i in 0..len {
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (folded_centers[i] - half, 0.0),
                    (folded_centers[i] + half, folded_mean[i]),
                ],
                PAL_H.mix(0.65).filled(),
            )))?;
            let y0 = (folded_mean[i] - folded_std[i]).max(0.0);
            let y1 = (folded_mean[i] + folded_std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(folded_centers[i], y0), (folded_centers[i], y1)],
                BLACK.mix(0.6),
            )))?;
        }
    }
    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_e2_figure1(
    out_path: &Path,
    baseline_panel_a_mean: &[f32],
    baseline_panel_a_ci95: &[f32],
    nohill_panel_a_mean: &[f32],
    nohill_panel_a_ci95: &[f32],
    baseline_stats: &E2SweepStats,
    nohill_stats: &E2SweepStats,
    baseline_ci95_g_scene: &[f32],
    nohill_ci95_g_scene: &[f32],
    diversity_rows: &[DiversityRow],
    trajectories: &[Vec<f32>],
    phase_mode: E2PhaseMode,
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (1860, 420)).into_drawing_area();
    root.fill(&WHITE)?;
    let (_, content) = root.split_horizontally(75);
    let (content, _) = content.split_horizontally(1710);
    let (wide, panel_d) = content.split_horizontally(1440);
    let panels = wide.split_evenly((1, 3));
    let traj_n_agents = trajectories.len().max(1);
    let c_len = baseline_panel_a_mean
        .len()
        .min(baseline_panel_a_ci95.len())
        .min(nohill_panel_a_mean.len())
        .min(nohill_panel_a_ci95.len());
    let traj_burn_in = e2_trajectory_burn_in_step(traj_n_agents);
    let traj_phase_switch = e2_trajectory_phase_switch_step(phase_mode, traj_n_agents);

    draw_e2_timeseries_pair_panel(
        &panels[0],
        "A. Mean LOO C_score",
        "mean LOO C_score",
        baseline_panel_a_mean,
        baseline_panel_a_ci95,
        nohill_panel_a_mean,
        nohill_panel_a_ci95,
        traj_burn_in,
        traj_phase_switch,
        0,
        c_len.saturating_sub(1),
        true,
        Some((0.0, 1.0)),
    )?;
    let g_len = baseline_stats
        .mean_g_scene
        .len()
        .min(baseline_ci95_g_scene.len())
        .min(nohill_stats.mean_g_scene.len())
        .min(nohill_ci95_g_scene.len());
    draw_e2_timeseries_pair_panel(
        &panels[1],
        "B. Mean scene consonance G(F)",
        "mean G(F) (a.u.)",
        &baseline_stats.mean_g_scene,
        baseline_ci95_g_scene,
        &nohill_stats.mean_g_scene,
        nohill_ci95_g_scene,
        traj_burn_in,
        traj_phase_switch,
        0,
        g_len.saturating_sub(1),
        true,
        None,
    )?;
    draw_trajectory_panel(
        &panels[2],
        "C. Local-search trajectories",
        trajectories,
        traj_burn_in,
        traj_phase_switch,
    )?;
    draw_diversity_metric_panel_impl_for_conditions(
        &panel_d,
        "D. Final unique bins",
        "bins",
        diversity_rows,
        |metrics| metrics.unique_bins as f32,
        29,
        18,
        22,
        &["baseline", "nohill"],
    )?;
    root.present()?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_e2_figure2(
    out_path: &Path,
    pairwise_centers: &[f32],
    pairwise_baseline_mean: &[f32],
    pairwise_baseline_std: &[f32],
    pairwise_nohill_mean: &[f32],
    _pairwise_norep_mean: &[f32],
    hist_rows: &[HistStructureRow],
    diversity_rows: &[DiversityRow],
) -> Result<(), Box<dyn Error>> {
    if pairwise_centers.is_empty() {
        return Ok(());
    }
    let zeros = vec![0.0; pairwise_nohill_mean.len()];
    let (display_centers, display_baseline_mean, display_baseline_std) = rebin_histogram_series(
        pairwise_centers,
        pairwise_baseline_mean,
        pairwise_baseline_std,
        E2_PAIRWISE_BIN_ST,
        E2_PAIRWISE_DISPLAY_BIN_ST,
    );
    let (display_nohill_centers, display_nohill_mean, _) = rebin_histogram_series(
        pairwise_centers,
        pairwise_nohill_mean,
        &zeros,
        E2_PAIRWISE_BIN_ST,
        E2_PAIRWISE_DISPLAY_BIN_ST,
    );
    let len = display_centers
        .len()
        .min(display_baseline_mean.len())
        .min(display_baseline_std.len())
        .min(display_nohill_centers.len())
        .min(display_nohill_mean.len());
    if len == 0 {
        return Ok(());
    }
    let root = bitmap_root(out_path, (1704, 400)).into_drawing_area();
    root.fill(&WHITE)?;
    let (panel_a, panel_bc) = root.split_horizontally(1254);
    let (panel_b, panel_c) = panel_bc.split_horizontally(225);

    // Convert semitone data to cents for display (×100)
    let st2c = 100.0f32;
    let bin_c = E2_PAIRWISE_DISPLAY_BIN_ST * st2c;

    {
        let x_min = display_centers[0] * st2c - 0.5 * bin_c;
        let x_max = display_centers[len - 1] * st2c + 0.5 * bin_c;
        let mut y_peak = 0.0f32;
        for i in 0..len {
            y_peak = y_peak
                .max(display_baseline_mean[i] + display_baseline_std[i])
                .max(display_nohill_mean[i]);
        }
        let y_max = (1.15 * y_peak.max(1e-4)).max(1e-4);
        let mut chart = ChartBuilder::on(&panel_a)
            .caption(
                "E. Interval histogram: local-search vs random-walk",
                ("sans-serif", 32),
            )
            .margin(10)
            .x_label_area_size(55)
            .y_label_area_size(70)
            .build_cartesian_2d(x_min..x_max, 0.0f32..y_max)?;
        chart
            .configure_mesh()
            .x_desc("cents")
            .y_desc("mean fraction")
            .label_style(("sans-serif", 20).into_font())
            .axis_desc_style(("sans-serif", 24).into_font())
            .draw()?;
        draw_e2_interval_guides_cents(&mut chart, y_max, Some(bin_c))?;
        let half = bin_c * 0.45;
        let baseline_bars = (0..len).map(|i| {
            let cx = display_centers[i] * st2c;
            Rectangle::new(
                [(cx - half, 0.0), (cx + half, display_baseline_mean[i])],
                PAL_H.mix(0.65).filled(),
            )
        });
        chart
            .draw_series(baseline_bars)?
            .label("local-search")
            .legend(|(x, y)| {
                Rectangle::new([(x, y - 5), (x + 18, y + 5)], PAL_H.mix(0.65).filled())
            });
        for i in 0..len {
            let cx = display_centers[i] * st2c;
            let y0 = (display_baseline_mean[i] - display_baseline_std[i]).max(0.0);
            let y1 = (display_baseline_mean[i] + display_baseline_std[i]).min(y_max);
            chart.draw_series(std::iter::once(PathElement::new(
                vec![(cx, y0), (cx, y1)],
                BLACK.mix(0.6),
            )))?;
        }
        let nohill_line = display_nohill_centers
            .iter()
            .take(len)
            .map(|&x| x * st2c)
            .zip(display_nohill_mean.iter().take(len).copied());
        chart
            .draw_series(LineSeries::new(nohill_line, PAL_R.stroke_width(3)))?
            .label("random-walk")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], PAL_R.stroke_width(3)));
        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .label_font(("sans-serif", 22).into_font())
            .draw()?;
    }

    draw_entropy_panel_for_conditions(
        &panel_b,
        "F. Entropy",
        hist_rows,
        &["baseline", "nohill"],
        32,
        20,
        24,
    )?;
    draw_polyphony_panel_for_conditions(
        &panel_c,
        "G. Polyphony",
        diversity_rows,
        &["baseline", "nohill"],
        32,
        20,
        24,
    )?;

    root.present()?;
    Ok(())
}

fn plot_e2_dense_sweep(
    out_dir: &Path,
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
    quick: bool,
) -> Result<(), Box<dyn Error>> {
    let seeds = e2_dense_seed_slice(quick);
    let max_worker_threads = if quick { Some(4) } else { None };
    let mut rows = Vec::new();
    let mut cases: Vec<(f32, usize, f32)> = Vec::new();

    for &range_oct in &E2_DENSE_SWEEP_RANGES_OCT {
        for &voices_per_oct in &E2_DENSE_SWEEP_VOICES_PER_OCT {
            let n_agents = e2_dense_n_agents(range_oct, voices_per_oct);
            cases.push((range_oct, n_agents, voices_per_oct));
        }
    }
    for &n_agents in &E2_DENSE_SWEEP_2OCT_EXTREME_N {
        cases.push((2.0, n_agents, n_agents as f32 / 2.0));
    }
    cases.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    cases.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && a.1 == b.1);

    for (range_oct, n_agents, voices_per_oct) in cases {
        for (condition, label) in [
            (E2Condition::Baseline, "baseline"),
            (E2Condition::NoCrowding, "nocrowd"),
        ] {
            eprintln!(
                "  Dense sweep: range={range_oct:.1}oct voices/oct={voices_per_oct:.1} N={n_agents} cond={label} seeds={}",
                seeds.len()
            );
            let (runs, _) = e2_seed_sweep_with_threads_for_seeds(
                space,
                anchor_hz,
                condition,
                E2_STEP_SEMITONES,
                phase_mode,
                None,
                0,
                n_agents,
                range_oct,
                seeds,
                max_worker_threads,
            );
            for run in &runs {
                rows.push(e2_dense_row_from_run(
                    run,
                    anchor_hz,
                    range_oct,
                    voices_per_oct,
                    label,
                ));
            }
        }
    }

    let summary_rows = e2_dense_sweep_summary_rows(&rows);
    write_with_log(
        out_dir.join("paper_e2_dense_sweep_by_seed.csv"),
        e2_dense_sweep_by_seed_csv(&rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_dense_sweep_summary.csv"),
        e2_dense_sweep_summary_csv(&summary_rows),
    )?;
    write_with_log(
        out_dir.join("paper_e2_dense_sweep_delta.csv"),
        e2_dense_delta_csv(&summary_rows),
    )?;
    let plot_path = out_dir.join("paper_e2_dense_tradeoff_scatter.svg");
    render_e2_dense_tradeoff_plot(&plot_path, &summary_rows)?;

    Ok(())
}

fn e2_c_snapshot(run: &E2Run) -> (f32, f32, f32) {
    let pre_idx = e2_pre_step();
    let post_idx = e2_post_step_for(E2_SWEEPS);
    e2_c_snapshot_at(&run.mean_c_series, pre_idx, post_idx)
}

fn e2_c_snapshot_at(series: &[f32], pre_idx: usize, post_idx: usize) -> (f32, f32, f32) {
    let init = series.first().copied().unwrap_or(0.0);
    let pre = series.get(pre_idx).copied().unwrap_or(init);
    let post = series
        .get(post_idx.min(series.len().saturating_sub(1)))
        .copied()
        .unwrap_or(pre);
    (init, pre, post)
}

#[cfg(test)]
fn e2_c_snapshot_series(
    series: &[f32],
    anchor_shift_enabled: bool,
    anchor_shift_step: usize,
    burn_in: usize,
    steps: usize,
) -> (f32, f32, f32) {
    let pre_idx = e2_pre_step_for(anchor_shift_enabled, anchor_shift_step, burn_in);
    let post_idx = e2_post_step_for(steps);
    e2_c_snapshot_at(series, pre_idx, post_idx)
}

fn mean_std_scalar(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().copied().sum::<f32>() / n;
    let var = values
        .iter()
        .map(|v| (*v - mean) * (*v - mean))
        .sum::<f32>()
        / n;
    (mean, var.max(0.0).sqrt())
}

fn render_series_plot_fixed_y(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series: &[(f32, f32)],
    markers: &[f32],
    mut y_lo: f32,
    mut y_hi: f32,
) -> Result<(), Box<dyn Error>> {
    if series.is_empty() {
        return Ok(());
    }
    if !matches!(y_lo.partial_cmp(&y_hi), Some(std::cmp::Ordering::Less)) {
        y_lo = 0.0;
        y_hi = 1.0;
    }
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0).max(1.0);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;
    chart.draw_series(LineSeries::new(series.iter().copied(), &PAL_H))?;
    root.present()?;
    Ok(())
}

fn render_series_plot_with_markers(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series: &[(f32, f32)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series.is_empty() {
        return Ok(());
    }
    let x_max = series.last().map(|(x, _)| *x).unwrap_or(0.0).max(1.0);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for &(_, y) in series {
        if y.is_finite() {
            y_min = y_min.min(y);
            y_max = y_max.max(y);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 {
        0.1 * range
    } else {
        0.1 * y_max.abs().max(1.0)
    };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;
    chart.draw_series(LineSeries::new(series.iter().copied(), &PAL_H))?;
    root.present()?;
    Ok(())
}

fn render_series_plot_with_band(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    mean: &[f32],
    std: &[f32],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if mean.is_empty() {
        return Ok(());
    }
    let len = mean.len().min(std.len());
    let x_max = (len.saturating_sub(1) as f32).max(1.0);

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for i in 0..len {
        let lo = mean[i] - std[i];
        let hi = mean[i] + std[i];
        if lo.is_finite() && hi.is_finite() {
            y_min = y_min.min(lo);
            y_max = y_max.max(hi);
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 {
        0.1 * range
    } else {
        0.1 * y_max.abs().max(1.0)
    };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    let mut band_points: Vec<(f32, f32)> = Vec::with_capacity(len * 2);
    for i in 0..len {
        let x = i as f32;
        band_points.push((x, mean[i] + std[i]));
    }
    for i in (0..len).rev() {
        let x = i as f32;
        band_points.push((x, mean[i] - std[i]));
    }
    chart.draw_series(std::iter::once(Polygon::new(
        band_points,
        PAL_H.mix(0.2).filled(),
    )))?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;
    let line = mean.iter().enumerate().map(|(i, &y)| (i as f32, y));
    chart.draw_series(LineSeries::new(line, &PAL_H))?;
    root.present()?;
    Ok(())
}

fn render_series_plot_multi(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series_list: &[(&str, &[f32], RGBColor)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series_list.is_empty() {
        return Ok(());
    }
    let mut x_max = 1.0f32;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, series, _) in series_list {
        if series.is_empty() {
            continue;
        }
        x_max = x_max.max(series.len().saturating_sub(1) as f32);
        for &val in *series {
            if val.is_finite() {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 {
        0.1 * range
    } else {
        0.1 * y_max.abs().max(1.0)
    };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;

    for &(label, series, color) in series_list {
        if series.is_empty() {
            continue;
        }
        let line = series.iter().enumerate().map(|(i, &y)| (i as f32, y));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_series_plot_multi_with_band(
    out_path: &Path,
    caption: &str,
    y_desc: &str,
    series_list: &[(&str, &[f32], &[f32], RGBColor)],
    markers: &[f32],
) -> Result<(), Box<dyn Error>> {
    if series_list.is_empty() {
        return Ok(());
    }
    let mut x_max = 1.0f32;
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for (_, mean, std, _) in series_list {
        let len = mean.len().min(std.len());
        if len == 0 {
            continue;
        }
        x_max = x_max.max(len.saturating_sub(1) as f32);
        for i in 0..len {
            let lo = mean[i] - std[i];
            let hi = mean[i] + std[i];
            if lo.is_finite() && hi.is_finite() {
                y_min = y_min.min(lo);
                y_max = y_max.max(hi);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = 0.0;
        y_max = 1.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 {
        0.1 * range
    } else {
        0.1 * y_max.abs().max(1.0)
    };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..x_max.max(1.0), y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc(y_desc)
        .draw()?;

    draw_vertical_guides(&mut chart, markers, y_lo, y_hi)?;

    for &(label, mean, std, color) in series_list {
        let len = mean.len().min(std.len());
        if len == 0 {
            continue;
        }
        let mut band_points: Vec<(f32, f32)> = Vec::with_capacity(len * 2);
        for i in 0..len {
            band_points.push((i as f32, mean[i] + std[i]));
        }
        for i in (0..len).rev() {
            band_points.push((i as f32, mean[i] - std[i]));
        }
        chart.draw_series(std::iter::once(Polygon::new(
            band_points,
            color.mix(0.2).filled(),
        )))?;
        let line = mean.iter().enumerate().map(|(i, &y)| (i as f32, y));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

fn render_agent_trajectories_plot(
    out_path: &Path,
    trajectories: &[Vec<f32>],
) -> Result<(), Box<dyn Error>> {
    if trajectories.is_empty() {
        return Ok(());
    }
    let steps = trajectories
        .iter()
        .map(|trace| trace.len())
        .max()
        .unwrap_or(0);
    if steps == 0 {
        return Ok(());
    }

    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for trace in trajectories {
        for &val in trace {
            if val.is_finite() {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
            }
        }
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -12.0;
        y_max = 12.0;
    }
    let range = (y_max - y_min).abs();
    let pad = if range > 1e-6 { 0.1 * range } else { 1.0 };
    let y_lo = y_min - pad;
    let y_hi = y_max + pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Agent Trajectories (Semitones)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(
            0.0f32..(steps.saturating_sub(1) as f32).max(1.0),
            y_lo..y_hi,
        )?;

    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("semitones")
        .draw()?;

    for (agent_id, trace) in trajectories.iter().enumerate() {
        if trace.is_empty() {
            continue;
        }
        let series = trace
            .iter()
            .enumerate()
            .map(|(step, &val)| (step as f32, val));
        let color = Palette99::pick(agent_id).mix(0.9);
        chart.draw_series(LineSeries::new(
            series,
            ShapeStyle::from(&color).stroke_width(2),
        ))?;
    }

    root.present()?;
    Ok(())
}

fn render_interval_histogram(
    out_path: &Path,
    caption: &str,
    values: &[f32],
    min: f32,
    max: f32,
    bin_width: f32,
    x_desc: &str,
) -> Result<(), Box<dyn Error>> {
    let counts = histogram_counts(values, min, max, bin_width);
    let y_max = counts
        .iter()
        .map(|(_, count)| *count as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(caption, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("count")
        .x_labels(25)
        .draw()?;

    for (bin_start, count) in counts {
        let x0 = bin_start;
        let x1 = bin_start + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f32)],
            PAL_H.mix(0.6).filled(),
        )))?;
    }

    root.present()?;
    Ok(())
}

fn render_e2_histogram_sweep(out_dir: &Path, run: &E2Run) -> Result<(), Box<dyn Error>> {
    let post_start = E2_BURN_IN;
    let post_end = E2_SWEEPS.saturating_sub(1);
    let ranges: Vec<(&str, usize, usize, &Vec<f32>)> = if e2_anchor_shift_enabled() {
        let pre_start = E2_BURN_IN;
        let pre_end = E2_ANCHOR_SHIFT_STEP.saturating_sub(1);
        let post_start = E2_ANCHOR_SHIFT_STEP;
        let post_end = E2_SWEEPS.saturating_sub(1);
        vec![
            ("pre", pre_start, pre_end, &run.semitone_samples_pre),
            ("post", post_start, post_end, &run.semitone_samples_post),
        ]
    } else {
        vec![("post", post_start, post_end, &run.semitone_samples_post)]
    };
    let bins = [0.5f32, 0.25f32];
    for (label, start, end, values) in ranges {
        for &bin_width in &bins {
            let fname = format!(
                "paper_e2_interval_histogram_{}_bw{}.svg",
                label,
                format_float_token(bin_width)
            );
            let phase_label = if label == "post" {
                e2_post_label()
            } else {
                label
            };
            let caption = format!(
                "Interval Histogram ({phase_label}, steps {start}-{end}, bin={bin_width:.2} ct)"
            );
            let out_path = out_dir.join(fname);
            render_interval_histogram(
                &out_path,
                &caption,
                values,
                -12.0,
                12.0,
                bin_width,
                "semitones",
            )?;
        }
    }
    Ok(())
}

fn render_e2_control_histograms(
    out_dir: &Path,
    baseline: &E2Run,
    nohill: &E2Run,
    norep: &E2Run,
) -> Result<(), Box<dyn Error>> {
    let bin_width = E2_ANCHOR_BIN_ST;
    let min = -12.0f32;
    let max = 12.0f32;
    let counts_base = histogram_counts(&baseline.semitone_samples_post, min, max, bin_width);
    let counts_nohill = histogram_counts(&nohill.semitone_samples_post, min, max, bin_width);
    let counts_norep = histogram_counts(&norep.semitone_samples_post, min, max, bin_width);

    let mut y_max = 0.0f32;
    for counts in [&counts_base, &counts_nohill, &counts_norep] {
        for &(_, count) in counts.iter() {
            y_max = y_max.max(count as f32);
        }
    }
    y_max = y_max.max(1.0);

    let out_path = out_dir.join("paper_e2_interval_histogram_post_controls_bw0p50.svg");
    let root = bitmap_root(&out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let post_label = e2_post_label();
    let mut chart = ChartBuilder::on(&root)
        .caption(
            format!("Interval Histogram ({post_label}, controls, bin=50 ct)"),
            ("sans-serif", 20),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(min..max, 0.0f32..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc("semitones")
        .y_desc("count")
        .x_labels(25)
        .draw()?;

    let counts = [&counts_base, &counts_nohill, &counts_norep];
    let colors = [PAL_H.mix(0.6), PAL_R.mix(0.6), PAL_CD.mix(0.6)];
    let sub_width = bin_width / 3.0;
    for bin_idx in 0..counts_base.len() {
        let bin_start = counts_base[bin_idx].0;
        for (j, counts_set) in counts.iter().enumerate() {
            let count = counts_set[bin_idx].1 as f32;
            let x0 = bin_start + sub_width * j as f32;
            let x1 = x0 + sub_width;
            chart.draw_series(std::iter::once(Rectangle::new(
                [(x0, 0.0), (x1, count)],
                colors[j].filled(),
            )))?;
        }
    }

    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 1.05), (min + 0.3, y_max * 1.05)],
            PAL_H,
        )))?
        .label("local-search")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_H));
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 1.02), (min + 0.3, y_max * 1.02)],
            PAL_R,
        )))?
        .label("random-walk")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_R));
    chart
        .draw_series(std::iter::once(PathElement::new(
            vec![(min, y_max * 0.99), (min + 0.3, y_max * 0.99)],
            PAL_CD,
        )))?
        .label("no-crowding")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_CD));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

struct CorrStats {
    n: usize,
    pearson_r: f32,
    pearson_p: f32,
    spearman_rho: f32,
    spearman_p: f32,
}

fn pearson_r(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len() as f32;
    let mean_x = x.iter().copied().sum::<f32>() / n;
    let mean_y = y.iter().copied().sum::<f32>() / n;
    let mut num = 0.0f32;
    let mut den_x = 0.0f32;
    let mut den_y = 0.0f32;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    let den = (den_x * den_y).sqrt();
    if den > 0.0 { num / den } else { 0.0 }
}

fn ranks(values: &[f32]) -> Vec<f32> {
    let mut indexed: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f32; values.len()];
    let mut i = 0usize;
    while i < indexed.len() {
        let start = i;
        let val = indexed[i].1;
        let mut end = i + 1;
        while end < indexed.len() && (indexed[end].1 - val).abs() < 1e-6 {
            end += 1;
        }
        let rank = (start + end - 1) as f32 * 0.5 + 1.0;
        for j in start..end {
            ranks[indexed[j].0] = rank;
        }
        i = end;
    }
    ranks
}

fn spearman_rho(x: &[f32], y: &[f32]) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let rx = ranks(x);
    let ry = ranks(y);
    pearson_r(&rx, &ry)
}

fn perm_pvalue(
    x: &[f32],
    y: &[f32],
    n_perm: usize,
    seed: u64,
    corr_fn: fn(&[f32], &[f32]) -> f32,
) -> f32 {
    if x.len() != y.len() || x.len() < 2 {
        return 1.0;
    }
    let obs = corr_fn(x, y).abs();
    let mut rng = seeded_rng(seed);
    let mut y_perm = y.to_vec();
    let mut count = 0usize;
    for _ in 0..n_perm {
        y_perm.shuffle(&mut rng);
        let r = corr_fn(x, &y_perm).abs();
        if r >= obs {
            count += 1;
        }
    }
    (count as f32 + 1.0) / (n_perm as f32 + 1.0)
}

fn format_p_value(p: f32) -> String {
    if !p.is_finite() {
        return "p=nan".to_string();
    }
    if p < 0.001 {
        "p<0.001".to_string()
    } else {
        format!("p={:.3}", p)
    }
}

fn corr_stats(x_raw: &[f32], y_raw: &[u32], seed: u64) -> CorrStats {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (x, y) in x_raw.iter().zip(y_raw.iter()) {
        if x.is_finite() {
            xs.push(*x);
            ys.push(*y as f32);
        }
    }
    let n = xs.len();
    let pearson = pearson_r(&xs, &ys);
    let spearman = spearman_rho(&xs, &ys);
    let pearson_p = perm_pvalue(&xs, &ys, 1000, seed ^ 0xA11CE_u64, pearson_r);
    let spearman_p = perm_pvalue(&xs, &ys, 1000, seed ^ 0xBEEF0_u64, spearman_rho);
    CorrStats {
        n,
        pearson_r: pearson,
        pearson_p,
        spearman_rho: spearman,
        spearman_p,
    }
}

struct ScatterData {
    points: Vec<(f32, f32)>,
    x_min: f32,
    x_max: f32,
    y_max: f32,
    stats: CorrStats,
}

fn build_scatter_data(x_values: &[f32], lifetimes: &[u32], seed: u64) -> ScatterData {
    let stats = corr_stats(x_values, lifetimes, seed);
    let mut points = Vec::new();
    for (x, y) in x_values.iter().zip(lifetimes.iter()) {
        if x.is_finite() {
            points.push((*x, *y as f32));
        }
    }
    let mut x_min = 0.0f32;
    let mut x_max = 1.0f32;
    let mut y_max = 1.0f32;
    if !points.is_empty() {
        x_min = f32::INFINITY;
        x_max = f32::NEG_INFINITY;
        y_max = f32::NEG_INFINITY;
        for &(x, y) in &points {
            x_min = x_min.min(x);
            x_max = x_max.max(x);
            y_max = y_max.max(y);
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            x_min = 0.0;
            x_max = 1.0;
        }
        if (x_max - x_min).abs() < 1e-6 {
            x_min -= 0.05;
            x_max += 0.05;
        }
        y_max = y_max.max(1.0);
    }
    ScatterData {
        points,
        x_min,
        x_max,
        y_max,
        stats,
    }
}

fn scatter_with_ranges(data: &ScatterData, x_min: f32, x_max: f32, y_max: f32) -> ScatterData {
    ScatterData {
        points: data.points.clone(),
        x_min,
        x_max,
        y_max,
        stats: CorrStats {
            n: data.stats.n,
            pearson_r: data.stats.pearson_r,
            pearson_p: data.stats.pearson_p,
            spearman_rho: data.stats.spearman_rho,
            spearman_p: data.stats.spearman_p,
        },
    }
}

fn force_scatter_x_range(data: &mut ScatterData, x_min: f32, x_max: f32) {
    data.x_min = x_min;
    data.x_max = x_max;
    if (data.x_max - data.x_min).abs() < 1e-6 {
        data.x_max = data.x_min + 1.0;
    }
}

fn draw_note_lines<DB: DrawingBackend, CT: CoordTranslate>(
    area: &DrawingArea<DB, CT>,
    lines: &[String],
    x_frac: f32,
    y_frac: f32,
    line_height_px: i32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    let screen = area.strip_coord_spec();
    let (w, h) = screen.dim_in_pixel();
    let x = (w as f32 * x_frac).round() as i32;
    let mut y = (h as f32 * y_frac).round() as i32;
    for line in lines {
        screen.draw(&Text::new(
            line.clone(),
            (x, y),
            ("sans-serif", 14).into_font(),
        ))?;
        y += line_height_px;
    }
    Ok(())
}

fn render_scatter_on_area(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    x_desc: &str,
    data: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 18))
        .margin(10)
        .x_label_area_size(35)
        .y_label_area_size(50)
        .build_cartesian_2d(data.x_min..data.x_max, 0.0f32..(data.y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("lifetime (steps)")
        .draw()?;

    if !data.points.is_empty() {
        chart.draw_series(
            data.points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, PAL_H.mix(0.5).filled())),
        )?;
    }
    let ref_x = if data.x_min <= 0.0 && data.x_max >= 0.0 {
        Some(0.0)
    } else if data.x_min <= 0.5 && data.x_max >= 0.5 {
        Some(0.5)
    } else {
        None
    };
    if let Some(ref_x) = ref_x {
        let y_top = data.y_max * 1.05;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(ref_x, 0.0), (ref_x, y_top)],
            BLACK.mix(0.3),
        )))?;
    }

    let pearson_p = format_p_value(data.stats.pearson_p);
    let spearman_p = format_p_value(data.stats.spearman_p);
    let lines = vec![
        format!("N={}", data.stats.n),
        format!("Pearson r={:.3} ({})", data.stats.pearson_r, pearson_p),
        format!("Spearman ρ={:.3} ({})", data.stats.spearman_rho, spearman_p),
    ];
    draw_note_lines(chart.plotting_area(), &lines, 0.02, 0.05, 16)?;
    Ok(())
}

fn render_e3_scatter_with_stats(
    out_path: &Path,
    caption: &str,
    x_desc: &str,
    x_values: &[f32],
    lifetimes: &[u32],
    seed: u64,
) -> Result<CorrStats, Box<dyn Error>> {
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut data = build_scatter_data(x_values, lifetimes, seed);
    force_scatter_x_range(&mut data, 0.0, 1.0);
    render_scatter_on_area(&root, caption, x_desc, &data)?;
    root.present()?;
    Ok(data.stats)
}

enum SplitKind {
    Median,
    Quartiles,
}

struct SurvivalStats {
    median_high: f32,
    median_low: f32,
    logrank_p: f32,
}

struct SurvivalData {
    series_high: Vec<(f32, f32)>,
    series_low: Vec<(f32, f32)>,
    x_max: f32,
    stats: SurvivalStats,
    n_high: usize,
    n_low: usize,
}

fn survival_with_x_max(data: &SurvivalData, x_max: f32) -> SurvivalData {
    SurvivalData {
        series_high: data.series_high.clone(),
        series_low: data.series_low.clone(),
        x_max,
        stats: SurvivalStats {
            median_high: data.stats.median_high,
            median_low: data.stats.median_low,
            logrank_p: data.stats.logrank_p,
        },
        n_high: data.n_high,
        n_low: data.n_low,
    }
}

fn median_u32(values: &[u32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] as f32 + sorted[mid] as f32) * 0.5
    } else {
        sorted[mid] as f32
    }
}

fn split_by_median(values: &[f32]) -> (Vec<usize>, Vec<usize>) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if sorted.is_empty() {
        0.0
    } else {
        sorted[sorted.len() / 2]
    };
    let mut high = Vec::new();
    let mut low = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v >= median {
            high.push(i);
        } else {
            low.push(i);
        }
    }
    (high, low)
}

fn split_by_quartiles(values: &[f32]) -> (Vec<usize>, Vec<usize>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q25 = sorted[(sorted.len() - 1) / 4];
    let q75 = sorted[(sorted.len() - 1) * 3 / 4];
    let mut high = Vec::new();
    let mut low = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v >= q75 {
            high.push(i);
        } else if v <= q25 {
            low.push(i);
        }
    }
    (high, low)
}

fn logrank_statistic(high: &[u32], low: &[u32]) -> f32 {
    if high.is_empty() || low.is_empty() {
        return 0.0;
    }
    let mut times: Vec<u32> = high.iter().chain(low.iter()).copied().collect();
    times.sort_unstable();
    times.dedup();
    let mut o_minus_e = 0.0f32;
    let mut var = 0.0f32;
    for &t in &times {
        let n1 = high.iter().filter(|&&v| v >= t).count();
        let n2 = low.iter().filter(|&&v| v >= t).count();
        let d1 = high.iter().filter(|&&v| v == t).count();
        let d2 = low.iter().filter(|&&v| v == t).count();
        let n = n1 + n2;
        let d = d1 + d2;
        if n <= 1 || d == 0 {
            continue;
        }
        let n1f = n1 as f32;
        let nf = n as f32;
        let df = d as f32;
        let expected = df * n1f / nf;
        let var_t = df * (n1f / nf) * (1.0 - n1f / nf) * ((nf - df) / (nf - 1.0));
        o_minus_e += d1 as f32 - expected;
        var += var_t;
    }
    if var > 0.0 {
        o_minus_e / var.sqrt()
    } else {
        0.0
    }
}

fn logrank_pvalue(high: &[u32], low: &[u32], n_perm: usize, seed: u64) -> f32 {
    if high.is_empty() || low.is_empty() {
        return 1.0;
    }
    let obs = logrank_statistic(high, low).abs();
    let mut rng = seeded_rng(seed);
    let mut combined: Vec<u32> = high.iter().chain(low.iter()).copied().collect();
    let n_high = high.len();
    let mut count = 0usize;
    for _ in 0..n_perm {
        combined.shuffle(&mut rng);
        let (a, b) = combined.split_at(n_high);
        let stat = logrank_statistic(a, b).abs();
        if stat >= obs {
            count += 1;
        }
    }
    (count as f32 + 1.0) / (n_perm as f32 + 1.0)
}

fn build_survival_data(
    lifetimes: &[u32],
    values: &[f32],
    split: SplitKind,
    seed: u64,
) -> SurvivalData {
    let mut filtered_lifetimes = Vec::new();
    let mut filtered_values = Vec::new();
    for (&lt, &val) in lifetimes.iter().zip(values.iter()) {
        if val.is_finite() {
            filtered_lifetimes.push(lt);
            filtered_values.push(val);
        }
    }

    let (high_idx, low_idx) = match split {
        SplitKind::Median => split_by_median(&filtered_values),
        SplitKind::Quartiles => split_by_quartiles(&filtered_values),
    };
    let mut high = Vec::new();
    let mut low = Vec::new();
    for &i in &high_idx {
        if let Some(&lt) = filtered_lifetimes.get(i) {
            high.push(lt);
        }
    }
    for &i in &low_idx {
        if let Some(&lt) = filtered_lifetimes.get(i) {
            low.push(lt);
        }
    }

    let max_t = high.iter().chain(low.iter()).copied().max().unwrap_or(0) as usize;
    let series_high = build_survival_series(&high, max_t);
    let series_low = build_survival_series(&low, max_t);
    let x_max = max_t.max(1) as f32;

    let median_high = median_u32(&high);
    let median_low = median_u32(&low);
    let logrank_p = logrank_pvalue(&high, &low, 1000, seed ^ 0xE3AA_u64);
    SurvivalData {
        series_high,
        series_low,
        x_max,
        stats: SurvivalStats {
            median_high,
            median_low,
            logrank_p,
        },
        n_high: high.len(),
        n_low: low.len(),
    }
}

fn render_survival_on_area(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    data: &SurvivalData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(55)
        .y_label_area_size(70)
        .build_cartesian_2d(0.0f32..data.x_max, 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival")
        .label_style(("sans-serif", 20).into_font())
        .axis_desc_style(("sans-serif", 24).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.series_high.clone(), &PAL_H))?
        .label("high")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_H));
    chart
        .draw_series(LineSeries::new(data.series_low.clone(), &PAL_R))?
        .label("low")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_R));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 24).into_font())
        .draw()?;

    Ok(())
}

fn render_survival_on_area_compact(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    data: &SurvivalData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 42))
        .margin(10)
        .margin_left(6)
        .margin_right(6)
        .x_label_area_size(65)
        .y_label_area_size(80)
        .build_cartesian_2d(0.0f32..data.x_max, 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival")
        .x_label_formatter(&|v| format!("{}", *v as i32))
        .x_labels(5)
        .y_labels(6)
        .label_style(("sans-serif", 28).into_font())
        .axis_desc_style(("sans-serif", 32).into_font())
        .draw()?;

    chart
        .draw_series(LineSeries::new(data.series_high.clone(), &PAL_H))?
        .label("high")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_H));
    chart
        .draw_series(LineSeries::new(data.series_low.clone(), &PAL_R))?
        .label("low")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_R));

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 30).into_font())
        .draw()?;

    Ok(())
}

fn render_scatter_on_area_compact(
    area: &DrawingArea<SVGBackend, Shift>,
    caption: &str,
    x_desc: &str,
    data: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    let mut chart = ChartBuilder::on(area)
        .caption(caption, ("sans-serif", 42))
        .margin(10)
        .margin_left(6)
        .margin_right(6)
        .x_label_area_size(65)
        .y_label_area_size(80)
        .build_cartesian_2d(data.x_min..data.x_max, 0.0f32..(data.y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc(x_desc)
        .y_desc("lifetime (steps)")
        .x_labels(5)
        .y_label_formatter(&|v| format!("{}", *v as i32))
        .y_labels(5)
        .label_style(("sans-serif", 28).into_font())
        .axis_desc_style(("sans-serif", 32).into_font())
        .draw()?;

    if !data.points.is_empty() {
        chart.draw_series(
            data.points
                .iter()
                .map(|(x, y)| Circle::new((*x, *y), 3, PAL_H.mix(0.5).filled())),
        )?;
    }
    if data.x_min <= 0.5 && data.x_max >= 0.5 {
        let y_top = data.y_max * 1.05;
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(0.5, 0.0), (0.5, y_top)],
            BLACK.mix(0.3),
        )))?;
    }

    Ok(())
}

fn render_survival_split_plot(
    out_path: &Path,
    caption: &str,
    lifetimes: &[u32],
    values: &[f32],
    split: SplitKind,
    seed: u64,
) -> Result<SurvivalStats, Box<dyn Error>> {
    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let data = build_survival_data(lifetimes, values, split, seed);
    render_survival_on_area(&root, caption, &data)?;
    root.present()?;
    Ok(data.stats)
}

fn render_scatter_compare(
    out_path: &Path,
    caption: &str,
    x_desc: &str,
    left_label: &str,
    left_data: &ScatterData,
    right_label: &str,
    right_data: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (1400, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((1, 2));
    let x_min = left_data.x_min.min(right_data.x_min);
    let x_max = left_data.x_max.max(right_data.x_max);
    let y_max = left_data.y_max.max(right_data.y_max);
    let left_common = scatter_with_ranges(left_data, x_min, x_max, y_max);
    let right_common = scatter_with_ranges(right_data, x_min, x_max, y_max);
    render_scatter_on_area(
        &areas[0],
        &format!("{caption} — {left_label}"),
        x_desc,
        &left_common,
    )?;
    render_scatter_on_area(
        &areas[1],
        &format!("{caption} — {right_label}"),
        x_desc,
        &right_common,
    )?;
    root.present()?;
    Ok(())
}

fn render_survival_compare(
    out_path: &Path,
    _caption: &str,
    left_label: &str,
    left_data: &SurvivalData,
    right_label: &str,
    right_data: &SurvivalData,
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (700, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let areas = root.split_evenly((2, 1));
    let x_max = left_data.x_max.max(right_data.x_max);
    let left_common = survival_with_x_max(left_data, x_max);
    let right_common = survival_with_x_max(right_data, x_max);
    render_survival_on_area(&areas[0], &format!("A. {left_label}"), &left_common)?;
    render_survival_on_area(&areas[1], &format!("B. {right_label}"), &right_common)?;
    root.present()?;
    Ok(())
}

fn render_e3_figure4(
    out_path: &Path,
    left_surv: &SurvivalData,
    right_surv: &SurvivalData,
    left_scatter: &ScatterData,
    right_scatter: &ScatterData,
) -> Result<(), Box<dyn Error>> {
    // 1×4 horizontal layout: A B C D
    let root = bitmap_root(out_path, (2160, 510)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 4));

    // Cap survival x-axis at 1000 steps
    let x_max_surv = 1000.0f32;
    let left_surv_common = survival_with_x_max(left_surv, x_max_surv);
    let right_surv_common = survival_with_x_max(right_surv, x_max_surv);
    render_survival_on_area_compact(&panels[0], "A. Baseline", &left_surv_common)?;
    render_survival_on_area_compact(&panels[1], "B. No recharge", &right_surv_common)?;

    let x_min = left_scatter.x_min.min(right_scatter.x_min);
    let x_max_s = left_scatter.x_max.max(right_scatter.x_max);
    let y_max_s = left_scatter.y_max.max(right_scatter.y_max);
    let left_sc = scatter_with_ranges(left_scatter, x_min, x_max_s, y_max_s);
    let right_sc = scatter_with_ranges(right_scatter, x_min, x_max_s, y_max_s);
    render_scatter_on_area_compact(&panels[2], "C. Baseline", "early C_score", &left_sc)?;
    render_scatter_on_area_compact(&panels[3], "D. No recharge", "early C_score", &right_sc)?;

    root.present()?;
    Ok(())
}

fn render_consonance_lifetime_scatter(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    let y_max = deaths
        .iter()
        .map(|(_, lifetime, _)| *lifetime as f32)
        .fold(0.0f32, f32::max)
        .max(1.0);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("C_level01 vs Lifetime", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..1.0f32, 0.0f32..(y_max * 1.05))?;

    chart
        .configure_mesh()
        .x_desc("avg C_level01")
        .y_desc("lifetime (steps)")
        .draw()?;

    let points = deaths
        .iter()
        .map(|(_, lifetime, avg_c_level)| (*avg_c_level, *lifetime as f32));
    chart.draw_series(points.map(|(x, y)| Circle::new((x, y), 3, PAL_R.filled())))?;

    root.present()?;
    Ok(())
}

fn render_survival_curve(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let lifetimes: Vec<u32> = deaths.iter().map(|(_, lifetime, _)| *lifetime).collect();
    let max_t = lifetimes.iter().copied().max().unwrap_or(0) as usize;
    let series = build_survival_series(&lifetimes, max_t);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Survival Curve", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..(max_t as f32), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival S(t)")
        .draw()?;

    chart.draw_series(LineSeries::new(series, &BLACK))?;
    root.present()?;
    Ok(())
}

fn render_survival_by_c_level(
    out_path: &Path,
    deaths: &[(usize, u32, f32)],
) -> Result<(), Box<dyn Error>> {
    if deaths.is_empty() {
        return Ok(());
    }
    let mut c_level_values: Vec<f32> = deaths.iter().map(|(_, _, c_level)| *c_level).collect();
    c_level_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if c_level_values.len().is_multiple_of(2) {
        let hi = c_level_values.len() / 2;
        let lo = hi.saturating_sub(1);
        0.5 * (c_level_values[lo] + c_level_values[hi])
    } else {
        c_level_values[c_level_values.len() / 2]
    };

    let mut high: Vec<u32> = Vec::new();
    let mut low: Vec<u32> = Vec::new();
    for (_, lifetime, c_level) in deaths {
        if *c_level >= median {
            high.push(*lifetime);
        } else {
            low.push(*lifetime);
        }
    }

    let max_t = deaths
        .iter()
        .map(|(_, lifetime, _)| *lifetime)
        .max()
        .unwrap_or(0) as usize;

    let high_series = build_survival_series(&high, max_t);
    let low_series = build_survival_series(&low, max_t);

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Survival by C_level01 (Median Split)", ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0f32..(max_t as f32), 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("time (steps)")
        .y_desc("survival S(t)")
        .draw()?;

    if !high_series.is_empty() {
        chart
            .draw_series(LineSeries::new(high_series, &PAL_H))?
            .label("avg C_level01 >= median")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_H));
    }
    if !low_series.is_empty() {
        chart
            .draw_series(LineSeries::new(low_series, &PAL_R))?
            .label("avg C_level01 < median")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], PAL_R));
    }

    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

// ── E5 Combined 3-Panel Figure ───────────────────────────────────
#[allow(clippy::type_complexity)]
fn render_e5_combined_figure(
    out_path: &Path,
    cond_series: &[(E5Condition, Vec<(f32, f32, f32)>)],
    delta_series: &[E5DeltaSeries],
    rep_vitality: Option<&E5VitalityResult>,
    rep_control: Option<&E5VitalityResult>,
    pearson_by_cond: &[(E5Condition, Vec<f32>)],
) -> Result<(), Box<dyn Error>> {
    let root = bitmap_root(out_path, (3600, 1100)).into_drawing_area();
    root.fill(&WHITE)?;
    let panels = root.split_evenly((1, 3));

    draw_e5_plv_panel(&panels[0], cond_series, delta_series)?;
    draw_e5_scatter_panel(&panels[1], rep_vitality, rep_control)?;
    draw_e5_pearson_panel(&panels[2], pearson_by_cond)?;

    root.present()?;
    Ok(())
}

/// Panel A: Group mean PLV over time with CI shading
#[allow(clippy::type_complexity)]
fn draw_e5_plv_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    cond_series: &[(E5Condition, Vec<(f32, f32, f32)>)],
    delta_series: &[E5DeltaSeries],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let panels = area.split_evenly((2, 1));
    draw_e5_plv_raw_subpanel(&panels[0], cond_series)?;
    draw_e5_plv_delta_subpanel(&panels[1], cond_series, delta_series)?;
    Ok(())
}

#[allow(clippy::type_complexity)]
fn draw_e5_plv_raw_subpanel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    cond_series: &[(E5Condition, Vec<(f32, f32, f32)>)],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let x_max = cond_series
        .iter()
        .flat_map(|(_, s)| s.iter().map(|(t, _, _)| *t))
        .fold(0.0f32, f32::max)
        .max(1.0);

    let mut chart = ChartBuilder::on(area)
        .caption("A1. Group PLV (raw)", ("sans-serif", 72))
        .margin(14)
        .x_label_area_size(64)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0f32..x_max, 0.0f32..1.0f32)?;

    chart
        .configure_mesh()
        .x_desc("")
        .y_desc("PLV")
        .y_label_formatter(&|v| format!("{:.1}", v))
        .label_style(("sans-serif", 40).into_font())
        .axis_desc_style(("sans-serif", 44).into_font())
        .draw()?;

    for (cond, series) in cond_series {
        if series.is_empty() {
            continue;
        }
        let color = cond.color();
        // CI band
        let mut band: Vec<(f32, f32)> = Vec::with_capacity(series.len() * 2);
        for &(x, mean, ci) in series {
            if mean.is_finite() {
                band.push((x, (mean + ci).clamp(0.0, 1.0)));
            }
        }
        for &(x, mean, ci) in series.iter().rev() {
            if mean.is_finite() {
                band.push((x, (mean - ci).clamp(0.0, 1.0)));
            }
        }
        if band.len() >= 3 {
            chart.draw_series(std::iter::once(Polygon::new(
                band,
                color.mix(0.18).filled(),
            )))?;
        }

        // Mean line
        let line_points: Vec<(f32, f32)> = series
            .iter()
            .filter(|(_, m, _)| m.is_finite())
            .map(|(x, m, _)| (*x, *m))
            .collect();
        chart
            .draw_series(LineSeries::new(
                line_points,
                ShapeStyle::from(&color).stroke_width(3),
            ))?
            .label(cond.label())
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 40).into_font())
        .draw()?;

    Ok(())
}

#[allow(clippy::type_complexity)]
fn draw_e5_plv_delta_subpanel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    cond_series: &[(E5Condition, Vec<(f32, f32, f32)>)],
    delta_series: &[E5DeltaSeries],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let x_max = cond_series
        .iter()
        .flat_map(|(_, s)| s.iter().map(|(t, _, _)| *t))
        .fold(0.0f32, f32::max)
        .max(1.0);

    let y_lo = -0.1f32;
    let y_hi = 0.6f32;

    let mut chart = ChartBuilder::on(area)
        .caption("A2. Paired contrast ΔPLV", ("sans-serif", 72))
        .margin(14)
        .x_label_area_size(96)
        .y_label_area_size(90)
        .build_cartesian_2d(0.0f32..x_max, y_lo..y_hi)?;

    chart
        .configure_mesh()
        .x_desc("time (s)")
        .y_desc("ΔPLV")
        .y_label_formatter(&|v| format!("{:.1}", v))
        .label_style(("sans-serif", 40).into_font())
        .axis_desc_style(("sans-serif", 44).into_font())
        .draw()?;

    chart.draw_series(std::iter::once(DashedPathElement::new(
        vec![(0.0f32, 0.0f32), (x_max, 0.0f32)],
        2,
        6,
        ShapeStyle::from(&BLACK.mix(0.35)).stroke_width(2),
    )))?;

    for ds in delta_series {
        if ds.series.is_empty() {
            continue;
        }

        let mut band: Vec<(f32, f32)> = Vec::with_capacity(ds.series.len() * 2);
        for &(x, mean, ci) in &ds.series {
            if mean.is_finite() {
                band.push((x, (mean + ci).clamp(y_lo, y_hi)));
            }
        }
        for &(x, mean, ci) in ds.series.iter().rev() {
            if mean.is_finite() {
                band.push((x, (mean - ci).clamp(y_lo, y_hi)));
            }
        }
        if band.len() >= 3 {
            chart.draw_series(std::iter::once(Polygon::new(
                band,
                ds.color.mix(0.18).filled(),
            )))?;
        }

        let line_points: Vec<(f32, f32)> = ds
            .series
            .iter()
            .filter(|(_, m, _)| m.is_finite())
            .map(|(x, m, _)| (*x, *m))
            .collect();
        chart
            .draw_series(LineSeries::new(
                line_points,
                ShapeStyle::from(&ds.color).stroke_width(3),
            ))?
            .label(ds.label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], ds.color));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 40).into_font())
        .draw()?;

    Ok(())
}

/// Panel B: Per-agent final PLV vs C_field scatter (representative seed)
fn draw_e5_scatter_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    rep_vitality: Option<&E5VitalityResult>,
    rep_control: Option<&E5VitalityResult>,
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let control_points: Vec<(f32, f32)> = rep_control
        .map(|res| {
            res.agent_final
                .iter()
                .filter(|a| a.plv.is_finite())
                .map(|a| (a.consonance, a.plv))
                .collect()
        })
        .unwrap_or_default();
    let vitality_points: Vec<(f32, f32)> = rep_vitality
        .map(|res| {
            res.agent_final
                .iter()
                .filter(|a| a.plv.is_finite())
                .map(|a| (a.consonance, a.plv))
                .collect()
        })
        .unwrap_or_default();

    let x_lo = -0.5f32;
    let x_hi = 1.0f32;

    let mut chart = ChartBuilder::on(area)
        .caption("B. PLV vs C_field", ("sans-serif", 72))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(x_lo..x_hi, 0.0f32..1.05f32)?;

    chart
        .configure_mesh()
        .x_desc("C_field")
        .y_desc("final PLV")
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    // Draw control first (behind), then vitality on top
    if rep_control.is_some() {
        chart
            .draw_series(control_points.iter().map(|&(x, y)| {
                Cross::new((x, y), 8, ShapeStyle::from(&PAL_E5_CONTROL).stroke_width(3))
            }))?
            .label("control")
            .legend(|(x, y)| {
                Cross::new((x, y), 9, ShapeStyle::from(&PAL_E5_CONTROL).stroke_width(3))
            });
    }
    if rep_vitality.is_some() {
        chart
            .draw_series(vitality_points.iter().map(|&(x, y)| {
                EmptyElement::at((x, y))
                    + PathElement::new(
                        vec![(-8, 0), (8, 0)],
                        ShapeStyle::from(&PAL_E5_VITALITY).stroke_width(3),
                    )
                    + PathElement::new(
                        vec![(0, -12), (0, 12)],
                        ShapeStyle::from(&PAL_E5_VITALITY).stroke_width(3),
                    )
            }))?
            .label("vitality")
            .legend(|(x, y)| {
                EmptyElement::at((x, y))
                    + PathElement::new(
                        vec![(-9, 0), (9, 0)],
                        ShapeStyle::from(&PAL_E5_VITALITY).stroke_width(3),
                    )
                    + PathElement::new(
                        vec![(0, -9), (0, 9)],
                        ShapeStyle::from(&PAL_E5_VITALITY).stroke_width(3),
                    )
            });
    }

    // Add fitted guides: vitality sigmoid (c50=0), control mean baseline
    if !vitality_points.is_empty()
        && let Some(fit) = fit_e5_sigmoid(&vitality_points)
    {
        let curve: Vec<(f32, f32)> = (0..=200)
            .map(|i| {
                let x = x_lo + (x_hi - x_lo) * (i as f32 / 200.0);
                let g = 1.0 / (1.0 + (-fit.steepness * x).exp());
                let y = (fit.base + fit.amplitude * g).clamp(0.0, 1.05);
                (x, y)
            })
            .collect();
        chart.draw_series(std::iter::once(DashedPathElement::new(
            curve,
            10,
            8,
            ShapeStyle::from(&PAL_E5_VITALITY.mix(0.98)).stroke_width(5),
        )))?;
    }
    if !control_points.is_empty() {
        let mean_control =
            control_points.iter().map(|(_, y)| *y).sum::<f32>() / control_points.len() as f32;
        // White underlay improves visibility against dense scatter points.
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x_lo, mean_control), (x_hi, mean_control)],
            ShapeStyle::from(&WHITE.mix(0.95)).stroke_width(11),
        )))?;
        chart.draw_series(std::iter::once(DashedPathElement::new(
            vec![(x_lo, mean_control), (x_hi, mean_control)],
            10,
            8,
            ShapeStyle::from(&PAL_E5_CONTROL.mix(0.98)).stroke_width(7),
        )))?;
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::Coordinate(130, 110))
        .background_style(WHITE.mix(0.88))
        .border_style(BLACK)
        .label_font(("sans-serif", 56).into_font())
        .draw()?;

    Ok(())
}

/// Panel C: Seed-level Pearson r dot + box summary
fn draw_e5_pearson_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    pearson_by_cond: &[(E5Condition, Vec<f32>)],
) -> Result<(), Box<dyn Error>>
where
    <DB as DrawingBackend>::ErrorType: 'static,
{
    let n_conds = pearson_by_cond.len() as f32;
    let y_lo = pearson_by_cond
        .iter()
        .flat_map(|(_, rs)| rs.iter().copied())
        .fold(0.0f32, f32::min)
        .min(-0.1)
        - 0.05;
    let y_hi = pearson_by_cond
        .iter()
        .flat_map(|(_, rs)| rs.iter().copied())
        .fold(0.0f32, f32::max)
        .max(0.6)
        + 0.05;

    let mut chart = ChartBuilder::on(area)
        .caption("C. Pearson r(C_field, PLV)", ("sans-serif", 72))
        .margin(20)
        .x_label_area_size(90)
        .y_label_area_size(120)
        .build_cartesian_2d(-0.5f32..(n_conds - 0.5), y_lo..y_hi)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .y_desc("Pearson r")
        .x_labels(pearson_by_cond.len())
        .x_label_formatter(&|x| {
            let idx = x.round() as usize;
            pearson_by_cond
                .get(idx)
                .map(|(c, _)| c.label().to_string())
                .unwrap_or_default()
        })
        .label_style(("sans-serif", 52).into_font())
        .axis_desc_style(("sans-serif", 56).into_font())
        .draw()?;

    // Zero line
    chart.draw_series(std::iter::once(DashedPathElement::new(
        vec![(-0.5f32, 0.0f32), (n_conds - 0.5, 0.0)],
        2,
        6,
        ShapeStyle::from(&BLACK.mix(0.3)).stroke_width(2),
    )))?;

    for (i, (cond, rs)) in pearson_by_cond.iter().enumerate() {
        let color = cond.color();
        let center = i as f32;

        if rs.is_empty() {
            continue;
        }

        // Individual dots with jitter
        let mut jitter_rng = seeded_rng(i as u64 + 0xD07);
        for &r in rs {
            let jx = jitter_rng.random_range(-0.15f32..0.15);
            chart.draw_series(std::iter::once(Circle::new(
                (center + jx, r),
                5,
                color.mix(0.5).filled(),
            )))?;
        }

        // Box: mean +/- std
        let (mean, std) = mean_std_scalar(rs);
        let x0 = center - 0.2;
        let x1 = center + 0.2;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, (mean - std).max(y_lo)), (x1, (mean + std).min(y_hi))],
            color.mix(0.25).filled(),
        )))?;
        // Mean line
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x0, mean), (x1, mean)],
            ShapeStyle::from(&color).stroke_width(4),
        )))?;
    }

    Ok(())
}

fn draw_vertical_guides<DB: DrawingBackend>(
    chart: &mut ChartContext<DB, Cartesian2d<RangedCoordf32, RangedCoordf32>>,
    xs: &[f32],
    y_min: f32,
    y_max: f32,
) -> Result<(), Box<dyn Error>>
where
    DB::ErrorType: 'static,
{
    for &x in xs {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(x, y_min), (x, y_max)],
            BLACK.mix(0.2),
        )))?;
    }
    Ok(())
}

fn histogram_counts(values: &[f32], min: f32, max: f32, bin_width: f32) -> Vec<(f32, usize)> {
    if bin_width <= 0.0 {
        return Vec::new();
    }
    let bins = ((max - min) / bin_width).ceil().max(1.0) as usize;
    let mut counts = vec![0usize; bins];
    for &value in values {
        if value < min || value > max {
            continue;
        }
        let mut idx = ((value - min) / bin_width).floor() as isize;
        if idx as usize >= bins {
            idx = (bins - 1) as isize;
        }
        if idx >= 0 {
            counts[idx as usize] += 1;
        }
    }
    (0..bins)
        .map(|i| (min + i as f32 * bin_width, counts[i]))
        .collect()
}

struct SlidingPlv {
    window: usize,
    buf_cos: Vec<f32>,
    buf_sin: Vec<f32>,
    sum_cos: f32,
    sum_sin: f32,
    idx: usize,
    len: usize,
}

impl SlidingPlv {
    fn new(window: usize) -> Self {
        let window = window.max(1);
        Self {
            window,
            buf_cos: vec![0.0; window],
            buf_sin: vec![0.0; window],
            sum_cos: 0.0,
            sum_sin: 0.0,
            idx: 0,
            len: 0,
        }
    }

    fn push(&mut self, angle: f32) {
        let c = angle.cos();
        let s = angle.sin();
        if self.len < self.window {
            self.buf_cos[self.idx] = c;
            self.buf_sin[self.idx] = s;
            self.sum_cos += c;
            self.sum_sin += s;
            self.len += 1;
            self.idx = (self.idx + 1) % self.window;
            return;
        }

        let old_c = self.buf_cos[self.idx];
        let old_s = self.buf_sin[self.idx];
        self.sum_cos -= old_c;
        self.sum_sin -= old_s;
        self.buf_cos[self.idx] = c;
        self.buf_sin[self.idx] = s;
        self.sum_cos += c;
        self.sum_sin += s;
        self.idx = (self.idx + 1) % self.window;
    }

    fn plv(&self) -> f32 {
        if self.len == 0 {
            return 0.0;
        }
        let inv = 1.0 / self.len as f32;
        let mean_cos = self.sum_cos * inv;
        let mean_sin = self.sum_sin * inv;
        (mean_cos * mean_cos + mean_sin * mean_sin).sqrt()
    }

    fn is_full(&self) -> bool {
        self.len >= self.window
    }
}

fn pairwise_interval_samples(semitones: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let eps = 1e-6f32;
    for i in 0..semitones.len() {
        for j in (i + 1)..semitones.len() {
            let diff = (semitones[i] - semitones[j]).abs();
            let mut folded = diff.rem_euclid(12.0);
            if folded >= 12.0 - eps {
                folded = 0.0;
            }
            out.push(folded);
        }
    }
    out
}

fn build_survival_series(lifetimes: &[u32], max_t: usize) -> Vec<(f32, f32)> {
    if lifetimes.is_empty() {
        return Vec::new();
    }
    let mut sorted = lifetimes.to_vec();
    sorted.sort_unstable();
    let total = sorted.len() as f32;
    let mut idx = 0usize;
    let mut series = Vec::with_capacity(max_t + 1);
    for t in 0..=max_t {
        let t_u32 = t as u32;
        while idx < sorted.len() && sorted[idx] < t_u32 {
            idx += 1;
        }
        let survivors = (sorted.len() - idx) as f32;
        series.push((t as f32, survivors / total));
    }
    series
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_e2_run(metric: f32, seed: u64) -> E2Run {
        E2Run {
            seed,
            mean_c_series: vec![metric],
            mean_c_level_series: vec![metric],
            mean_c_score_loo_series: vec![metric],
            mean_c_score_chosen_loo_series: vec![metric],
            mean_score_series: vec![0.0],
            mean_crowding_series: vec![0.0],
            moved_frac_series: vec![0.0],
            accepted_worse_frac_series: vec![0.0],
            attempted_update_frac_series: vec![0.0],
            moved_given_attempt_frac_series: vec![0.0],
            mean_abs_delta_semitones_series: vec![0.0],
            mean_abs_delta_semitones_moved_series: vec![0.0],
            semitone_samples_pre: Vec::new(),
            semitone_samples_post: Vec::new(),
            final_semitones: Vec::new(),
            final_freqs_hz: Vec::new(),
            final_log2_ratios: Vec::new(),
            trajectory_semitones: Vec::new(),
            trajectory_c_level: Vec::new(),
            anchor_shift: E2AnchorShiftStats {
                step: 0,
                anchor_hz_before: 0.0,
                anchor_hz_after: 0.0,
                count_min: 0,
                count_max: 0,
                respawned: 0,
            },
            density_mass_mean: 0.0,
            density_mass_min: 0.0,
            density_mass_max: 0.0,
            r_state01_min: 0.0,
            r_state01_mean: 0.0,
            r_state01_max: 0.0,
            r_ref_peak: 0.0,
            roughness_k: 0.0,
            roughness_ref_eps: 0.0,
            fixed_drone_hz: E4_ANCHOR_HZ,
            n_agents: 0,
            k_bins: 0,
        }
    }

    #[test]
    fn e2_c_snapshot_uses_anchor_shift_pre_when_enabled() {
        let series: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (init, pre, post) = e2_c_snapshot_series(&series, true, 6, 3, series.len());
        assert!((init - 0.0).abs() < 1e-6);
        assert!((pre - 5.0).abs() < 1e-6);
        assert!((post - 9.0).abs() < 1e-6);
    }

    #[test]
    fn e2_c_snapshot_uses_burnin_end_when_shift_disabled() {
        let series: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let (init, pre, post) = e2_c_snapshot_series(&series, false, 6, 3, series.len());
        assert!((init - 0.0).abs() < 1e-6);
        assert!((pre - 2.0).abs() < 1e-6);
        assert!((post - 9.0).abs() < 1e-6);
    }

    #[test]
    fn e2_density_normalization_invariant_to_scale() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid(&space);
        let anchor_idx = space.n_bins() / 2;
        let (env_scan, mut density_scan) = build_env_scans(&space, anchor_idx, &[], &du_scan);

        let (_, c_level_a, _, _) =
            compute_c_score_level_scans(&space, &workspace, &env_scan, &density_scan, &du_scan);

        for v in density_scan.iter_mut() {
            *v *= 4.0;
        }
        let (_, c_level_b, _, _) =
            compute_c_score_level_scans(&space, &workspace, &env_scan, &density_scan, &du_scan);

        assert_eq!(c_level_a.len(), c_level_b.len());
        for i in 0..c_level_a.len() {
            assert!(
                (c_level_a[i] - c_level_b[i]).abs() < 1e-5,
                "i={i} a={} b={}",
                c_level_a[i],
                c_level_b[i]
            );
        }
    }

    #[test]
    fn r_state01_stats_clamp_range() {
        let scan = [-0.5f32, 0.2, 1.2];
        let stats = r_state01_stats(&scan);
        assert!(stats.min >= 0.0 && stats.min <= 1.0);
        assert!(stats.mean >= 0.0 && stats.mean <= 1.0);
        assert!(stats.max >= 0.0 && stats.max <= 1.0);
    }

    #[test]
    fn consonant_exclusion_flags_targets() {
        for &st in &[0.0, 3.0, 4.0, 7.0, 12.0] {
            assert!(is_consonant_near(st), "expected consonant near {st}");
        }
    }

    #[test]
    fn consonant_exclusion_rejects_clear_outside() {
        for &st in &[1.0, 2.0, 6.0, 11.0] {
            assert!(!is_consonant_near(st), "expected non-consonant near {st}");
        }
    }

    #[test]
    fn histogram_counts_include_endpoints() {
        let values = [0.0f32, 1.0f32];
        let counts = histogram_counts(&values, 0.0, 1.0, 0.5);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[0].1, 1);
        assert_eq!(counts[1].1, 1);
    }

    #[test]
    fn histogram_counts_sum_matches_in_range() {
        let values = [0.0f32, 0.25, 0.5, 1.0, 1.5, -0.2];
        let counts = histogram_counts(&values, 0.0, 1.0, 0.5);
        let sum: usize = counts.iter().map(|(_, c)| *c).sum();
        assert_eq!(sum, 4);
    }

    #[test]
    fn pairwise_intervals_have_expected_count_and_range() {
        let semitones = [0.0f32, 3.0, 4.0, 7.0];
        let pairs = pairwise_interval_samples(&semitones);
        assert_eq!(pairs.len(), 6);
        for &v in &pairs {
            assert!(v >= 0.0 && v <= 12.0 + 1e-6, "value out of range: {v}");
        }
    }

    #[test]
    fn pairwise_interval_fold_maps_octave_to_zero() {
        let semitones = [0.0f32, 12.0];
        let pairs = pairwise_interval_samples(&semitones);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].abs() < 1e-6, "expected 0, got {}", pairs[0]);
    }

    #[test]
    fn pairwise_interval_octave_hits_zero_hist_bin() {
        let semitones = [0.0f32, 12.0];
        let pairs = pairwise_interval_samples(&semitones);
        let hist = histogram_counts_fixed(&pairs, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].abs() < 1e-6);
        assert!(
            !hist.is_empty() && hist[0].1 > 0.5,
            "expected first bin to receive octave sample"
        );
        let tail = hist.last().map(|(_, c)| *c).unwrap_or(0.0);
        assert!(tail < 0.5, "expected last bin not to receive octave sample");
    }

    #[test]
    fn close_pair_fraction_counts_absolute_near_unisons_only() {
        let semitones = [0.0f32, 0.20, 0.60, 7.0];
        let frac = close_pair_fraction_for_semitones(&semitones, 0.5);
        assert!(
            (frac - (2.0 / 6.0)).abs() < 1e-6,
            "expected 2 close pairs out of 6, got {frac}"
        );
    }

    #[test]
    fn fixed_drone_is_included_in_field_scans() {
        let space = Log2Space::new(200.0, 400.0, 12);
        let (_erb_scan, du_scan) = erb_grid(&space);
        let drone = e2_fixed_drone(&space, E4_ANCHOR_HZ);
        let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
            &space,
            std::slice::from_ref(&drone.idx),
            &[],
            &du_scan,
        );
        assert!(
            env_scan[drone.idx] > 0.0,
            "expected fixed drone in env scan"
        );
        assert!(
            density_scan[drone.idx] > 0.0,
            "expected fixed drone in density scan"
        );
    }

    #[test]
    fn fixed_drone_is_audible_but_excluded_from_adaptive_scene_metrics() {
        let mut run = test_e2_run(0.0, 0xD06E);
        run.fixed_drone_hz = E4_ANCHOR_HZ;
        run.n_agents = 1;
        run.trajectory_semitones = vec![vec![12.0]];
        run.trajectory_c_level = vec![vec![0.5]];
        run.final_semitones = vec![12.0];
        run.final_log2_ratios = vec![1.0];
        run.final_freqs_hz = vec![E4_ANCHOR_HZ * 2.0];

        let adaptive_freqs = e2_freqs_at_step(&run, 0);
        assert_eq!(adaptive_freqs.len(), 1);
        assert!((adaptive_freqs[0] - E4_ANCHOR_HZ * 2.0).abs() < 1e-4);

        let render_freqs = e2_trajectory_freqs_at_step(
            &run.trajectory_semitones,
            run.fixed_drone_hz,
            0,
            None,
            Some(run.fixed_drone_hz),
        );
        assert_eq!(render_freqs.len(), 2);
        assert!((render_freqs[0] - E4_ANCHOR_HZ * 2.0).abs() < 1e-4);
        assert!((render_freqs[1] - E4_ANCHOR_HZ).abs() < 1e-4);

        let csv = e2_scene_metrics_csv(&run, E4_ANCHOR_HZ);
        let row = csv
            .lines()
            .nth(1)
            .expect("expected one scene-metrics data row");
        let fields: Vec<&str> = row.split(',').collect();
        assert_eq!(fields.last().copied(), Some("1"));
    }

    #[test]
    fn g_scene_is_finite_and_uses_audible_fixed_drone() {
        let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid(&space);
        let freqs = [E4_ANCHOR_HZ, E4_ANCHOR_HZ * 1.5];
        let point = e2_scene_g_point_for_freqs(&space, &workspace, &du_scan, &freqs);
        assert!(point.h_scene.is_finite() && point.h_scene >= 0.0);
        assert!(point.h_single_mean.is_finite() && point.h_single_mean >= 0.0);
        assert!(point.h_social.is_finite());
        assert!(point.r_scene.is_finite() && point.r_scene >= 0.0);
        assert!(point.r_single_mean.is_finite() && point.r_single_mean >= 0.0);
        assert!(point.r_social.is_finite());
        assert!(point.g_scene.is_finite());
        assert!(
            point.h_scene > 0.0 && point.r_scene > 0.0,
            "unexpectedly pathological G(F) components: {:?}",
            point
        );
    }

    #[test]
    fn scene_roughness_feature_penalizes_close_intervals() {
        let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
        let workspace = build_consonance_workspace(&space);
        let (_erb_scan, du_scan) = erb_grid(&space);
        let close = [E4_ANCHOR_HZ, E4_ANCHOR_HZ * 2.0_f32.powf(1.0 / 12.0)];
        let consonant = [E4_ANCHOR_HZ, E4_ANCHOR_HZ * 1.5];
        let p_close = e2_scene_g_point_for_freqs(&space, &workspace, &du_scan, &close);
        let p_consonant = e2_scene_g_point_for_freqs(&space, &workspace, &du_scan, &consonant);
        assert!(
            p_close.r_social > p_consonant.r_social + 1e-3,
            "expected close interval to have higher social roughness: close={:?} consonant={:?}",
            p_close,
            p_consonant
        );
    }

    #[test]
    fn rebin_histogram_series_merges_contiguous_bins() {
        let centers = vec![0.025, 0.075, 0.125, 0.175, 0.225];
        let mean = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let std = vec![0.01, 0.02, 0.03, 0.04, 0.05];
        let (centers_out, mean_out, std_out) =
            rebin_histogram_series(&centers, &mean, &std, 0.05, 0.25);
        assert_eq!(centers_out.len(), 1);
        assert!((centers_out[0] - 0.125).abs() < 1e-6);
        assert!((mean_out[0] - 1.5).abs() < 1e-6);
        let std_expected = (0.01f32.powi(2)
            + 0.02f32.powi(2)
            + 0.03f32.powi(2)
            + 0.04f32.powi(2)
            + 0.05f32.powi(2))
        .sqrt();
        assert!((std_out[0] - std_expected).abs() < 1e-6);
    }

    #[test]
    fn snap_to_hist_bin_center_targets_display_centers() {
        assert!((snap_to_hist_bin_center(386.314, 25.0) - 387.5).abs() < 1e-3);
        assert!((snap_to_hist_bin_center(498.045, 25.0) - 487.5).abs() < 1e-3);
        assert!((snap_to_hist_bin_center(701.955, 25.0) - 712.5).abs() < 1e-3);
    }

    #[test]
    fn rhai_e6_segment_tracks_life_births_and_deaths() {
        let snaps = vec![
            E6PitchSnapshot {
                step: 0,
                freqs_hz: vec![220.0],
                agents: vec![E6AgentSnapshot {
                    life_id: 10,
                    agent_id: 0,
                    freq_hz: 220.0,
                }],
            },
            E6PitchSnapshot {
                step: 25,
                freqs_hz: vec![233.08],
                agents: vec![E6AgentSnapshot {
                    life_id: 10,
                    agent_id: 0,
                    freq_hz: 233.08,
                }],
            },
            E6PitchSnapshot {
                step: 50,
                freqs_hz: vec![330.0],
                agents: vec![E6AgentSnapshot {
                    life_id: 11,
                    agent_id: 0,
                    freq_hz: 330.0,
                }],
            },
        ];
        let refs: Vec<&E6PitchSnapshot> = snaps.iter().collect();
        let rhai = rhai_e6_segment("test", &refs, 0.12, None);
        assert!(rhai.contains("let l10 = create(e6_voice, 1).freq(220.00).amp("));
        assert!(rhai.contains("let l10 = l10.freq(233.08);"));
        assert!(rhai.contains("release(l10);"));
        assert!(rhai.contains("let l11 = create(e6_voice, 1).freq(330.00).amp("));
        assert!(rhai.contains("// terminal_release"));
    }

    #[test]
    fn e2_burn_band_x_uses_post_switch_window() {
        let band = e2_burn_band_x(10, Some(25), 0, 49);
        assert_eq!(band, Some((25.0, 35.0)));
    }

    #[test]
    fn interval_distance_mod_12_maps_octave_equivalence() {
        let d = interval_distance_mod_12(12.0, 0.0);
        assert!(
            d.abs() < 1e-6,
            "expected octave equivalence distance 0, got {d}"
        );
        let d = interval_distance_mod_12(11.9, 0.0);
        assert!(
            (d - 0.1).abs() < 1e-4,
            "expected wrap-around distance 0.1, got {d}"
        );
    }

    #[test]
    fn consonant_mass_for_intervals_counts_target_windows() {
        // Values near JI targets: 3.16 ≈ 6:5, 3.90 ≈ 5:4, 7.05 ≈ 3:2
        let intervals = [3.16f32, 3.90, 6.0, 7.05, 11.9, 0.1, 12.0];
        let core_mass = consonant_mass_for_intervals(&intervals, &E2_CONSONANT_TARGETS_CORE, 0.25);
        assert!(
            (core_mass - 3.0 / 7.0).abs() < 1e-6,
            "core_mass={core_mass}"
        );

        let octave_mass = consonant_mass_for_intervals(&intervals, &[0.0], 0.25);
        assert!(
            (octave_mass - 3.0 / 7.0).abs() < 1e-6,
            "octave_mass={octave_mass}"
        );
    }

    #[test]
    fn fold_hist_abs_semitones_merges_sign_pairs() {
        let centers = [-1.0f32, 0.0, 1.0];
        let mean = [0.2f32, 0.1, 0.3];
        let std = [0.01f32, 0.02, 0.03];
        let (fold_c, fold_m, fold_s) = fold_hist_abs_semitones(&centers, &mean, &std, 1.0);
        assert_eq!(fold_c, vec![0.0, 1.0]);
        assert!((fold_m[0] - 0.1).abs() < 1e-6);
        assert!((fold_m[1] - 0.5).abs() < 1e-6);
        let expected_std = (0.01f32 * 0.01 + 0.03 * 0.03).sqrt();
        assert!((fold_s[0] - 0.02).abs() < 1e-6);
        assert!((fold_s[1] - expected_std).abs() < 1e-6);
    }

    #[test]
    fn exact_permutation_pvalue_mean_diff_small_sample() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0];
        let p = exact_permutation_pvalue_mean_diff(&a, &b);
        assert!((p - (3.0 / 21.0)).abs() < 1e-6, "unexpected exact p={p}");
    }

    #[test]
    fn permutation_pvalue_mean_diff_small_sample_uses_exact() {
        let a = [1.0f32, 1.0, 1.0];
        let b = [0.0f32, 0.0, 0.0];
        let p_exact = exact_permutation_pvalue_mean_diff(&a, &b);
        let (p, method, n_perm) = permutation_pvalue_mean_diff(&a, &b, 1_000, 2_000, 1234);
        assert_eq!(method, "exact");
        assert_eq!(n_perm, 20);
        assert!(
            (p - p_exact).abs() < 1e-6,
            "auto exact p={p} differed from exact p={p_exact}"
        );
    }

    #[test]
    fn split_by_median_and_quartiles_sizes() {
        let values = vec![0.1f32, 0.2, 0.3, 0.4];
        let (high, low) = split_by_median(&values);
        assert_eq!(high.len(), 2);
        assert_eq!(low.len(), 2);
        let (high_q, low_q) = split_by_quartiles(&values);
        assert_eq!(high_q.len(), 2);
        assert_eq!(low_q.len(), 1);
    }

    #[test]
    fn float_key_roundtrip_is_stable() {
        let values = [0.02f32, 0.25, 0.98, 1.0];
        for value in values {
            let key = float_key(value);
            let expected = (value * FLOAT_KEY_SCALE).round() as i32;
            assert_eq!(key, expected);
            let roundtrip = float_from_key(key);
            assert!((roundtrip - value).abs() < 1e-6);
        }
    }

    #[test]
    fn logrank_statistic_zero_when_equal() {
        let high = vec![1u32, 2, 3];
        let low = vec![1u32, 2, 3];
        let stat = logrank_statistic(&high, &low);
        assert!(stat.abs() < 1e-6);
    }

    #[test]
    fn shift_indices_by_ratio_respawns_and_clamps() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let n = space.n_bins();
        assert!(n > 6);
        let mut rng = seeded_rng(42);
        let min_idx = 2usize;
        let max_idx = 5usize;
        let mut indices = vec![0usize, 1, n - 2, n - 1];
        let (_count_min, _count_max, respawned) =
            shift_indices_by_ratio(&space, &mut indices, 10.0, min_idx, max_idx, &mut rng);
        assert!(respawned > 0);
        assert!(indices.iter().all(|&idx| idx >= min_idx && idx <= max_idx));
    }
}

// ── Independent consonance metric: Just Intonation ratio-complexity ──

/// JI ratio table: (interval_semitones, weight = 1/(p*q) for ratio p:q)
const JI_RATIOS: [(f32, f32); 8] = [
    (0.0, 1.000),    // 1:1
    (12.0, 0.500),   // 2:1
    (7.0196, 0.167), // 3:2
    (4.9804, 0.083), // 4:3
    (3.8631, 0.050), // 5:4
    (8.8436, 0.067), // 5:3
    (3.1564, 0.033), // 6:5
    (8.1369, 0.025), // 8:5
];
/// Gaussian sharpness for JI proximity (half-width ~0.42 semitones)
const JI_ALPHA: f32 = 4.0;

/// Score a single interval (in semitones, mod 12) against the JI ratio table.
/// Returns the weighted Gaussian proximity to the nearest JI ratio.
fn ji_interval_consonance(interval_st: f32) -> f32 {
    let mut iv = interval_st.abs() % 12.0;
    if iv > 6.0 {
        iv = 12.0 - iv; // fold to [0, 6]
    }
    let mut best = 0.0f32;
    for &(ji_st, weight) in &JI_RATIOS {
        let mut ji = ji_st % 12.0;
        if ji > 6.0 {
            ji = 12.0 - ji;
        }
        let dist = (iv - ji).abs();
        let score = weight * (-JI_ALPHA * dist * dist).exp();
        if score > best {
            best = score;
        }
    }
    best
}

/// Mean pairwise JI consonance score for a population of frequencies.
fn ji_population_score(freqs_hz: &[f32], anchor_hz: f32) -> f32 {
    let semitones: Vec<f32> = freqs_hz
        .iter()
        .filter(|f| f.is_finite() && **f > 0.0)
        .map(|f| 12.0 * (*f / anchor_hz).log2())
        .collect();
    if semitones.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0f32;
    let mut count = 0u32;
    for i in 0..semitones.len() {
        for j in (i + 1)..semitones.len() {
            total += ji_interval_consonance(semitones[i] - semitones[j]);
            count += 1;
        }
    }
    if count > 0 { total / count as f32 } else { 0.0 }
}

fn e2_freqs_at_step(run: &E2Run, step: usize) -> Vec<f32> {
    run.trajectory_semitones
        .iter()
        .filter_map(|trace| {
            trace.get(step).copied().and_then(|st| {
                if st.is_finite() {
                    Some(run.fixed_drone_hz * 2.0_f32.powf(st / 12.0))
                } else {
                    None
                }
            })
        })
        .collect()
}

fn e2_scene_ji_series(run: &E2Run) -> Vec<f32> {
    let n_steps = run
        .trajectory_semitones
        .iter()
        .map(|trace| trace.len())
        .min()
        .unwrap_or(0);
    let mut series = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let freqs = e2_freqs_at_step(run, step);
        series.push(ji_population_score(&freqs, run.fixed_drone_hz));
    }
    series
}

fn e2_scene_g_series(run: &E2Run, space: &Log2Space) -> Vec<E2SceneGPoint> {
    let workspace = build_consonance_workspace(space);
    let (_erb_scan, du_scan) = erb_grid(space);
    let n_steps = run
        .trajectory_semitones
        .iter()
        .map(|trace| trace.len())
        .min()
        .unwrap_or(0);
    let mut series = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let freqs = e2_trajectory_freqs_at_step(
            &run.trajectory_semitones,
            run.fixed_drone_hz,
            step,
            None,
            Some(run.fixed_drone_hz),
        );
        series.push(e2_scene_g_point_for_freqs(
            space, &workspace, &du_scan, &freqs,
        ));
    }
    series
}

fn e2_scene_g_pair_csv(baseline: &[E2SceneGPoint], nohill: &[E2SceneGPoint]) -> String {
    let len = baseline.len().min(nohill.len());
    let mut out = String::from(
        "# G(F)=a*H_soc+b*R_soc+c*H_soc*R_soc; H_soc=H_scene-mean_i H(singleton_i); R_soc=R_scene-mean_i R(singleton_i); scene includes adaptive voices plus fixed drone\n\
step,baseline_g_scene,baseline_h_scene,baseline_h_single_mean,baseline_h_social,baseline_r_scene,baseline_r_single_mean,baseline_r_social,nohill_g_scene,nohill_h_scene,nohill_h_single_mean,nohill_h_social,nohill_r_scene,nohill_r_single_mean,nohill_r_social\n",
    );
    for step in 0..len {
        let b = baseline[step];
        let n = nohill[step];
        out.push_str(&format!(
            "{step},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
            b.g_scene,
            b.h_scene,
            b.h_single_mean,
            b.h_social,
            b.r_scene,
            b.r_single_mean,
            b.r_social,
            n.g_scene,
            n.h_scene,
            n.h_single_mean,
            n.h_social,
            n.r_scene,
            n.r_single_mean,
            n.r_social,
        ));
    }
    out
}

fn render_e2_scene_g_pair_plot(
    out_path: &Path,
    baseline: &[E2SceneGPoint],
    nohill: &[E2SceneGPoint],
    phase_mode: E2PhaseMode,
    n_agents: usize,
) -> Result<(), Box<dyn Error>> {
    if baseline.is_empty() || nohill.is_empty() {
        return Ok(());
    }
    let len = baseline.len().min(nohill.len());
    let x_max = len.saturating_sub(1);
    let mut y_min = f32::INFINITY;
    let mut y_max = f32::NEG_INFINITY;
    for point in baseline.iter().take(len).chain(nohill.iter().take(len)) {
        y_min = y_min.min(point.g_scene);
        y_max = y_max.max(point.g_scene);
    }
    if !y_min.is_finite() || !y_max.is_finite() {
        y_min = -1.0;
        y_max = 1.0;
    }
    let pad = ((y_max - y_min).abs() * 0.1).max(1e-3);
    y_min -= pad;
    y_max += pad;

    let root = bitmap_root(out_path, (1200, 700)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("Scene Consonance G(F) Over Time", ("sans-serif", 32))
        .margin(10)
        .x_label_area_size(60)
        .y_label_area_size(75)
        .build_cartesian_2d(0f32..x_max.max(1) as f32, y_min..y_max)?;
    chart
        .configure_mesh()
        .x_desc("step")
        .y_desc("G(F) (a.u.)")
        .label_style(("sans-serif", 28).into_font())
        .axis_desc_style(("sans-serif", 28).into_font())
        .draw()?;

    if let Some((burn_x0, burn_x1)) = e2_burn_band_x(
        e2_trajectory_burn_in_step(n_agents),
        e2_trajectory_phase_switch_step(phase_mode, n_agents),
        0,
        x_max,
    ) {
        chart.draw_series(std::iter::once(Rectangle::new(
            [(burn_x0, y_min), (burn_x1, y_max)],
            RGBColor(180, 180, 180).mix(0.15).filled(),
        )))?;
    }
    if let Some(step) = e2_trajectory_phase_switch_step(phase_mode, n_agents)
        && step <= x_max
    {
        chart.draw_series(std::iter::once(PathElement::new(
            vec![(step as f32, y_min), (step as f32, y_max)],
            ShapeStyle::from(&BLACK.mix(0.55)).stroke_width(2),
        )))?;
    }

    for (condition, series, color) in [
        ("baseline", baseline, e2_condition_color("baseline")),
        ("nohill", nohill, e2_condition_color("nohill")),
    ] {
        let line = (0..len).map(|i| (i as f32, series[i].g_scene));
        chart
            .draw_series(LineSeries::new(line, color))?
            .label(e2_condition_display(condition))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 18, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .label_font(("sans-serif", 28).into_font())
        .draw()?;

    root.present()?;
    Ok(())
}

fn e2_scene_metrics_csv(run: &E2Run, _anchor_hz: f32) -> String {
    let ji_series = e2_scene_ji_series(run);
    let n_steps = run
        .mean_c_series
        .len()
        .min(run.mean_c_level_series.len())
        .min(run.mean_c_score_loo_series.len())
        .min(run.mean_c_score_chosen_loo_series.len())
        .min(run.mean_score_series.len())
        .min(run.mean_crowding_series.len())
        .min(run.moved_frac_series.len())
        .min(ji_series.len());
    let mut out = String::from(
        "step,mean_c_field,mean_c_level,mean_c_score_loo_current,mean_c_score_loo_chosen,mean_score,mean_crowding,moved_frac,ji_scene_score,pairwise_entropy,unique_bins_25ct\n",
    );
    for step in 0..n_steps {
        let semitones: Vec<f32> = run
            .trajectory_semitones
            .iter()
            .filter_map(|trace| trace.get(step).copied())
            .filter(|v| v.is_finite())
            .collect();
        let pairwise = pairwise_interval_samples(&semitones);
        let pairwise_probs =
            histogram_probabilities_fixed(&pairwise, 0.0, 12.0, E2_PAIRWISE_BIN_ST);
        let hist = hist_structure_metrics_from_probs(&pairwise_probs);
        let diversity = diversity_metrics_for_semitones(&semitones);
        out.push_str(&format!(
            "{step},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
            run.mean_c_series[step],
            run.mean_c_level_series[step],
            run.mean_c_score_loo_series[step],
            run.mean_c_score_chosen_loo_series[step],
            run.mean_score_series[step],
            run.mean_crowding_series[step],
            run.moved_frac_series[step],
            ji_series[step],
            hist.entropy,
            diversity.unique_bins
        ));
    }
    out
}

// ── Audio supplement Rhai generation ─────────────────────────────────────────

/// Audio replay timing constants (shared between E2 and E6 replay scripts)
const AUDIO_STEP_SEC: f32 = 0.12;
const AUDIO_GAP_SEC: f32 = 1.0;
const AUDIO_QUICKLISTEN_GAP_SEC: f32 = AUDIO_GAP_SEC;
const AUDIO_QUICKLISTEN_FADE_SEC: f32 = 1.0;
const AUDIO_INTEGRATION_FADE_SEC: f32 = 0.2;
const AUDIO_E2_POLYPHONY_TARGET_SEGMENT_SEC: f32 = 8.0;
const AUDIO_N_SELECT_E2: usize = 8;
const AUDIO_E2_AGENT_AMP: f32 = 0.10;
const AUDIO_E2_AGENT_ATTACK_SEC: f32 = 0.30;
const AUDIO_E2_AGENT_DECAY_SEC: f32 = 0.35;
const AUDIO_E2_AGENT_SUSTAIN_LEVEL: f32 = 0.50;
const AUDIO_E2_AGENT_RELEASE_SEC: f32 = 0.8;
const AUDIO_E2_AGENT_PARTIALS: usize = E4_ENV_PARTIALS_DEFAULT as usize;
const AUDIO_E6_STEP_SEC: f32 = 0.40;
/// How many E6 snapshots to replay per segment (last N of run)
const AUDIO_E6_TAIL_SNAPSHOTS: usize = 30;
const AUDIO_E6_AGENT_AMP: f32 = 0.070;
const AUDIO_E6_AGENT_PARTIALS: usize = E4_ENV_PARTIALS_DEFAULT as usize;
const AUDIO_E6_AGENT_SUSTAIN_DRIVE: f32 = 0.001;
const AUDIO_E6_AGENT_ATTACK_SEC: f32 = 0.08;
const AUDIO_E6_AGENT_DECAY_SEC: f32 = 0.45;
const AUDIO_E6_AGENT_SUSTAIN_LEVEL: f32 = 0.50;
const AUDIO_E6_AGENT_RELEASE_SEC: f32 = 0.45;
const AUDIO_E6_HS_GAIN: f32 = 1.0;

fn audio_landscape_agent_brightness() -> f32 {
    // HarmonicBody maps brightness linearly to spectral slope in [1.2, 0.2].
    const DARKEST_SLOPE: f32 = 1.2;
    const BRIGHTEST_SLOPE: f32 = 0.2;
    let slope = E4_ENV_PARTIAL_DECAY_DEFAULT.clamp(BRIGHTEST_SLOPE, DARKEST_SLOPE);
    ((DARKEST_SLOPE - slope) / (DARKEST_SLOPE - BRIGHTEST_SLOPE)).clamp(0.0, 1.0)
}

/// Header for generated Rhai replay scripts.
fn rhai_replay_header(title: &str, detail: &str) -> String {
    let audio_landscape_agent_brightness = audio_landscape_agent_brightness();
    format!(
        "// {title}\n\
         // Audio supplement for Conchordal (ALIFE 2026)\n\
         //\n\
         // AUTO-GENERATED by `cargo run --release --bin paper -- --audio-rhai`\n\
         // DO NOT EDIT\n\
         //\n\
         {detail}\n\n\
         let agent_spec = derive(harmonic)\n    \
             .brain(\"drone\")\n    \
             .pitch_mode(\"lock\")\n    \
             .amp({:.3})\n    \
             .modes(harmonic_modes().count({}))\n    \
             .brightness({:.3})\n    \
             .sustain_drive(0.001)\n    \
             .pitch_smooth(0.12)\n    \
             .adsr({:.3}, {:.3}, {:.3}, {:.3});\n\n",
        AUDIO_E2_AGENT_AMP,
        AUDIO_E2_AGENT_PARTIALS,
        audio_landscape_agent_brightness,
        AUDIO_E2_AGENT_ATTACK_SEC,
        AUDIO_E2_AGENT_DECAY_SEC,
        AUDIO_E2_AGENT_SUSTAIN_LEVEL,
        AUDIO_E2_AGENT_RELEASE_SEC
    )
}

/// Emit Rhai scene for an E2 segment: create agents, step through trajectory, release.
/// Linearly interpolates between simulation steps to reduce grid-quantisation beating.
fn e2_trajectory_freqs_at_step(
    trajectory_semitones: &[Vec<f32>],
    reference_hz: f32,
    step: usize,
    selected: Option<&[usize]>,
    fixed_drone_hz: Option<f32>,
) -> Vec<f32> {
    let mut freqs = Vec::new();
    match selected {
        Some(selected) => {
            freqs.reserve(selected.len() + usize::from(fixed_drone_hz.is_some()));
            for &agent_i in selected {
                let Some(trace) = trajectory_semitones.get(agent_i) else {
                    continue;
                };
                let Some(&st) = trace.get(step) else {
                    continue;
                };
                if st.is_finite() {
                    freqs.push(reference_hz * 2.0_f32.powf(st / 12.0));
                }
            }
        }
        None => {
            freqs.reserve(trajectory_semitones.len() + usize::from(fixed_drone_hz.is_some()));
            for trace in trajectory_semitones {
                let Some(&st) = trace.get(step) else {
                    continue;
                };
                if st.is_finite() {
                    freqs.push(reference_hz * 2.0_f32.powf(st / 12.0));
                }
            }
        }
    }
    if let Some(hz) = fixed_drone_hz {
        freqs.push(hz.max(1.0));
    }
    freqs
}

fn rhai_e2_segment(
    scene_name: &str,
    trajectory_semitones: &[Vec<f32>],
    anchor_hz: f32,
    selected: &[usize],
    fixed_drone_hz: Option<f32>,
    step_sec: f32,
) -> String {
    let mut s = format!("scene(\"{scene_name}\", || {{\n");

    // Create agents at initial freq
    for (i, &agent_i) in selected.iter().enumerate() {
        let st = trajectory_semitones[agent_i][0];
        let freq = anchor_hz * 2.0_f32.powf(st / 12.0);
        s.push_str(&format!(
            "    let a{i} = create(agent_spec, 1).freq({freq:.2});\n"
        ));
    }
    if let Some(hz) = fixed_drone_hz {
        s.push_str(&format!(
            "    let d0 = create(agent_spec, 1).freq({hz:.2});\n"
        ));
    }
    s.push_str("    flush();\n");

    let n_steps = trajectory_semitones[selected[0]].len();
    // Track previous emitted semitone per agent to suppress no-change updates
    let mut prev_emitted: Vec<f32> = selected
        .iter()
        .map(|&ai| trajectory_semitones[ai][0])
        .collect();

    for step in 1..n_steps {
        s.push_str(&format!("    wait({step_sec:.4});\n"));
        let mut any_changed = false;
        for (i, &agent_i) in selected.iter().enumerate() {
            let st1 = trajectory_semitones[agent_i][step];
            if (st1 - prev_emitted[i]).abs() > 1e-4 {
                let freq = anchor_hz * 2.0_f32.powf(st1 / 12.0);
                s.push_str(&format!("    a{i}.freq({freq:.2});\n"));
                prev_emitted[i] = st1;
                any_changed = true;
            }
        }
        if any_changed {
            s.push_str("    flush();\n");
        }
    }

    // Final hold
    s.push_str(&format!("    wait({step_sec:.3});\n"));
    s.push_str("});\n");
    s
}

fn downsample_e2_trajectory_for_audio(
    trajectory_semitones: &[Vec<f32>],
    max_steps: usize,
) -> Vec<Vec<f32>> {
    if trajectory_semitones.is_empty() || max_steps == 0 {
        return trajectory_semitones.to_vec();
    }
    let n_steps = trajectory_semitones[0].len();
    if n_steps <= max_steps || max_steps <= 1 {
        return trajectory_semitones.to_vec();
    }

    let mut sampled_indices = Vec::with_capacity(max_steps);
    for out_idx in 0..max_steps {
        let t = out_idx as f32 / (max_steps - 1) as f32;
        let src_idx = (t * (n_steps - 1) as f32).round() as usize;
        sampled_indices.push(src_idx.min(n_steps - 1));
    }

    trajectory_semitones
        .iter()
        .map(|trace| {
            sampled_indices
                .iter()
                .map(|&idx| {
                    trace
                        .get(idx)
                        .copied()
                        .unwrap_or(*trace.last().unwrap_or(&f32::NAN))
                })
                .collect()
        })
        .collect()
}

fn e6_audio_agents_at_snapshot<'a>(
    snapshot: &'a E6PitchSnapshot,
    selected_agent_ids: Option<&std::collections::BTreeSet<usize>>,
) -> Vec<&'a E6AgentSnapshot> {
    let mut agents: Vec<&E6AgentSnapshot> = snapshot
        .agents
        .iter()
        .filter(|agent| agent.freq_hz.is_finite() && agent.freq_hz > 20.0)
        .filter(|agent| {
            selected_agent_ids
                .map(|ids| ids.contains(&agent.agent_id))
                .unwrap_or(true)
        })
        .collect();
    agents.sort_by_key(|agent| (agent.agent_id, agent.life_id));
    agents
}

/// Emit Rhai scene for an E6 segment as per-life sustained voices.
/// Each life gets a fresh attack/decay on birth, sustains while alive, and releases on death.
fn rhai_e6_segment(
    scene_name: &str,
    snapshots: &[&E6PitchSnapshot],
    step_sec: f32,
    selected_agent_ids: Option<&std::collections::BTreeSet<usize>>,
) -> String {
    rhai_e6_segment_with_gain(scene_name, snapshots, step_sec, selected_agent_ids, 1.0)
}

fn rhai_e6_segment_with_gain(
    scene_name: &str,
    snapshots: &[&E6PitchSnapshot],
    step_sec: f32,
    selected_agent_ids: Option<&std::collections::BTreeSet<usize>>,
    amp_scale: f32,
) -> String {
    let mut s = format!("scene(\"{scene_name}\", || {{\n");
    let audio_e6_agent_brightness = audio_landscape_agent_brightness();
    let voice_amp = AUDIO_E6_AGENT_AMP * amp_scale.max(0.0);

    if snapshots.is_empty() {
        s.push_str("    wait(1.0);\n});\n");
        return s;
    }
    s.push_str(&format!(
        "    let e6_voice = derive(harmonic)\n        .brain(\"drone\")\n        .pitch_mode(\"lock\")\n        .amp({:.3})\n        .modes(harmonic_modes().count({}))\n        .brightness({:.3})\n        .sustain_drive({:.3})\n        .pitch_smooth(0.12)\n        .adsr({:.3}, {:.3}, {:.3}, {:.3});\n",
        voice_amp,
        AUDIO_E6_AGENT_PARTIALS,
        audio_e6_agent_brightness,
        AUDIO_E6_AGENT_SUSTAIN_DRIVE,
        AUDIO_E6_AGENT_ATTACK_SEC,
        AUDIO_E6_AGENT_DECAY_SEC,
        AUDIO_E6_AGENT_SUSTAIN_LEVEL,
        AUDIO_E6_AGENT_RELEASE_SEC
    ));

    let initial_agents = e6_audio_agents_at_snapshot(snapshots[0], selected_agent_ids);
    let mut active_freqs: std::collections::BTreeMap<u64, f32> = std::collections::BTreeMap::new();
    for agent in initial_agents {
        active_freqs.insert(agent.life_id, agent.freq_hz);
        s.push_str(&format!(
            "    let l{} = create(e6_voice, 1).freq({:.2}).amp({:.3});\n",
            agent.life_id, agent.freq_hz, voice_amp
        ));
    }
    s.push_str("    flush();\n");

    for snap in snapshots.iter().skip(1) {
        let agents = e6_audio_agents_at_snapshot(snap, selected_agent_ids);
        let mut current_freqs: std::collections::BTreeMap<u64, f32> =
            std::collections::BTreeMap::new();
        let mut any_changed = false;
        s.push_str(&format!("    wait({step_sec:.4});\n"));
        for agent in agents {
            current_freqs.insert(agent.life_id, agent.freq_hz);
            match active_freqs.get(&agent.life_id) {
                Some(prev_freq_hz) => {
                    if (agent.freq_hz - *prev_freq_hz).abs() > 1e-4 {
                        s.push_str(&format!(
                            "    let l{} = l{}.freq({:.2});\n",
                            agent.life_id, agent.life_id, agent.freq_hz
                        ));
                        any_changed = true;
                    }
                }
                None => {
                    s.push_str(&format!(
                        "    let l{} = create(e6_voice, 1).freq({:.2}).amp({:.3});\n",
                        agent.life_id, agent.freq_hz, voice_amp
                    ));
                    any_changed = true;
                }
            }
        }
        for life_id in active_freqs.keys() {
            if !current_freqs.contains_key(life_id) {
                s.push_str(&format!("    release(l{life_id});\n"));
                any_changed = true;
            }
        }
        if any_changed {
            s.push_str("    flush();\n");
        }
        active_freqs = current_freqs;
    }

    let final_release_wait = AUDIO_E6_AGENT_RELEASE_SEC;
    s.push_str(&format!("    wait({step_sec:.3});\n"));
    s.push_str("    // terminal_release\n");
    for life_id in active_freqs.keys() {
        s.push_str(&format!("    release(l{life_id});\n"));
    }
    s.push_str("    flush();\n");
    s.push_str(&format!("    wait({final_release_wait:.3});\n"));
    s.push_str("});\n");
    s
}

#[derive(Debug, Clone)]
struct AudioSegmentWindow {
    label: String,
    start_sec: f32,
    end_sec: f32,
}

fn parse_rhai_scene_windows(rhai_path: &Path) -> io::Result<Vec<AudioSegmentWindow>> {
    let text = read_to_string(rhai_path)?;
    let mut windows = Vec::new();
    let mut current_label: Option<String> = None;
    let mut current_start_sec = 0.0f32;
    let mut current_time_sec = 0.0f32;
    let mut brace_depth: i32 = 0;
    let mut current_has_explicit_release = false;
    let mut current_first_release_sec: Option<f32> = None;
    let mut current_terminal_release_sec: Option<f32> = None;

    for line in text.lines() {
        let trimmed = line.trim();
        if current_label.is_none()
            && trimmed.starts_with("scene(\"")
            && let Some(rest) = trimmed.strip_prefix("scene(\"")
            && let Some((label, _)) = rest.split_once("\", || {")
        {
            current_label = Some(label.to_string());
            current_start_sec = current_time_sec;
            current_has_explicit_release = false;
            current_first_release_sec = None;
            current_terminal_release_sec = None;
        }

        for (idx, _) in trimmed.match_indices("wait(") {
            let start = idx + "wait(".len();
            if let Some(end_rel) = trimmed[start..].find(')') {
                let wait_str = &trimmed[start..start + end_rel];
                if let Ok(wait_sec) = wait_str.parse::<f32>() {
                    current_time_sec += wait_sec;
                }
            }
        }

        if current_label.is_some() {
            if trimmed == "// terminal_release" {
                current_terminal_release_sec = Some(current_time_sec);
            }
            if trimmed.contains("release(") {
                current_has_explicit_release = true;
                if current_first_release_sec.is_none() {
                    current_first_release_sec = Some(current_time_sec);
                }
            }
            brace_depth += trimmed.matches('{').count() as i32;
            brace_depth -= trimmed.matches('}').count() as i32;
            if brace_depth == 0 {
                windows.push(AudioSegmentWindow {
                    label: current_label.take().unwrap_or_default(),
                    start_sec: current_start_sec,
                    end_sec: if let Some(terminal_release_sec) = current_terminal_release_sec {
                        terminal_release_sec
                    } else if current_has_explicit_release {
                        current_first_release_sec.unwrap_or(current_time_sec)
                    } else {
                        current_time_sec + AUDIO_E2_AGENT_RELEASE_SEC
                    },
                });
            }
        }
    }

    Ok(windows)
}

fn resolve_audio_postprocess_paths(wav_stem: &str) -> (PathBuf, PathBuf, PathBuf) {
    let wav_name = format!("{wav_stem}.wav");
    let rhai_name = format!("{wav_stem}.rhai");
    if Path::new("supplementary_audio").is_dir() {
        (
            PathBuf::from("supplementary_audio/audio").join(&wav_name),
            PathBuf::from("supplementary_audio/scenarios").join(&rhai_name),
            PathBuf::from("supplementary_audio/manifest.csv"),
        )
    } else {
        (
            PathBuf::from("audio").join(&wav_name),
            PathBuf::from("scenarios").join(&rhai_name),
            PathBuf::from("manifest.csv"),
        )
    }
}

fn rewrite_audio_manifest(
    manifest_path: &Path,
    wav_name: &str,
    windows: &[AudioSegmentWindow],
) -> io::Result<()> {
    let text = read_to_string(manifest_path)?;
    fn condition_for_audio_label(label: &str) -> Option<&'static str> {
        match label {
            "shared" => Some("shared"),
            "scrambled" => Some("scrambled"),
            "off" => Some("off"),
            "heredity_selection" => Some("heredity+selection"),
            "local_search_excerpt" => Some("local-search"),
            "showcase_local_search" => Some("local-search"),
            "showcase_heredity_selection" => Some("heredity+selection"),
            "neither" => Some("neither"),
            "no_search_excerpt" => Some("no-search"),
            "controls_neither" => Some("neither"),
            "controls_no_search" => Some("no-search"),
            _ if label.starts_with("local_search_") => Some("local-search"),
            _ if label.starts_with("no_search_") => Some("no-search"),
            _ if label.starts_with("heredity_selection_") => Some("heredity+selection"),
            _ if label.starts_with("heredity_only_") => Some("heredity-only"),
            _ if label.starts_with("selection_only_") => Some("selection-only"),
            _ if label.starts_with("neither_") => Some("neither"),
            _ => None,
        }
    }
    fn assay_for_audio_label(label: &str) -> Option<&'static str> {
        match label {
            _ if label.starts_with("local_search_") => Some("Consonance Search"),
            _ if label.starts_with("no_search_") => Some("Consonance Search"),
            _ if label.starts_with("heredity_") => Some("Hereditary Adaptation"),
            _ if label.starts_with("selection_only_") => Some("Hereditary Adaptation"),
            _ if label.starts_with("neither_") => Some("Hereditary Adaptation"),
            "showcase_local_search" => Some("Consonance Search"),
            "showcase_heredity_selection" => Some("Hereditary Adaptation"),
            "controls_neither" => Some("Hereditary Adaptation"),
            "controls_no_search" => Some("Consonance Search"),
            "shared" | "scrambled" | "off" => Some("Temporal Scaffold"),
            _ => None,
        }
    }
    fn seed_for_audio_label(label: &str, existing: &str) -> String {
        match label {
            "showcase_heredity_selection" => E6B_SEEDS[0].to_string(),
            "showcase_local_search" => E2_SEEDS[3].to_string(),
            "controls_neither" => E6B_SEEDS[0].to_string(),
            "controls_no_search" => E2_SEEDS[0].to_string(),
            _ if label.ends_with("_seed0") => {
                if label.starts_with("heredity_")
                    || label.starts_with("selection_only_")
                    || label.starts_with("neither_")
                {
                    E6B_SEEDS[0].to_string()
                } else {
                    E2_SEEDS[0].to_string()
                }
            }
            _ if label.ends_with("_seed1") => {
                if label.starts_with("heredity_")
                    || label.starts_with("selection_only_")
                    || label.starts_with("neither_")
                {
                    E6B_SEEDS[1].to_string()
                } else {
                    E2_SEEDS[1].to_string()
                }
            }
            _ if label.ends_with("_seed2") => {
                if label.starts_with("heredity_")
                    || label.starts_with("selection_only_")
                    || label.starts_with("neither_")
                {
                    E6B_SEEDS[2].to_string()
                } else {
                    E2_SEEDS[2].to_string()
                }
            }
            _ if label.ends_with("_seed10") => {
                if label.starts_with("heredity_")
                    || label.starts_with("selection_only_")
                    || label.starts_with("neither_")
                {
                    E6B_SEEDS[10].to_string()
                } else {
                    E2_SEEDS[10].to_string()
                }
            }
            _ => existing.to_string(),
        }
    }

    let mut out = String::new();
    let mut match_idx = 0usize;
    for line in text.lines() {
        if !line.starts_with(&format!("{wav_name},")) {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let mut cols: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
        if cols.len() >= 9
            && let Some(window) = windows.get(match_idx)
        {
            cols[2] = window.label.clone();
            if let Some(condition) = condition_for_audio_label(&window.label) {
                cols[3] = condition.to_string();
            }
            if let Some(assay) = assay_for_audio_label(&window.label) {
                cols[7] = assay.to_string();
            }
            cols[4] = seed_for_audio_label(&window.label, &cols[4]);
            cols[5] = format!("{:.1}", window.start_sec);
            cols[6] = format!("{:.1}", window.end_sec);
        }
        out.push_str(&cols.join(","));
        out.push('\n');
        match_idx += 1;
    }
    std::fs::write(manifest_path, out)?;
    Ok(())
}

fn apply_segment_fades_i16(
    wav_path: &Path,
    windows: &[AudioSegmentWindow],
    fade_sec: f32,
) -> io::Result<()> {
    let mut reader = hound::WavReader::open(wav_path).map_err(io::Error::other)?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate == 0 || spec.bits_per_sample != 16 {
        return Err(io::Error::other(format!(
            "unsupported quicklisten WAV format: channels={}, rate={}, bits={}",
            spec.channels, spec.sample_rate, spec.bits_per_sample
        )));
    }
    let mut samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<Vec<_>, _>>()
        .map_err(io::Error::other)?;
    drop(reader);

    let sr = spec.sample_rate as f32;
    for (window_idx, window) in windows.iter().enumerate() {
        let start = (window.start_sec * sr).round().max(0.0) as usize;
        let end = (window.end_sec * sr).round().max(0.0) as usize;
        if end <= start || start >= samples.len() {
            continue;
        }
        let end = end.min(samples.len());
        let seg_len = end - start;
        let fade_len = ((fade_sec * sr).round() as usize).min(seg_len / 2).max(1);

        if fade_len == 1 {
            samples[start] = 0;
            samples[end - 1] = 0;
            continue;
        }

        let fade_den = (fade_len - 1) as f32;
        for i in 0..fade_len {
            let gain = i as f32 / fade_den;
            let idx = start + i;
            samples[idx] = ((samples[idx] as f32) * gain).round() as i16;
        }
        for i in 0..fade_len {
            let gain = (fade_len - 1 - i) as f32 / fade_den;
            let idx = end - fade_len + i;
            samples[idx] = ((samples[idx] as f32) * gain).round() as i16;
        }

        let zero_until = windows
            .get(window_idx + 1)
            .map(|next| (next.start_sec * sr).round().max(0.0) as usize)
            .unwrap_or(samples.len())
            .min(samples.len());
        let zero_from = end.saturating_sub(16);
        if zero_until > zero_from {
            samples[zero_from..zero_until].fill(0);
        }
    }

    let mut writer = hound::WavWriter::create(wav_path, spec).map_err(io::Error::other)?;
    for sample in samples {
        writer.write_sample(sample).map_err(io::Error::other)?;
    }
    writer.finalize().map_err(io::Error::other)?;
    Ok(())
}

fn postprocess_named_wav_default(wav_stem: &str, fade_sec: f32) -> io::Result<()> {
    let (wav_path, rhai_path, manifest_path) = resolve_audio_postprocess_paths(wav_stem);
    let windows = parse_rhai_scene_windows(&rhai_path)?;
    let wav_name = format!("{wav_stem}.wav");
    rewrite_audio_manifest(&manifest_path, &wav_name, &windows)?;
    apply_segment_fades_i16(&wav_path, &windows, fade_sec)?;
    eprintln!(
        "postprocessed {wav_name}: {} segments, {:.1}s fades",
        windows.len(),
        fade_sec
    );
    Ok(())
}

fn postprocess_quicklisten_showcase_wav_default() -> io::Result<()> {
    postprocess_named_wav_default("showcase", AUDIO_QUICKLISTEN_FADE_SEC)
}

fn postprocess_quicklisten_controls_wav_default() -> io::Result<()> {
    postprocess_named_wav_default("controls", AUDIO_QUICKLISTEN_FADE_SEC)
}

fn postprocess_e6b_named_wav_default(wav_stem: &str) -> io::Result<()> {
    let (wav_path, rhai_path, manifest_path) = resolve_audio_postprocess_paths(wav_stem);
    let windows = parse_rhai_scene_windows(&rhai_path)?;
    let wav_name = format!("{wav_stem}.wav");
    rewrite_audio_manifest(&manifest_path, &wav_name, &windows)?;
    apply_segment_fades_i16(&wav_path, &windows, AUDIO_INTEGRATION_FADE_SEC)?;
    eprintln!(
        "postprocessed {wav_name}: {} segments, {:.1}s fades",
        windows.len(),
        AUDIO_INTEGRATION_FADE_SEC
    );
    Ok(())
}

fn postprocess_e6b_wav_default() -> io::Result<()> {
    postprocess_e6b_named_wav_default("hereditary_adaptation")
}

fn postprocess_hereditary_controls_wav_default() -> io::Result<()> {
    postprocess_e6b_named_wav_default("hereditary_adaptation_controls")
}

fn rewrite_manifest_only_for_stem(wav_stem: &str) -> io::Result<()> {
    let (wav_path, rhai_path, manifest_path) = resolve_audio_postprocess_paths(wav_stem);
    let windows = parse_rhai_scene_windows(&rhai_path)?;
    let wav_name = format!("{wav_stem}.wav");
    rewrite_audio_manifest(&manifest_path, &wav_name, &windows)?;
    eprintln!(
        "updated manifest for {wav_name}: {} segments",
        windows.len()
    );
    let _ = wav_path;
    Ok(())
}

fn postprocess_polyphony_wav_default() -> io::Result<()> {
    rewrite_manifest_only_for_stem("self_organized_polyphony")
}

fn postprocess_polyphony_no_hill_wav_default() -> io::Result<()> {
    rewrite_manifest_only_for_stem("self_organized_polyphony_no_hill")
}

fn run_audio_e2_jobs_parallel(
    jobs: &[(u64, E2Condition, &'static str)],
    space: &Log2Space,
    anchor_hz: f32,
    phase_mode: E2PhaseMode,
) -> Vec<E2Run> {
    if jobs.is_empty() {
        return Vec::new();
    }
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len())
        .max(1);
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<E2Run>>> = Mutex::new((0..jobs.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs.len() {
                        break;
                    }
                    let (seed, cond, label) = jobs[idx];
                    eprintln!("    E2 {label}");
                    let run = run_e2_once(
                        space,
                        anchor_hz,
                        seed,
                        cond,
                        E2_STEP_SEMITONES,
                        phase_mode,
                        None,
                        0,
                    );
                    slots.lock().unwrap()[idx] = Some(run);
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|slot| slot.expect("missing E2 audio output"))
        .collect()
}

fn run_audio_e6b_jobs_parallel(jobs: Vec<(String, E6bRunConfig)>) -> Vec<(String, E6bRunResult)> {
    if jobs.is_empty() {
        return Vec::new();
    }
    let worker_count = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len())
        .max(1);
    let next = AtomicUsize::new(0);
    let slots: Mutex<Vec<Option<(String, E6bRunResult)>>> =
        Mutex::new((0..jobs.len()).map(|_| None).collect());
    std::thread::scope(|scope| {
        for _ in 0..worker_count {
            scope.spawn(|| {
                loop {
                    let idx = next.fetch_add(1, Ordering::Relaxed);
                    if idx >= jobs.len() {
                        break;
                    }
                    let (label, cfg) = &jobs[idx];
                    eprintln!("    E6b {label}");
                    let result = run_e6b(cfg);
                    slots.lock().unwrap()[idx] = Some((label.clone(), result));
                }
            });
        }
    });
    slots
        .into_inner()
        .unwrap()
        .into_iter()
        .map(|slot| slot.expect("missing E6b audio output"))
        .collect()
}

/// Generate all audio supplement Rhai scripts from simulation data.
pub fn generate_audio_replay_rhai() -> io::Result<()> {
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
    let anchor_hz: f32 = E4_ANCHOR_HZ;
    let phase_mode = E2PhaseMode::DissonanceThenConsonance;

    let out_dir = Path::new("supplementary_audio/scenarios");
    create_dir_all(out_dir)?;

    // Temporal scaffold uses the same public regeneration path as the other
    // audio supplement scenarios: emit Rhai here, then render it uniformly
    // through conchordal-render in supplementary_audio/render.sh.
    let cfg_shared = crate::sim::E3AudioConfig::default_shared();
    let cfg_scrambled = crate::sim::E3AudioConfig::default_scrambled();
    let cfg_off = crate::sim::E3AudioConfig::default_off();
    crate::sim::generate_e3_rhai(&cfg_shared, &out_dir.join("temporal_scaffold_shared.rhai"))?;
    crate::sim::generate_e3_rhai(
        &cfg_scrambled,
        &out_dir.join("temporal_scaffold_scrambled.rhai"),
    )?;
    crate::sim::generate_e3_rhai(&cfg_off, &out_dir.join("temporal_scaffold_off.rhai"))?;

    // Select 8 agents evenly from 24
    let e2_selected: Vec<usize> = (0..AUDIO_N_SELECT_E2)
        .map(|k| k * E2_N_AGENTS / AUDIO_N_SELECT_E2)
        .collect();

    // ── E2 runs ──
    let e2_segments: &[(u64, E2Condition, &str)] = &[
        (E2_SEEDS[0], E2Condition::Baseline, "seed0_baseline"),
        (E2_SEEDS[0], E2Condition::NoHillClimb, "seed0_nohill"),
        (E2_SEEDS[0], E2Condition::NoCrowding, "seed0_norep"),
        (E2_SEEDS[3], E2Condition::Baseline, "seed3_baseline"),
        (E2_SEEDS[10], E2Condition::Baseline, "seed10_baseline"),
        (E2_SEEDS[10], E2Condition::NoHillClimb, "seed10_nohill"),
        (E2_SEEDS[10], E2Condition::NoCrowding, "seed10_norep"),
    ];
    let e2_local_search_segment_indices: &[usize] = &[0, 4];
    let e2_no_hill_segment_indices: &[usize] = &[1, 5];

    eprintln!("  [audio-rhai] Running E2 simulations...");
    let e2_runs = run_audio_e2_jobs_parallel(e2_segments, &space, anchor_hz, phase_mode);

    // ── self_organized_polyphony.rhai ──
    {
        let polyphony_max_steps =
            ((AUDIO_E2_POLYPHONY_TARGET_SEGMENT_SEC - AUDIO_E2_AGENT_RELEASE_SEC) / AUDIO_STEP_SEC)
                .floor()
                .max(2.0) as usize;
        let detail = format!(
            "// {} adaptive agents (selected from {}) + 1 fixed drone at {:.1} Hz, {} sweeps, step={} ct\n\
             // Phase switch at step {} (dissonance -> consonance)\n\
             // audio replay downsampled to at most {} updates per segment (~{:.1}s each)\n\
             //\n\
             // 2 segments: local-search, seed 0 and seed 10\n\
             //   shared seed list index 0  = 0x{:X} = {}\n\
             //   shared seed list index 10 = 0x{:X} = {}",
            AUDIO_N_SELECT_E2,
            E2_N_AGENTS,
            anchor_hz,
            E2_SWEEPS,
            E2_STEP_SEMITONES,
            E2_PHASE_SWITCH_STEP,
            polyphony_max_steps,
            AUDIO_E2_POLYPHONY_TARGET_SEGMENT_SEC,
            E2_SEEDS[0],
            E2_SEEDS[0],
            E2_SEEDS[10],
            E2_SEEDS[10],
        );
        let mut rhai = rhai_replay_header(
            "self_organized_polyphony.rhai -- Consonance Search evidence",
            &detail,
        );
        for (scene_idx, &seg_idx) in e2_local_search_segment_indices.iter().enumerate() {
            let scene = match seg_idx {
                0 => "local_search_seed0",
                4 => "local_search_seed10",
                _ => unreachable!("unexpected polyphony audio segment index"),
            };
            let trajectory = downsample_e2_trajectory_for_audio(
                &e2_runs[seg_idx].trajectory_semitones,
                polyphony_max_steps,
            );
            rhai.push_str(&rhai_e2_segment(
                scene,
                &trajectory,
                anchor_hz,
                &e2_selected,
                Some(e2_runs[seg_idx].fixed_drone_hz),
                AUDIO_STEP_SEC,
            ));
            if scene_idx + 1 < e2_local_search_segment_indices.len() {
                rhai.push_str(&format!("wait({AUDIO_GAP_SEC:.1});\n\n"));
            }
        }
        write_with_log(out_dir.join("self_organized_polyphony.rhai"), rhai)?;
    }

    // ── self_organized_polyphony_no_hill.rhai ──
    {
        let polyphony_max_steps =
            ((AUDIO_E2_POLYPHONY_TARGET_SEGMENT_SEC - AUDIO_E2_AGENT_RELEASE_SEC) / AUDIO_STEP_SEC)
                .floor()
                .max(2.0) as usize;
        let detail = format!(
            "// {} adaptive agents (selected from {}) + 1 fixed drone at {:.1} Hz, {} sweeps, step={} ct\n\
             // Phase switch at step {} (dissonance -> consonance)\n\
             // audio replay downsampled to at most {} updates per segment (~{:.1}s each)\n\
             //\n\
             // 2 segments: no-search, seed 0 and seed 10\n\
             //   shared seed list index 0  = 0x{:X} = {}\n\
             //   shared seed list index 10 = 0x{:X} = {}",
            AUDIO_N_SELECT_E2,
            E2_N_AGENTS,
            anchor_hz,
            E2_SWEEPS,
            E2_STEP_SEMITONES,
            E2_PHASE_SWITCH_STEP,
            polyphony_max_steps,
            AUDIO_E2_POLYPHONY_TARGET_SEGMENT_SEC,
            E2_SEEDS[0],
            E2_SEEDS[0],
            E2_SEEDS[10],
            E2_SEEDS[10],
        );
        let mut rhai = rhai_replay_header(
            "self_organized_polyphony_no_hill.rhai -- No-search control replay",
            &detail,
        );
        for (scene_idx, &seg_idx) in e2_no_hill_segment_indices.iter().enumerate() {
            let scene = match seg_idx {
                1 => "no_search_seed0",
                5 => "no_search_seed10",
                _ => unreachable!("unexpected no-hill audio segment index"),
            };
            let trajectory = downsample_e2_trajectory_for_audio(
                &e2_runs[seg_idx].trajectory_semitones,
                polyphony_max_steps,
            );
            rhai.push_str(&rhai_e2_segment(
                scene,
                &trajectory,
                anchor_hz,
                &e2_selected,
                Some(e2_runs[seg_idx].fixed_drone_hz),
                AUDIO_STEP_SEC,
            ));
            if scene_idx + 1 < e2_no_hill_segment_indices.len() {
                rhai.push_str(&format!("wait({AUDIO_GAP_SEC:.1});\n\n"));
            }
        }
        write_with_log(out_dir.join("self_organized_polyphony_no_hill.rhai"), rhai)?;
    }

    // ── E6b runs ──
    struct E6bAudioSeg {
        label: &'static str,
        condition: E6Condition,
        selection_enabled: bool,
    }
    let e6b_conditions = [
        E6bAudioSeg {
            label: "heredity_sel",
            condition: E6Condition::Heredity,
            selection_enabled: true,
        },
        E6bAudioSeg {
            label: "heredity_nosel",
            condition: E6Condition::Heredity,
            selection_enabled: false,
        },
        E6bAudioSeg {
            label: "random_sel",
            condition: E6Condition::Random,
            selection_enabled: true,
        },
        E6bAudioSeg {
            label: "random_nosel",
            condition: E6Condition::Random,
            selection_enabled: false,
        },
    ];
    let e6b_seed_a = E6B_SEEDS[0];
    let e6b_seed_b = E6B_SEEDS[10];
    let e6b_quick_seed_a = E6B_SEEDS[1];
    let e6b_quick_seed_b = E6B_SEEDS[2];
    let e6b_seeds: [u64; 4] = [e6b_seed_a, e6b_seed_b, e6b_quick_seed_a, e6b_quick_seed_b];

    eprintln!("  [audio-rhai] Running E6b simulations...");
    let mut e6b_jobs: Vec<(String, E6bRunConfig)> = Vec::new();
    for cond in &e6b_conditions {
        for &seed in &e6b_seeds {
            let label = format!(
                "seed{}_{}",
                if seed == e6b_seed_a {
                    "0"
                } else if seed == e6b_seed_b {
                    "10"
                } else if seed == e6b_quick_seed_a {
                    "1"
                } else {
                    "2"
                },
                cond.label
            );
            let cfg = E6bRunConfig {
                seed,
                steps_cap: E6B_STEPS_CAP,
                min_deaths: E6B_MIN_DEATHS,
                pop_size: E6B_DEFAULT_POP_SIZE,
                first_k: E6B_FIRST_K,
                condition: cond.condition,
                snapshot_interval: E6B_SNAPSHOT_INTERVAL,
                selection_enabled: cond.selection_enabled,
                shuffle_landscape: false,
                polyphonic_crowding_weight_override: None,
                polyphonic_overcapacity_weight_override: None,
                polyphonic_capacity_radius_cents_override: None,
                polyphonic_capacity_free_voices_override: None,
                polyphonic_parent_share_weight_override: None,
                polyphonic_parent_energy_weight_override: None,
                juvenile_contextual_tuning_ticks_override: None,
                juvenile_contextual_settlement_enabled_override: None,
                survival_score_low_override: None,
                survival_score_high_override: None,
                survival_recharge_per_sec_override: None,
                background_death_rate_per_sec_override: None,
                respawn_parent_prior_mix_override: None,
                respawn_same_band_discount_override: None,
                respawn_octave_discount_override: None,
                parent_proposal_kind_override: None,
                parent_proposal_sigma_st_override: None,
                parent_proposal_unison_notch_gain_override: None,
                parent_proposal_unison_notch_sigma_st_override: None,
                parent_proposal_candidate_count_override: None,
                azimuth_mode_override: None,
                random_baseline_mode: E6bRandomBaselineMode::LogRandomFiltered,
            };
            e6b_jobs.push((label, cfg));
        }
    }
    let e6b_results = run_audio_e6b_jobs_parallel(e6b_jobs);

    // ── hereditary_adaptation.rhai ──
    {
        let detail = format!(
            "// per-life replay from the {}-agent hereditary adaptation assay,\n\
             // tail {} snapshots per segment, snapshot_interval={} steps, replay_step_sec={:.3}\n\
             // each life uses harmonic ADSR atk={:.2}s dec={:.2}s sus={:.2} rel={:.2}s\n\
             //\n\
             // 4 segments: heredity + selection and matched-random + selection, seeds 0 and 10\n\
             // main evidence contrast: lineage-biased respawn versus matched random baseline\n\
             // under the same metabolic selection regime\n\
             //   shared seed list index 0  = {}\n\
             //   shared seed list index 10 = {}",
            E6B_DEFAULT_POP_SIZE,
            AUDIO_E6_TAIL_SNAPSHOTS,
            E6B_SNAPSHOT_INTERVAL,
            AUDIO_E6_STEP_SEC,
            AUDIO_E6_AGENT_ATTACK_SEC,
            AUDIO_E6_AGENT_DECAY_SEC,
            AUDIO_E6_AGENT_SUSTAIN_LEVEL,
            AUDIO_E6_AGENT_RELEASE_SEC,
            e6b_seed_a,
            e6b_seed_b,
        );
        let mut rhai = rhai_replay_header(
            "hereditary_adaptation.rhai -- Hereditary adaptation replay",
            &detail,
        );
        let e6b_order = [
            "seed0_heredity_sel",
            "seed0_random_sel",
            "seed10_heredity_sel",
            "seed10_random_sel",
        ];
        for (i, label) in e6b_order.iter().enumerate() {
            let (_, result) = e6b_results
                .iter()
                .find(|(candidate, _)| candidate == label)
                .unwrap_or_else(|| panic!("missing E6b audio result for label {label}"));
            let snaps = &result.snapshots;
            let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
            let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
            let scene = match *label {
                "seed0_heredity_sel" => "heredity_selection_seed0",
                "seed0_random_sel" => "selection_only_seed0",
                "seed10_heredity_sel" => "heredity_selection_seed10",
                "seed10_random_sel" => "selection_only_seed10",
                _ => unreachable!("unexpected hereditary audio label"),
            };
            let amp_scale = if label.ends_with("heredity_sel") {
                AUDIO_E6_HS_GAIN
            } else {
                1.0
            };
            rhai.push_str(&rhai_e6_segment_with_gain(
                scene,
                &tail,
                AUDIO_E6_STEP_SEC,
                None,
                amp_scale,
            ));
            if i + 1 < e6b_order.len() {
                rhai.push_str(&format!("wait({AUDIO_GAP_SEC:.1});\n\n"));
            }
        }
        write_with_log(out_dir.join("hereditary_adaptation.rhai"), rhai)?;
    }

    // ── hereditary_adaptation_controls.rhai ──
    {
        let detail = format!(
            "// per-life replay from the {}-agent hereditary adaptation assay,\n\
             // tail {} snapshots per segment, snapshot_interval={} steps, replay_step_sec={:.3}\n\
             // each life uses harmonic ADSR atk={:.2}s dec={:.2}s sus={:.2} rel={:.2}s\n\
             //\n\
             // 4 auxiliary segments: heredity only and neither, seeds 0 and 10\n\
             // this isolates lineage bias without selection, and matched random without selection\n\
             //   shared seed list index 0  = {}\n\
             //   shared seed list index 10 = {}",
            E6B_DEFAULT_POP_SIZE,
            AUDIO_E6_TAIL_SNAPSHOTS,
            E6B_SNAPSHOT_INTERVAL,
            AUDIO_E6_STEP_SEC,
            AUDIO_E6_AGENT_ATTACK_SEC,
            AUDIO_E6_AGENT_DECAY_SEC,
            AUDIO_E6_AGENT_SUSTAIN_LEVEL,
            AUDIO_E6_AGENT_RELEASE_SEC,
            e6b_seed_a,
            e6b_seed_b,
        );
        let mut rhai = rhai_replay_header(
            "hereditary_adaptation_controls.rhai -- Hereditary adaptation controls",
            &detail,
        );
        let e6b_order = [
            "seed0_heredity_nosel",
            "seed0_random_nosel",
            "seed10_heredity_nosel",
            "seed10_random_nosel",
        ];
        for (i, label) in e6b_order.iter().enumerate() {
            let (_, result) = e6b_results
                .iter()
                .find(|(candidate, _)| candidate == label)
                .unwrap_or_else(|| panic!("missing E6b audio result for label {label}"));
            let snaps = &result.snapshots;
            let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
            let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
            let scene = match *label {
                "seed0_heredity_nosel" => "heredity_only_seed0",
                "seed0_random_nosel" => "neither_seed0",
                "seed10_heredity_nosel" => "heredity_only_seed10",
                "seed10_random_nosel" => "neither_seed10",
                _ => unreachable!("unexpected hereditary control label"),
            };
            rhai.push_str(&rhai_e6_segment(scene, &tail, AUDIO_E6_STEP_SEC, None));
            if i + 1 < e6b_order.len() {
                rhai.push_str(&format!("wait({AUDIO_GAP_SEC:.1});\n\n"));
            }
        }
        write_with_log(out_dir.join("hereditary_adaptation_controls.rhai"), rhai)?;
    }

    // ── showcase.rhai ──
    // 2 segments: hereditary + selection and local search
    {
        let detail = format!(
            "// 2 showcase segments from hereditary adaptation:\n\
             //   1. Heredity + selection (seed 1) -- family-biased respawn + local azimuth search\n\
             //   2. Heredity + selection (seed 2) -- family-biased respawn + local azimuth search",
        );
        let mut rhai = rhai_replay_header("showcase.rhai -- Orientation showcase", &detail);

        // Segment 1: hereditary + selection, seed 1
        let e6b_sel_hered_seed1 = &e6b_results
            .iter()
            .find(|(label, _)| label == "seed1_heredity_sel")
            .expect("missing E6b audio result for seed1_heredity_sel")
            .1;
        let snaps = &e6b_sel_hered_seed1.snapshots;
        let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
        let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
        rhai.push_str(&rhai_e6_segment_with_gain(
            "heredity_selection_seed1",
            &tail,
            AUDIO_E6_STEP_SEC,
            None,
            AUDIO_E6_HS_GAIN,
        ));
        rhai.push_str(&format!("wait({AUDIO_QUICKLISTEN_GAP_SEC:.1});\n\n"));

        // Segment 2: hereditary + selection, seed 2
        let e6b_sel_hered_seed2 = &e6b_results
            .iter()
            .find(|(label, _)| label == "seed2_heredity_sel")
            .expect("missing E6b audio result for seed2_heredity_sel")
            .1;
        let snaps = &e6b_sel_hered_seed2.snapshots;
        let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
        let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
        rhai.push_str(&rhai_e6_segment_with_gain(
            "heredity_selection_seed2",
            &tail,
            AUDIO_E6_STEP_SEC,
            None,
            AUDIO_E6_HS_GAIN,
        ));
        write_with_log(out_dir.join("showcase.rhai"), rhai)?;
    }

    // ── controls.rhai ──
    // 2 segments: neither, seed 0 and seed 10
    {
        let detail = format!(
            "// 2 control segments from hereditary adaptation:\n\
             //   1. Neither (seed 1) -- matched random respawn, no selection\n\
             //   2. Neither (seed 2) -- matched random respawn, no selection",
        );
        let mut rhai = rhai_replay_header("controls.rhai -- Orientation controls", &detail);

        // Segment 1: random + no selection, seed 1
        let e6b_random_nosel_seed1 = &e6b_results
            .iter()
            .find(|(label, _)| label == "seed1_random_nosel")
            .expect("missing E6b audio result for seed1_random_nosel")
            .1;
        let snaps = &e6b_random_nosel_seed1.snapshots;
        let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
        let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
        rhai.push_str(&rhai_e6_segment(
            "neither_seed1",
            &tail,
            AUDIO_E6_STEP_SEC,
            None,
        ));
        rhai.push_str(&format!("wait({AUDIO_QUICKLISTEN_GAP_SEC:.1});\n\n"));

        // Segment 2: random + no selection, seed 2
        let e6b_random_nosel_seed2 = &e6b_results
            .iter()
            .find(|(label, _)| label == "seed2_random_nosel")
            .expect("missing E6b audio result for seed2_random_nosel")
            .1;
        let snaps = &e6b_random_nosel_seed2.snapshots;
        let tail_start = snaps.len().saturating_sub(AUDIO_E6_TAIL_SNAPSHOTS);
        let tail: Vec<&E6PitchSnapshot> = snaps[tail_start..].iter().collect();
        rhai.push_str(&rhai_e6_segment(
            "neither_seed2",
            &tail,
            AUDIO_E6_STEP_SEC,
            None,
        ));

        write_with_log(out_dir.join("controls.rhai"), rhai)?;
    }

    eprintln!("Audio replay Rhai scripts written to supplementary_audio/scenarios/");
    Ok(())
}

// ── E2 trajectory replay ── direct WAV synthesis ────────────────────────────

/// Generate a WAV file that replays E2 simulation trajectories as pure sine
/// tones.  Each sweep step is rendered as a chord impulse (attack + decay),
/// producing a rhythmic sequence that makes pitch convergence audible.
pub fn generate_e2_replay_wav(output_path: &Path) -> io::Result<()> {
    use std::f32::consts::TAU;

    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
    let anchor_hz: f32 = E4_ANCHOR_HZ; // 220.0
    let phase_mode = E2PhaseMode::DissonanceThenConsonance;
    let sample_rate: u32 = 48000;
    let sr = sample_rate as f32;

    let segments: &[(u64, E2Condition, &str)] = &[
        (E2_SEEDS[0], E2Condition::Baseline, "seed0_baseline"),
        (E2_SEEDS[0], E2Condition::NoHillClimb, "seed0_nohill"),
        (E2_SEEDS[0], E2Condition::NoCrowding, "seed0_norep"),
        (E2_SEEDS[10], E2Condition::Baseline, "seed10_baseline"),
        (E2_SEEDS[10], E2Condition::NoHillClimb, "seed10_nohill"),
        (E2_SEEDS[10], E2Condition::NoCrowding, "seed10_norep"),
    ];

    // Timing
    let step_sec: f32 = 0.12; // per sweep step
    let attack_sec: f32 = 0.005; // sharp attack
    let sustain_sec: f32 = 0.06; // sustain plateau
    let _release_sec: f32 = 0.055; // fade to silence before next step
    let gap_sec: f32 = 1.0; // silence between segments
    let agent_amp: f32 = 0.12; // per-agent amplitude

    let step_samples = (step_sec * sr) as usize;
    let attack_samples = (attack_sec * sr) as usize;
    let sustain_samples = (sustain_sec * sr) as usize;
    let release_start = attack_samples + sustain_samples;
    let release_samples = step_samples.saturating_sub(release_start);
    let gap_samples = (gap_sec * sr) as usize;

    // Select 8 agents evenly from 24
    let n_select = 8usize;
    let selected: Vec<usize> = (0..n_select).map(|k| k * E2_N_AGENTS / n_select).collect();

    // Collect all samples
    let mut all_samples: Vec<f32> = Vec::new();

    for (seg_idx, &(seed, condition, label)) in segments.iter().enumerate() {
        eprintln!(
            "  [e2-replay-wav] segment {}/{}: {}",
            seg_idx + 1,
            segments.len(),
            label
        );

        let run = run_e2_once(
            &space,
            anchor_hz,
            seed,
            condition,
            E2_STEP_SEMITONES,
            phase_mode,
            None,
            0,
        );

        let n_steps = run.trajectory_semitones[0].len();
        // Track oscillator phases across steps for phase continuity
        let mut phases = vec![0.0f32; n_select + 1];

        for step in 0..n_steps {
            // Frequencies for this step
            let freqs = e2_trajectory_freqs_at_step(
                &run.trajectory_semitones,
                anchor_hz,
                step,
                Some(&selected),
                Some(run.fixed_drone_hz),
            );

            // Render step_samples of audio
            for s in 0..step_samples {
                // Envelope: attack → sustain → release
                let env = if s < attack_samples {
                    s as f32 / attack_samples as f32
                } else if s < release_start {
                    1.0
                } else {
                    let t = (s - release_start) as f32 / release_samples.max(1) as f32;
                    (1.0 - t).max(0.0)
                };

                // Additive sine synthesis
                let mut sample = 0.0f32;
                for (i, &freq) in freqs.iter().enumerate() {
                    phases[i] += freq * TAU / sr;
                    if phases[i] > TAU {
                        phases[i] -= TAU;
                    }
                    sample += phases[i].sin() * agent_amp * env;
                }
                all_samples.push(sample);
            }
        }

        // Gap between segments
        if seg_idx + 1 < segments.len() {
            all_samples.extend(std::iter::repeat(0.0f32).take(gap_samples));
        }
    }

    // Normalize to avoid clipping
    let peak = all_samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.95 {
        let scale = 0.9 / peak;
        for s in &mut all_samples {
            *s *= scale;
        }
    }

    // Write WAV
    if let Some(parent) = output_path.parent() {
        create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    for &s in &all_samples {
        let i16_val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(i16_val)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    }
    writer
        .finalize()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let duration_sec = all_samples.len() as f32 / sr;
    eprintln!(
        "Wrote {} ({:.1}s, {} samples, peak={:.3})",
        output_path.display(),
        duration_sec,
        all_samples.len(),
        peak
    );
    Ok(())
}

fn e2_range_oct_tag(range_oct: f32) -> String {
    if (range_oct - range_oct.round()).abs() < 1e-6 {
        format!("r{}oct", range_oct.round() as i32)
    } else {
        format!("r{:.1}oct", range_oct).replace('.', "p")
    }
}

fn e2_baseline_summary_csv(
    stem: &str,
    mode: &str,
    run: &E2Run,
    anchor_hz: f32,
    render_partials: u32,
    range_oct: f32,
) -> String {
    e2_render_summary_csv(
        stem,
        mode,
        "baseline",
        run,
        anchor_hz,
        render_partials,
        range_oct,
    )
}

fn e2_moves_csv(run: &E2Run) -> String {
    let mut out = String::from("agent_id,step,semitone_before,semitone_after,delta_semitones\n");
    for (agent_id, semis) in run.trajectory_semitones.iter().enumerate() {
        for step in 1..semis.len() {
            let before = semis[step - 1];
            let after = semis[step];
            let delta = after - before;
            if delta.abs() > E2_SEMITONE_EPS {
                out.push_str(&format!(
                    "{agent_id},{step},{before:.6},{after:.6},{delta:.6}\n"
                ));
            }
        }
    }
    out
}

fn render_e2_trajectory_wav(
    output_path: &Path,
    trajectory_semitones: &[Vec<f32>],
    anchor_hz: f32,
    render_partials: u32,
    fixed_drone_hz: Option<f32>,
    continuous: bool,
) -> io::Result<()> {
    use std::f32::consts::TAU;

    if trajectory_semitones.is_empty() {
        return Ok(());
    }
    let n_agents = trajectory_semitones.len() + usize::from(fixed_drone_hz.is_some());
    let n_steps = trajectory_semitones[0].len();
    if n_steps == 0 {
        return Ok(());
    }

    let sample_rate: u32 = 48_000;
    let sr = sample_rate as f32;
    let step_sec: f32 = 0.12;
    let step_samples = (step_sec * sr) as usize;
    let attack_sec: f32 = 0.006;
    let sustain_sec: f32 = 0.06;
    let attack_samples = (attack_sec * sr) as usize;
    let sustain_samples = (sustain_sec * sr) as usize;
    let release_start = attack_samples + sustain_samples;
    let release_samples = step_samples.saturating_sub(release_start).max(1);
    let partials = render_partials.clamp(1, 16) as usize;
    let harmonic_norm = (1..=partials)
        .map(|k| 1.0 / k as f32)
        .sum::<f32>()
        .max(1e-6);
    let agent_amp = 0.22f32;
    let mut phases = vec![0.0f32; n_agents * partials];
    let mut all_samples = Vec::with_capacity(n_steps * step_samples);

    for step in 0..n_steps {
        let current_freqs = e2_trajectory_freqs_at_step(
            trajectory_semitones,
            anchor_hz,
            step,
            None,
            fixed_drone_hz,
        );
        let next_freqs = if continuous && step + 1 < n_steps {
            e2_trajectory_freqs_at_step(
                trajectory_semitones,
                anchor_hz,
                step + 1,
                None,
                fixed_drone_hz,
            )
        } else {
            current_freqs.clone()
        };

        for s in 0..step_samples {
            let env = if continuous {
                1.0
            } else if s < attack_samples {
                s as f32 / attack_samples.max(1) as f32
            } else if s < release_start {
                1.0
            } else {
                let t = (s - release_start) as f32 / release_samples as f32;
                (1.0 - t).max(0.0)
            };
            let t = s as f32 / step_samples.max(1) as f32;
            let mut sample = 0.0f32;
            for (agent_i, (&freq0, &freq1)) in
                current_freqs.iter().zip(next_freqs.iter()).enumerate()
            {
                let base_freq = if continuous {
                    freq0 + (freq1 - freq0) * t
                } else {
                    freq0
                };
                for partial_idx in 0..partials {
                    let harmonic = (partial_idx + 1) as f32;
                    let freq = base_freq * harmonic;
                    let phase_slot = agent_i * partials + partial_idx;
                    phases[phase_slot] += freq * TAU / sr;
                    if phases[phase_slot] > TAU {
                        phases[phase_slot] -= TAU;
                    }
                    let weight = (1.0 / harmonic) / harmonic_norm;
                    sample += phases[phase_slot].sin() * agent_amp * weight * env;
                }
            }
            all_samples.push(sample);
        }
    }

    let peak = all_samples
        .iter()
        .map(|s| s.abs())
        .fold(0.0f32, f32::max)
        .max(1e-6);
    let scale = if peak > 0.95 { 0.9 / peak } else { 1.0 };
    for s in &mut all_samples {
        *s *= scale;
    }

    if let Some(parent) = output_path.parent() {
        create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec).map_err(io::Error::other)?;
    for &s in &all_samples {
        let i16_val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(i16_val).map_err(io::Error::other)?;
    }
    writer.finalize().map_err(io::Error::other)?;
    eprintln!("Wrote {}", output_path.display());
    Ok(())
}

fn build_pitch_core_landscape(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    env_scan: &[f32],
    density_scan: &[f32],
    du_scan: &[f32],
) -> (Landscape, Vec<f32>, Vec<f32>, f32, RState01Stats) {
    let (c_score_scan, c_level_scan, density_mass, r_state_stats) =
        compute_c_score_level_scans(space, workspace, env_scan, density_scan, du_scan);
    let landscape =
        build_pitch_core_landscape_from_scans(space, &c_score_scan, &c_level_scan, env_scan);
    (
        landscape,
        c_score_scan,
        c_level_scan,
        density_mass,
        r_state_stats,
    )
}

fn build_pitch_core_landscape_from_scans(
    space: &Log2Space,
    c_score_scan: &[f32],
    c_level_scan: &[f32],
    env_scan: &[f32],
) -> Landscape {
    let mut landscape = Landscape::new(space.clone());
    landscape.roughness_kernel_params = KernelParams::default();
    landscape.consonance_field_score = c_score_scan.to_vec();
    landscape.consonance_field_level = c_level_scan.to_vec();
    landscape.subjective_intensity = env_scan.to_vec();
    landscape
}

fn build_e2_pitch_cores(
    space: &Log2Space,
    agent_indices: &[usize],
    step_semitones: f32,
) -> Vec<PitchHillClimbPitchCore> {
    let mut pitch_cores = Vec::with_capacity(agent_indices.len());
    for &idx in agent_indices {
        let mut core = PitchHillClimbPitchCore::new(
            step_semitones * 100.0,
            space.centers_hz[idx].log2(),
            0.0,
            0.02,
            0.0,
            0.5,
        );
        configure_shared_hillclimb_core(&mut core, 1.0);
        pitch_cores.push(core);
    }
    pitch_cores
}

#[allow(clippy::too_many_arguments)]
fn e2_objective_score_at_log2(
    space: &Log2Space,
    c_score_scan: &[f32],
    erb_scan: &[f32],
    prev_erb: &[f32],
    self_agent_i: usize,
    current_pitch_log2: f32,
    pitch_log2: f32,
    min_idx: usize,
    max_idx: usize,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
) -> (usize, f32, f32) {
    let idx = space
        .index_of_log2(pitch_log2)
        .unwrap_or_else(|| space.nearest_index(2.0_f32.powf(pitch_log2)))
        .clamp(min_idx, max_idx);
    let cand_erb = erb_scan[idx];
    let mut crowding = 0.0f32;
    for (j, &other_erb) in prev_erb.iter().enumerate() {
        if j == self_agent_i {
            continue;
        }
        crowding += crowding_runtime_delta_erb(kernel_params, cand_erb - other_erb);
    }
    let move_cost = shared_hill_move_cost_coeff() * (pitch_log2 - current_pitch_log2).abs();
    let score = score_sign * c_score_scan[idx] - crowding_weight * crowding - move_cost;
    (idx, score, crowding)
}

fn e2_scene_apply_move(
    env_total: &mut [f32],
    density_total: &mut [f32],
    du_scan: &[f32],
    from_idx: usize,
    to_idx: usize,
) {
    if from_idx == to_idx {
        return;
    }
    env_total[from_idx] = (env_total[from_idx] - 1.0).max(0.0);
    let from_denom = du_scan[from_idx].max(1e-12);
    density_total[from_idx] = (density_total[from_idx] - 1.0 / from_denom).max(0.0);
    env_total[to_idx] += 1.0;
    density_total[to_idx] += 1.0 / du_scan[to_idx].max(1e-12);
}

fn record_e2_trajectory_snapshot(
    trajectory_semitones: &mut [Vec<f32>],
    indices: &[usize],
    log2_ratio_scan: &[f32],
) {
    for (agent_id, &idx) in indices.iter().enumerate() {
        let semitone = 12.0 * log2_ratio_scan[idx];
        if let Some(trace) = trajectory_semitones.get_mut(agent_id) {
            trace.push(semitone);
        }
    }
}

fn compute_e2_trajectory_c_levels(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    fixed_drone_idx: usize,
    anchor_hz: f32,
    trajectory_semitones: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    if trajectory_semitones.is_empty() {
        return Vec::new();
    }
    let n_agents = trajectory_semitones.len();
    let n_steps = trajectory_semitones[0].len();
    let mut trajectory_c_level = (0..n_agents)
        .map(|_| Vec::with_capacity(n_steps))
        .collect::<Vec<_>>();
    for step in 0..n_steps {
        let mut indices = Vec::with_capacity(n_agents);
        for trace in trajectory_semitones {
            let st = trace.get(step).copied().unwrap_or(f32::NAN);
            let freq_hz = anchor_hz * 2.0_f32.powf(st / 12.0);
            indices.push(space.nearest_index(freq_hz));
        }
        let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
            space,
            std::slice::from_ref(&fixed_drone_idx),
            &indices,
            du_scan,
        );
        let (_, c_level_scan, _, _) =
            compute_c_score_level_scans(space, workspace, &env_scan, &density_scan, du_scan);
        for (agent_id, &idx) in indices.iter().enumerate() {
            trajectory_c_level[agent_id].push(c_level_scan[idx]);
        }
    }
    trajectory_c_level
}

fn compute_e2_trajectory_mean_c_score_loo(
    space: &Log2Space,
    workspace: &ConsonanceWorkspace,
    du_scan: &[f32],
    fixed_drone_idx: usize,
    anchor_hz: f32,
    trajectory_semitones: &[Vec<f32>],
) -> Vec<f32> {
    if trajectory_semitones.is_empty() {
        return Vec::new();
    }
    let n_agents = trajectory_semitones.len();
    let n_steps = trajectory_semitones[0].len();
    let mut mean_c_score = Vec::with_capacity(n_steps);
    let mut env_loo = Vec::new();
    let mut density_loo = Vec::new();
    for step in 0..n_steps {
        let mut indices = Vec::with_capacity(n_agents);
        for trace in trajectory_semitones {
            let st = trace.get(step).copied().unwrap_or(f32::NAN);
            let freq_hz = anchor_hz * 2.0_f32.powf(st / 12.0);
            indices.push(space.nearest_index(freq_hz));
        }
        let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
            space,
            std::slice::from_ref(&fixed_drone_idx),
            &indices,
            du_scan,
        );
        let (current_mean, _) = mean_c_score_loo_pair_at_indices_with_prev_reused(
            space,
            workspace,
            &env_scan,
            &density_scan,
            du_scan,
            &indices,
            &indices,
            &indices,
            &mut env_loo,
            &mut density_loo,
        );
        mean_c_score.push(current_mean);
    }
    mean_c_score
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_pitch_core_proposal(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    _prev_indices: &[usize],
    pitch_cores: &mut [PitchHillClimbPitchCore],
    space: &Log2Space,
    landscape: &Landscape,
    workspace: &ConsonanceWorkspace,
    env_total: &[f32],
    density_total: &[f32],
    du_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    temperature: f32,
    sweep: usize,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    mut trajectory_semitones: Option<&mut [Vec<f32>]>,
    rng: &mut StdRng,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }

    debug_assert_eq!(indices.len(), pitch_cores.len());
    let order = build_e2_update_order(schedule, indices.len(), sweep, rng);
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let u01_by_agent: Vec<f32> = (0..indices.len()).map(|_| rng.random::<f32>()).collect();
    let mut env_current = env_total.to_vec();
    let mut density_current = density_total.to_vec();
    let mut env_loo = Vec::new();
    let mut density_loo = Vec::new();
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut accepted_worse_count = 0usize;

    for &agent_i in &order {
        let base_update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle | E2UpdateSchedule::SequentialRotate => true,
        };

        let agent_idx = indices[agent_i];
        let loo_c_score_scan = e2_loo_c_score_scan_for_agent_reused(
            space,
            workspace,
            &env_current,
            &density_current,
            du_scan,
            agent_idx,
            &mut env_loo,
            &mut density_loo,
        );
        let current_pitch_log2 = space.centers_hz[agent_idx].log2();
        let current_erb: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
        let (_, current_score, current_crowding) = e2_objective_score_at_log2(
            space,
            &loo_c_score_scan,
            erb_scan,
            &current_erb,
            agent_i,
            current_pitch_log2,
            current_pitch_log2,
            min_idx,
            max_idx,
            score_sign,
            crowding_weight,
            kernel_params,
        );
        let c_score_current = loo_c_score_scan[agent_idx];
        let backtrack_target = if block_backtrack {
            backtrack_targets.and_then(|prev| prev.get(agent_i).copied())
        } else {
            None
        };
        let update_allowed = base_update_allowed;
        if update_allowed {
            attempt_count += 1;
        }

        let (chosen_idx, chosen_score_val, chosen_crowding_val, accepted_worse) = if update_allowed
        {
            let neighbor_pitch_log2: Vec<f32> = indices
                .iter()
                .enumerate()
                .filter_map(|(j, &idx)| {
                    if j == agent_i {
                        None
                    } else {
                        Some(space.centers_hz[idx].log2())
                    }
                })
                .collect();
            let proposal = pitch_cores[agent_i].propose_with_scorer(
                current_pitch_log2,
                current_pitch_log2,
                landscape,
                &neighbor_pitch_log2,
                rng,
                |pitch_log2| {
                    e2_objective_score_at_log2(
                        space,
                        &loo_c_score_scan,
                        erb_scan,
                        &current_erb,
                        agent_i,
                        current_pitch_log2,
                        pitch_log2,
                        min_idx,
                        max_idx,
                        score_sign,
                        crowding_weight,
                        kernel_params,
                    )
                    .1
                },
            );
            let (proposed_idx, cand_score, cand_crowding) = e2_objective_score_at_log2(
                space,
                &loo_c_score_scan,
                erb_scan,
                &current_erb,
                agent_i,
                current_pitch_log2,
                proposal.target_pitch_log2,
                min_idx,
                max_idx,
                score_sign,
                crowding_weight,
                kernel_params,
            );
            if let Some(prev_idx) = backtrack_target
                && proposed_idx == prev_idx
                && (cand_score - current_score) <= E2_BACKTRACK_ALLOW_EPS
            {
                (agent_idx, current_score, current_crowding, false)
            } else {
                let delta = cand_score - current_score;
                if delta > E2_SCORE_IMPROVE_EPS {
                    (proposed_idx, cand_score, cand_crowding, false)
                } else if delta < 0.0 {
                    let (accept, acc_worse) =
                        metropolis_accept(delta, temperature, u01_by_agent[agent_i]);
                    if accept {
                        (proposed_idx, cand_score, cand_crowding, acc_worse)
                    } else {
                        (agent_idx, current_score, current_crowding, false)
                    }
                } else {
                    (agent_idx, current_score, current_crowding, false)
                }
            }
        } else {
            (agent_idx, current_score, current_crowding, false)
        };

        if chosen_idx != agent_idx {
            e2_scene_apply_move(
                &mut env_current,
                &mut density_current,
                du_scan,
                agent_idx,
                chosen_idx,
            );
            indices[agent_i] = chosen_idx;
        }
        let moved = chosen_idx != agent_idx;
        let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[agent_idx]);
        let abs_delta = delta_semitones.abs();
        let abs_delta_moved = if moved { abs_delta } else { 0.0 };
        let c_score_chosen = loo_c_score_scan[chosen_idx];

        if let Some(trace) = trajectory_semitones.as_deref_mut() {
            record_e2_trajectory_snapshot(trace, indices, log2_ratio_scan);
        }

        if moved {
            moved_count += 1;
        }
        if accepted_worse {
            accepted_worse_count += 1;
        }
        abs_delta_sum += abs_delta;
        abs_delta_moved_sum += abs_delta_moved;
        if c_score_current.is_finite() {
            c_score_current_sum += c_score_current;
            c_score_current_count += 1;
        }
        if c_score_chosen.is_finite() {
            c_score_chosen_sum += c_score_chosen;
            c_score_chosen_count += 1;
        }
        if chosen_score_val.is_finite() {
            score_sum += chosen_score_val;
            score_count += 1;
        }
        if chosen_crowding_val.is_finite() {
            crowding_sum += chosen_crowding_val;
            crowding_count += 1;
        }
    }

    let n = indices.len() as f32;
    UpdateStats {
        mean_c_score_current_loo: if c_score_current_count > 0 {
            c_score_current_sum / c_score_current_count as f32
        } else {
            0.0
        },
        mean_c_score_chosen_loo: if c_score_chosen_count > 0 {
            c_score_chosen_sum / c_score_chosen_count as f32
        } else {
            0.0
        },
        mean_score: if score_count > 0 {
            score_sum / score_count as f32
        } else {
            0.0
        },
        mean_crowding: if crowding_count > 0 {
            crowding_sum / crowding_count as f32
        } else {
            0.0
        },
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: accepted_worse_count as f32 / n,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones: abs_delta_sum / n,
        mean_abs_delta_semitones_moved: if moved_count > 0 {
            abs_delta_moved_sum / moved_count as f32
        } else {
            0.0
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn update_e2_sweep_pitch_core_proposal_prescored(
    schedule: E2UpdateSchedule,
    indices: &mut [usize],
    _prev_indices: &[usize],
    pitch_cores: &mut [PitchHillClimbPitchCore],
    space: &Log2Space,
    landscape: &Landscape,
    prescored_c_score_scan: &[f32],
    erb_scan: &[f32],
    log2_ratio_scan: &[f32],
    min_idx: usize,
    max_idx: usize,
    score_sign: f32,
    crowding_weight: f32,
    kernel_params: &KernelParams,
    temperature: f32,
    sweep: usize,
    block_backtrack: bool,
    backtrack_targets: Option<&[usize]>,
    mut trajectory_semitones: Option<&mut [Vec<f32>]>,
    rng: &mut StdRng,
) -> UpdateStats {
    if indices.is_empty() {
        return UpdateStats {
            mean_c_score_current_loo: 0.0,
            mean_c_score_chosen_loo: 0.0,
            mean_score: 0.0,
            mean_crowding: 0.0,
            moved_frac: 0.0,
            accepted_worse_frac: 0.0,
            attempted_update_frac: 0.0,
            moved_given_attempt_frac: 0.0,
            mean_abs_delta_semitones: 0.0,
            mean_abs_delta_semitones_moved: 0.0,
        };
    }

    debug_assert_eq!(indices.len(), pitch_cores.len());
    let order = build_e2_update_order(schedule, indices.len(), sweep, rng);
    let u_move_by_agent: Vec<f32> = if matches!(schedule, E2UpdateSchedule::Lazy) {
        (0..indices.len()).map(|_| rng.random::<f32>()).collect()
    } else {
        vec![0.0; indices.len()]
    };
    let u01_by_agent: Vec<f32> = (0..indices.len()).map(|_| rng.random::<f32>()).collect();
    let mut c_score_current_sum = 0.0f32;
    let mut c_score_current_count = 0u32;
    let mut c_score_chosen_sum = 0.0f32;
    let mut c_score_chosen_count = 0u32;
    let mut score_sum = 0.0f32;
    let mut score_count = 0u32;
    let mut crowding_sum = 0.0f32;
    let mut crowding_count = 0u32;
    let mut abs_delta_sum = 0.0f32;
    let mut abs_delta_moved_sum = 0.0f32;
    let mut attempt_count = 0usize;
    let mut moved_count = 0usize;
    let mut accepted_worse_count = 0usize;

    for &agent_i in &order {
        let base_update_allowed = match schedule {
            E2UpdateSchedule::Checkerboard => (agent_i + sweep).is_multiple_of(2),
            E2UpdateSchedule::Lazy => u_move_by_agent[agent_i] < E2_LAZY_MOVE_PROB.clamp(0.0, 1.0),
            E2UpdateSchedule::RandomSingle | E2UpdateSchedule::SequentialRotate => true,
        };

        let agent_idx = indices[agent_i];
        let current_pitch_log2 = space.centers_hz[agent_idx].log2();
        let current_erb: Vec<f32> = indices.iter().map(|&idx| erb_scan[idx]).collect();
        let (_, current_score, current_crowding) = e2_objective_score_at_log2(
            space,
            prescored_c_score_scan,
            erb_scan,
            &current_erb,
            agent_i,
            current_pitch_log2,
            current_pitch_log2,
            min_idx,
            max_idx,
            score_sign,
            crowding_weight,
            kernel_params,
        );
        let c_score_current = prescored_c_score_scan[agent_idx];
        let backtrack_target = if block_backtrack {
            backtrack_targets.and_then(|prev| prev.get(agent_i).copied())
        } else {
            None
        };
        let update_allowed = base_update_allowed;
        if update_allowed {
            attempt_count += 1;
        }

        let (chosen_idx, chosen_score_val, chosen_crowding_val, accepted_worse) = if update_allowed
        {
            let neighbor_pitch_log2: Vec<f32> = indices
                .iter()
                .enumerate()
                .filter_map(|(j, &idx)| {
                    if j == agent_i {
                        None
                    } else {
                        Some(space.centers_hz[idx].log2())
                    }
                })
                .collect();
            let proposal = pitch_cores[agent_i].propose_with_scorer(
                current_pitch_log2,
                current_pitch_log2,
                landscape,
                &neighbor_pitch_log2,
                rng,
                |pitch_log2| {
                    e2_objective_score_at_log2(
                        space,
                        prescored_c_score_scan,
                        erb_scan,
                        &current_erb,
                        agent_i,
                        current_pitch_log2,
                        pitch_log2,
                        min_idx,
                        max_idx,
                        score_sign,
                        crowding_weight,
                        kernel_params,
                    )
                    .1
                },
            );
            let (proposed_idx, cand_score, cand_crowding) = e2_objective_score_at_log2(
                space,
                prescored_c_score_scan,
                erb_scan,
                &current_erb,
                agent_i,
                current_pitch_log2,
                proposal.target_pitch_log2,
                min_idx,
                max_idx,
                score_sign,
                crowding_weight,
                kernel_params,
            );
            if let Some(prev_idx) = backtrack_target
                && proposed_idx == prev_idx
                && (cand_score - current_score) <= E2_BACKTRACK_ALLOW_EPS
            {
                (agent_idx, current_score, current_crowding, false)
            } else {
                let delta = cand_score - current_score;
                if delta > E2_SCORE_IMPROVE_EPS {
                    (proposed_idx, cand_score, cand_crowding, false)
                } else if delta < 0.0 {
                    let (accept, acc_worse) =
                        metropolis_accept(delta, temperature, u01_by_agent[agent_i]);
                    if accept {
                        (proposed_idx, cand_score, cand_crowding, acc_worse)
                    } else {
                        (agent_idx, current_score, current_crowding, false)
                    }
                } else {
                    (agent_idx, current_score, current_crowding, false)
                }
            }
        } else {
            (agent_idx, current_score, current_crowding, false)
        };

        indices[agent_i] = chosen_idx;
        let moved = chosen_idx != agent_idx;
        let delta_semitones = 12.0 * (log2_ratio_scan[chosen_idx] - log2_ratio_scan[agent_idx]);
        let abs_delta = delta_semitones.abs();
        let abs_delta_moved = if moved { abs_delta } else { 0.0 };
        let c_score_chosen = prescored_c_score_scan[chosen_idx];

        if let Some(trace) = trajectory_semitones.as_deref_mut() {
            record_e2_trajectory_snapshot(trace, indices, log2_ratio_scan);
        }

        if moved {
            moved_count += 1;
        }
        if accepted_worse {
            accepted_worse_count += 1;
        }
        abs_delta_sum += abs_delta;
        abs_delta_moved_sum += abs_delta_moved;
        if c_score_current.is_finite() {
            c_score_current_sum += c_score_current;
            c_score_current_count += 1;
        }
        if c_score_chosen.is_finite() {
            c_score_chosen_sum += c_score_chosen;
            c_score_chosen_count += 1;
        }
        if chosen_score_val.is_finite() {
            score_sum += chosen_score_val;
            score_count += 1;
        }
        if chosen_crowding_val.is_finite() {
            crowding_sum += chosen_crowding_val;
            crowding_count += 1;
        }
    }

    let n = indices.len() as f32;
    UpdateStats {
        mean_c_score_current_loo: if c_score_current_count > 0 {
            c_score_current_sum / c_score_current_count as f32
        } else {
            0.0
        },
        mean_c_score_chosen_loo: if c_score_chosen_count > 0 {
            c_score_chosen_sum / c_score_chosen_count as f32
        } else {
            0.0
        },
        mean_score: if score_count > 0 {
            score_sum / score_count as f32
        } else {
            0.0
        },
        mean_crowding: if crowding_count > 0 {
            crowding_sum / crowding_count as f32
        } else {
            0.0
        },
        moved_frac: moved_count as f32 / n,
        accepted_worse_frac: accepted_worse_count as f32 / n,
        attempted_update_frac: attempt_count as f32 / n,
        moved_given_attempt_frac: if attempt_count > 0 {
            moved_count as f32 / attempt_count as f32
        } else {
            0.0
        },
        mean_abs_delta_semitones: abs_delta_sum / n,
        mean_abs_delta_semitones_moved: if moved_count > 0 {
            abs_delta_moved_sum / moved_count as f32
        } else {
            0.0
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn run_e2_once_proposal_cfg(
    space: &Log2Space,
    anchor_hz: f32,
    seed: u64,
    step_semitones: f32,
    phase_mode: E2PhaseMode,
    n_agents: usize,
    range_oct: f32,
) -> E2Run {
    let mut rng = seeded_rng(seed);
    let fixed_drone = e2_fixed_drone(space, anchor_hz);
    let anchor_hz_current = anchor_hz;
    let log2_ratio_scan = build_log2_ratio_scan(space, anchor_hz_current);
    let half_range_oct = 0.5 * range_oct.max(0.01);
    let (min_idx, max_idx) = log2_ratio_bounds(&log2_ratio_scan, -half_range_oct, half_range_oct);
    let mut agent_indices = match E2_INIT_MODE {
        E2InitMode::Uniform => init_e2_agent_indices_uniform(&mut rng, min_idx, max_idx, n_agents),
        E2InitMode::RejectConsonant => init_e2_agent_indices_reject_consonant(
            &mut rng,
            min_idx,
            max_idx,
            &log2_ratio_scan,
            n_agents,
        ),
    };

    let (erb_scan, du_scan) = erb_grid(space);
    let kernel_params = KernelParams::default();
    let workspace = build_consonance_workspace(space);
    let k_bins = k_from_semitones(step_semitones);
    let mut pitch_cores = build_e2_pitch_cores(space, &agent_indices, step_semitones);

    let mut mean_c_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_level_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_c_score_chosen_loo_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_score_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_crowding_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut accepted_worse_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut attempted_update_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut moved_given_attempt_frac_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_series = Vec::with_capacity(E2_SWEEPS);
    let mut mean_abs_delta_semitones_moved_series = Vec::with_capacity(E2_SWEEPS);
    let mut semitone_samples_pre = Vec::new();
    let mut semitone_samples_post = Vec::new();
    let mut density_mass_sum = 0.0f32;
    let mut density_mass_min = f32::INFINITY;
    let mut density_mass_max = 0.0f32;
    let mut density_mass_count = 0u32;
    let mut r_state01_min = f32::INFINITY;
    let mut r_state01_max = f32::NEG_INFINITY;
    let mut r_state01_mean_sum = 0.0f32;
    let mut r_state01_mean_count = 0u32;
    let trajectory_steps_cap = 1 + E2_SWEEPS.saturating_mul(e2_microsteps_per_sweep(n_agents));
    let mut trajectory_semitones = (0..n_agents)
        .map(|_| Vec::with_capacity(trajectory_steps_cap))
        .collect::<Vec<_>>();
    let mut backtrack_targets = agent_indices.clone();
    let phase_switch_step = phase_mode.switch_step();

    for sweep in 0..E2_SWEEPS {
        if let Some(switch_step) = phase_switch_step
            && sweep == switch_step
        {
            backtrack_targets.clone_from_slice(&agent_indices);
        }
        let score_sign = phase_mode.score_sign(sweep);
        let (env_scan, density_scan) = build_env_scans_with_fixed_sources(
            space,
            std::slice::from_ref(&fixed_drone.idx),
            &agent_indices,
            &du_scan,
        );
        let (landscape, c_score_scan, c_level_scan, density_mass, r_state_stats) =
            build_pitch_core_landscape(space, &workspace, &env_scan, &density_scan, &du_scan);

        if density_mass.is_finite() {
            density_mass_sum += density_mass;
            density_mass_min = density_mass_min.min(density_mass);
            density_mass_max = density_mass_max.max(density_mass);
            density_mass_count += 1;
        }
        if r_state_stats.mean.is_finite() {
            r_state01_min = r_state01_min.min(r_state_stats.min);
            r_state01_max = r_state01_max.max(r_state_stats.max);
            r_state01_mean_sum += r_state_stats.mean;
            r_state01_mean_count += 1;
        }

        let mean_c = mean_at_indices(&c_score_scan, &agent_indices);
        let mean_c_level = mean_at_indices(&c_level_scan, &agent_indices);
        mean_c_series.push(mean_c);
        mean_c_level_series.push(mean_c_level);

        if sweep == 0 {
            record_e2_trajectory_snapshot(
                &mut trajectory_semitones,
                &agent_indices,
                &log2_ratio_scan,
            );
        }
        if sweep >= E2_BURN_IN {
            let target = if sweep < E2_PHASE_SWITCH_STEP {
                &mut semitone_samples_pre
            } else {
                &mut semitone_samples_post
            };
            target.extend(agent_indices.iter().map(|&idx| 12.0 * log2_ratio_scan[idx]));
        }

        let temperature = e2_accept_temperature(sweep, phase_mode);
        let block_backtrack = e2_should_block_backtrack(phase_mode, sweep);
        let positions_before_update = agent_indices.clone();
        let stats = if score_sign < 0.0 {
            update_e2_sweep_scored_loo(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
                space,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &erb_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                k_bins,
                score_sign,
                E2_CROWDING_WEIGHT,
                &kernel_params,
                temperature,
                sweep,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                Some(trajectory_semitones.as_mut_slice()),
                &mut rng,
            )
        } else {
            update_e2_sweep_pitch_core_proposal(
                E2_UPDATE_SCHEDULE,
                &mut agent_indices,
                &positions_before_update,
                &mut pitch_cores,
                space,
                &landscape,
                &workspace,
                &env_scan,
                &density_scan,
                &du_scan,
                &erb_scan,
                &log2_ratio_scan,
                min_idx,
                max_idx,
                score_sign,
                E2_CROWDING_WEIGHT,
                &kernel_params,
                temperature,
                sweep,
                block_backtrack,
                if block_backtrack {
                    Some(backtrack_targets.as_slice())
                } else {
                    None
                },
                Some(trajectory_semitones.as_mut_slice()),
                &mut rng,
            )
        };
        if block_backtrack {
            e2_update_backtrack_targets(
                &mut backtrack_targets,
                &positions_before_update,
                &agent_indices,
            );
        }
        mean_c_score_loo_series.push(stats.mean_c_score_current_loo);
        mean_c_score_chosen_loo_series.push(stats.mean_c_score_chosen_loo);
        mean_score_series.push(stats.mean_score);
        mean_crowding_series.push(stats.mean_crowding);
        moved_frac_series.push(stats.moved_frac);
        accepted_worse_frac_series.push(stats.accepted_worse_frac);
        attempted_update_frac_series.push(stats.attempted_update_frac);
        moved_given_attempt_frac_series.push(stats.moved_given_attempt_frac);
        mean_abs_delta_semitones_series.push(stats.mean_abs_delta_semitones);
        mean_abs_delta_semitones_moved_series.push(stats.mean_abs_delta_semitones_moved);
    }

    let final_semitones: Vec<f32> = agent_indices
        .iter()
        .map(|&idx| 12.0 * log2_ratio_scan[idx])
        .collect();
    let final_freqs_hz: Vec<f32> = final_semitones
        .iter()
        .map(|&st| anchor_hz_current * 2.0_f32.powf(st / 12.0))
        .collect();
    let final_log2_ratios: Vec<f32> = final_semitones.iter().map(|&st| st / 12.0).collect();
    let trajectory_c_level = compute_e2_trajectory_c_levels(
        space,
        &workspace,
        &du_scan,
        fixed_drone.idx,
        anchor_hz_current,
        &trajectory_semitones,
    );

    E2Run {
        seed,
        mean_c_series,
        mean_c_level_series,
        mean_c_score_loo_series,
        mean_c_score_chosen_loo_series,
        mean_score_series,
        mean_crowding_series,
        moved_frac_series,
        accepted_worse_frac_series,
        attempted_update_frac_series,
        moved_given_attempt_frac_series,
        mean_abs_delta_semitones_series,
        mean_abs_delta_semitones_moved_series,
        semitone_samples_pre,
        semitone_samples_post,
        final_semitones,
        final_freqs_hz,
        final_log2_ratios,
        trajectory_semitones,
        trajectory_c_level,
        anchor_shift: E2AnchorShiftStats {
            step: E2_ANCHOR_SHIFT_STEP,
            anchor_hz_before: anchor_hz_current,
            anchor_hz_after: anchor_hz_current * E2_ANCHOR_SHIFT_RATIO,
            count_min: 0,
            count_max: 0,
            respawned: 0,
        },
        density_mass_mean: if density_mass_count > 0 {
            density_mass_sum / density_mass_count as f32
        } else {
            0.0
        },
        density_mass_min: if density_mass_count > 0 {
            density_mass_min
        } else {
            0.0
        },
        density_mass_max: if density_mass_count > 0 {
            density_mass_max
        } else {
            0.0
        },
        r_state01_min: if r_state01_mean_count > 0 {
            r_state01_min
        } else {
            0.0
        },
        r_state01_mean: if r_state01_mean_count > 0 {
            r_state01_mean_sum / r_state01_mean_count as f32
        } else {
            0.0
        },
        r_state01_max: if r_state01_mean_count > 0 {
            r_state01_max
        } else {
            0.0
        },
        r_ref_peak: workspace.r_ref_peak,
        roughness_k: workspace.params.roughness_k,
        roughness_ref_eps: workspace.params.roughness_ref_eps,
        fixed_drone_hz: fixed_drone.hz,
        n_agents,
        k_bins: k_from_semitones(step_semitones),
    }
}

pub fn generate_e2_baseline_render(
    n_agents: usize,
    render_partials: u32,
    range_oct: f32,
) -> io::Result<()> {
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
    let anchor_hz = E4_ANCHOR_HZ;
    let seed = E2_SEEDS[0];
    let run = run_e2_once_cfg(
        &space,
        anchor_hz,
        seed,
        E2Condition::Baseline,
        E2_STEP_SEMITONES,
        E2PhaseMode::DissonanceThenConsonance,
        None,
        0,
        n_agents,
        range_oct,
    );

    let out_dir = Path::new("supplementary_audio/audio");
    create_dir_all(out_dir)?;
    let stem = format!(
        "15_exp1_replay_seed0_baseline_n{}_p{}_{}",
        n_agents,
        render_partials,
        e2_range_oct_tag(range_oct)
    );
    let wav_path = out_dir.join(format!("{stem}.wav"));
    let csv_path = out_dir.join(format!("{stem}.csv"));
    let moves_path = out_dir.join(format!("{stem}_moves.csv"));
    let scene_metrics_path = out_dir.join(format!("{stem}_scene_metrics.csv"));

    render_e2_trajectory_wav(
        &wav_path,
        &run.trajectory_semitones,
        anchor_hz,
        render_partials,
        Some(run.fixed_drone_hz),
        true,
    )?;
    write_with_log(
        &csv_path,
        e2_baseline_summary_csv(
            &format!("{stem}.wav"),
            "continuous",
            &run,
            anchor_hz,
            render_partials,
            range_oct,
        ),
    )?;
    write_with_log(&moves_path, e2_moves_csv(&run))?;
    write_with_log(&scene_metrics_path, e2_scene_metrics_csv(&run, anchor_hz))?;
    Ok(())
}

pub fn generate_e2_proposal_render(
    n_agents: usize,
    render_partials: u32,
    range_oct: f32,
) -> io::Result<()> {
    let space = Log2Space::new(20.0, 8000.0, SPACE_BINS_PER_OCT);
    let anchor_hz = E4_ANCHOR_HZ;
    let seed = E2_SEEDS[0];
    let run = run_e2_once_proposal_cfg(
        &space,
        anchor_hz,
        seed,
        E2_STEP_SEMITONES,
        E2PhaseMode::DissonanceThenConsonance,
        n_agents,
        range_oct,
    );

    let out_dir = Path::new("supplementary_audio/audio");
    create_dir_all(out_dir)?;
    let stem = format!(
        "15_exp1_replay_seed0_proposal_n{}_p{}_{}",
        n_agents,
        render_partials,
        e2_range_oct_tag(range_oct)
    );
    let wav_path = out_dir.join(format!("{stem}.wav"));
    let csv_path = out_dir.join(format!("{stem}.csv"));
    let moves_path = out_dir.join(format!("{stem}_moves.csv"));
    let scene_metrics_path = out_dir.join(format!("{stem}_scene_metrics.csv"));

    render_e2_trajectory_wav(
        &wav_path,
        &run.trajectory_semitones,
        anchor_hz,
        render_partials,
        Some(run.fixed_drone_hz),
        true,
    )?;
    write_with_log(
        &csv_path,
        e2_baseline_summary_csv(
            &format!("{stem}.wav"),
            "continuous",
            &run,
            anchor_hz,
            render_partials,
            range_oct,
        ),
    )?;
    write_with_log(&moves_path, e2_moves_csv(&run))?;
    write_with_log(&scene_metrics_path, e2_scene_metrics_csv(&run, anchor_hz))?;
    Ok(())
}

#[cfg(test)]
mod ji_tests {
    use super::*;

    #[test]
    fn ji_unison_is_max() {
        let score = ji_interval_consonance(0.0);
        assert!(score > 0.9, "unison score should be ~1.0, got {}", score);
    }

    #[test]
    fn ji_perfect_fifth_moderate() {
        let score = ji_interval_consonance(7.02); // ~3:2
        assert!(
            score > 0.1,
            "perfect fifth score should be > 0.1, got {}",
            score
        );
    }

    #[test]
    fn ji_tritone_is_low() {
        let score = ji_interval_consonance(6.0); // tritone
        assert!(
            score < 0.01,
            "tritone score should be < 0.01, got {}",
            score
        );
    }

    #[test]
    fn ji_population_all_unisons() {
        let freqs = vec![220.0, 220.0, 220.0, 220.0];
        let score = ji_population_score(&freqs, 220.0);
        assert!(
            score > 0.9,
            "all-unison population should score > 0.9, got {}",
            score
        );
    }
}

#[cfg(test)]
mod sign_flip_tests {
    use super::*;

    #[test]
    fn permutation_pvalue_one_sample_basic() {
        // All positive values: only 1 of 16 sign patterns yields mean ≥ observed
        let x = [1.0f32, 1.0, 1.0, 1.0];
        let (p, method) = permutation_pvalue_one_sample(&x, 10_000, 42);
        assert_eq!(method, "exact");
        // exact: 2^4 = 16 patterns; only all-positive gives mean = 1.0,
        // and all-negative gives |mean| = 1.0, so p = 2/16 = 0.125
        assert!((p - 2.0 / 16.0).abs() < 1e-6, "expected p=0.125, got {}", p);
    }

    #[test]
    fn permutation_pvalue_one_sample_symmetric() {
        // Symmetric around zero: mean ≈ 0, p should be high
        let x = [1.0f32, -1.0, 1.0, -1.0];
        let (p, method) = permutation_pvalue_one_sample(&x, 10_000, 42);
        assert_eq!(method, "exact");
        assert!(p > 0.5, "symmetric data should give large p, got {}", p);
    }

    #[test]
    fn permutation_pvalue_one_sample_empty() {
        let (p, _) = permutation_pvalue_one_sample(&[], 10_000, 42);
        assert!((p - 1.0).abs() < 1e-6);
    }
}
