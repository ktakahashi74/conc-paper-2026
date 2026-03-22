use conchordal::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{
    Landscape, LandscapeParams, LandscapeUpdate, RoughnessScalarMode,
};
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::psycho_state::{compute_roughness_reference, r_pot_scan_to_r_state01_scan};
use conchordal::core::roughness_kernel::{
    KernelParams, RoughnessKernel, crowding_runtime_delta_erb, erb_grid,
};
use conchordal::core::timebase::Timebase;
use conchordal::life::articulation_core::{
    AnyArticulationCore, ArticulationState, kuramoto_phase_step,
};
use conchordal::life::control::{AgentControl, LeaveSelfOutMode, PitchCoreKind, PitchMode};
use conchordal::life::individual::{Individual, PitchHillClimbPitchCore, SoundBody};
use conchordal::life::lifecycle::LifecycleConfig;
use conchordal::life::metabolism_policy::MetabolismPolicy;
use conchordal::life::population::{ControlUpdateMode, Population};
use conchordal::life::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, PhonationSpec, RhythmCouplingMode, SpawnSpec,
    SpawnStrategy,
};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

pub const E4_ANCHOR_HZ: f32 = 220.0;
#[allow(dead_code)]
pub const E4_WINDOW_CENTS: f32 = 50.0;

const E4_GROUP_ANCHOR: u64 = 0;
const E4_GROUP_VOICES: u64 = 1;
const E4_ENV_PARTIALS_DEFAULT: u32 = 6;
const E4_ENV_PARTIAL_DECAY_DEFAULT: f32 = 1.0;

const E3_GROUP_AGENTS: u64 = 2;
const E3_FS: f32 = 48_000.0;
const E3_HOP: usize = 512;
pub(crate) const E3_BINS_PER_OCT: u32 = 96;
pub(crate) const E3_FMIN: f32 = 40.0;
pub(crate) const E3_FMAX: f32 = 2000.0;
const E3_RANGE_OCT: f32 = 2.0; // +/- 1 octave around anchor
const E3_THETA_FREQ_HZ: f32 = 1.0;
const E3_METABOLISM_RATE: f32 = 0.5;
const E3_SELECTION_RECHARGE_PER_SEC: f32 = 0.40;
const E6_INITIAL_ENERGY: f32 = 0.05;
const E6_METABOLISM_RATE: f32 = 0.12;
const E6_SELECTION_RECHARGE_PER_SEC: f32 = 0.10;
const E6_FERTILITY_ENERGY_EXPONENT: f32 = 2.0;
const E6_HEREDITY_MUTATION_MAX_ATTEMPTS: usize = 24;
const E6_HEREDITY_PEAK_MERGE_ST: f32 = 0.25;
const E6_HEREDITY_PEAK_MIN_RELATIVE_MASS: f32 = 0.20;
const E6_HEREDITY_INHERIT_SAME_FAMILY_PROB: f32 = 0.80;
const E6_HEREDITY_AZIMUTH_SIGMA_ST: f32 = 0.25;
const E6_HEREDITY_AZIMUTH_CLIP_ST: f32 = 1.5;
const E6_HEREDITY_FAMILY_OCCUPANCY_SIGMA_ST: f32 = 0.5;
const E6_HEREDITY_FAMILY_OCCUPANCY_STRENGTH: f32 = 0.0;
const E6_HEREDITY_FAMILY_CONTEXTUAL_BETA: f32 = 2.5;
const E6_POST_RESPAWN_AZIMUTH_TUNING_RADIUS_ST: f32 = 0.5;
const E6_JUVENILE_CONTEXTUAL_TUNING_RADIUS_ST: f32 = 0.5;
const E6_JUVENILE_CONTEXTUAL_TUNING_GRID_ST: f32 = 0.25;
const E6_JUVENILE_CONTEXTUAL_TUNING_TICKS: u32 = 96;
const E6_JUVENILE_CONTEXTUAL_TUNING_CROWDING_WEIGHT: f32 = 0.05;
const E6_JUVENILE_CONTEXTUAL_TUNING_IMPROVEMENT_THRESHOLD: f32 = 0.01;
const E6_JUVENILE_CULL_TICKS: u32 = 20;
const E6_JUVENILE_CULL_C_LEVEL_THRESHOLD: f32 = 0.75;
const E6_JUVENILE_CULL_PROB_PER_TICK: f32 = 0.25;
const E2_MATCH_CROWDING_WEIGHT: f32 = 0.15;
const E4_MATCH_SIGMA_CENTS: f32 = 15.0;
const E2_E6_HILL_NEIGHBOR_STEP_CENTS: f32 = 25.0;
const E2_E6_HILL_EXPLORATION: f32 = 0.0;
const E2_E6_HILL_PERSISTENCE: f32 = 0.5;
const E2_E6_HILL_MOVE_COST_COEFF: f32 = 0.75;
const E2_E6_HILL_IMPROVEMENT_THRESHOLD: f32 = 0.04;
const E6_HARD_ANTI_FUSION_EPS: f32 = 1e-6;
const E6_NICHE_PARENT_PRIOR_FLOOR: f32 = 0.05;
const E6_NICHE_PARENT_PRIOR_EXPONENT: f32 = 2.0;
pub const E6B_DEFAULT_RANGE_OCT: f32 = 4.0;
pub const E6B_DEFAULT_POP_SIZE: usize = 16;
const E6B_FUSION_MIN_SEPARATION_CENTS: f32 = 10.0;
const E6B_SELECTION_CROWDING_WEIGHT: f32 = 0.03;
const E6B_LOCAL_CAPACITY_RADIUS_CENTS: f32 = 35.0;
const E6B_LOCAL_CAPACITY_FREE_VOICES: usize = 3;
const E6B_LOCAL_CAPACITY_WEIGHT: f32 = 0.0;

#[derive(Clone, Copy, Debug, Default)]
pub struct E4RuntimeOverrides {
    pub env_partials: Option<u32>,
    pub env_partial_decay: Option<f32>,
    pub exploration: Option<f32>,
    pub persistence: Option<f32>,
    pub neighbor_step_cents: Option<f32>,
}

static E4_RUNTIME_OVERRIDES: OnceLock<Mutex<E4RuntimeOverrides>> = OnceLock::new();

fn e4_runtime_overrides_store() -> &'static Mutex<E4RuntimeOverrides> {
    E4_RUNTIME_OVERRIDES.get_or_init(|| Mutex::new(E4RuntimeOverrides::default()))
}

pub fn set_e4_runtime_overrides(overrides: E4RuntimeOverrides) {
    let mut guard = e4_runtime_overrides_store()
        .lock()
        .expect("e4 runtime overrides lock poisoned");
    *guard = overrides;
}

fn get_e4_runtime_overrides() -> E4RuntimeOverrides {
    *e4_runtime_overrides_store()
        .lock()
        .expect("e4 runtime overrides lock poisoned")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E3Condition {
    Baseline,
    NoRecharge,
}

impl E3Condition {
    pub fn label(self) -> &'static str {
        match self {
            E3Condition::Baseline => "baseline",
            E3Condition::NoRecharge => "norecharge",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct E3RunConfig {
    pub seed: u64,
    pub steps_cap: usize,
    pub min_deaths: usize,
    pub pop_size: usize,
    pub first_k: usize,
    pub condition: E3Condition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6Condition {
    Heredity,
    Random,
}

impl E6Condition {
    pub fn label(self) -> &'static str {
        match self {
            E6Condition::Heredity => "heredity",
            E6Condition::Random => "random",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6FamilyNfdMode {
    Off,
    SameFamilyOnly,
    SameFamilyAndAlt,
}

impl E6FamilyNfdMode {
    pub fn label(self) -> &'static str {
        match self {
            E6FamilyNfdMode::Off => "off",
            E6FamilyNfdMode::SameFamilyOnly => "same_only",
            E6FamilyNfdMode::SameFamilyAndAlt => "same_plus_alt",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6SelectionScoreMode {
    LegacyAnchorContextMix,
    PolyphonicLooCrowding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6RespawnMode {
    LegacyFamilyAzimuth,
    VacantNicheByParentPrior,
}

#[derive(Clone, Debug)]
pub struct E6RunConfig {
    pub seed: u64,
    pub steps_cap: usize,
    pub min_deaths: usize,
    pub pop_size: usize,
    pub first_k: usize,
    pub condition: E6Condition,
    pub snapshot_interval: usize,
    pub landscape_weight: f32,
    pub shuffle_landscape: bool,
    pub env_partials: Option<u32>,
    pub oracle_azimuth_radius_st: Option<f32>,
    pub oracle_global_anchor_c: bool,
    pub oracle_freeze_pitch_after_respawn: bool,
    pub crowding_strength_override: Option<f32>,
    pub adaptation_enabled_override: Option<bool>,
    pub range_oct_override: Option<f32>,
    pub e2_aligned_exact_local_search_radius_st: Option<f32>,
    pub disable_within_life_pitch_movement: bool,
    pub selection_enabled: bool,
    pub selection_contextual_mix_weight: f32,
    pub selection_score_mode: E6SelectionScoreMode,
    pub polyphonic_crowding_weight_override: Option<f32>,
    pub juvenile_contextual_settlement_enabled: bool,
    pub juvenile_cull_enabled: bool,
    pub record_life_diagnostics: bool,
    pub family_occupancy_strength_override: Option<f32>,
    pub family_nfd_mode: E6FamilyNfdMode,
    pub respawn_mode: E6RespawnMode,
    pub legacy_family_min_spacing_cents: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct E6AgentSnapshot {
    pub life_id: u64,
    pub agent_id: usize,
    pub freq_hz: f32,
}

#[derive(Clone, Debug)]
pub struct E6PitchSnapshot {
    pub step: usize,
    pub freqs_hz: Vec<f32>,
    pub agents: Vec<E6AgentSnapshot>,
    /// Kuramoto order parameter R ∈ [0,1]: population phase coherence
    pub phase_coherence: f32,
}

#[derive(Clone, Debug)]
pub struct E6RunResult {
    pub deaths: Vec<E3DeathRecord>,
    pub life_checks: Vec<E6LifeCheckRecord>,
    pub snapshots: Vec<E6PitchSnapshot>,
    pub respawns: Vec<E6RespawnRecord>,
    pub total_deaths: usize,
}

#[derive(Clone, Debug)]
pub struct E6bRunConfig {
    pub seed: u64,
    pub steps_cap: usize,
    pub min_deaths: usize,
    pub pop_size: usize,
    pub first_k: usize,
    pub condition: E6Condition,
    pub snapshot_interval: usize,
    pub selection_enabled: bool,
    pub shuffle_landscape: bool,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E6bRunResult {
    pub deaths: Vec<E3DeathRecord>,
    pub life_checks: Vec<E6LifeCheckRecord>,
    pub snapshots: Vec<E6PitchSnapshot>,
    pub respawns: Vec<E6RespawnRecord>,
    pub total_deaths: usize,
}

impl From<E6RunResult> for E6bRunResult {
    fn from(value: E6RunResult) -> Self {
        Self {
            deaths: value.deaths,
            life_checks: value.life_checks,
            snapshots: value.snapshots,
            respawns: value.respawns,
            total_deaths: value.total_deaths,
        }
    }
}

#[derive(Clone, Debug)]
pub struct E6SamplerDebugScan {
    pub parent_freq_hz: f32,
    pub parent_semitones_from_anchor: f32,
    pub delta_semitones: Vec<f32>,
    pub harmonicity_pmf: Vec<f32>,
    pub harmonicity_local_pmf: Vec<f32>,
    pub final_pmf: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct E6RespawnRecord {
    pub step: usize,
    pub dead_agent_id: usize,
    pub child_life_id: u64,
    pub parent_agent_id: Option<usize>,
    pub parent_life_id: Option<u64>,
    pub parent_freq_hz: Option<f32>,
    pub parent_energy: Option<f32>,
    pub parent_c_level: Option<f32>,
    pub candidate_count: usize,
    pub candidate_energy_mean: f32,
    pub candidate_energy_std: f32,
    pub candidate_c_level_mean: f32,
    pub chosen_selection_prob: Option<f32>,
    pub offspring_freq_hz: f32,
    pub spawn_freq_hz: f32,
    pub spawn_c_level: f32,
    pub parent_family_center_hz: Option<f32>,
    pub parent_azimuth_st: Option<f32>,
    pub child_family_center_hz: Option<f32>,
    pub child_azimuth_st: Option<f32>,
    pub family_inherited: Option<bool>,
    pub family_mutated: Option<bool>,
    pub oracle_applied: bool,
    pub oracle_freq_hz: Option<f32>,
    pub oracle_c_score: Option<f32>,
    pub oracle_c_level: Option<f32>,
    pub oracle_delta_st: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct E6LifeCheckRecord {
    pub life_id: u64,
    pub agent_id: usize,
    pub birth_step: u32,
    pub death_step: u32,
    pub lifetime_steps: u32,
    pub spawn_c_score: f32,
    pub spawn_c_level: f32,
    pub step1_c_score: Option<f32>,
    pub step1_c_level: Option<f32>,
    pub step10_c_score: Option<f32>,
    pub step10_c_level: Option<f32>,
    pub step100_c_score: Option<f32>,
    pub step100_c_level: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct E3DeathRecord {
    pub condition: String,
    pub seed: u64,
    pub life_id: u64,
    pub agent_id: usize,
    pub birth_step: u32,
    pub death_step: u32,
    pub lifetime_steps: u32,
    pub c_score_birth: f32,
    pub c_score_firstk: f32,
    pub avg_c_score_tick: f32,
    pub c_score_std_over_life: f32,
    pub avg_c_score_attack: f32,
    pub c_level_birth: f32,
    pub c_level_firstk: f32,
    pub avg_c_level_tick: f32,
    pub c_level_std_over_life: f32,
    pub avg_c_level_attack: f32,
    pub attack_tick_count: u32,
}

#[derive(Clone, Debug)]
pub struct E3PolicyParams {
    pub condition: String,
    pub dt_sec: f32,
    pub basal_cost_per_sec: f32,
    pub action_cost_per_attack: f32,
    pub recharge_per_attack: f32,
    pub continuous_recharge_per_sec: f32,
}

#[derive(Clone, Copy, Debug)]
struct E3LifeState {
    life_id: u64,
    birth_step: u32,
    ticks: u32,
    sum_c_score_tick: f32,
    sum_c_score_tick_sq: f32,
    sum_c_score_firstk: f32,
    c_score_birth: f32,
    sum_c_score_attack: f32,
    sum_c_level_tick: f32,
    sum_c_level_tick_sq: f32,
    sum_c_level_firstk: f32,
    firstk_count: u32,
    c_level_birth: f32,
    sum_c_level_attack: f32,
    attack_tick_count: u32,
    spawn_c_score: Option<f32>,
    spawn_c_level: Option<f32>,
    step1_c_score: Option<f32>,
    step1_c_level: Option<f32>,
    step10_c_score: Option<f32>,
    step10_c_level: Option<f32>,
    step100_c_score: Option<f32>,
    step100_c_level: Option<f32>,
    freeze_freq_hz: Option<f32>,
    pending_birth: bool,
    was_alive: bool,
}

impl E3LifeState {
    fn new(life_id: u64) -> Self {
        Self {
            life_id,
            birth_step: 0,
            ticks: 0,
            sum_c_score_tick: 0.0,
            sum_c_score_tick_sq: 0.0,
            sum_c_score_firstk: 0.0,
            c_score_birth: 0.0,
            sum_c_score_attack: 0.0,
            sum_c_level_tick: 0.0,
            sum_c_level_tick_sq: 0.0,
            sum_c_level_firstk: 0.0,
            firstk_count: 0,
            c_level_birth: 0.0,
            sum_c_level_attack: 0.0,
            attack_tick_count: 0,
            spawn_c_score: None,
            spawn_c_level: None,
            step1_c_score: None,
            step1_c_level: None,
            step10_c_score: None,
            step10_c_level: None,
            step100_c_score: None,
            step100_c_level: None,
            freeze_freq_hz: None,
            pending_birth: true,
            was_alive: true,
        }
    }

    fn reset_for_new_life(&mut self, life_id: u64) {
        self.life_id = life_id;
        self.birth_step = 0;
        self.ticks = 0;
        self.sum_c_score_tick = 0.0;
        self.sum_c_score_tick_sq = 0.0;
        self.sum_c_score_firstk = 0.0;
        self.c_score_birth = 0.0;
        self.sum_c_score_attack = 0.0;
        self.sum_c_level_tick = 0.0;
        self.sum_c_level_tick_sq = 0.0;
        self.sum_c_level_firstk = 0.0;
        self.firstk_count = 0;
        self.c_level_birth = 0.0;
        self.sum_c_level_attack = 0.0;
        self.attack_tick_count = 0;
        self.spawn_c_score = None;
        self.spawn_c_level = None;
        self.step1_c_score = None;
        self.step1_c_level = None;
        self.step10_c_score = None;
        self.step10_c_level = None;
        self.step100_c_score = None;
        self.step100_c_level = None;
        self.freeze_freq_hz = None;
        self.pending_birth = true;
        self.was_alive = false;
    }
}

pub fn configure_shared_hillclimb_control(control: &mut AgentControl, landscape_weight: f32) {
    control.pitch.mode = PitchMode::Free;
    control.pitch.core_kind = PitchCoreKind::HillClimb;
    control.pitch.landscape_weight = landscape_weight;
    control.pitch.gravity = 0.0;
    control.pitch.neighbor_step_cents = Some(E2_E6_HILL_NEIGHBOR_STEP_CENTS);
    control.pitch.exploration = E2_E6_HILL_EXPLORATION;
    control.pitch.persistence = E2_E6_HILL_PERSISTENCE;
    control.pitch.crowding_strength = E2_MATCH_CROWDING_WEIGHT;
    control.pitch.crowding_sigma_cents = E4_MATCH_SIGMA_CENTS;
    control.pitch.crowding_sigma_from_roughness = false;
    control.pitch.leave_self_out = true;
    control.pitch.leave_self_out_mode = LeaveSelfOutMode::ApproxHarmonics;
    control.pitch.anneal_temp = 0.0;
    control.pitch.move_cost_coeff = E2_E6_HILL_MOVE_COST_COEFF;
    control.pitch.improvement_threshold = E2_E6_HILL_IMPROVEMENT_THRESHOLD;
    control.pitch.global_peak_count = 0;
    control.pitch.use_ratio_candidates = false;
    control.pitch.ratio_candidate_count = 0;
}

pub fn configure_shared_hillclimb_core(core: &mut PitchHillClimbPitchCore, landscape_weight: f32) {
    core.set_exploration(E2_E6_HILL_EXPLORATION);
    core.set_persistence(E2_E6_HILL_PERSISTENCE);
    core.set_move_cost_coeff(E2_E6_HILL_MOVE_COST_COEFF);
    core.set_improvement_threshold(E2_E6_HILL_IMPROVEMENT_THRESHOLD);
    core.set_leave_self_out(true);
    core.set_leave_self_out_mode(LeaveSelfOutMode::ApproxHarmonics);
    core.set_crowding(E2_MATCH_CROWDING_WEIGHT, E4_MATCH_SIGMA_CENTS, false);
    core.set_landscape_weight(landscape_weight);
    core.set_global_peaks(0, 0.0);
    core.set_ratio_candidates(false, 0);
}

pub fn shared_hill_move_cost_coeff() -> f32 {
    E2_E6_HILL_MOVE_COST_COEFF
}

fn apply_e6_shuffle_permutation(landscape: &mut Landscape, perm: &[usize]) {
    let apply = |v: &[f32]| -> Vec<f32> { perm.iter().map(|&i| v[i]).collect() };
    landscape.harmonicity = apply(&landscape.harmonicity);
    landscape.harmonicity_path_a = apply(&landscape.harmonicity_path_a);
    landscape.harmonicity_path_b = apply(&landscape.harmonicity_path_b);
    landscape.harmonicity01 = apply(&landscape.harmonicity01);
    landscape.roughness = apply(&landscape.roughness);
    landscape.roughness01 = apply(&landscape.roughness01);
    landscape.consonance_field_score = apply(&landscape.consonance_field_score);
    landscape.consonance_field_level = apply(&landscape.consonance_field_level);
    landscape.consonance_field_energy = apply(&landscape.consonance_field_energy);
    landscape.consonance_density_mass = apply(&landscape.consonance_density_mass);
    landscape.consonance_density_pmf = apply(&landscape.consonance_density_pmf);
}

#[derive(Clone, Copy, Debug)]
struct E4SimConfig {
    anchor_hz: f32,
    center_cents: f32,
    range_oct: f32,
    voice_count: usize,
    fs: f32,
    hop: usize,
    steps: u32,
    bins_per_oct: u32,
    fmin: f32,
    fmax: f32,
    min_dist_erb: f32,
    exploration: f32,
    persistence: f32,
    theta_freq_hz: f32,
    neighbor_step_cents: f32,
    baseline_mirror_weight: f32,
    burn_in_steps: u32,
    roughness_weight_scale: f32,
    env_partials: u32,
    env_partial_decay: f32,
}

impl E4SimConfig {
    fn paper_defaults() -> Self {
        let mut cfg = Self {
            anchor_hz: E4_ANCHOR_HZ,
            center_cents: 350.0,
            range_oct: 0.5,
            voice_count: 32,
            fs: 48_000.0,
            hop: 512,
            steps: 1200,
            bins_per_oct: 96,
            fmin: 80.0,
            fmax: 2000.0,
            min_dist_erb: 0.0,
            exploration: 0.8,
            persistence: 0.2,
            theta_freq_hz: 6.0,
            neighbor_step_cents: 50.0,
            baseline_mirror_weight: 0.5,
            burn_in_steps: 600,
            roughness_weight_scale: 1.0,
            env_partials: 1, // SineBody: single partial per agent
            env_partial_decay: E4_ENV_PARTIAL_DECAY_DEFAULT,
        };
        let overrides = get_e4_runtime_overrides();
        if let Some(partials) = overrides.env_partials {
            cfg.env_partials = sanitize_env_partials(partials);
        }
        if let Some(decay) = overrides.env_partial_decay {
            cfg.env_partial_decay = sanitize_env_partial_decay(decay);
        }
        if let Some(exploration) = overrides.exploration
            && exploration.is_finite()
        {
            cfg.exploration = exploration.clamp(0.0, 1.0);
        }
        if let Some(persistence) = overrides.persistence
            && persistence.is_finite()
        {
            cfg.persistence = persistence.clamp(0.0, 1.0);
        }
        if let Some(step) = overrides.neighbor_step_cents {
            cfg.neighbor_step_cents = sanitize_neighbor_step_cents(step);
        }
        cfg
    }

    #[cfg(test)]
    fn test_defaults() -> Self {
        Self {
            voice_count: 6,
            steps: 420,
            burn_in_steps: 180,
            env_partials: E4_ENV_PARTIALS_DEFAULT,
            ..Self::paper_defaults()
        }
    }

    fn center_hz(&self) -> f32 {
        let ratio = 2.0f32.powf(self.center_cents / 1200.0);
        self.anchor_hz * ratio
    }

    fn range_bounds_hz(&self) -> (f32, f32) {
        let half = self.range_oct * 0.5;
        let center = self.center_hz();
        let lo = center * 2.0f32.powf(-half);
        let hi = center * 2.0f32.powf(half);
        (lo.min(hi), lo.max(hi))
    }
}

#[derive(Clone, Copy, Debug, Default)]
#[allow(dead_code)]
pub struct IntervalMetrics {
    pub mass_maj3: f32,
    pub mass_min3: f32,
    pub mass_p5: f32,
    pub n_voices: usize,
}

#[allow(dead_code)]
pub fn interval_metrics(anchor_hz: f32, freqs_hz: &[f32], window_cents: f32) -> IntervalMetrics {
    let mut metrics = IntervalMetrics {
        n_voices: freqs_hz.len(),
        ..IntervalMetrics::default()
    };
    if !anchor_hz.is_finite() || anchor_hz <= 0.0 {
        return metrics;
    }
    let w = window_cents.max(0.0);
    for &freq in freqs_hz {
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        let ratio = freq / anchor_hz;
        if !ratio.is_finite() || ratio <= 0.0 {
            continue;
        }
        let cents = 1200.0 * ratio.log2();
        let cents_mod = cents.rem_euclid(1200.0);
        if (cents_mod - 400.0).abs() <= w {
            metrics.mass_maj3 += 1.0;
        }
        if (cents_mod - 300.0).abs() <= w {
            metrics.mass_min3 += 1.0;
        }
        if (cents_mod - 700.0).abs() <= w {
            metrics.mass_p5 += 1.0;
        }
    }
    metrics
}

pub fn run_e3_collect_deaths(cfg: &E3RunConfig) -> Vec<E3DeathRecord> {
    let mut deaths = Vec::new();
    if cfg.pop_size == 0 || cfg.steps_cap == 0 {
        return deaths;
    }

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let mut landscape = build_anchor_landscape(
        &space,
        &params,
        anchor_hz,
        E4_ENV_PARTIALS_DEFAULT,
        E4_ENV_PARTIAL_DECAY_DEFAULT,
    );
    landscape.rhythm = init_rhythms(E3_THETA_FREQ_HZ);

    let mut pop = Population::new(Timebase {
        fs: E3_FS,
        hop: E3_HOP,
    });
    pop.set_seed(cfg.seed);
    pop.set_control_update_mode(ControlUpdateMode::SequentialRotating);
    pop.set_current_frame(0);

    let spec = e3_spawn_spec(cfg.condition, anchor_hz);
    let strategy = e3_spawn_strategy(anchor_hz, &space);
    let ids: Vec<u64> = (0..cfg.pop_size as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E3_GROUP_AGENTS,
            ids,
            spec: spec.clone(),
            strategy: Some(strategy.clone()),
        },
        &landscape,
        None,
    );

    let mut next_life_id = 0u64;
    let mut states: Vec<E3LifeState> = (0..cfg.pop_size)
        .map(|_| {
            let life_id = next_life_id;
            next_life_id += 1;
            E3LifeState::new(life_id)
        })
        .collect();

    let dt = E3_HOP as f32 / E3_FS;
    let first_k = cfg.first_k as u32;

    for step in 0..cfg.steps_cap {
        if matches!(cfg.condition, E3Condition::Baseline) {
            for agent in pop.individuals.iter_mut() {
                if !agent.is_alive() {
                    continue;
                }
                let c_score = landscape.evaluate_pitch_score(agent.body.base_freq_hz());
                apply_selection_recharge(agent, c_score, E3_SELECTION_RECHARGE_PER_SEC, dt);
            }
        }

        pop.advance(E3_HOP, E3_FS, step as u64, dt, &landscape);

        let mut respawn_ids: Vec<u64> = Vec::new();
        for agent in pop.individuals.iter_mut() {
            let id = agent.id();
            let idx = id as usize;
            if idx >= states.len() {
                continue;
            }
            let state = &mut states[idx];
            let alive = agent.is_alive();

            if alive {
                let c_score = landscape.evaluate_pitch_score(agent.body.base_freq_hz());
                let c_level = landscape.evaluate_pitch_level(agent.body.base_freq_hz());
                if state.pending_birth {
                    state.birth_step = step as u32;
                    state.c_score_birth = c_score;
                    state.c_level_birth = c_level;
                    state.pending_birth = false;
                }
                state.ticks = state.ticks.saturating_add(1);
                state.sum_c_score_tick += c_score;
                state.sum_c_score_tick_sq += c_score * c_score;
                state.sum_c_level_tick += c_level;
                state.sum_c_level_tick_sq += c_level * c_level;
                if state.firstk_count < first_k {
                    state.sum_c_score_firstk += c_score;
                    state.sum_c_level_firstk += c_level;
                    state.firstk_count += 1;
                }

                // Attack-time consonance
                if let AnyArticulationCore::Entrain(ref core) = agent.articulation.core {
                    if core.state == ArticulationState::Attack {
                        state.sum_c_score_attack += c_score;
                        state.sum_c_level_attack += c_level;
                        state.attack_tick_count += 1;
                    }
                }
            }

            if state.was_alive && !alive {
                let ticks = state.ticks.max(1);
                let avg_c_level_tick = if state.ticks > 0 {
                    state.sum_c_level_tick / state.ticks as f32
                } else {
                    0.0
                };
                let avg_c_score_tick = if state.ticks > 0 {
                    state.sum_c_score_tick / state.ticks as f32
                } else {
                    0.0
                };
                let c_score_std_over_life = if state.ticks > 0 {
                    let mean_sq = state.sum_c_score_tick_sq / state.ticks as f32;
                    let var = (mean_sq - avg_c_score_tick * avg_c_score_tick).max(0.0);
                    var.sqrt()
                } else {
                    0.0
                };
                let c_level_std_over_life = if state.ticks > 0 {
                    let mean_sq = state.sum_c_level_tick_sq / state.ticks as f32;
                    let var = (mean_sq - avg_c_level_tick * avg_c_level_tick).max(0.0);
                    var.sqrt()
                } else {
                    0.0
                };
                let c_score_firstk = if state.firstk_count > 0 {
                    state.sum_c_score_firstk / state.firstk_count as f32
                } else {
                    0.0
                };
                let c_level_firstk = if state.firstk_count > 0 {
                    state.sum_c_level_firstk / state.firstk_count as f32
                } else {
                    0.0
                };
                let avg_c_score_attack = if state.attack_tick_count > 0 {
                    state.sum_c_score_attack / state.attack_tick_count as f32
                } else {
                    avg_c_score_tick
                };
                let avg_c_level_attack = if state.attack_tick_count > 0 {
                    state.sum_c_level_attack / state.attack_tick_count as f32
                } else {
                    avg_c_level_tick // fallback
                };
                let attack_tick_count = state.attack_tick_count;

                deaths.push(E3DeathRecord {
                    condition: cfg.condition.label().to_string(),
                    seed: cfg.seed,
                    life_id: state.life_id,
                    agent_id: idx,
                    birth_step: state.birth_step,
                    death_step: step as u32,
                    lifetime_steps: ticks,
                    c_score_birth: state.c_score_birth,
                    c_score_firstk,
                    avg_c_score_tick,
                    c_score_std_over_life,
                    avg_c_score_attack,
                    c_level_birth: state.c_level_birth,
                    c_level_firstk,
                    avg_c_level_tick,
                    c_level_std_over_life,
                    avg_c_level_attack,
                    attack_tick_count,
                });

                respawn_ids.push(id);
                state.reset_for_new_life(next_life_id);
                next_life_id += 1;
            } else {
                state.was_alive = alive;
            }
        }

        for id in respawn_ids {
            pop.remove_agent(id);
            pop.apply_action(
                Action::Spawn {
                    group_id: E3_GROUP_AGENTS,
                    ids: vec![id],
                    spec: spec.clone(),
                    strategy: Some(strategy.clone()),
                },
                &landscape,
                None,
            );
        }

        landscape.rhythm.advance_in_place(dt);

        if deaths.len() >= cfg.min_deaths {
            break;
        }
    }

    deaths
}

pub fn run_e6(cfg: &E6RunConfig) -> E6RunResult {
    let mut out = E6RunResult {
        deaths: Vec::new(),
        life_checks: Vec::new(),
        snapshots: Vec::new(),
        respawns: Vec::new(),
        total_deaths: 0,
    };
    if cfg.pop_size == 0 || cfg.steps_cap == 0 {
        return out;
    }

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let partials = cfg.env_partials.unwrap_or(E4_ENV_PARTIALS_DEFAULT);
    let selection_reference_landscape = e3_reference_landscape_with_partials(anchor_hz, partials);
    let mut landscape = build_anchor_landscape(
        &space,
        &params,
        anchor_hz,
        partials,
        E4_ENV_PARTIAL_DECAY_DEFAULT,
    );
    landscape.rhythm = init_rhythms(E3_THETA_FREQ_HZ);

    let shuffle_perm: Option<Vec<usize>> = if cfg.shuffle_landscape {
        let mut rng_shuffle = SmallRng::seed_from_u64(cfg.seed ^ 0x5E6F_E6E6);
        let n = landscape.consonance_field_level.len();
        let mut perm: Vec<usize> = (0..n).collect();
        perm.shuffle(&mut rng_shuffle);
        apply_e6_shuffle_permutation(&mut landscape, &perm);
        Some(perm)
    } else {
        None
    };

    let mut pop = Population::new(Timebase {
        fs: E3_FS,
        hop: E3_HOP,
    });
    pop.set_seed(cfg.seed);
    pop.set_current_frame(0);

    let range_oct = cfg.range_oct_override.unwrap_or(E3_RANGE_OCT).max(1e-6);
    let spec = e6_spawn_spec(
        anchor_hz,
        cfg.landscape_weight,
        cfg.crowding_strength_override,
        cfg.adaptation_enabled_override,
        Some(range_oct),
        cfg.e2_aligned_exact_local_search_radius_st.is_some(),
        cfg.disable_within_life_pitch_movement,
    );
    let strategy = e3_spawn_strategy_with_range(anchor_hz, &space, range_oct);
    let ids: Vec<u64> = (0..cfg.pop_size as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E3_GROUP_AGENTS,
            ids,
            spec: spec.clone(),
            strategy: Some(strategy.clone()),
        },
        &landscape,
        None,
    );

    let mut next_life_id = 0u64;
    let mut states: Vec<E3LifeState> = (0..cfg.pop_size)
        .map(|_| {
            let life_id = next_life_id;
            next_life_id += 1;
            E3LifeState::new(life_id)
        })
        .collect();

    let dt = E3_HOP as f32 / E3_FS;
    let first_k = cfg.first_k as u32;
    let snapshot_interval = cfg.snapshot_interval.max(1);
    let family_occupancy_strength = cfg
        .family_occupancy_strength_override
        .unwrap_or(E6_HEREDITY_FAMILY_OCCUPANCY_STRENGTH)
        .max(0.0);
    let family_nfd_mode = cfg.family_nfd_mode;
    let selection_score_mode = cfg.selection_score_mode;
    let respawn_mode = cfg.respawn_mode;
    let record_life_diagnostics = cfg.record_life_diagnostics;
    let polyphonic_crowding_weight = cfg
        .polyphonic_crowding_weight_override
        .unwrap_or(E2_MATCH_CROWDING_WEIGHT)
        .max(0.0);
    let polyphonic_overcapacity_weight = if cfg.legacy_family_min_spacing_cents.is_some() {
        E6B_LOCAL_CAPACITY_WEIGHT
    } else {
        0.0
    };
    let mut rng = SmallRng::seed_from_u64(cfg.seed ^ 0xE600_5EED_u64);
    let (tessitura_min_hz, tessitura_max_hz) =
        e3_tessitura_bounds_for_range(anchor_hz, &space, range_oct);

    for step in 0..cfg.steps_cap {
        // Recompute landscape from all alive agents on every update (includes roughness).
        if step > 0 || cfg.selection_contextual_mix_weight > 0.0 {
            update_e4_landscape_from_population(
                &space,
                &params,
                &pop,
                &mut landscape,
                partials,
                E4_ENV_PARTIAL_DECAY_DEFAULT,
            );
            if let Some(ref perm) = shuffle_perm {
                apply_e6_shuffle_permutation(&mut landscape, perm);
            }
        }
        if let Some(radius_st) = cfg.e2_aligned_exact_local_search_radius_st {
            e6_apply_exact_e2_aligned_local_search(
                &mut pop,
                &selection_reference_landscape,
                step,
                radius_st,
                tessitura_min_hz,
                tessitura_max_hz,
            );
        }
        let selection_scene_landscape = if matches!(
            selection_score_mode,
            E6SelectionScoreMode::PolyphonicLooCrowding
        ) {
            Some(build_e6_scene_landscape_with_anchor(
                &space,
                &params,
                &selection_reference_landscape,
                &collect_alive_freqs(&pop),
                partials,
                E4_ENV_PARTIAL_DECAY_DEFAULT,
            ))
        } else {
            None
        };
        if cfg.disable_within_life_pitch_movement
            && cfg.e2_aligned_exact_local_search_radius_st.is_none()
            && cfg.selection_enabled
            && cfg.juvenile_contextual_settlement_enabled
        {
            match selection_score_mode {
                E6SelectionScoreMode::LegacyAnchorContextMix
                    if cfg.selection_contextual_mix_weight > 0.0 =>
                {
                    e6_apply_juvenile_contextual_local_search(
                        &mut pop,
                        &states,
                        &selection_reference_landscape,
                        &landscape,
                        &space,
                        &params,
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                        step,
                        tessitura_min_hz,
                        tessitura_max_hz,
                        cfg.selection_contextual_mix_weight,
                    );
                }
                E6SelectionScoreMode::PolyphonicLooCrowding => {
                    e6_apply_juvenile_polyphonic_local_search(
                        &mut pop,
                        &states,
                        selection_scene_landscape
                            .as_ref()
                            .expect("polyphonic juvenile scene landscape"),
                        &space,
                        &params,
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                        step,
                        tessitura_min_hz,
                        tessitura_max_hz,
                        cfg.legacy_family_min_spacing_cents,
                        polyphonic_crowding_weight,
                        polyphonic_overcapacity_weight,
                    );
                }
                _ => {}
            }
        }
        for agent_idx in 0..pop.individuals.len() {
            if !cfg.selection_enabled || !pop.individuals[agent_idx].is_alive() {
                continue;
            }
            let freq_hz = pop.individuals[agent_idx].body.base_freq_hz();
            let c_score = match selection_score_mode {
                E6SelectionScoreMode::LegacyAnchorContextMix => {
                    if cfg.selection_contextual_mix_weight > 0.0 {
                        let loo_contextual_landscape = build_e6_contextual_loo_landscape(
                            &space,
                            &params,
                            &landscape,
                            freq_hz,
                            partials,
                            E4_ENV_PARTIAL_DECAY_DEFAULT,
                        );
                        e6_selection_score(
                            &selection_reference_landscape,
                            &loo_contextual_landscape,
                            freq_hz,
                            cfg.selection_contextual_mix_weight,
                        )
                    } else {
                        selection_reference_landscape.evaluate_pitch_score(freq_hz)
                    }
                }
                E6SelectionScoreMode::PolyphonicLooCrowding => {
                    let mut other_freqs_hz = Vec::new();
                    for (other_idx, other_agent) in pop.individuals.iter().enumerate() {
                        if other_idx == agent_idx || !other_agent.is_alive() {
                            continue;
                        }
                        let other_freq_hz = other_agent.body.base_freq_hz();
                        if other_freq_hz.is_finite() && other_freq_hz > 0.0 {
                            other_freqs_hz.push(other_freq_hz);
                        }
                    }
                    let loo_scene_landscape = build_e6_contextual_loo_landscape(
                        &space,
                        &params,
                        selection_scene_landscape
                            .as_ref()
                            .expect("polyphonic scene landscape"),
                        freq_hz,
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                    );
                    e6_polyphonic_selection_score(
                        &loo_scene_landscape,
                        freq_hz,
                        &other_freqs_hz,
                        polyphonic_crowding_weight,
                        polyphonic_overcapacity_weight,
                    )
                }
            };
            let agent = &mut pop.individuals[agent_idx];
            apply_selection_recharge(agent, c_score, E6_SELECTION_RECHARGE_PER_SEC, dt);
        }
        pop.advance(E3_HOP, E3_FS, step as u64, dt, &landscape);
        if cfg.oracle_freeze_pitch_after_respawn {
            for agent in pop.individuals.iter_mut() {
                if !agent.is_alive() {
                    continue;
                }
                let idx = agent.id() as usize;
                if let Some(freq_hz) = states.get(idx).and_then(|state| state.freeze_freq_hz) {
                    agent.body.set_freq(freq_hz);
                }
            }
        }

        if step % snapshot_interval == 0 {
            e6_push_snapshot(&mut out, &pop, &states, step, landscape.rhythm.theta.phase);
        }

        let (post_advance_c_scores, post_advance_c_levels) = if record_life_diagnostics {
            let post_advance_selection_scene_landscape = if matches!(
                selection_score_mode,
                E6SelectionScoreMode::PolyphonicLooCrowding
            ) {
                Some(build_e6_scene_landscape_with_anchor(
                    &space,
                    &params,
                    &selection_reference_landscape,
                    &collect_alive_freqs(&pop),
                    partials,
                    E4_ENV_PARTIAL_DECAY_DEFAULT,
                ))
            } else {
                None
            };
            let c_scores: Vec<Option<f32>> = pop
                .individuals
                .iter()
                .enumerate()
                .map(|(agent_idx, agent)| {
                    if !agent.is_alive() {
                        return None;
                    }
                    let freq_hz = agent.body.base_freq_hz();
                    let score = match selection_score_mode {
                        E6SelectionScoreMode::LegacyAnchorContextMix => {
                            selection_reference_landscape.evaluate_pitch_score(freq_hz)
                        }
                        E6SelectionScoreMode::PolyphonicLooCrowding => {
                            let other_freqs_hz = pop
                                .individuals
                                .iter()
                                .enumerate()
                                .filter_map(|(other_idx, other)| {
                                    if other_idx == agent_idx || !other.is_alive() {
                                        return None;
                                    }
                                    let other_freq_hz = other.body.base_freq_hz();
                                    (other_freq_hz.is_finite() && other_freq_hz > 0.0)
                                        .then_some(other_freq_hz)
                                })
                                .collect::<Vec<_>>();
                            let loo_scene_landscape = build_e6_contextual_loo_landscape(
                                &space,
                                &params,
                                post_advance_selection_scene_landscape
                                    .as_ref()
                                    .expect("polyphonic post-advance scene landscape"),
                                freq_hz,
                                partials,
                                E4_ENV_PARTIAL_DECAY_DEFAULT,
                            );
                            e6_polyphonic_selection_score(
                                &loo_scene_landscape,
                                freq_hz,
                                &other_freqs_hz,
                                polyphonic_crowding_weight,
                                polyphonic_overcapacity_weight,
                            )
                        }
                    };
                    Some(score)
                })
                .collect();
            let c_levels: Vec<Option<f32>> = pop
                .individuals
                .iter()
                .map(|agent| {
                    agent.is_alive().then_some(
                        selection_reference_landscape
                            .evaluate_pitch_level(agent.body.base_freq_hz())
                            .clamp(0.0, 1.0),
                    )
                })
                .collect();
            (Some(c_scores), Some(c_levels))
        } else {
            (None, None)
        };
        let mut respawn_ids: Vec<u64> = Vec::new();
        for agent in pop.individuals.iter_mut() {
            let id = agent.id();
            let idx = id as usize;
            if idx >= states.len() {
                continue;
            }
            let state = &mut states[idx];
            let alive = agent.is_alive();

            if alive {
                if record_life_diagnostics {
                    let c_score = post_advance_c_scores
                        .as_ref()
                        .and_then(|scores| scores.get(idx))
                        .and_then(|score| *score)
                        .unwrap_or(0.0);
                    let c_level = post_advance_c_levels
                        .as_ref()
                        .and_then(|levels| levels.get(idx))
                        .and_then(|level| *level)
                        .unwrap_or(0.0);
                    if cfg.juvenile_cull_enabled
                        && state.ticks < E6_JUVENILE_CULL_TICKS
                        && c_level < E6_JUVENILE_CULL_C_LEVEL_THRESHOLD
                        && rng.random::<f32>() < E6_JUVENILE_CULL_PROB_PER_TICK
                    {
                        agent.start_remove_fade(0.0);
                    }
                    if state.pending_birth {
                        state.birth_step = step as u32;
                        state.c_score_birth = c_score;
                        state.c_level_birth = c_level;
                        state.pending_birth = false;
                    }
                    state.ticks = state.ticks.saturating_add(1);
                    state.sum_c_score_tick += c_score;
                    state.sum_c_score_tick_sq += c_score * c_score;
                    state.sum_c_level_tick += c_level;
                    state.sum_c_level_tick_sq += c_level * c_level;
                    if state.step1_c_level.is_none() {
                        state.step1_c_score = Some(c_score);
                        state.step1_c_level = Some(c_level);
                    }
                    if state.ticks >= 10 && state.step10_c_level.is_none() {
                        state.step10_c_score = Some(c_score);
                        state.step10_c_level = Some(c_level);
                    }
                    if state.ticks >= 100 && state.step100_c_level.is_none() {
                        state.step100_c_score = Some(c_score);
                        state.step100_c_level = Some(c_level);
                    }
                    if state.firstk_count < first_k {
                        state.sum_c_score_firstk += c_score;
                        state.sum_c_level_firstk += c_level;
                        state.firstk_count += 1;
                    }

                    if let AnyArticulationCore::Entrain(ref core) = agent.articulation.core {
                        if core.state == ArticulationState::Attack {
                            state.sum_c_score_attack += c_score;
                            state.sum_c_level_attack += c_level;
                            state.attack_tick_count += 1;
                        }
                    }
                } else {
                    if state.pending_birth {
                        state.birth_step = step as u32;
                        state.c_score_birth = state.spawn_c_score.unwrap_or(0.0);
                        state.c_level_birth = state.spawn_c_level.unwrap_or(0.0);
                        state.pending_birth = false;
                    }
                    state.ticks = state.ticks.saturating_add(1);
                }
            }

            if !alive && state.was_alive {
                if record_life_diagnostics {
                    if state.pending_birth {
                        state.birth_step = step as u32;
                        state.c_level_birth = selection_reference_landscape
                            .evaluate_pitch_level(agent.body.base_freq_hz())
                            .clamp(0.0, 1.0);
                        state.pending_birth = false;
                    }
                    let ticks = state.ticks.max(1);
                    let avg_c_level_tick = if state.ticks > 0 {
                        state.sum_c_level_tick / state.ticks as f32
                    } else {
                        0.0
                    };
                    let avg_c_score_tick = if state.ticks > 0 {
                        state.sum_c_score_tick / state.ticks as f32
                    } else {
                        0.0
                    };
                    let c_score_std_over_life = if state.ticks > 0 {
                        let mean_sq = state.sum_c_score_tick_sq / state.ticks as f32;
                        let var = (mean_sq - avg_c_score_tick * avg_c_score_tick).max(0.0);
                        var.sqrt()
                    } else {
                        0.0
                    };
                    let c_level_std_over_life = if state.ticks > 0 {
                        let mean_sq = state.sum_c_level_tick_sq / state.ticks as f32;
                        let var = (mean_sq - avg_c_level_tick * avg_c_level_tick).max(0.0);
                        var.sqrt()
                    } else {
                        0.0
                    };
                    let c_score_firstk = if state.firstk_count > 0 {
                        state.sum_c_score_firstk / state.firstk_count as f32
                    } else {
                        0.0
                    };
                    let c_level_firstk = if state.firstk_count > 0 {
                        state.sum_c_level_firstk / state.firstk_count as f32
                    } else {
                        0.0
                    };
                    let avg_c_score_attack = if state.attack_tick_count > 0 {
                        state.sum_c_score_attack / state.attack_tick_count as f32
                    } else {
                        avg_c_score_tick
                    };
                    let avg_c_level_attack = if state.attack_tick_count > 0 {
                        state.sum_c_level_attack / state.attack_tick_count as f32
                    } else {
                        avg_c_level_tick
                    };
                    let attack_tick_count = state.attack_tick_count;

                    out.deaths.push(E3DeathRecord {
                        condition: cfg.condition.label().to_string(),
                        seed: cfg.seed,
                        life_id: state.life_id,
                        agent_id: idx,
                        birth_step: state.birth_step,
                        death_step: step as u32,
                        lifetime_steps: ticks,
                        c_score_birth: state.c_score_birth,
                        c_score_firstk,
                        avg_c_score_tick,
                        c_score_std_over_life,
                        avg_c_score_attack,
                        c_level_birth: state.c_level_birth,
                        c_level_firstk,
                        avg_c_level_tick,
                        c_level_std_over_life,
                        avg_c_level_attack,
                        attack_tick_count,
                    });
                    out.life_checks.push(E6LifeCheckRecord {
                        life_id: state.life_id,
                        agent_id: idx,
                        birth_step: state.birth_step,
                        death_step: step as u32,
                        lifetime_steps: ticks,
                        spawn_c_score: state.spawn_c_score.unwrap_or(0.0),
                        spawn_c_level: state.spawn_c_level.unwrap_or(0.0),
                        step1_c_score: state.step1_c_score,
                        step1_c_level: state.step1_c_level,
                        step10_c_score: state.step10_c_score,
                        step10_c_level: state.step10_c_level,
                        step100_c_score: state.step100_c_score,
                        step100_c_level: state.step100_c_level,
                    });
                } else {
                    if state.pending_birth {
                        state.birth_step = step as u32;
                        state.c_score_birth = state.spawn_c_score.unwrap_or(0.0);
                        state.c_level_birth = state.spawn_c_level.unwrap_or(0.0);
                        state.pending_birth = false;
                    }
                    out.deaths.push(E3DeathRecord {
                        condition: cfg.condition.label().to_string(),
                        seed: cfg.seed,
                        life_id: state.life_id,
                        agent_id: idx,
                        birth_step: state.birth_step,
                        death_step: step as u32,
                        lifetime_steps: state.ticks.max(1),
                        c_score_birth: state.c_score_birth,
                        c_score_firstk: 0.0,
                        avg_c_score_tick: 0.0,
                        c_score_std_over_life: 0.0,
                        avg_c_score_attack: 0.0,
                        c_level_birth: state.c_level_birth,
                        c_level_firstk: 0.0,
                        avg_c_level_tick: 0.0,
                        c_level_std_over_life: 0.0,
                        avg_c_level_attack: 0.0,
                        attack_tick_count: 0,
                    });
                }

                respawn_ids.push(id);
                state.reset_for_new_life(next_life_id);
                next_life_id += 1;
            } else {
                state.was_alive = alive;
            }
        }

        let respawn_set: HashSet<u64> = respawn_ids.iter().copied().collect();
        for id in respawn_ids {
            pop.remove_agent(id);
            let child_idx = id as usize;
            let child_life_id = states
                .get(child_idx)
                .map(|state| state.life_id)
                .unwrap_or(u64::MAX);
            let oracle_radius_st = cfg.oracle_azimuth_radius_st.filter(|r| *r > 0.0);
            let oracle_global_anchor_c = cfg.oracle_global_anchor_c;
            let oracle_freeze_pitch_after_respawn = cfg.oracle_freeze_pitch_after_respawn;
            let post_respawn_azimuth_tuning_radius_st = if oracle_global_anchor_c
                || oracle_radius_st.is_some()
                || cfg.e2_aligned_exact_local_search_radius_st.is_some()
                || matches!(respawn_mode, E6RespawnMode::VacantNicheByParentPrior)
            {
                None
            } else {
                Some(E6_POST_RESPAWN_AZIMUTH_TUNING_RADIUS_ST)
            };

            let offspring_strategy = match cfg.condition {
                E6Condition::Random => {
                    let vacant_niche_sample =
                        if matches!(respawn_mode, E6RespawnMode::VacantNicheByParentPrior) {
                            let alive_freqs_hz = collect_alive_freqs(&pop);
                            let scene_landscape = build_e6_scene_landscape_with_anchor(
                                &space,
                                &params,
                                &selection_reference_landscape,
                                &alive_freqs_hz,
                                partials,
                                E4_ENV_PARTIAL_DECAY_DEFAULT,
                            );
                            let niches = e6_extract_vacant_niches(
                                &space,
                                &scene_landscape,
                                &alive_freqs_hz,
                                None,
                                tessitura_min_hz,
                                tessitura_max_hz,
                                polyphonic_crowding_weight,
                                polyphonic_overcapacity_weight,
                            );
                            Some(sample_from_vacant_niches(
                                &niches,
                                sample_random_log_freq(
                                    &mut rng,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                ),
                                false,
                                &mut rng,
                            ))
                        } else {
                            None
                        };
                    let offspring_freq = vacant_niche_sample
                        .map(|sample| sample.offspring_freq_hz)
                        .unwrap_or_else(|| {
                            sample_random_log_freq(&mut rng, tessitura_min_hz, tessitura_max_hz)
                        });
                    if oracle_global_anchor_c {
                        let (spawn_freq, oracle_c_score, oracle_c_level) =
                            oracle_global_anchor_search(
                                &selection_reference_landscape,
                                tessitura_min_hz,
                                tessitura_max_hz,
                            );
                        let oracle_delta_st = 12.0 * (spawn_freq / offspring_freq.max(1e-6)).log2();
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            child_life_id,
                            parent_agent_id: None,
                            parent_life_id: None,
                            parent_freq_hz: None,
                            parent_energy: None,
                            parent_c_level: None,
                            candidate_count: 0,
                            candidate_energy_mean: 0.0,
                            candidate_energy_std: 0.0,
                            candidate_c_level_mean: 0.0,
                            chosen_selection_prob: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level: oracle_c_level,
                            parent_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            oracle_applied: true,
                            oracle_freq_hz: Some(spawn_freq),
                            oracle_c_score: Some(oracle_c_score),
                            oracle_c_level: Some(oracle_c_level),
                            oracle_delta_st: Some(oracle_delta_st),
                        });
                        pop.apply_action(
                            Action::Spawn {
                                group_id: E3_GROUP_AGENTS,
                                ids: vec![id],
                                spec: spec.clone(),
                                strategy: Some(SpawnStrategy::RandomLog {
                                    min_freq: spawn_freq,
                                    max_freq: spawn_freq,
                                }),
                            },
                            &landscape,
                            None,
                        );
                        let _ = e6_record_child_spawn_state(
                            &mut states,
                            child_idx,
                            spawn_freq,
                            &selection_reference_landscape,
                            oracle_freeze_pitch_after_respawn,
                        );
                        continue;
                    }
                    if let Some(radius_st) = oracle_radius_st {
                        let (spawn_freq, oracle_c_score, oracle_c_level) = oracle_azimuth_search(
                            &selection_reference_landscape,
                            offspring_freq,
                            radius_st,
                            tessitura_min_hz,
                            tessitura_max_hz,
                        );
                        let oracle_delta_st = 12.0 * (spawn_freq / offspring_freq.max(1e-6)).log2();
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            child_life_id,
                            parent_agent_id: None,
                            parent_life_id: None,
                            parent_freq_hz: None,
                            parent_energy: None,
                            parent_c_level: None,
                            candidate_count: 0,
                            candidate_energy_mean: 0.0,
                            candidate_energy_std: 0.0,
                            candidate_c_level_mean: 0.0,
                            chosen_selection_prob: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level: oracle_c_level,
                            parent_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            oracle_applied: true,
                            oracle_freq_hz: Some(spawn_freq),
                            oracle_c_score: Some(oracle_c_score),
                            oracle_c_level: Some(oracle_c_level),
                            oracle_delta_st: Some(oracle_delta_st),
                        });
                        pop.apply_action(
                            Action::Spawn {
                                group_id: E3_GROUP_AGENTS,
                                ids: vec![id],
                                spec: spec.clone(),
                                strategy: Some(SpawnStrategy::RandomLog {
                                    min_freq: spawn_freq,
                                    max_freq: spawn_freq,
                                }),
                            },
                            &landscape,
                            None,
                        );
                        let _ = e6_record_child_spawn_state(
                            &mut states,
                            child_idx,
                            spawn_freq,
                            &selection_reference_landscape,
                            oracle_freeze_pitch_after_respawn,
                        );
                        continue;
                    }
                    if let Some(radius_st) = post_respawn_azimuth_tuning_radius_st {
                        let (spawn_freq, _, _) = oracle_azimuth_search(
                            &selection_reference_landscape,
                            offspring_freq,
                            radius_st,
                            tessitura_min_hz,
                            tessitura_max_hz,
                        );
                        let spawn_c_level = selection_reference_landscape
                            .evaluate_pitch_level(spawn_freq)
                            .clamp(0.0, 1.0);
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            child_life_id,
                            parent_agent_id: None,
                            parent_life_id: None,
                            parent_freq_hz: None,
                            parent_energy: None,
                            parent_c_level: None,
                            candidate_count: 0,
                            candidate_energy_mean: 0.0,
                            candidate_energy_std: 0.0,
                            candidate_c_level_mean: 0.0,
                            chosen_selection_prob: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level,
                            parent_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            oracle_applied: false,
                            oracle_freq_hz: None,
                            oracle_c_score: None,
                            oracle_c_level: None,
                            oracle_delta_st: None,
                        });
                        pop.apply_action(
                            Action::Spawn {
                                group_id: E3_GROUP_AGENTS,
                                ids: vec![id],
                                spec: spec.clone(),
                                strategy: Some(SpawnStrategy::RandomLog {
                                    min_freq: spawn_freq,
                                    max_freq: spawn_freq,
                                }),
                            },
                            &landscape,
                            None,
                        );
                        let _ = e6_record_child_spawn_state(
                            &mut states,
                            child_idx,
                            spawn_freq,
                            &selection_reference_landscape,
                            oracle_freeze_pitch_after_respawn,
                        );
                        continue;
                    }
                    if matches!(respawn_mode, E6RespawnMode::VacantNicheByParentPrior) {
                        pop.apply_action(
                            Action::Spawn {
                                group_id: E3_GROUP_AGENTS,
                                ids: vec![id],
                                spec: spec.clone(),
                                strategy: Some(SpawnStrategy::RandomLog {
                                    min_freq: offspring_freq,
                                    max_freq: offspring_freq,
                                }),
                            },
                            &landscape,
                            None,
                        );
                        let _ = e6_record_child_spawn_state(
                            &mut states,
                            child_idx,
                            offspring_freq,
                            &selection_reference_landscape,
                            oracle_freeze_pitch_after_respawn,
                        );
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            child_life_id,
                            parent_agent_id: None,
                            parent_life_id: None,
                            parent_freq_hz: None,
                            parent_energy: None,
                            parent_c_level: None,
                            candidate_count: 0,
                            candidate_energy_mean: 0.0,
                            candidate_energy_std: 0.0,
                            candidate_c_level_mean: 0.0,
                            chosen_selection_prob: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: offspring_freq,
                            spawn_c_level: selection_reference_landscape
                                .evaluate_pitch_level(offspring_freq)
                                .clamp(0.0, 1.0),
                            parent_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: vacant_niche_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: vacant_niche_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            oracle_applied: false,
                            oracle_freq_hz: None,
                            oracle_c_score: None,
                            oracle_c_level: None,
                            oracle_delta_st: None,
                        });
                        continue;
                    }
                    strategy.clone()
                }
                E6Condition::Heredity => {
                    let alive_parents: Vec<(usize, u64, f32, f32, f32, f32)> = pop
                        .individuals
                        .iter()
                        .filter(|a| a.is_alive() && !respawn_set.contains(&a.id()))
                        .filter_map(|a| {
                            let freq = a.body.base_freq_hz();
                            if !freq.is_finite() || freq <= 0.0 {
                                return None;
                            }
                            let energy = e6_agent_energy(a)?;
                            let weight = e6_parent_fertility_weight(a)?;
                            let parent_idx = a.id() as usize;
                            let parent_life_id = states
                                .get(parent_idx)
                                .map(|state| state.life_id)
                                .unwrap_or(u64::MAX);
                            let parent_c_level = selection_reference_landscape
                                .evaluate_pitch_level(freq)
                                .clamp(0.0, 1.0);
                            Some((
                                parent_idx,
                                parent_life_id,
                                freq,
                                energy,
                                parent_c_level,
                                weight,
                            ))
                        })
                        .collect();
                    if alive_parents.is_empty() {
                        strategy.clone()
                    } else {
                        let candidate_count = alive_parents.len();
                        let total_weight: f32 =
                            alive_parents.iter().map(|(_, _, _, _, _, w)| *w).sum();
                        let mean_energy = alive_parents
                            .iter()
                            .map(|(_, _, _, energy, _, _)| *energy)
                            .sum::<f32>()
                            / candidate_count as f32;
                        let mean_energy_sq = alive_parents
                            .iter()
                            .map(|(_, _, _, energy, _, _)| energy * energy)
                            .sum::<f32>()
                            / candidate_count as f32;
                        let energy_std =
                            (mean_energy_sq - mean_energy * mean_energy).max(0.0).sqrt();
                        let mean_c_level = alive_parents
                            .iter()
                            .map(|(_, _, _, _, c_level, _)| *c_level)
                            .sum::<f32>()
                            / candidate_count as f32;
                        let parent_idx = if cfg.selection_enabled {
                            if let Ok(dist) = WeightedIndex::new(
                                alive_parents.iter().map(|(_, _, _, _, _, w)| *w),
                            ) {
                                dist.sample(&mut rng)
                            } else {
                                rng.random_range(0..alive_parents.len())
                            }
                        } else {
                            rng.random_range(0..alive_parents.len())
                        };
                        let (
                            parent_agent_id,
                            parent_life_id,
                            parent_freq,
                            parent_energy,
                            parent_c_level,
                            parent_weight,
                        ) = alive_parents[parent_idx];
                        let chosen_prob = if !cfg.selection_enabled {
                            Some(1.0 / candidate_count as f32)
                        } else if total_weight > 0.0 && total_weight.is_finite() {
                            Some((parent_weight / total_weight).clamp(0.0, 1.0))
                        } else {
                            None
                        };
                        let mut parent_landscape = build_e6_parent_harmonicity_landscape(
                            &space,
                            &params,
                            parent_freq,
                            partials,
                            E4_ENV_PARTIAL_DECAY_DEFAULT,
                        );
                        if let Some(ref perm) = shuffle_perm {
                            apply_e6_shuffle_permutation(&mut parent_landscape, perm);
                        }
                        let alive_freqs_hz = collect_alive_freqs(&pop);
                        let heredity_sample = match respawn_mode {
                            E6RespawnMode::LegacyFamilyAzimuth => sample_from_parent_harmonicity(
                                &parent_landscape,
                                &landscape,
                                &alive_freqs_hz,
                                parent_freq,
                                spec.control.pitch.crowding_sigma_cents,
                                tessitura_min_hz,
                                tessitura_max_hz,
                                &mut rng,
                                family_occupancy_strength,
                                family_nfd_mode,
                                cfg.legacy_family_min_spacing_cents,
                            ),
                            E6RespawnMode::VacantNicheByParentPrior => {
                                let scene_landscape = build_e6_scene_landscape_with_anchor(
                                    &space,
                                    &params,
                                    &selection_reference_landscape,
                                    &alive_freqs_hz,
                                    partials,
                                    E4_ENV_PARTIAL_DECAY_DEFAULT,
                                );
                                let niches = e6_extract_vacant_niches(
                                    &space,
                                    &scene_landscape,
                                    &alive_freqs_hz,
                                    Some(&parent_landscape),
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                );
                                sample_from_vacant_niches(&niches, parent_freq, true, &mut rng)
                            }
                        };
                        let offspring_freq = heredity_sample.offspring_freq_hz;
                        let (spawn_freq, oracle_applied, oracle_c_score, oracle_c_level) =
                            if oracle_global_anchor_c {
                                let (freq, score, level) = oracle_global_anchor_search(
                                    &selection_reference_landscape,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                );
                                (freq, true, Some(score), Some(level))
                            } else if let Some(radius_st) = oracle_radius_st {
                                let (freq, score, level) = oracle_azimuth_search(
                                    &selection_reference_landscape,
                                    offspring_freq,
                                    radius_st,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                );
                                (freq, true, Some(score), Some(level))
                            } else if let Some(radius_st) = post_respawn_azimuth_tuning_radius_st {
                                let (freq, _, _) = oracle_azimuth_search(
                                    &selection_reference_landscape,
                                    offspring_freq,
                                    radius_st,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                );
                                (freq, false, None, None)
                            } else {
                                (offspring_freq, false, None, None)
                            };
                        let spawn_c_level = oracle_c_level.unwrap_or_else(|| {
                            selection_reference_landscape
                                .evaluate_pitch_level(spawn_freq)
                                .clamp(0.0, 1.0)
                        });
                        let oracle_delta_st = oracle_c_level
                            .map(|_| 12.0 * (spawn_freq / offspring_freq.max(1e-6)).log2());
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            child_life_id,
                            parent_agent_id: Some(parent_agent_id),
                            parent_life_id: Some(parent_life_id),
                            parent_freq_hz: Some(parent_freq),
                            parent_energy: Some(parent_energy),
                            parent_c_level: Some(parent_c_level),
                            candidate_count,
                            candidate_energy_mean: mean_energy,
                            candidate_energy_std: energy_std,
                            candidate_c_level_mean: mean_c_level,
                            chosen_selection_prob: chosen_prob,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level,
                            parent_family_center_hz: Some(heredity_sample.parent_family_center_hz),
                            parent_azimuth_st: Some(heredity_sample.parent_azimuth_st),
                            child_family_center_hz: Some(heredity_sample.child_family_center_hz),
                            child_azimuth_st: Some(heredity_sample.child_azimuth_st),
                            family_inherited: Some(heredity_sample.family_inherited),
                            family_mutated: Some(heredity_sample.family_mutated),
                            oracle_applied,
                            oracle_freq_hz: oracle_applied.then_some(spawn_freq),
                            oracle_c_score,
                            oracle_c_level,
                            oracle_delta_st,
                        });
                        pop.apply_action(
                            Action::Spawn {
                                group_id: E3_GROUP_AGENTS,
                                ids: vec![id],
                                spec: spec.clone(),
                                strategy: Some(SpawnStrategy::RandomLog {
                                    min_freq: spawn_freq,
                                    max_freq: spawn_freq,
                                }),
                            },
                            &landscape,
                            None,
                        );
                        let _ = e6_record_child_spawn_state(
                            &mut states,
                            child_idx,
                            spawn_freq,
                            &selection_reference_landscape,
                            oracle_freeze_pitch_after_respawn,
                        );
                        continue;
                    }
                }
            };

            pop.apply_action(
                Action::Spawn {
                    group_id: E3_GROUP_AGENTS,
                    ids: vec![id],
                    spec: spec.clone(),
                    strategy: Some(offspring_strategy),
                },
                &landscape,
                None,
            );
            let spawn_freq = pop
                .individuals
                .iter()
                .find(|a| a.id() == id)
                .map(|a| a.body.base_freq_hz())
                .unwrap_or(anchor_hz);
            let spawn_c_level = selection_reference_landscape
                .evaluate_pitch_level(spawn_freq)
                .clamp(0.0, 1.0);
            let _ = e6_record_child_spawn_state(
                &mut states,
                child_idx,
                spawn_freq,
                &selection_reference_landscape,
                oracle_freeze_pitch_after_respawn,
            );
            out.respawns.push(E6RespawnRecord {
                step,
                dead_agent_id: id as usize,
                child_life_id,
                parent_agent_id: None,
                parent_life_id: None,
                parent_freq_hz: None,
                parent_energy: None,
                parent_c_level: None,
                candidate_count: 0,
                candidate_energy_mean: 0.0,
                candidate_energy_std: 0.0,
                candidate_c_level_mean: 0.0,
                chosen_selection_prob: None,
                offspring_freq_hz: spawn_freq,
                spawn_freq_hz: spawn_freq,
                spawn_c_level,
                parent_family_center_hz: None,
                parent_azimuth_st: None,
                child_family_center_hz: None,
                child_azimuth_st: None,
                family_inherited: None,
                family_mutated: None,
                oracle_applied: false,
                oracle_freq_hz: None,
                oracle_c_score: None,
                oracle_c_level: None,
                oracle_delta_st: None,
            });
        }

        landscape.rhythm.advance_in_place(dt);

        if out.deaths.len() >= cfg.min_deaths {
            if cfg.legacy_family_min_spacing_cents.is_some()
                && cfg.selection_enabled
                && cfg.juvenile_contextual_settlement_enabled
            {
                for settle_pass in 0..4 {
                    update_e4_landscape_from_population(
                        &space,
                        &params,
                        &pop,
                        &mut landscape,
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                    );
                    if let Some(ref perm) = shuffle_perm {
                        apply_e6_shuffle_permutation(&mut landscape, perm);
                    }
                    let settle_scene_landscape = build_e6_scene_landscape_with_anchor(
                        &space,
                        &params,
                        &selection_reference_landscape,
                        &collect_alive_freqs(&pop),
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                    );
                    e6_apply_juvenile_polyphonic_local_search(
                        &mut pop,
                        &states,
                        &settle_scene_landscape,
                        &space,
                        &params,
                        partials,
                        E4_ENV_PARTIAL_DECAY_DEFAULT,
                        step + settle_pass + 1,
                        tessitura_min_hz,
                        tessitura_max_hz,
                        cfg.legacy_family_min_spacing_cents,
                        polyphonic_crowding_weight,
                        polyphonic_overcapacity_weight,
                    );
                }
                e6_push_snapshot(
                    &mut out,
                    &pop,
                    &states,
                    step + 1,
                    landscape.rhythm.theta.phase,
                );
            }
            break;
        }
    }

    out.total_deaths = out.deaths.len();
    out
}

pub fn run_e6b(cfg: &E6bRunConfig) -> E6bRunResult {
    let shared_cfg = E6RunConfig {
        seed: cfg.seed,
        steps_cap: cfg.steps_cap,
        min_deaths: cfg.min_deaths,
        pop_size: cfg.pop_size,
        first_k: cfg.first_k,
        condition: cfg.condition,
        snapshot_interval: cfg.snapshot_interval,
        landscape_weight: 0.0,
        shuffle_landscape: cfg.shuffle_landscape,
        env_partials: None,
        oracle_azimuth_radius_st: None,
        oracle_global_anchor_c: false,
        oracle_freeze_pitch_after_respawn: false,
        crowding_strength_override: Some(0.0),
        adaptation_enabled_override: Some(false),
        range_oct_override: Some(E6B_DEFAULT_RANGE_OCT),
        e2_aligned_exact_local_search_radius_st: None,
        disable_within_life_pitch_movement: true,
        selection_enabled: cfg.selection_enabled,
        selection_contextual_mix_weight: 0.0,
        selection_score_mode: E6SelectionScoreMode::PolyphonicLooCrowding,
        polyphonic_crowding_weight_override: Some(E6B_SELECTION_CROWDING_WEIGHT),
        juvenile_contextual_settlement_enabled: true,
        juvenile_cull_enabled: false,
        record_life_diagnostics: false,
        family_occupancy_strength_override: None,
        family_nfd_mode: E6FamilyNfdMode::Off,
        respawn_mode: E6RespawnMode::LegacyFamilyAzimuth,
        legacy_family_min_spacing_cents: Some(E6B_FUSION_MIN_SEPARATION_CENTS),
    };
    run_e6(&shared_cfg).into()
}

pub fn e3_reference_landscape(anchor_hz: f32) -> Landscape {
    e3_reference_landscape_with_partials(anchor_hz, E4_ENV_PARTIALS_DEFAULT)
}

pub fn e3_reference_landscape_with_partials(anchor_hz: f32, env_partials: u32) -> Landscape {
    let anchor_hz = anchor_hz.max(1.0);
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let mut landscape = build_anchor_landscape(
        &space,
        &params,
        anchor_hz,
        env_partials,
        E4_ENV_PARTIAL_DECAY_DEFAULT,
    );
    landscape.rhythm = init_rhythms(E3_THETA_FREQ_HZ);
    landscape
}

#[allow(dead_code)]
pub fn run_e4_condition(mirror_weight: f32, seed: u64) -> Vec<f32> {
    run_e4_condition_with_config(mirror_weight, seed, &E4SimConfig::paper_defaults())
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E4TailSamples {
    pub mirror_weight: f32,
    pub seed: u64,
    pub anchor_hz: f32,
    pub burn_in_steps: u32,
    pub steps_total: u32,
    pub tail_window: u32,
    pub freqs_by_step: Vec<Vec<f32>>,
    pub agent_freqs_by_step: Vec<Vec<E4AgentFreq>>,
    pub landscape_metrics_by_step: Vec<E4LandscapeMetrics>,
}

#[derive(Hash, Eq, PartialEq)]
struct E4TailSamplesCacheKey {
    mirror_weight_bits: u32,
    seed: u64,
    tail_window: u32,
    steps: u32,
    fmin_bits: u32,
    fmax_bits: u32,
    anchor_hz_bits: u32,
    center_cents_bits: u32,
    range_oct_bits: u32,
    fs_bits: u32,
    hop: u32,
    bins_per_oct: u32,
    voice_count: u32,
    min_dist_erb_bits: u32,
    exploration_bits: u32,
    persistence_bits: u32,
    theta_freq_hz_bits: u32,
    neighbor_step_cents_bits: u32,
    baseline_mirror_weight_bits: u32,
    burn_in_steps: u32,
    roughness_weight_scale_bits: u32,
    env_partials: u32,
    env_partial_decay_bits: u32,
}

type E4TailSamplesCache = std::collections::HashMap<E4TailSamplesCacheKey, Arc<E4TailSamples>>;
static E4_TAIL_SAMPLES_CACHE: OnceLock<Mutex<E4TailSamplesCache>> = OnceLock::new();

fn e4_tail_samples_cache() -> &'static Mutex<E4TailSamplesCache> {
    E4_TAIL_SAMPLES_CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()))
}

fn e4_tail_samples_cache_key(
    cfg: &E4SimConfig,
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
) -> E4TailSamplesCacheKey {
    E4TailSamplesCacheKey {
        mirror_weight_bits: mirror_weight.to_bits(),
        seed,
        tail_window,
        steps: cfg.steps,
        fmin_bits: cfg.fmin.to_bits(),
        fmax_bits: cfg.fmax.to_bits(),
        anchor_hz_bits: cfg.anchor_hz.to_bits(),
        center_cents_bits: cfg.center_cents.to_bits(),
        range_oct_bits: cfg.range_oct.to_bits(),
        fs_bits: cfg.fs.to_bits(),
        hop: cfg.hop as u32,
        bins_per_oct: cfg.bins_per_oct,
        voice_count: cfg.voice_count as u32,
        min_dist_erb_bits: cfg.min_dist_erb.to_bits(),
        exploration_bits: cfg.exploration.to_bits(),
        persistence_bits: cfg.persistence.to_bits(),
        theta_freq_hz_bits: cfg.theta_freq_hz.to_bits(),
        neighbor_step_cents_bits: cfg.neighbor_step_cents.to_bits(),
        baseline_mirror_weight_bits: cfg.baseline_mirror_weight.to_bits(),
        burn_in_steps: cfg.burn_in_steps,
        roughness_weight_scale_bits: cfg.roughness_weight_scale.to_bits(),
        env_partials: cfg.env_partials,
        env_partial_decay_bits: cfg.env_partial_decay.to_bits(),
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E4MirrorScheduleSamples {
    pub freqs_by_step: Vec<Vec<f32>>,
    pub mirror_weight_by_step: Vec<f32>,
    pub agent_freqs_by_step: Vec<Vec<E4AgentFreq>>,
    pub landscape_metrics_by_step: Vec<E4LandscapeMetrics>,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
pub struct E4PaperMeta {
    pub anchor_hz: f32,
    pub center_cents: f32,
    pub range_oct: f32,
    pub voice_count: usize,
    pub fs: f32,
    pub hop: usize,
    pub steps: u32,
    pub bins_per_oct: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub min_dist_erb: f32,
    pub exploration: f32,
    pub persistence: f32,
    pub theta_freq_hz: f32,
    pub neighbor_step_cents: f32,
    pub baseline_mirror_weight: f32,
    pub burn_in_steps: u32,
    pub roughness_weight_scale: f32,
    pub env_partials: u32,
    pub env_partial_decay: f32,
}

#[allow(dead_code)]
pub fn e4_paper_meta() -> E4PaperMeta {
    let cfg = E4SimConfig::paper_defaults();
    E4PaperMeta {
        anchor_hz: cfg.anchor_hz,
        center_cents: cfg.center_cents,
        range_oct: cfg.range_oct,
        voice_count: cfg.voice_count,
        fs: cfg.fs,
        hop: cfg.hop,
        steps: cfg.steps,
        bins_per_oct: cfg.bins_per_oct,
        fmin: cfg.fmin,
        fmax: cfg.fmax,
        min_dist_erb: cfg.min_dist_erb,
        exploration: cfg.exploration,
        persistence: cfg.persistence,
        theta_freq_hz: cfg.theta_freq_hz,
        neighbor_step_cents: cfg.neighbor_step_cents,
        baseline_mirror_weight: cfg.baseline_mirror_weight,
        burn_in_steps: cfg.burn_in_steps,
        roughness_weight_scale: cfg.roughness_weight_scale,
        env_partials: cfg.env_partials,
        env_partial_decay: cfg.env_partial_decay,
    }
}

#[derive(Clone, Copy, Debug)]
pub struct E4AgentFreq {
    pub agent_id: u64,
    pub freq_hz: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct E4LandscapeMetrics {
    pub root_affinity: f32,
    pub overtone_affinity: f32,
    pub binding_strength: f32,
    pub harmonic_tilt: f32,
}

pub fn run_e4_condition_tail_samples(
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
) -> Arc<E4TailSamples> {
    run_e4_condition_tail_samples_with_config(
        mirror_weight,
        seed,
        &E4SimConfig::paper_defaults(),
        tail_window,
    )
}

pub fn run_e4_condition_tail_samples_with_wr(
    mirror_weight: f32,
    seed: u64,
    tail_window: u32,
    roughness_weight_scale: f32,
) -> Arc<E4TailSamples> {
    let mut cfg = E4SimConfig::paper_defaults();
    cfg.roughness_weight_scale = sanitize_roughness_weight_scale(roughness_weight_scale);
    run_e4_condition_tail_samples_with_config(mirror_weight, seed, &cfg, tail_window)
}

pub fn run_e4_mirror_schedule_samples(
    seed: u64,
    steps_total: u32,
    schedule: &[(u32, f32)],
) -> E4MirrorScheduleSamples {
    run_e4_mirror_schedule_samples_with_config(
        seed,
        steps_total,
        schedule,
        &E4SimConfig::paper_defaults(),
    )
}

#[allow(dead_code)]
fn run_e4_condition_with_config(mirror_weight: f32, seed: u64, cfg: &E4SimConfig) -> Vec<f32> {
    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs, cfg.roughness_weight_scale);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

    let mut pop = Population::new(Timebase {
        fs: cfg.fs,
        hop: cfg.hop,
    });
    pop.set_seed(seed);

    let baseline_mirror = cfg.baseline_mirror_weight.clamp(0.0, 1.0);
    let target_mirror = mirror_weight.clamp(0.0, 1.0);
    let burn_in_steps = cfg.burn_in_steps.min(cfg.steps);
    let mut target_applied = burn_in_steps == 0;
    let initial_mirror = if target_applied {
        target_mirror
    } else {
        baseline_mirror
    };
    apply_mirror_weight(&mut pop, &landscape, &mut params, initial_mirror);

    landscape = build_anchor_landscape(
        &space,
        &params,
        cfg.anchor_hz,
        cfg.env_partials,
        cfg.env_partial_decay,
    );

    let anchor_spec = SpawnSpec {
        control: anchor_control(cfg.anchor_hz),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_ANCHOR,
            ids: vec![0],
            spec: anchor_spec,
            strategy: None,
        },
        &landscape,
        None,
    );

    let (min_freq, max_freq) = cfg.range_bounds_hz();
    let voice_spec = SpawnSpec {
        control: voice_control(cfg),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    let ids: Vec<u64> = (1..=cfg.voice_count as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let dt = cfg.hop as f32 / cfg.fs;
    for step in 0..cfg.steps {
        if !target_applied && step >= burn_in_steps {
            apply_mirror_weight(&mut pop, &landscape, &mut params, target_mirror);
            target_applied = true;
        }
        update_e4_landscape_from_population(
            &space,
            &params,
            &pop,
            &mut landscape,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false, &landscape);
        rhythms.advance_in_place(dt);
    }

    let mut freqs = Vec::with_capacity(cfg.voice_count);
    for agent in &pop.individuals {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let freq = agent.body.base_freq_hz();
        if freq.is_finite() && freq > 0.0 {
            freqs.push(freq);
        }
    }
    freqs
}

#[allow(dead_code)]
fn run_e4_condition_tail_samples_with_config(
    mirror_weight: f32,
    seed: u64,
    cfg: &E4SimConfig,
    tail_window: u32,
) -> Arc<E4TailSamples> {
    let key = e4_tail_samples_cache_key(cfg, mirror_weight, seed, tail_window);
    {
        let cache = e4_tail_samples_cache()
            .lock()
            .expect("tail samples cache poisoned");
        if let Some(samples) = cache.get(&key) {
            return Arc::clone(samples);
        }
    }

    let samples =
        run_e4_condition_tail_samples_with_config_uncached(mirror_weight, seed, cfg, tail_window);
    let mut cache = e4_tail_samples_cache()
        .lock()
        .expect("tail samples cache poisoned");
    if let Some(samples) = cache.get(&key) {
        return Arc::clone(samples);
    }
    let arc = Arc::new(samples);
    cache.insert(key, Arc::clone(&arc));
    arc
}

#[allow(dead_code)]
fn run_e4_condition_tail_samples_with_config_uncached(
    mirror_weight: f32,
    seed: u64,
    cfg: &E4SimConfig,
    tail_window: u32,
) -> E4TailSamples {
    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs, cfg.roughness_weight_scale);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

    let mut pop = Population::new(Timebase {
        fs: cfg.fs,
        hop: cfg.hop,
    });
    pop.set_seed(seed);

    let baseline_mirror = cfg.baseline_mirror_weight.clamp(0.0, 1.0);
    let target_mirror = mirror_weight.clamp(0.0, 1.0);
    let burn_in_steps = cfg.burn_in_steps.min(cfg.steps);
    let mut target_applied = burn_in_steps == 0;
    let initial_mirror = if target_applied {
        target_mirror
    } else {
        baseline_mirror
    };
    apply_mirror_weight(&mut pop, &landscape, &mut params, initial_mirror);

    landscape = build_anchor_landscape(
        &space,
        &params,
        cfg.anchor_hz,
        cfg.env_partials,
        cfg.env_partial_decay,
    );

    let anchor_spec = SpawnSpec {
        control: anchor_control(cfg.anchor_hz),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_ANCHOR,
            ids: vec![0],
            spec: anchor_spec,
            strategy: None,
        },
        &landscape,
        None,
    );

    let (min_freq, max_freq) = cfg.range_bounds_hz();
    let voice_spec = SpawnSpec {
        control: voice_control(cfg),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    let ids: Vec<u64> = (1..=cfg.voice_count as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let response_steps = cfg.steps.saturating_sub(burn_in_steps).max(1);
    let tail_window = tail_window.min(response_steps).max(1);
    let start_step = cfg.steps.saturating_sub(tail_window);
    let dt = cfg.hop as f32 / cfg.fs;
    let mut freqs_by_step: Vec<Vec<f32>> = Vec::with_capacity(tail_window as usize);
    let mut agent_freqs_by_step: Vec<Vec<E4AgentFreq>> = Vec::with_capacity(tail_window as usize);
    let mut landscape_metrics_by_step: Vec<E4LandscapeMetrics> =
        Vec::with_capacity(tail_window as usize);

    for step in 0..cfg.steps {
        if !target_applied && step >= burn_in_steps {
            apply_mirror_weight(&mut pop, &landscape, &mut params, target_mirror);
            target_applied = true;
        }
        update_e4_landscape_from_population(
            &space,
            &params,
            &pop,
            &mut landscape,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false, &landscape);
        rhythms.advance_in_place(dt);

        if step >= start_step {
            update_e4_landscape_from_population(
                &space,
                &params,
                &pop,
                &mut landscape,
                cfg.env_partials,
                cfg.env_partial_decay,
            );
            landscape.rhythm = rhythms;
            landscape_metrics_by_step.push(E4LandscapeMetrics {
                root_affinity: landscape.root_affinity,
                overtone_affinity: landscape.overtone_affinity,
                binding_strength: landscape.binding_strength,
                harmonic_tilt: landscape.harmonic_tilt,
            });
            let agent_freqs = collect_voice_freqs_with_ids(&pop);
            if agent_freqs.len() != cfg.voice_count {
                panic!(
                    "E4 protocol violation: seed={seed} step={step} voices={} expected={}",
                    agent_freqs.len(),
                    cfg.voice_count
                );
            }
            freqs_by_step.push(agent_freqs.iter().map(|row| row.freq_hz).collect());
            agent_freqs_by_step.push(agent_freqs);
        }
    }

    E4TailSamples {
        mirror_weight: target_mirror,
        seed,
        anchor_hz: cfg.anchor_hz,
        burn_in_steps,
        steps_total: cfg.steps,
        tail_window,
        freqs_by_step,
        agent_freqs_by_step,
        landscape_metrics_by_step,
    }
}

fn run_e4_mirror_schedule_samples_with_config(
    seed: u64,
    steps_total: u32,
    schedule: &[(u32, f32)],
    cfg: &E4SimConfig,
) -> E4MirrorScheduleSamples {
    if steps_total == 0 {
        return E4MirrorScheduleSamples {
            freqs_by_step: Vec::new(),
            mirror_weight_by_step: Vec::new(),
            agent_freqs_by_step: Vec::new(),
            landscape_metrics_by_step: Vec::new(),
        };
    }
    let schedule = normalize_mirror_schedule(schedule, 0.0);
    let mut sched_idx = 0usize;
    let mut current_weight = schedule[0].1;

    let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
    let mut params = make_landscape_params(&space, cfg.fs, cfg.roughness_weight_scale);
    let mut landscape = Landscape::new(space.clone());
    let mut rhythms = init_e4_rhythms(cfg);

    let mut pop = Population::new(Timebase {
        fs: cfg.fs,
        hop: cfg.hop,
    });
    pop.set_seed(seed);
    apply_mirror_weight(&mut pop, &landscape, &mut params, current_weight);

    landscape = build_anchor_landscape(
        &space,
        &params,
        cfg.anchor_hz,
        cfg.env_partials,
        cfg.env_partial_decay,
    );

    let anchor_spec = SpawnSpec {
        control: anchor_control(cfg.anchor_hz),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_ANCHOR,
            ids: vec![0],
            spec: anchor_spec,
            strategy: None,
        },
        &landscape,
        None,
    );

    let (min_freq, max_freq) = cfg.range_bounds_hz();
    let voice_spec = SpawnSpec {
        control: voice_control(cfg),
        articulation: ArticulationCoreConfig::Drone {
            sway: None,
            breath_gain_init: Some(1.0),
            envelope: None,
        },
    };
    let ids: Vec<u64> = (1..=cfg.voice_count as u64).collect();
    pop.apply_action(
        Action::Spawn {
            group_id: E4_GROUP_VOICES,
            ids,
            spec: voice_spec,
            strategy: None,
        },
        &landscape,
        None,
    );
    apply_e4_initial_pitches(&mut pop, &space, &landscape, cfg, seed, min_freq, max_freq);

    let dt = cfg.hop as f32 / cfg.fs;
    let mut freqs_by_step = Vec::with_capacity(steps_total as usize);
    let mut mirror_weight_by_step = Vec::with_capacity(steps_total as usize);
    let mut agent_freqs_by_step = Vec::with_capacity(steps_total as usize);
    let mut landscape_metrics_by_step = Vec::with_capacity(steps_total as usize);

    for step in 0..steps_total {
        while sched_idx + 1 < schedule.len() && step >= schedule[sched_idx + 1].0 {
            sched_idx += 1;
            current_weight = schedule[sched_idx].1;
            apply_mirror_weight(&mut pop, &landscape, &mut params, current_weight);
        }

        update_e4_landscape_from_population(
            &space,
            &params,
            &pop,
            &mut landscape,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        landscape.rhythm = rhythms;
        pop.advance(cfg.hop, cfg.fs, step as u64, dt, &landscape);
        pop.cleanup_dead(step as u64, dt, false, &landscape);
        rhythms.advance_in_place(dt);
        update_e4_landscape_from_population(
            &space,
            &params,
            &pop,
            &mut landscape,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        landscape.rhythm = rhythms;
        landscape_metrics_by_step.push(E4LandscapeMetrics {
            root_affinity: landscape.root_affinity,
            overtone_affinity: landscape.overtone_affinity,
            binding_strength: landscape.binding_strength,
            harmonic_tilt: landscape.harmonic_tilt,
        });

        let agent_freqs = collect_voice_freqs_with_ids(&pop);
        if agent_freqs.len() != cfg.voice_count {
            panic!(
                "E4 schedule protocol violation: seed={seed} step={step} voices={} expected={}",
                agent_freqs.len(),
                cfg.voice_count
            );
        }
        mirror_weight_by_step.push(current_weight);
        freqs_by_step.push(agent_freqs.iter().map(|row| row.freq_hz).collect());
        agent_freqs_by_step.push(agent_freqs);
    }

    E4MirrorScheduleSamples {
        freqs_by_step,
        mirror_weight_by_step,
        agent_freqs_by_step,
        landscape_metrics_by_step,
    }
}

fn normalize_mirror_schedule(schedule: &[(u32, f32)], default_weight: f32) -> Vec<(u32, f32)> {
    let mut points: Vec<(u32, f32)> = schedule
        .iter()
        .map(|(step, w)| (*step, w.clamp(0.0, 1.0)))
        .collect();
    points.sort_by_key(|(step, _)| *step);
    points.dedup_by(|a, b| a.0 == b.0);
    if points.is_empty() {
        points.push((0, default_weight.clamp(0.0, 1.0)));
    } else if points[0].0 != 0 {
        points.insert(0, (0, points[0].1));
    }
    points
}

fn apply_mirror_weight(
    pop: &mut Population,
    landscape: &Landscape,
    params: &mut LandscapeParams,
    mirror_weight: f32,
) {
    let update = LandscapeUpdate {
        mirror: Some(mirror_weight.clamp(0.0, 1.0)),
        ..LandscapeUpdate::default()
    };
    pop.apply_action(Action::SetHarmonicityParams { update }, landscape, None);
    if let Some(update) = pop.take_pending_update() {
        apply_params_update(params, &update);
    }
}

#[allow(dead_code)]
fn collect_voice_freqs(pop: &Population) -> Vec<f32> {
    collect_voice_freqs_with_ids(pop)
        .into_iter()
        .map(|row| row.freq_hz)
        .collect()
}

fn collect_voice_freqs_with_ids(pop: &Population) -> Vec<E4AgentFreq> {
    let mut freqs = Vec::new();
    for agent in &pop.individuals {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let freq = agent.body.base_freq_hz();
        if freq.is_finite() && freq > 0.0 {
            freqs.push(E4AgentFreq {
                agent_id: agent.id,
                freq_hz: freq,
            });
        }
    }
    freqs.sort_by_key(|row| row.agent_id);
    freqs
}

fn sanitize_roughness_weight_scale(scale: f32) -> f32 {
    // Keep E4 wr-sweep robust against NaN/negative inputs.
    if scale.is_finite() {
        scale.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn sanitize_env_partials(partials: u32) -> u32 {
    partials.clamp(1, 32)
}

fn sanitize_env_partial_decay(decay: f32) -> f32 {
    if decay.is_finite() {
        decay.clamp(0.0, 4.0)
    } else {
        E4_ENV_PARTIAL_DECAY_DEFAULT
    }
}

fn sanitize_neighbor_step_cents(step: f32) -> f32 {
    if step.is_finite() { step.max(0.0) } else { 0.0 }
}

fn make_landscape_params(
    space: &Log2Space,
    fs: f32,
    _roughness_weight_scale: f32,
) -> LandscapeParams {
    let mut harmonicity_params = HarmonicityParams::default();
    harmonicity_params.rho_common_overtone = harmonicity_params.rho_common_root;
    harmonicity_params.mirror_weight = 0.5;
    LandscapeParams {
        fs,
        max_hist_cols: 1,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
        harmonicity_kernel: HarmonicityKernel::new(space, harmonicity_params),
        consonance_kernel: ConsonanceKernel::default(),
        consonance_representation: ConsonanceRepresentationParams::default(),
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
    }
}

fn apply_params_update(params: &mut LandscapeParams, upd: &LandscapeUpdate) {
    if let Some(m) = upd.mirror {
        params.harmonicity_kernel.params.mirror_weight = m;
    }
    if let Some(k) = upd.roughness_k {
        params.roughness_k = k.max(1e-6);
    }
}

fn add_harmonic_partials_to_env(
    space: &Log2Space,
    env_scan: &mut [f32],
    f0_hz: f32,
    base_gain: f32,
    partials: u32,
    partial_decay: f32,
) {
    if !f0_hz.is_finite() || f0_hz <= 0.0 || !base_gain.is_finite() || base_gain <= 0.0 {
        return;
    }
    let decay = sanitize_env_partial_decay(partial_decay);
    let partials = sanitize_env_partials(partials);
    for k in 1..=partials {
        let freq = f0_hz * k as f32;
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        let Some(idx) = space.index_of_freq(freq) else {
            continue;
        };
        let weight = base_gain / (k as f32).powf(decay);
        if weight.is_finite() && weight > 0.0 {
            env_scan[idx] += weight;
        }
    }
}

fn subtract_harmonic_partials_from_env(
    space: &Log2Space,
    env_scan: &mut [f32],
    f0_hz: f32,
    base_gain: f32,
    partials: u32,
    partial_decay: f32,
) {
    if !f0_hz.is_finite() || f0_hz <= 0.0 || !base_gain.is_finite() || base_gain <= 0.0 {
        return;
    }
    let decay = sanitize_env_partial_decay(partial_decay);
    let partials = sanitize_env_partials(partials);
    for k in 1..=partials {
        let freq = f0_hz * k as f32;
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        let Some(idx) = space.index_of_freq(freq) else {
            continue;
        };
        let weight = base_gain / (k as f32).powf(decay);
        if weight.is_finite() && weight > 0.0 {
            env_scan[idx] = (env_scan[idx] - weight).max(0.0);
        }
    }
}

fn build_anchor_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    anchor_hz: f32,
    env_partials: u32,
    env_partial_decay: f32,
) -> Landscape {
    let mut landscape = Landscape::new(space.clone());
    let mut anchor_env_scan = vec![0.0f32; space.n_bins()];
    add_harmonic_partials_to_env(
        space,
        &mut anchor_env_scan,
        anchor_hz,
        1.0,
        env_partials,
        env_partial_decay,
    );
    space.assert_scan_len_named(&anchor_env_scan, "anchor_env_scan");

    let h_dual = params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(&anchor_env_scan, space);
    space.assert_scan_len_named(&h_dual.blended, "perc_h_pot_scan");
    space.assert_scan_len_named(&h_dual.path_a, "perc_h_path_a_scan");
    space.assert_scan_len_named(&h_dual.path_b, "perc_h_path_b_scan");

    landscape.subjective_intensity = anchor_env_scan.clone();
    landscape.nsgt_power = anchor_env_scan;
    landscape.harmonicity = h_dual.blended;
    landscape.harmonicity_path_a = h_dual.path_a;
    landscape.harmonicity_path_b = h_dual.path_b;
    landscape.root_affinity = h_dual.metrics.root_affinity;
    landscape.overtone_affinity = h_dual.metrics.overtone_affinity;
    landscape.binding_strength = h_dual.metrics.binding_strength;
    landscape.harmonic_tilt = h_dual.metrics.harmonic_tilt;
    landscape.harmonicity_mirror_weight = params.harmonicity_kernel.params.mirror_weight;
    landscape.roughness.fill(0.0);
    landscape.roughness01.fill(0.0);
    landscape.recompute_consonance(params);
    landscape
}

fn build_density_landscape_from_freqs(
    space: &Log2Space,
    params: &LandscapeParams,
    freqs_hz: &[f32],
    env_partials: u32,
    env_partial_decay: f32,
) -> Landscape {
    let mut landscape = Landscape::new(space.clone());
    let mut env_scan = vec![0.0f32; space.n_bins()];
    for &freq_hz in freqs_hz {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            continue;
        }
        add_harmonic_partials_to_env(
            space,
            &mut env_scan,
            freq_hz,
            1.0,
            env_partials,
            env_partial_decay,
        );
    }
    space.assert_scan_len_named(&env_scan, "combined_env_scan");

    let h_dual = params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(&env_scan, space);
    landscape.subjective_intensity = env_scan.clone();
    landscape.nsgt_power = env_scan;
    landscape.harmonicity = h_dual.blended;
    landscape.harmonicity_path_a = h_dual.path_a;
    landscape.harmonicity_path_b = h_dual.path_b;
    landscape.root_affinity = h_dual.metrics.root_affinity;
    landscape.overtone_affinity = h_dual.metrics.overtone_affinity;
    landscape.binding_strength = h_dual.metrics.binding_strength;
    landscape.harmonic_tilt = h_dual.metrics.harmonic_tilt;
    landscape.harmonicity_mirror_weight = params.harmonicity_kernel.params.mirror_weight;
    // Compute roughness from multi-source spectrum
    compute_roughness_for_landscape(space, params, &mut landscape);
    landscape
}

fn build_density_landscape_from_env_scan(
    space: &Log2Space,
    params: &LandscapeParams,
    env_scan: Vec<f32>,
) -> Landscape {
    let mut landscape = Landscape::new(space.clone());
    space.assert_scan_len_named(&env_scan, "combined_env_scan");

    let h_dual = params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(&env_scan, space);
    landscape.subjective_intensity = env_scan.clone();
    landscape.nsgt_power = env_scan;
    landscape.harmonicity = h_dual.blended;
    landscape.harmonicity_path_a = h_dual.path_a;
    landscape.harmonicity_path_b = h_dual.path_b;
    landscape.root_affinity = h_dual.metrics.root_affinity;
    landscape.overtone_affinity = h_dual.metrics.overtone_affinity;
    landscape.binding_strength = h_dual.metrics.binding_strength;
    landscape.harmonic_tilt = h_dual.metrics.harmonic_tilt;
    landscape.harmonicity_mirror_weight = params.harmonicity_kernel.params.mirror_weight;
    compute_roughness_for_landscape(space, params, &mut landscape);
    landscape
}

fn build_e6_contextual_loo_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    contextual_landscape: &Landscape,
    self_freq_hz: f32,
    env_partials: u32,
    env_partial_decay: f32,
) -> Landscape {
    if !self_freq_hz.is_finite() || self_freq_hz <= 0.0 {
        return contextual_landscape.clone();
    }
    let mut env_scan = contextual_landscape.subjective_intensity.clone();
    subtract_harmonic_partials_from_env(
        space,
        &mut env_scan,
        self_freq_hz,
        1.0,
        env_partials,
        env_partial_decay,
    );
    build_density_landscape_from_env_scan(space, params, env_scan)
}

fn compute_roughness_for_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    landscape: &mut Landscape,
) {
    let (_erb, du) = erb_grid(space);
    let (density, _mass) = conchordal::core::psycho_state::normalize_density(
        &landscape.subjective_intensity,
        &du,
        1e-12,
    );
    let (r_pot, _r_total) = params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&density, space);
    let r_ref = compute_roughness_reference(params, space);
    landscape.roughness = r_pot;
    let mut r01 = vec![0.0f32; space.n_bins()];
    r_pot_scan_to_r_state01_scan(
        &landscape.roughness,
        r_ref.peak,
        params.roughness_k,
        &mut r01,
    );
    landscape.roughness01 = r01;
    landscape.recompute_consonance(params);
}

fn build_e6_parent_harmonicity_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    parent_freq_hz: f32,
    env_partials: u32,
    env_partial_decay: f32,
) -> Landscape {
    build_density_landscape_from_freqs(
        space,
        params,
        &[parent_freq_hz],
        env_partials,
        env_partial_decay,
    )
}

const E6_HEREDITY_PARENT_WINDOW_SIGMA_OCT: f32 = 0.35;
const E6_HEREDITY_SOCIAL_ROUGHNESS_WEIGHT: f32 = 0.0;

fn parent_window_and_notch_weights(
    space: &Log2Space,
    parent_freq_hz: f32,
    crowding_sigma_cents: f32,
) -> Vec<f32> {
    let parent_hz = parent_freq_hz.max(1e-6);
    let parent_log2 = parent_hz.log2();
    let parent_erb = hz_to_erb(parent_hz);
    let window_sigma_oct = E6_HEREDITY_PARENT_WINDOW_SIGMA_OCT.max(1e-6);
    let notch_sigma_erb = crowding_sigma_erb_from_hz(parent_hz, crowding_sigma_cents).max(1e-6);
    space
        .centers_hz
        .iter()
        .zip(space.centers_log2.iter())
        .map(|(&candidate_hz, &candidate_log2)| {
            let delta_oct = candidate_log2 - parent_log2;
            let window = (-0.5 * (delta_oct / window_sigma_oct).powi(2)).exp();
            let delta_erb = hz_to_erb(candidate_hz.max(1e-6)) - parent_erb;
            let notch = 1.0 - (-0.5 * (delta_erb / notch_sigma_erb).powi(2)).exp();
            window * notch.max(0.0)
        })
        .collect()
}

fn weak_social_roughness_weight(current_landscape: &Landscape, idx: usize) -> f32 {
    let roughness01 = current_landscape
        .roughness01
        .get(idx)
        .copied()
        .unwrap_or(0.0);
    (1.0 - E6_HEREDITY_SOCIAL_ROUGHNESS_WEIGHT * roughness01.clamp(0.0, 1.0)).max(0.0)
}

fn family_occupancy_score(center_freq_hz: f32, alive_freqs_hz: &[f32]) -> f32 {
    if !center_freq_hz.is_finite() || center_freq_hz <= 0.0 || alive_freqs_hz.is_empty() {
        return 0.0;
    }
    let center_log2 = center_freq_hz.max(1e-6).log2();
    let sigma_oct = (E6_HEREDITY_FAMILY_OCCUPANCY_SIGMA_ST / 12.0).max(1e-6);
    alive_freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .map(|freq_hz| {
            let dz = (freq_hz.log2() - center_log2) / sigma_oct;
            (-0.5 * dz * dz).exp()
        })
        .sum::<f32>()
}

fn family_occupancy_weight(
    center_freq_hz: f32,
    alive_freqs_hz: &[f32],
    occupancy_strength: f32,
) -> f32 {
    let occupancy = family_occupancy_score(center_freq_hz, alive_freqs_hz).max(0.0);
    (1.0 / (1.0 + occupancy_strength.max(0.0) * occupancy)).clamp(0.1, 1.0)
}

fn family_contextual_weight(center_freq_hz: f32, current_landscape: &Landscape) -> f32 {
    if !center_freq_hz.is_finite() || center_freq_hz <= 0.0 {
        return 1.0;
    }
    let contextual_score = current_landscape.evaluate_pitch_score(center_freq_hz);
    (E6_HEREDITY_FAMILY_CONTEXTUAL_BETA * contextual_score)
        .exp()
        .clamp(0.1, 4.0)
}

fn family_choice_weight(
    center_freq_hz: f32,
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    occupancy_strength: f32,
    nfd_mode: E6FamilyNfdMode,
    apply_nfd: bool,
) -> f32 {
    let occupancy_weight = match nfd_mode {
        E6FamilyNfdMode::Off => 1.0,
        E6FamilyNfdMode::SameFamilyOnly => {
            if apply_nfd {
                family_occupancy_weight(center_freq_hz, alive_freqs_hz, occupancy_strength)
            } else {
                1.0
            }
        }
        E6FamilyNfdMode::SameFamilyAndAlt => {
            family_occupancy_weight(center_freq_hz, alive_freqs_hz, occupancy_strength)
        }
    };
    family_contextual_weight(center_freq_hz, current_landscape) * occupancy_weight
}

fn sample_from_weight_scan(
    space: &Log2Space,
    weights: &[f32],
    rng: &mut impl Rng,
    fallback_hz: f32,
) -> f32 {
    if weights.len() != space.n_bins() || weights.is_empty() {
        return fallback_hz;
    }
    let total: f32 = weights.iter().copied().filter(|w| w.is_finite()).sum();
    if total <= 0.0 {
        return fallback_hz;
    }
    let target = rng.random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (idx, &weight) in weights.iter().enumerate() {
        let w = weight.max(0.0);
        cumulative += w;
        if target <= cumulative {
            return space.centers_hz[idx];
        }
    }
    space.centers_hz[space.n_bins() - 1]
}

fn sample_random_log_freq(rng: &mut impl Rng, min_hz: f32, max_hz: f32) -> f32 {
    let lo = min_hz.min(max_hz).max(1e-6);
    let hi = max_hz.max(min_hz).max(lo);
    if (hi - lo).abs() <= f32::EPSILON {
        return lo;
    }
    let lo_log2 = lo.log2();
    let hi_log2 = hi.log2();
    2.0f32.powf(rng.random_range(lo_log2..=hi_log2))
}

fn tessitura_masked_weights(
    space: &Log2Space,
    weights: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
) -> Vec<f32> {
    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    space
        .centers_hz
        .iter()
        .zip(weights.iter())
        .map(|(&freq_hz, &weight)| {
            if freq_hz >= min_hz && freq_hz <= max_hz {
                weight
            } else {
                0.0
            }
        })
        .collect()
}

#[derive(Clone, Copy, Debug)]
struct E6PeakFamily {
    center_idx: usize,
    mass: f32,
}

#[derive(Clone, Copy, Debug)]
struct E6HereditySample {
    offspring_freq_hz: f32,
    parent_family_center_hz: f32,
    parent_azimuth_st: f32,
    child_family_center_hz: f32,
    child_azimuth_st: f32,
    family_inherited: bool,
    family_mutated: bool,
}

#[derive(Clone, Copy, Debug)]
struct E6VacantNiche {
    center_idx: usize,
    center_freq_hz: f32,
    base_score: f32,
    parent_prior: f32,
}

fn build_e6_scene_landscape_with_anchor(
    space: &Log2Space,
    params: &LandscapeParams,
    selection_reference_landscape: &Landscape,
    freqs_hz: &[f32],
    env_partials: u32,
    env_partial_decay: f32,
) -> Landscape {
    let mut env_scan = selection_reference_landscape.subjective_intensity.clone();
    for &freq_hz in freqs_hz {
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            continue;
        }
        add_harmonic_partials_to_env(
            space,
            &mut env_scan,
            freq_hz,
            1.0,
            env_partials,
            env_partial_decay,
        );
    }
    build_density_landscape_from_env_scan(space, params, env_scan)
}

fn e6_runtime_crowding(freq_hz: f32, other_freqs_hz: &[f32]) -> f32 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    let kernel_params = KernelParams::default();
    let candidate_erb = hz_to_erb(freq_hz.max(1e-6));
    other_freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .map(|other_freq_hz| {
            let other_erb = hz_to_erb(other_freq_hz.max(1e-6));
            crowding_runtime_delta_erb(&kernel_params, candidate_erb - other_erb)
        })
        .sum()
}

fn e6_respects_hard_anti_fusion(freq_hz: f32, occupied_freqs_hz: &[f32]) -> bool {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return false;
    }
    let kernel_params = KernelParams::default();
    let candidate_erb = hz_to_erb(freq_hz.max(1e-6));
    occupied_freqs_hz.iter().copied().all(|other_freq_hz| {
        if !other_freq_hz.is_finite() || other_freq_hz <= 0.0 {
            return true;
        }
        let other_erb = hz_to_erb(other_freq_hz.max(1e-6));
        crowding_runtime_delta_erb(&kernel_params, candidate_erb - other_erb)
            <= E6_HARD_ANTI_FUSION_EPS
    })
}

fn e6_respects_min_spacing_cents(
    freq_hz: f32,
    occupied_freqs_hz: &[f32],
    min_spacing_cents: f32,
) -> bool {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return false;
    }
    let min_spacing_cents = min_spacing_cents.max(0.0);
    occupied_freqs_hz.iter().copied().all(|other_freq_hz| {
        if !other_freq_hz.is_finite() || other_freq_hz <= 0.0 {
            return true;
        }
        (1200.0 * (freq_hz / other_freq_hz).log2().abs()) + 1e-6 >= min_spacing_cents
    })
}

fn e6_runtime_overcapacity(
    freq_hz: f32,
    other_freqs_hz: &[f32],
    radius_cents: f32,
    free_voices: usize,
) -> f32 {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0.0;
    }
    let radius_cents = radius_cents.max(0.0);
    let local_occupancy = 1 + other_freqs_hz
        .iter()
        .copied()
        .filter(|other_freq_hz| other_freq_hz.is_finite() && *other_freq_hz > 0.0)
        .filter(|&other_freq_hz| {
            1200.0 * (freq_hz / other_freq_hz).log2().abs() <= radius_cents + 1e-6
        })
        .count();
    local_occupancy.saturating_sub(free_voices.max(1)) as f32
}

fn e6_polyphonic_selection_score(
    loo_scene_landscape: &Landscape,
    freq_hz: f32,
    other_freqs_hz: &[f32],
    crowding_weight: f32,
    overcapacity_weight: f32,
) -> f32 {
    loo_scene_landscape.evaluate_pitch_score(freq_hz)
        - crowding_weight.max(0.0) * e6_runtime_crowding(freq_hz, other_freqs_hz)
        - overcapacity_weight.max(0.0)
            * e6_runtime_overcapacity(
                freq_hz,
                other_freqs_hz,
                E6B_LOCAL_CAPACITY_RADIUS_CENTS,
                E6B_LOCAL_CAPACITY_FREE_VOICES,
            )
}

fn e6_extract_vacant_niches(
    space: &Log2Space,
    scene_landscape: &Landscape,
    occupied_freqs_hz: &[f32],
    parent_prior_landscape: Option<&Landscape>,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    crowding_weight: f32,
    overcapacity_weight: f32,
) -> Vec<E6VacantNiche> {
    if space.n_bins() == 0 {
        return Vec::new();
    }

    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let mut score_scan = vec![f32::NEG_INFINITY; space.n_bins()];
    for (idx, &freq_hz) in space.centers_hz.iter().enumerate() {
        if freq_hz < min_hz || freq_hz > max_hz {
            continue;
        }
        if !e6_respects_hard_anti_fusion(freq_hz, occupied_freqs_hz) {
            continue;
        }
        score_scan[idx] = e6_polyphonic_selection_score(
            scene_landscape,
            freq_hz,
            occupied_freqs_hz,
            crowding_weight,
            overcapacity_weight,
        );
    }

    let best_idx = score_scan
        .iter()
        .enumerate()
        .filter(|(_, score)| score.is_finite())
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx);
    let Some(best_idx) = best_idx else {
        return Vec::new();
    };

    let mut peak_indices: Vec<usize> = Vec::new();
    for idx in 0..score_scan.len() {
        let here = score_scan[idx];
        if !here.is_finite() {
            continue;
        }
        let prev = if idx == 0 {
            f32::NEG_INFINITY
        } else {
            score_scan[idx - 1]
        };
        let next = if idx + 1 >= score_scan.len() {
            f32::NEG_INFINITY
        } else {
            score_scan[idx + 1]
        };
        let prev_finite = if prev.is_finite() {
            prev
        } else {
            f32::NEG_INFINITY
        };
        let next_finite = if next.is_finite() {
            next
        } else {
            f32::NEG_INFINITY
        };
        if here >= prev_finite && here > next_finite {
            peak_indices.push(idx);
        }
    }
    if peak_indices.is_empty() {
        peak_indices.push(best_idx);
    }

    peak_indices
        .into_iter()
        .map(|idx| {
            let freq_hz = space.centers_hz[idx];
            let parent_prior = parent_prior_landscape
                .map(|landscape| landscape.evaluate_pitch_score(freq_hz))
                .unwrap_or(1.0)
                .max(E6_NICHE_PARENT_PRIOR_FLOOR);
            E6VacantNiche {
                center_idx: idx,
                center_freq_hz: freq_hz,
                base_score: score_scan[idx],
                parent_prior,
            }
        })
        .collect()
}

fn sample_from_vacant_niches(
    niches: &[E6VacantNiche],
    parent_freq_hz: f32,
    heredity_enabled: bool,
    rng: &mut impl Rng,
) -> E6HereditySample {
    if niches.is_empty() {
        return E6HereditySample {
            offspring_freq_hz: parent_freq_hz,
            parent_family_center_hz: parent_freq_hz,
            parent_azimuth_st: 0.0,
            child_family_center_hz: parent_freq_hz,
            child_azimuth_st: 0.0,
            family_inherited: false,
            family_mutated: false,
        };
    }

    let preferred_idx = niches
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            a.parent_prior
                .partial_cmp(&b.parent_prior)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    a.base_score
                        .partial_cmp(&b.base_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        })
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let chosen_idx = if heredity_enabled {
        let weights: Vec<f32> = niches
            .iter()
            .map(|niche| {
                niche
                    .parent_prior
                    .powf(E6_NICHE_PARENT_PRIOR_EXPONENT)
                    .max(1e-6)
            })
            .collect();
        if let Ok(dist) = WeightedIndex::new(&weights) {
            dist.sample(rng)
        } else {
            preferred_idx
        }
    } else {
        rng.random_range(0..niches.len())
    };
    let preferred = niches[preferred_idx];
    let chosen = niches[chosen_idx];
    let family_inherited = heredity_enabled && chosen.center_idx == preferred.center_idx;
    E6HereditySample {
        offspring_freq_hz: chosen.center_freq_hz,
        parent_family_center_hz: preferred.center_freq_hz,
        parent_azimuth_st: 0.0,
        child_family_center_hz: chosen.center_freq_hz,
        child_azimuth_st: 0.0,
        family_inherited,
        family_mutated: heredity_enabled && !family_inherited,
    }
}

fn semitone_distance(space: &Log2Space, idx_a: usize, idx_b: usize) -> f32 {
    12.0 * (space.centers_log2[idx_a] - space.centers_log2[idx_b]).abs()
}

fn local_peak_mass(weights: &[f32], idx: usize) -> f32 {
    let start = idx.saturating_sub(1);
    let end = (idx + 1).min(weights.len().saturating_sub(1));
    weights[start..=end]
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum()
}

fn extract_peak_families(space: &Log2Space, weights: &[f32]) -> Vec<E6PeakFamily> {
    if weights.is_empty() || weights.len() != space.n_bins() {
        return Vec::new();
    }

    let mut peaks = Vec::new();
    for idx in 0..weights.len() {
        let here = weights[idx].max(0.0);
        if here <= 0.0 || !here.is_finite() {
            continue;
        }
        let prev = if idx == 0 {
            f32::NEG_INFINITY
        } else {
            weights[idx - 1].max(0.0)
        };
        let next = if idx + 1 >= weights.len() {
            f32::NEG_INFINITY
        } else {
            weights[idx + 1].max(0.0)
        };
        if here >= prev && here > next {
            peaks.push(E6PeakFamily {
                center_idx: idx,
                mass: local_peak_mass(weights, idx),
            });
        }
    }
    if peaks.is_empty() {
        return peaks;
    }

    peaks.sort_by_key(|peak| peak.center_idx);
    let mut merged: Vec<E6PeakFamily> = Vec::new();
    for peak in peaks {
        if let Some(last) = merged.last_mut() {
            if semitone_distance(space, last.center_idx, peak.center_idx)
                <= E6_HEREDITY_PEAK_MERGE_ST
            {
                if peak.mass > last.mass {
                    last.center_idx = peak.center_idx;
                }
                last.mass += peak.mass;
                continue;
            }
        }
        merged.push(peak);
    }

    let max_mass = merged.iter().map(|peak| peak.mass).fold(0.0f32, f32::max);
    if max_mass <= 0.0 || !max_mass.is_finite() {
        return Vec::new();
    }
    merged
        .into_iter()
        .filter(|peak| peak.mass >= E6_HEREDITY_PEAK_MIN_RELATIVE_MASS * max_mass)
        .collect()
}

fn nearest_peak_family_idx(
    space: &Log2Space,
    peaks: &[E6PeakFamily],
    target_freq_hz: f32,
) -> Option<usize> {
    if peaks.is_empty() || !target_freq_hz.is_finite() || target_freq_hz <= 0.0 {
        return None;
    }
    let target_log2 = target_freq_hz.log2();
    peaks
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            let da = (space.centers_log2[a.center_idx] - target_log2).abs();
            let db = (space.centers_log2[b.center_idx] - target_log2).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

fn sample_mutated_peak_family_idx(
    peaks: &[E6PeakFamily],
    excluded_idx: usize,
    rng: &mut impl Rng,
) -> Option<usize> {
    if peaks.len() <= 1 {
        return None;
    }
    let candidate_indices: Vec<usize> = peaks
        .iter()
        .enumerate()
        .filter_map(|(idx, _)| (idx != excluded_idx).then_some(idx))
        .collect();
    let dist = WeightedIndex::new(
        candidate_indices
            .iter()
            .map(|&idx| peaks[idx].mass.max(0.0)),
    )
    .ok()?;
    Some(candidate_indices[dist.sample(rng)])
}

fn freq_from_family_and_azimuth(center_freq_hz: f32, azimuth_st: f32) -> f32 {
    center_freq_hz.max(1e-6) * 2.0f32.powf(azimuth_st / 12.0)
}

fn e6_find_spaced_family_frequency(
    center_freq_hz: f32,
    preferred_azimuth_st: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    occupied_freqs_hz: &[f32],
    min_spacing_cents: f32,
) -> Option<f32> {
    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let grid_st = 0.05f32;
    let radius_steps = (E6_HEREDITY_AZIMUTH_CLIP_ST / grid_st).ceil() as i32;
    for offset_steps in 0..=radius_steps {
        for signed_steps in [offset_steps, -offset_steps] {
            if offset_steps == 0 && signed_steps < 0 {
                continue;
            }
            let azimuth_st = (preferred_azimuth_st + signed_steps as f32 * grid_st)
                .clamp(-E6_HEREDITY_AZIMUTH_CLIP_ST, E6_HEREDITY_AZIMUTH_CLIP_ST);
            let freq_hz = freq_from_family_and_azimuth(center_freq_hz, azimuth_st);
            if !freq_hz.is_finite() || freq_hz < min_hz || freq_hz > max_hz {
                continue;
            }
            if e6_respects_min_spacing_cents(freq_hz, occupied_freqs_hz, min_spacing_cents) {
                return Some(freq_hz);
            }
        }
    }
    None
}

fn oracle_azimuth_search(
    reference_landscape: &Landscape,
    center_freq_hz: f32,
    radius_st: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
) -> (f32, f32, f32) {
    if !center_freq_hz.is_finite() || center_freq_hz <= 0.0 {
        return (center_freq_hz, 0.0, 0.0);
    }

    let radius_oct = (radius_st.abs() / 12.0).max(0.0);
    let min_log2 = tessitura_min_hz.max(1e-6).log2();
    let max_log2 = tessitura_max_hz.max(tessitura_min_hz.max(1e-6)).log2();
    let center_log2 = center_freq_hz.max(1e-6).log2();
    let search_lo = (center_log2 - radius_oct).max(min_log2);
    let search_hi = (center_log2 + radius_oct).min(max_log2);

    let mut best_freq = center_freq_hz;
    let mut best_score = reference_landscape.evaluate_pitch_score(center_freq_hz);
    let mut best_level = reference_landscape
        .evaluate_pitch_level(center_freq_hz)
        .clamp(0.0, 1.0);
    let mut best_abs_delta = 0.0f32;

    for (&candidate_hz, &candidate_log2) in reference_landscape
        .space
        .centers_hz
        .iter()
        .zip(reference_landscape.space.centers_log2.iter())
    {
        if candidate_log2 < search_lo || candidate_log2 > search_hi {
            continue;
        }
        let candidate_score = reference_landscape.evaluate_pitch_score(candidate_hz);
        let candidate_level = reference_landscape
            .evaluate_pitch_level(candidate_hz)
            .clamp(0.0, 1.0);
        let candidate_abs_delta = (candidate_log2 - center_log2).abs();
        if candidate_score > best_score + 1e-6
            || ((candidate_score - best_score).abs() <= 1e-6
                && candidate_abs_delta < best_abs_delta)
        {
            best_freq = candidate_hz;
            best_score = candidate_score;
            best_level = candidate_level;
            best_abs_delta = candidate_abs_delta;
        }
    }

    (best_freq, best_score, best_level)
}

fn collect_alive_freqs(pop: &Population) -> Vec<f32> {
    pop.individuals
        .iter()
        .filter(|agent| agent.is_alive())
        .filter_map(|agent| {
            let freq_hz = agent.body.base_freq_hz();
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                return None;
            }
            Some(freq_hz)
        })
        .collect()
}

fn e6_push_snapshot(
    out: &mut E6RunResult,
    pop: &Population,
    states: &[E3LifeState],
    step: usize,
    theta_phase: f32,
) {
    let mut freqs_hz: Vec<f32> = Vec::new();
    let mut agents: Vec<E6AgentSnapshot> = Vec::new();
    let mut phase_sin_sum = 0.0f32;
    let mut phase_cos_sum = 0.0f32;
    let mut phase_count = 0u32;
    for agent in pop.individuals.iter() {
        if !agent.is_alive() {
            continue;
        }
        let idx = agent.id() as usize;
        if idx >= states.len() {
            continue;
        }
        let f = agent.body.base_freq_hz();
        if !f.is_finite() || f <= 0.0 {
            continue;
        }
        freqs_hz.push(f);
        agents.push(E6AgentSnapshot {
            life_id: states[idx].life_id,
            agent_id: idx,
            freq_hz: f,
        });
        if let AnyArticulationCore::Entrain(ref k) = agent.articulation.core {
            let phase_err = k.rhythm_phase - theta_phase;
            phase_sin_sum += phase_err.sin();
            phase_cos_sum += phase_err.cos();
            phase_count += 1;
        }
    }
    agents.sort_by_key(|agent| agent.life_id);
    let phase_coherence = if phase_count > 0 {
        let n = phase_count as f32;
        ((phase_sin_sum / n).powi(2) + (phase_cos_sum / n).powi(2)).sqrt()
    } else {
        0.0
    };
    out.snapshots.push(E6PitchSnapshot {
        step,
        freqs_hz,
        agents,
        phase_coherence,
    });
}

fn e2_aligned_azimuth_search(
    reference_landscape: &Landscape,
    center_freq_hz: f32,
    radius_st: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    neighbor_erbs: &[f32],
) -> (f32, f32, f32) {
    if !center_freq_hz.is_finite() || center_freq_hz <= 0.0 {
        return (center_freq_hz, 0.0, 0.0);
    }

    let current_pitch_log2 = center_freq_hz.max(1e-6).log2();
    let current_erb = hz_to_erb(center_freq_hz.max(1e-6));
    let kernel_params = KernelParams::default();
    let mut current_crowding = 0.0f32;
    for &other_erb in neighbor_erbs {
        current_crowding += crowding_runtime_delta_erb(&kernel_params, current_erb - other_erb);
    }

    let mut best_freq = center_freq_hz;
    let mut best_raw_score = reference_landscape.evaluate_pitch_score(center_freq_hz);
    let mut best_level = reference_landscape
        .evaluate_pitch_level(center_freq_hz)
        .clamp(0.0, 1.0);
    let mut best_objective = best_raw_score - E2_MATCH_CROWDING_WEIGHT * current_crowding;
    let mut best_abs_delta = 0.0f32;

    let radius_steps = (radius_st.max(0.0) / 0.25).round() as i32;
    let step_log2 = 0.25f32 / 12.0;
    for offset_steps in -radius_steps..=radius_steps {
        let candidate_pitch_log2 = current_pitch_log2 + offset_steps as f32 * step_log2;
        let candidate_freq_hz = 2.0f32
            .powf(candidate_pitch_log2)
            .clamp(tessitura_min_hz, tessitura_max_hz);
        if !candidate_freq_hz.is_finite() || candidate_freq_hz <= 0.0 {
            continue;
        }

        let candidate_erb = hz_to_erb(candidate_freq_hz.max(1e-6));
        let mut candidate_crowding = 0.0f32;
        for &other_erb in neighbor_erbs {
            candidate_crowding +=
                crowding_runtime_delta_erb(&kernel_params, candidate_erb - other_erb);
        }

        let candidate_raw_score = reference_landscape.evaluate_pitch_score(candidate_freq_hz);
        let candidate_level = reference_landscape
            .evaluate_pitch_level(candidate_freq_hz)
            .clamp(0.0, 1.0);
        let candidate_abs_delta = (candidate_pitch_log2 - current_pitch_log2).abs();
        let candidate_objective = candidate_raw_score
            - E2_MATCH_CROWDING_WEIGHT * candidate_crowding
            - shared_hill_move_cost_coeff() * candidate_abs_delta;

        if candidate_objective > best_objective + 1e-6
            || ((candidate_objective - best_objective).abs() <= 1e-6
                && candidate_abs_delta < best_abs_delta)
        {
            best_freq = candidate_freq_hz;
            best_raw_score = candidate_raw_score;
            best_level = candidate_level;
            best_objective = candidate_objective;
            best_abs_delta = candidate_abs_delta;
        }
    }

    (best_freq, best_raw_score, best_level)
}

fn oracle_global_anchor_search(
    reference_landscape: &Landscape,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
) -> (f32, f32, f32) {
    let min_log2 = tessitura_min_hz.max(1e-6).log2();
    let max_log2 = tessitura_max_hz.max(tessitura_min_hz.max(1e-6)).log2();

    let mut best_freq = tessitura_min_hz.max(1e-6);
    let mut best_score = f32::NEG_INFINITY;
    let mut best_level = 0.0f32;

    for (&candidate_hz, &candidate_log2) in reference_landscape
        .space
        .centers_hz
        .iter()
        .zip(reference_landscape.space.centers_log2.iter())
    {
        if candidate_log2 < min_log2 || candidate_log2 > max_log2 {
            continue;
        }
        let candidate_score = reference_landscape.evaluate_pitch_score(candidate_hz);
        let candidate_level = reference_landscape
            .evaluate_pitch_level(candidate_hz)
            .clamp(0.0, 1.0);
        if candidate_score > best_score {
            best_freq = candidate_hz;
            best_score = candidate_score;
            best_level = candidate_level;
        }
    }

    if !best_score.is_finite() {
        let fallback_score = reference_landscape.evaluate_pitch_score(best_freq);
        let fallback_level = reference_landscape
            .evaluate_pitch_level(best_freq)
            .clamp(0.0, 1.0);
        return (best_freq, fallback_score, fallback_level);
    }

    (best_freq, best_score, best_level)
}

/// Sample from the parent's harmonicity profile by separating coarse interval-family
/// inheritance from fine azimuth inheritance. The family is drawn from the parent's
/// local H-weighted PMF, while the fine offset is inherited from the parent with only
/// a small mutation.
fn sample_from_parent_harmonicity(
    parent_landscape: &Landscape,
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    parent_freq_hz: f32,
    crowding_sigma_cents: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    rng: &mut impl Rng,
    family_occupancy_strength: f32,
    family_nfd_mode: E6FamilyNfdMode,
    min_spacing_cents: Option<f32>,
) -> E6HereditySample {
    let space = &parent_landscape.space;
    let n = space.n_bins();
    if n == 0 || parent_landscape.harmonicity01.len() != n {
        return E6HereditySample {
            offspring_freq_hz: parent_freq_hz,
            parent_family_center_hz: parent_freq_hz,
            parent_azimuth_st: 0.0,
            child_family_center_hz: parent_freq_hz,
            child_azimuth_st: 0.0,
            family_inherited: false,
            family_mutated: false,
        };
    }

    let local_weights =
        parent_window_and_notch_weights(space, parent_freq_hz, crowding_sigma_cents);
    let mut weights = Vec::with_capacity(n);
    for (idx, &local_weight) in local_weights.iter().enumerate() {
        let h = parent_landscape.harmonicity01[idx].max(0.0);
        let social_weight = weak_social_roughness_weight(current_landscape, idx);
        weights.push(h * local_weight * social_weight);
    }
    let mut fallback_weights = Vec::with_capacity(n);
    for (idx, &local_weight) in local_weights.iter().enumerate() {
        let social_weight = weak_social_roughness_weight(current_landscape, idx);
        fallback_weights.push(local_weight * social_weight);
    }
    let truncated_weights =
        tessitura_masked_weights(space, &weights, tessitura_min_hz, tessitura_max_hz);
    let truncated_fallback_weights =
        tessitura_masked_weights(space, &fallback_weights, tessitura_min_hz, tessitura_max_hz);
    let fallback = sample_from_weight_scan(space, &truncated_fallback_weights, rng, parent_freq_hz);
    let peaks = extract_peak_families(space, &truncated_weights);
    let Some(parent_peak_idx) = nearest_peak_family_idx(space, &peaks, parent_freq_hz) else {
        let offspring_freq_hz = sample_from_weight_scan(space, &truncated_weights, rng, fallback);
        return E6HereditySample {
            offspring_freq_hz,
            parent_family_center_hz: parent_freq_hz,
            parent_azimuth_st: 0.0,
            child_family_center_hz: offspring_freq_hz,
            child_azimuth_st: 0.0,
            family_inherited: false,
            family_mutated: false,
        };
    };

    let parent_family_center_hz = space.centers_hz[peaks[parent_peak_idx].center_idx];
    let parent_azimuth_st = (12.0 * (parent_freq_hz / parent_family_center_hz.max(1e-6)).log2())
        .clamp(-E6_HEREDITY_AZIMUTH_CLIP_ST, E6_HEREDITY_AZIMUTH_CLIP_ST);

    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);

    for _ in 0..E6_HEREDITY_MUTATION_MAX_ATTEMPTS {
        let parent_family_weight = family_choice_weight(
            parent_family_center_hz,
            current_landscape,
            alive_freqs_hz,
            family_occupancy_strength,
            family_nfd_mode,
            true,
        );
        let inherit_same_family = peaks.len() == 1
            || rng.random::<f32>()
                < (E6_HEREDITY_INHERIT_SAME_FAMILY_PROB * parent_family_weight).clamp(0.0, 1.0);
        let child_peak_idx = if inherit_same_family {
            parent_peak_idx
        } else {
            let candidate_indices: Vec<usize> = (0..peaks.len())
                .filter(|&idx| idx != parent_peak_idx)
                .collect();
            let candidate_weights: Vec<f32> = candidate_indices
                .iter()
                .map(|&idx| {
                    let center_hz = space.centers_hz[peaks[idx].center_idx];
                    let family_weight = family_choice_weight(
                        center_hz,
                        current_landscape,
                        alive_freqs_hz,
                        family_occupancy_strength,
                        family_nfd_mode,
                        false,
                    );
                    (peaks[idx].mass.max(1e-6) * family_weight).max(0.0)
                })
                .collect();
            if let Ok(dist) = WeightedIndex::new(&candidate_weights) {
                let local_idx = dist.sample(rng);
                candidate_indices[local_idx]
            } else {
                sample_mutated_peak_family_idx(&peaks, parent_peak_idx, rng)
                    .unwrap_or(parent_peak_idx)
            }
        };
        let family_inherited = child_peak_idx == parent_peak_idx;
        let child_family_center_hz = space.centers_hz[peaks[child_peak_idx].center_idx];
        let child_azimuth_st = (parent_azimuth_st
            + sample_standard_normal(rng) * E6_HEREDITY_AZIMUTH_SIGMA_ST)
            .clamp(-E6_HEREDITY_AZIMUTH_CLIP_ST, E6_HEREDITY_AZIMUTH_CLIP_ST);
        let offspring_freq_hz = if let Some(min_spacing) = min_spacing_cents {
            match e6_find_spaced_family_frequency(
                child_family_center_hz,
                child_azimuth_st,
                min_hz,
                max_hz,
                alive_freqs_hz,
                min_spacing,
            ) {
                Some(freq_hz) => freq_hz,
                None => continue,
            }
        } else {
            freq_from_family_and_azimuth(child_family_center_hz, child_azimuth_st)
        };
        if offspring_freq_hz.is_finite()
            && offspring_freq_hz >= min_hz
            && offspring_freq_hz <= max_hz
        {
            return E6HereditySample {
                offspring_freq_hz,
                parent_family_center_hz,
                parent_azimuth_st,
                child_family_center_hz,
                child_azimuth_st,
                family_inherited,
                family_mutated: !family_inherited,
            };
        }
    }

    let offspring_freq_hz = sample_from_weight_scan(space, &truncated_weights, rng, fallback);
    let child_peak_idx =
        nearest_peak_family_idx(space, &peaks, offspring_freq_hz).unwrap_or(parent_peak_idx);
    let child_family_center_hz = space.centers_hz[peaks[child_peak_idx].center_idx];
    let child_azimuth_st = (12.0 * (offspring_freq_hz / child_family_center_hz.max(1e-6)).log2())
        .clamp(-E6_HEREDITY_AZIMUTH_CLIP_ST, E6_HEREDITY_AZIMUTH_CLIP_ST);
    let family_inherited = child_peak_idx == parent_peak_idx;
    let offspring_freq_hz = min_spacing_cents
        .and_then(|min_spacing| {
            e6_find_spaced_family_frequency(
                child_family_center_hz,
                child_azimuth_st,
                min_hz,
                max_hz,
                alive_freqs_hz,
                min_spacing,
            )
        })
        .unwrap_or(offspring_freq_hz);
    E6HereditySample {
        offspring_freq_hz,
        parent_family_center_hz,
        parent_azimuth_st,
        child_family_center_hz,
        child_azimuth_st,
        family_inherited,
        family_mutated: !family_inherited,
    }
}

fn sample_standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f32 {
    let u1 = rng.random_range(f32::EPSILON..1.0);
    let u2 = rng.random_range(0.0..1.0);
    let mag = (-2.0 * u1.ln()).sqrt();
    let angle = core::f32::consts::TAU * u2;
    mag * angle.cos()
}

pub(crate) fn e3_tessitura_bounds_for_range(
    anchor_hz: f32,
    space: &Log2Space,
    range_oct: f32,
) -> (f32, f32) {
    let half = 0.5 * range_oct.max(1e-6);
    let min_freq = anchor_hz * 2.0f32.powf(-half);
    let max_freq = anchor_hz * 2.0f32.powf(half);
    let min_freq = min_freq.clamp(space.fmin, space.fmax);
    let max_freq = max_freq.clamp(space.fmin, space.fmax);
    (min_freq.min(max_freq), max_freq.max(min_freq))
}

fn e3_spawn_strategy_with_range(
    anchor_hz: f32,
    space: &Log2Space,
    range_oct: f32,
) -> SpawnStrategy {
    let (min_freq, max_freq) = e3_tessitura_bounds_for_range(anchor_hz, space, range_oct);
    SpawnStrategy::RandomLog {
        min_freq: min_freq.min(max_freq),
        max_freq: max_freq.max(min_freq),
    }
}

fn e6_apply_exact_e2_aligned_local_search(
    pop: &mut Population,
    selection_reference_landscape: &Landscape,
    step: usize,
    radius_st: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
) {
    let mut alive_ids: Vec<usize> = pop
        .individuals
        .iter()
        .filter(|agent| agent.is_alive())
        .map(|agent| agent.id() as usize)
        .collect();
    if alive_ids.is_empty() {
        return;
    }
    alive_ids.sort_unstable();
    let target_agent_id = alive_ids[step % alive_ids.len()];
    let Some(agent_pos) = pop
        .individuals
        .iter()
        .position(|agent| agent.id() as usize == target_agent_id)
    else {
        return;
    };

    let current_freq_hz = pop.individuals[agent_pos].body.base_freq_hz();
    if !current_freq_hz.is_finite() || current_freq_hz <= 0.0 {
        return;
    }
    let current_pitch_log2 = current_freq_hz.log2();
    let kernel_params = KernelParams::default();
    let current_erb = hz_to_erb(current_freq_hz.max(1e-6));
    let neighbor_erbs: Vec<f32> = pop
        .individuals
        .iter()
        .enumerate()
        .filter_map(|(idx, agent)| {
            if idx == agent_pos || !agent.is_alive() {
                return None;
            }
            let freq_hz = agent.body.base_freq_hz();
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                return None;
            }
            Some(hz_to_erb(freq_hz))
        })
        .collect();

    let mut current_crowding = 0.0f32;
    for &other_erb in &neighbor_erbs {
        current_crowding += crowding_runtime_delta_erb(&kernel_params, current_erb - other_erb);
    }
    let current_score = selection_reference_landscape.evaluate_pitch_score(current_freq_hz)
        - E2_MATCH_CROWDING_WEIGHT * current_crowding;

    let radius_steps = (radius_st.max(0.0) / 0.25).round() as i32;
    let step_log2 = (0.25f32 / 12.0).max(1e-6);
    let mut best_freq_hz = current_freq_hz;
    let mut best_score = current_score;

    for offset_steps in -radius_steps..=radius_steps {
        let cand_pitch_log2 = current_pitch_log2 + offset_steps as f32 * step_log2;
        let cand_freq_hz = 2.0f32
            .powf(cand_pitch_log2)
            .clamp(tessitura_min_hz, tessitura_max_hz);
        if !cand_freq_hz.is_finite() || cand_freq_hz <= 0.0 {
            continue;
        }
        let cand_erb = hz_to_erb(cand_freq_hz.max(1e-6));
        let mut cand_crowding = 0.0f32;
        for &other_erb in &neighbor_erbs {
            cand_crowding += crowding_runtime_delta_erb(&kernel_params, cand_erb - other_erb);
        }
        let move_cost =
            shared_hill_move_cost_coeff() * (cand_pitch_log2 - current_pitch_log2).abs();
        let cand_score = selection_reference_landscape.evaluate_pitch_score(cand_freq_hz)
            - E2_MATCH_CROWDING_WEIGHT * cand_crowding
            - move_cost;
        if cand_score > best_score {
            best_score = cand_score;
            best_freq_hz = cand_freq_hz;
        }
    }

    if best_score > current_score + E2_E6_HILL_IMPROVEMENT_THRESHOLD
        && let Some(agent) = pop.individuals.get_mut(agent_pos)
    {
        agent.body.set_freq(best_freq_hz);
    }
}

fn e6_apply_juvenile_contextual_local_search(
    pop: &mut Population,
    states: &[E3LifeState],
    selection_reference_landscape: &Landscape,
    contextual_landscape: &Landscape,
    space: &Log2Space,
    params: &LandscapeParams,
    env_partials: u32,
    env_partial_decay: f32,
    step: usize,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    contextual_mix_weight: f32,
) {
    if contextual_mix_weight <= 0.0 {
        return;
    }
    let mut eligible_ids: Vec<usize> = pop
        .individuals
        .iter()
        .filter(|agent| agent.is_alive())
        .filter_map(|agent| {
            let idx = agent.id() as usize;
            let state = states.get(idx)?;
            (state.ticks < E6_JUVENILE_CONTEXTUAL_TUNING_TICKS).then_some(idx)
        })
        .collect();
    if eligible_ids.is_empty() {
        return;
    }
    eligible_ids.sort_unstable();
    let target_agent_id = eligible_ids[step % eligible_ids.len()];
    let Some(agent_pos) = pop
        .individuals
        .iter()
        .position(|agent| agent.id() as usize == target_agent_id)
    else {
        return;
    };

    let current_freq_hz = pop.individuals[agent_pos].body.base_freq_hz();
    if !current_freq_hz.is_finite() || current_freq_hz <= 0.0 {
        return;
    }
    let current_pitch_log2 = current_freq_hz.log2();
    let kernel_params = KernelParams::default();
    let current_erb = hz_to_erb(current_freq_hz.max(1e-6));
    let neighbor_erbs: Vec<f32> = pop
        .individuals
        .iter()
        .enumerate()
        .filter_map(|(idx, agent)| {
            if idx == agent_pos || !agent.is_alive() {
                return None;
            }
            let freq_hz = agent.body.base_freq_hz();
            if !freq_hz.is_finite() || freq_hz <= 0.0 {
                return None;
            }
            Some(hz_to_erb(freq_hz))
        })
        .collect();

    let mut current_crowding = 0.0f32;
    for &other_erb in &neighbor_erbs {
        current_crowding += crowding_runtime_delta_erb(&kernel_params, current_erb - other_erb);
    }
    let loo_contextual_landscape = build_e6_contextual_loo_landscape(
        space,
        params,
        contextual_landscape,
        current_freq_hz,
        env_partials,
        env_partial_decay,
    );
    let current_score = e6_selection_score(
        selection_reference_landscape,
        &loo_contextual_landscape,
        current_freq_hz,
        contextual_mix_weight,
    ) - E6_JUVENILE_CONTEXTUAL_TUNING_CROWDING_WEIGHT * current_crowding;

    let grid_st = E6_JUVENILE_CONTEXTUAL_TUNING_GRID_ST.max(0.25);
    let radius_steps = (E6_JUVENILE_CONTEXTUAL_TUNING_RADIUS_ST.max(0.0) / grid_st).round() as i32;
    let step_log2 = (grid_st / 12.0).max(1e-6);
    let mut best_freq_hz = current_freq_hz;
    let mut best_score = current_score;

    for offset_steps in -radius_steps..=radius_steps {
        let cand_pitch_log2 = current_pitch_log2 + offset_steps as f32 * step_log2;
        let cand_freq_hz = 2.0f32
            .powf(cand_pitch_log2)
            .clamp(tessitura_min_hz, tessitura_max_hz);
        if !cand_freq_hz.is_finite() || cand_freq_hz <= 0.0 {
            continue;
        }
        let cand_erb = hz_to_erb(cand_freq_hz.max(1e-6));
        let mut cand_crowding = 0.0f32;
        for &other_erb in &neighbor_erbs {
            cand_crowding += crowding_runtime_delta_erb(&kernel_params, cand_erb - other_erb);
        }
        let move_cost =
            shared_hill_move_cost_coeff() * (cand_pitch_log2 - current_pitch_log2).abs();
        let cand_score = e6_selection_score(
            selection_reference_landscape,
            &loo_contextual_landscape,
            cand_freq_hz,
            contextual_mix_weight,
        ) - E6_JUVENILE_CONTEXTUAL_TUNING_CROWDING_WEIGHT * cand_crowding
            - move_cost;
        if cand_score > best_score {
            best_score = cand_score;
            best_freq_hz = cand_freq_hz;
        }
    }

    if best_score > current_score + E6_JUVENILE_CONTEXTUAL_TUNING_IMPROVEMENT_THRESHOLD
        && let Some(agent) = pop.individuals.get_mut(agent_pos)
    {
        agent.body.set_freq(best_freq_hz);
    }
}

fn e6_apply_juvenile_polyphonic_local_search(
    pop: &mut Population,
    states: &[E3LifeState],
    scene_landscape: &Landscape,
    space: &Log2Space,
    params: &LandscapeParams,
    env_partials: u32,
    env_partial_decay: f32,
    step: usize,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    min_spacing_cents: Option<f32>,
    crowding_weight: f32,
    overcapacity_weight: f32,
) {
    let mut eligible_ids: Vec<usize> = pop
        .individuals
        .iter()
        .filter(|agent| agent.is_alive())
        .filter_map(|agent| {
            let idx = agent.id() as usize;
            let state = states.get(idx)?;
            (state.ticks < E6_JUVENILE_CONTEXTUAL_TUNING_TICKS).then_some(idx)
        })
        .collect();
    if eligible_ids.is_empty() {
        return;
    }
    eligible_ids.sort_unstable();
    let target_agent_ids: Vec<usize> = if min_spacing_cents.is_some() {
        eligible_ids.clone()
    } else {
        vec![eligible_ids[step % eligible_ids.len()]]
    };

    for target_agent_id in target_agent_ids {
        let Some(agent_pos) = pop
            .individuals
            .iter()
            .position(|agent| agent.id() as usize == target_agent_id)
        else {
            continue;
        };

        let current_freq_hz = pop.individuals[agent_pos].body.base_freq_hz();
        if !current_freq_hz.is_finite() || current_freq_hz <= 0.0 {
            continue;
        }
        let other_freqs_hz = pop
            .individuals
            .iter()
            .enumerate()
            .filter_map(|(idx, agent)| {
                if idx == agent_pos || !agent.is_alive() {
                    return None;
                }
                let freq_hz = agent.body.base_freq_hz();
                (freq_hz.is_finite() && freq_hz > 0.0).then_some(freq_hz)
            })
            .collect::<Vec<_>>();
        let loo_scene_landscape = build_e6_contextual_loo_landscape(
            space,
            params,
            scene_landscape,
            current_freq_hz,
            env_partials,
            env_partial_decay,
        );
        let current_pitch_log2 = current_freq_hz.log2();
        let current_spacing_ok = min_spacing_cents.is_some_and(|min_spacing| {
            e6_respects_min_spacing_cents(current_freq_hz, &other_freqs_hz, min_spacing)
        }) || min_spacing_cents.is_none()
            && e6_respects_hard_anti_fusion(current_freq_hz, &other_freqs_hz);
        let current_score = e6_polyphonic_selection_score(
            &loo_scene_landscape,
            current_freq_hz,
            &other_freqs_hz,
            crowding_weight,
            overcapacity_weight,
        ) - if current_spacing_ok { 0.0 } else { 1.0 };

        let grid_st = E6_JUVENILE_CONTEXTUAL_TUNING_GRID_ST.max(0.25);
        let radius_steps =
            (E6_JUVENILE_CONTEXTUAL_TUNING_RADIUS_ST.max(0.0) / grid_st).round() as i32;
        let step_log2 = (grid_st / 12.0).max(1e-6);
        let mut best_freq_hz = current_freq_hz;
        let mut best_score = current_score;
        let mut nearest_spaced_freq_hz: Option<f32> = None;
        let mut nearest_spaced_delta = f32::INFINITY;

        for offset_steps in -radius_steps..=radius_steps {
            let cand_pitch_log2 = current_pitch_log2 + offset_steps as f32 * step_log2;
            let cand_freq_hz = 2.0f32
                .powf(cand_pitch_log2)
                .clamp(tessitura_min_hz, tessitura_max_hz);
            if !cand_freq_hz.is_finite() || cand_freq_hz <= 0.0 {
                continue;
            }
            let respects_spacing = min_spacing_cents.is_some_and(|min_spacing| {
                e6_respects_min_spacing_cents(cand_freq_hz, &other_freqs_hz, min_spacing)
            }) || min_spacing_cents.is_none()
                && e6_respects_hard_anti_fusion(cand_freq_hz, &other_freqs_hz);
            if !respects_spacing {
                continue;
            }
            let move_delta = (cand_pitch_log2 - current_pitch_log2).abs();
            if move_delta > 0.0 && move_delta < nearest_spaced_delta {
                nearest_spaced_delta = move_delta;
                nearest_spaced_freq_hz = Some(cand_freq_hz);
            }
            let move_cost =
                shared_hill_move_cost_coeff() * (cand_pitch_log2 - current_pitch_log2).abs();
            let cand_score = e6_polyphonic_selection_score(
                &loo_scene_landscape,
                cand_freq_hz,
                &other_freqs_hz,
                crowding_weight,
                overcapacity_weight,
            ) - move_cost;
            if cand_score > best_score {
                best_score = cand_score;
                best_freq_hz = cand_freq_hz;
            }
        }

        if !current_spacing_ok
            && let Some(freq_hz) = nearest_spaced_freq_hz
            && let Some(agent) = pop.individuals.get_mut(agent_pos)
        {
            agent.body.set_freq(freq_hz);
        } else if best_score > current_score + E6_JUVENILE_CONTEXTUAL_TUNING_IMPROVEMENT_THRESHOLD
            && let Some(agent) = pop.individuals.get_mut(agent_pos)
        {
            agent.body.set_freq(best_freq_hz);
        }
    }
}

fn apply_selection_recharge(agent: &mut Individual, c_score: f32, rate_per_sec: f32, dt_sec: f32) {
    let delta = rate_per_sec * c_score.max(0.0) * dt_sec.max(0.0);
    if delta <= 0.0 || !delta.is_finite() {
        return;
    }
    if let AnyArticulationCore::Entrain(ref mut core) = agent.articulation.core {
        core.energy = (core.energy + delta).max(0.0);
    }
}

fn e6_selection_score(
    selection_reference_landscape: &Landscape,
    contextual_landscape: &Landscape,
    freq_hz: f32,
    contextual_mix_weight: f32,
) -> f32 {
    let anchor_score = selection_reference_landscape.evaluate_pitch_score(freq_hz);
    let w = contextual_mix_weight.clamp(0.0, 1.0);
    if w <= 0.0 {
        return anchor_score;
    }
    let contextual_score = contextual_landscape.evaluate_pitch_score(freq_hz);
    (1.0 - w) * anchor_score + w * contextual_score
}

pub fn e6_mean_selection_score_for_freqs(
    freqs_hz: &[f32],
    anchor_hz: f32,
    contextual_mix_weight: f32,
    env_partials: Option<u32>,
    env_partial_decay: Option<f32>,
) -> f32 {
    e6_mean_selection_score_for_freqs_mode(
        freqs_hz,
        anchor_hz,
        E6SelectionScoreMode::LegacyAnchorContextMix,
        contextual_mix_weight,
        env_partials,
        env_partial_decay,
    )
}

pub fn e6_mean_selection_score_for_freqs_mode(
    freqs_hz: &[f32],
    anchor_hz: f32,
    selection_score_mode: E6SelectionScoreMode,
    contextual_mix_weight: f32,
    env_partials: Option<u32>,
    env_partial_decay: Option<f32>,
) -> f32 {
    let clean_freqs: Vec<f32> = freqs_hz
        .iter()
        .copied()
        .filter(|f| f.is_finite() && *f > 0.0)
        .collect();
    if clean_freqs.is_empty() {
        return 0.0;
    }

    let partials = sanitize_env_partials(env_partials.unwrap_or(E4_ENV_PARTIALS_DEFAULT));
    let decay =
        sanitize_env_partial_decay(env_partial_decay.unwrap_or(E4_ENV_PARTIAL_DECAY_DEFAULT));
    let selection_reference_landscape = e3_reference_landscape_with_partials(anchor_hz, partials);
    let space = selection_reference_landscape.space.clone();
    let params = make_landscape_params(&space, E3_FS, 1.0);
    match selection_score_mode {
        E6SelectionScoreMode::LegacyAnchorContextMix => {
            let mut env_scan = vec![0.0f32; space.n_bins()];
            for &freq in &clean_freqs {
                add_harmonic_partials_to_env(&space, &mut env_scan, freq, 1.0, partials, decay);
            }
            let contextual_landscape =
                build_density_landscape_from_env_scan(&space, &params, env_scan);
            clean_freqs
                .iter()
                .copied()
                .map(|freq| {
                    let loo_contextual_landscape = build_e6_contextual_loo_landscape(
                        &space,
                        &params,
                        &contextual_landscape,
                        freq,
                        partials,
                        decay,
                    );
                    e6_selection_score(
                        &selection_reference_landscape,
                        &loo_contextual_landscape,
                        freq,
                        contextual_mix_weight,
                    )
                })
                .sum::<f32>()
                / clean_freqs.len() as f32
        }
        E6SelectionScoreMode::PolyphonicLooCrowding => {
            let scene_landscape = build_e6_scene_landscape_with_anchor(
                &space,
                &params,
                &selection_reference_landscape,
                &clean_freqs,
                partials,
                decay,
            );
            clean_freqs
                .iter()
                .enumerate()
                .map(|(idx, &freq)| {
                    let other_freqs_hz = clean_freqs
                        .iter()
                        .enumerate()
                        .filter_map(|(other_idx, &other_freq)| {
                            (other_idx != idx).then_some(other_freq)
                        })
                        .collect::<Vec<_>>();
                    let loo_scene_landscape = build_e6_contextual_loo_landscape(
                        &space,
                        &params,
                        &scene_landscape,
                        freq,
                        partials,
                        decay,
                    );
                    e6_polyphonic_selection_score(
                        &loo_scene_landscape,
                        freq,
                        &other_freqs_hz,
                        E2_MATCH_CROWDING_WEIGHT,
                        0.0,
                    )
                })
                .sum::<f32>()
                / clean_freqs.len() as f32
        }
    }
}

fn e6_agent_energy(agent: &Individual) -> Option<f32> {
    let AnyArticulationCore::Entrain(ref core) = agent.articulation.core else {
        return None;
    };
    let energy = core.energy.max(0.0);
    if energy.is_finite() {
        Some(energy)
    } else {
        None
    }
}

fn e6_parent_fertility_weight(agent: &Individual) -> Option<f32> {
    let energy = e6_agent_energy(agent)?;
    let weight = energy.powf(E6_FERTILITY_ENERGY_EXPONENT);
    Some(weight.max(1e-6))
}

fn e6_record_child_spawn_state(
    states: &mut [E3LifeState],
    child_idx: usize,
    spawn_freq_hz: f32,
    selection_reference_landscape: &Landscape,
    freeze_pitch_after_respawn: bool,
) -> (f32, f32) {
    let spawn_c_score = selection_reference_landscape.evaluate_pitch_score(spawn_freq_hz);
    let spawn_c_level = selection_reference_landscape
        .evaluate_pitch_level(spawn_freq_hz)
        .clamp(0.0, 1.0);
    if let Some(state) = states.get_mut(child_idx) {
        state.spawn_c_score = Some(spawn_c_score);
        state.spawn_c_level = Some(spawn_c_level);
        state.step1_c_score = None;
        state.step1_c_level = None;
        state.step10_c_score = None;
        state.step10_c_level = None;
        state.step100_c_score = None;
        state.step100_c_level = None;
        state.freeze_freq_hz = freeze_pitch_after_respawn.then_some(spawn_freq_hz);
    }
    (spawn_c_score, spawn_c_level)
}

fn normalize_scan_to_pmf(scan: &[f32]) -> Vec<f32> {
    let total: f32 = scan
        .iter()
        .copied()
        .filter(|w| w.is_finite() && *w > 0.0)
        .sum();
    if total <= 0.0 {
        return vec![0.0; scan.len()];
    }
    scan.iter()
        .map(|&w| {
            if w.is_finite() && w > 0.0 {
                w / total
            } else {
                0.0
            }
        })
        .collect()
}

pub fn e6_sampler_debug_scan(
    parent_freq_hz: f32,
    current_freqs_hz: &[f32],
    env_partials: Option<u32>,
    crowding_sigma_cents: f32,
) -> E6SamplerDebugScan {
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let partials = env_partials.unwrap_or(E4_ENV_PARTIALS_DEFAULT);
    let parent_landscape = build_e6_parent_harmonicity_landscape(
        &space,
        &params,
        parent_freq_hz,
        partials,
        E4_ENV_PARTIAL_DECAY_DEFAULT,
    );
    let current_landscape = if current_freqs_hz.is_empty() {
        build_anchor_landscape(
            &space,
            &params,
            E4_ANCHOR_HZ,
            partials,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        )
    } else {
        build_density_landscape_from_freqs(
            &space,
            &params,
            current_freqs_hz,
            partials,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        )
    };

    let local_weights =
        parent_window_and_notch_weights(&space, parent_freq_hz, crowding_sigma_cents);
    let parent_log2 = parent_freq_hz.max(1e-6).log2();
    let (tessitura_min_hz, tessitura_max_hz) = e3_tessitura_bounds(E4_ANCHOR_HZ, &space);

    let mut harmonicity = Vec::with_capacity(space.n_bins());
    let mut harmonicity_local = Vec::with_capacity(space.n_bins());
    let mut final_weights = Vec::with_capacity(space.n_bins());
    let mut delta_semitones = Vec::with_capacity(space.n_bins());

    for (idx, &local_weight) in local_weights.iter().enumerate() {
        let delta_oct = space.centers_log2[idx] - parent_log2;
        let delta_st = 12.0 * delta_oct;
        let h = parent_landscape.harmonicity01[idx].max(0.0);
        let w_h_local = h * local_weight;
        let social_weight = weak_social_roughness_weight(&current_landscape, idx);
        let w_final = w_h_local * social_weight;

        harmonicity.push(h);
        harmonicity_local.push(w_h_local);
        final_weights.push(w_final);
        delta_semitones.push(delta_st);
    }
    let final_weights =
        tessitura_masked_weights(&space, &final_weights, tessitura_min_hz, tessitura_max_hz);

    E6SamplerDebugScan {
        parent_freq_hz,
        parent_semitones_from_anchor: 12.0 * (parent_freq_hz / E4_ANCHOR_HZ).log2(),
        delta_semitones,
        harmonicity_pmf: normalize_scan_to_pmf(&harmonicity),
        harmonicity_local_pmf: normalize_scan_to_pmf(&harmonicity_local),
        final_pmf: normalize_scan_to_pmf(&final_weights),
    }
}

fn crowding_sigma_erb_from_hz(freq_hz: f32, sigma_cents: f32) -> f32 {
    let sigma_log2 = (sigma_cents.max(1e-3)) / 1200.0;
    let base_hz = freq_hz.max(1e-6);
    let plus_hz = base_hz * 2.0f32.powf(sigma_log2);
    (hz_to_erb(plus_hz) - hz_to_erb(base_hz)).abs().max(1e-6)
}

fn init_e4_rhythms(cfg: &E4SimConfig) -> NeuralRhythms {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = cfg.theta_freq_hz.max(0.1);
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.delta.freq_hz = 1.0;
    rhythms.delta.mag = 1.0;
    rhythms.delta.alpha = 1.0;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    rhythms
}

fn apply_e4_initial_pitches(
    pop: &mut Population,
    space: &Log2Space,
    landscape: &Landscape,
    cfg: &E4SimConfig,
    seed: u64,
    min_freq: f32,
    max_freq: f32,
) {
    let mut rng = SmallRng::seed_from_u64(seed ^ 0xE4E4_5EED);
    let min_freq = min_freq.clamp(space.fmin, space.fmax);
    let max_freq = max_freq.clamp(space.fmin, space.fmax);
    for agent in pop.individuals.iter_mut() {
        if agent.metadata.group_id != E4_GROUP_VOICES {
            continue;
        }
        let log2_freq =
            sample_e4_initial_pitch_log2(&mut rng, space, landscape, min_freq, max_freq);
        agent.force_set_pitch_log2(log2_freq);
        agent.set_neighbor_step_cents(cfg.neighbor_step_cents);
    }
}

fn sample_e4_initial_pitch_log2(
    rng: &mut impl Rng,
    space: &Log2Space,
    landscape: &Landscape,
    min_freq: f32,
    max_freq: f32,
) -> f32 {
    let min_idx = space.index_of_freq(min_freq).unwrap_or(0);
    let max_idx = space
        .index_of_freq(max_freq)
        .unwrap_or(space.n_bins().saturating_sub(1));
    let (min_idx, max_idx) = if min_idx <= max_idx {
        (min_idx, max_idx)
    } else {
        (max_idx, min_idx)
    };
    let mut weights = Vec::with_capacity(max_idx - min_idx + 1);
    for i in min_idx..=max_idx {
        let w = landscape
            .consonance_density_pmf
            .get(i)
            .copied()
            .unwrap_or(0.0)
            .max(0.0);
        weights.push(w);
    }
    if let Ok(dist) = WeightedIndex::new(&weights) {
        let idx = min_idx + dist.sample(rng);
        return space.freq_of_index(idx).log2();
    }
    let min_l = min_freq.log2();
    let max_l = max_freq.log2();
    if min_l.is_finite() && max_l.is_finite() && min_l < max_l {
        let r = rng.random_range(min_l..max_l);
        return r;
    }
    min_freq.max(1.0).log2()
}

fn update_e4_landscape_from_population(
    space: &Log2Space,
    params: &LandscapeParams,
    pop: &Population,
    landscape: &mut Landscape,
    env_partials: u32,
    env_partial_decay: f32,
) {
    let mut env_scan = vec![0.0f32; space.n_bins()];
    for agent in &pop.individuals {
        let freq = agent.body.base_freq_hz();
        if !freq.is_finite() || freq <= 0.0 {
            continue;
        }
        add_harmonic_partials_to_env(
            space,
            &mut env_scan,
            freq,
            1.0,
            env_partials,
            env_partial_decay,
        );
    }
    space.assert_scan_len_named(&env_scan, "e4_env_scan");
    let h_dual = params
        .harmonicity_kernel
        .potential_h_dual_from_log2_spectrum(&env_scan, space);
    landscape.subjective_intensity = env_scan.clone();
    landscape.nsgt_power = env_scan;
    landscape.harmonicity = h_dual.blended;
    landscape.harmonicity_path_a = h_dual.path_a;
    landscape.harmonicity_path_b = h_dual.path_b;
    landscape.root_affinity = h_dual.metrics.root_affinity;
    landscape.overtone_affinity = h_dual.metrics.overtone_affinity;
    landscape.binding_strength = h_dual.metrics.binding_strength;
    landscape.harmonic_tilt = h_dual.metrics.harmonic_tilt;
    landscape.harmonicity_mirror_weight = params.harmonicity_kernel.params.mirror_weight;
    compute_roughness_for_landscape(space, params, landscape);
}

fn init_rhythms(theta_freq_hz: f32) -> NeuralRhythms {
    NeuralRhythms {
        theta: RhythmBand {
            phase: 0.0,
            freq_hz: theta_freq_hz.max(0.1),
            mag: 1.0,
            alpha: 1.0,
            beta: 0.0,
        },
        delta: RhythmBand {
            phase: 0.0,
            freq_hz: 1.0,
            mag: 0.0,
            alpha: 0.0,
            beta: 0.0,
        },
        env_open: 1.0,
        env_level: 1.0,
    }
}

/// Spawn spec for E6: HillClimb with configurable landscape_weight.
/// When landscape_weight=0 agents move via density-based repulsion only,
/// isolating hereditary selection as the sole source of consonance improvement.
fn e6_spawn_spec(
    anchor_hz: f32,
    landscape_weight: f32,
    crowding_strength_override: Option<f32>,
    adaptation_enabled_override: Option<bool>,
    range_oct_override: Option<f32>,
    exact_e2_alignment: bool,
    disable_within_life_pitch_movement: bool,
) -> SpawnSpec {
    let mut control = AgentControl::default();
    configure_shared_hillclimb_control(&mut control, landscape_weight);
    if let Some(crowding_strength) = crowding_strength_override {
        control.pitch.crowding_strength = crowding_strength.max(0.0);
    }
    if let Some(enabled) = adaptation_enabled_override {
        control.adaptation.enabled = enabled;
    }
    if exact_e2_alignment || disable_within_life_pitch_movement {
        control.pitch.mode = PitchMode::Lock;
    }
    control.pitch.freq = anchor_hz.max(1.0);
    control.pitch.range_oct = range_oct_override.unwrap_or(E3_RANGE_OCT).max(1e-6);
    control.phonation.spec = PhonationSpec::default();

    let lifecycle = e6_lifecycle();
    let articulation = ArticulationCoreConfig::Entrain {
        lifecycle,
        rhythm_freq: Some(E3_THETA_FREQ_HZ),
        rhythm_sensitivity: None,
        rhythm_coupling: RhythmCouplingMode::TemporalOnly,
        rhythm_reward: None,
        breath_gain_init: None,
        k_omega: None,
        base_sigma: None,
        gate_thresholds: None,
        energy_cap: None,
    };

    SpawnSpec {
        control,
        articulation,
    }
}

fn e3_spawn_spec(condition: E3Condition, anchor_hz: f32) -> SpawnSpec {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Lock;
    control.pitch.freq = anchor_hz.max(1.0);
    control.phonation.spec = PhonationSpec::default();
    let lifecycle = e3_lifecycle(condition);
    let articulation = ArticulationCoreConfig::Entrain {
        lifecycle,
        rhythm_freq: Some(E3_THETA_FREQ_HZ),
        rhythm_sensitivity: None,
        rhythm_coupling: RhythmCouplingMode::TemporalOnly,
        rhythm_reward: None,
        breath_gain_init: None,
        k_omega: None,
        base_sigma: None,
        gate_thresholds: None,
        energy_cap: None,
    };

    SpawnSpec {
        control,
        articulation,
    }
}

fn e3_spawn_strategy(anchor_hz: f32, space: &Log2Space) -> SpawnStrategy {
    e3_spawn_strategy_with_range(anchor_hz, space, E3_RANGE_OCT)
}

fn e3_tessitura_bounds(anchor_hz: f32, space: &Log2Space) -> (f32, f32) {
    e3_tessitura_bounds_for_range(anchor_hz, space, E3_RANGE_OCT)
}

fn e3_lifecycle(_condition: E3Condition) -> LifecycleConfig {
    LifecycleConfig::Sustain {
        initial_energy: 1.0,
        metabolism_rate: E3_METABOLISM_RATE,
        recharge_rate: Some(0.0),
        action_cost: Some(0.0),
        continuous_recharge_rate: None,
        envelope: EnvelopeConfig::default(),
    }
}

fn e6_lifecycle() -> LifecycleConfig {
    LifecycleConfig::Sustain {
        initial_energy: E6_INITIAL_ENERGY,
        metabolism_rate: E6_METABOLISM_RATE,
        recharge_rate: Some(0.0),
        action_cost: Some(0.0),
        continuous_recharge_rate: None,
        envelope: EnvelopeConfig::default(),
    }
}

pub fn e3_policy_params(condition: E3Condition) -> E3PolicyParams {
    let lifecycle = e3_lifecycle(condition);
    let policy = MetabolismPolicy::from_lifecycle(&lifecycle);
    E3PolicyParams {
        condition: condition.label().to_string(),
        dt_sec: E3_HOP as f32 / E3_FS,
        basal_cost_per_sec: policy.basal_cost_per_sec,
        action_cost_per_attack: policy.action_cost_per_attack,
        recharge_per_attack: policy.recharge_per_attack,
        continuous_recharge_per_sec: match condition {
            E3Condition::Baseline => E3_SELECTION_RECHARGE_PER_SEC,
            E3Condition::NoRecharge => 0.0,
        },
    }
}

fn anchor_control(anchor_hz: f32) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Lock;
    control.pitch.freq = anchor_hz.max(1.0);
    control.phonation.spec = PhonationSpec::default();
    control
}

fn voice_control(cfg: &E4SimConfig) -> AgentControl {
    let mut control = AgentControl::default();
    control.pitch.mode = PitchMode::Free;
    control.pitch.core_kind = PitchCoreKind::PeakSampler;
    control.pitch.freq = cfg.center_hz().max(1.0);
    control.pitch.range_oct = cfg.range_oct;
    control.pitch.gravity = 0.0;
    control.pitch.exploration = cfg.exploration;
    control.pitch.persistence = cfg.persistence;
    control.phonation.spec = PhonationSpec::default();
    control
}

// ─── E3 audio rendering ───

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum E3AudioCondition {
    Shared,
    Scrambled,
    Off,
}

impl E3AudioCondition {
    pub fn label(self) -> &'static str {
        match self {
            Self::Shared => "shared",
            Self::Scrambled => "scrambled",
            Self::Off => "off",
        }
    }
}

#[derive(Clone, Debug)]
pub struct E3AudioConfig {
    pub seed: u64,
    pub pop_size: usize,
    pub duration_sec: f32,
    pub condition: E3AudioCondition,
    pub anchor_hz: f32,
    pub theta_freq_hz: f32,
}

impl E3AudioConfig {
    pub fn default_shared() -> Self {
        Self {
            seed: 42,
            pop_size: 48,
            duration_sec: 8.0,
            condition: E3AudioCondition::Shared,
            anchor_hz: E4_ANCHOR_HZ,
            theta_freq_hz: 2.0,
        }
    }

    pub fn default_scrambled() -> Self {
        Self {
            condition: E3AudioCondition::Scrambled,
            ..Self::default_shared()
        }
    }

    pub fn default_off() -> Self {
        Self {
            condition: E3AudioCondition::Off,
            ..Self::default_shared()
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct E3AudioAttack {
    time: f32,
    agent: usize,
}

fn validate_e3_audio_config(cfg: &E3AudioConfig) -> std::io::Result<()> {
    if cfg.pop_size == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "E3 audio requires pop_size > 0",
        ));
    }
    if cfg.duration_sec <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "E3 audio requires duration_sec > 0",
        ));
    }
    if cfg.anchor_hz <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "E3 audio requires anchor_hz > 0",
        ));
    }
    if cfg.theta_freq_hz <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "E3 audio requires theta_freq_hz > 0",
        ));
    }
    Ok(())
}

fn simulate_e3_audio_attacks(
    cfg: &E3AudioConfig,
) -> std::io::Result<(Vec<f32>, Vec<E3AudioAttack>)> {
    use std::f32::consts::TAU;

    const SIM_DT: f32 = 0.002;
    const K_TIME: f32 = 3.0;
    const AGENT_OMEGA_MEAN: f32 = TAU * 1.8;
    const AGENT_JITTER: f32 = 0.02;

    validate_e3_audio_config(cfg)?;

    let mut rng = SmallRng::seed_from_u64(cfg.seed);
    let kick_freq_hz = cfg.theta_freq_hz;
    let kick_omega = TAU * kick_freq_hz;
    let steps_per_cycle = ((1.0 / kick_freq_hz) / SIM_DT).round().max(1.0) as usize;

    let log2_anchor = cfg.anchor_hz.log2();
    let pitches_hz: Vec<f32> = (0..cfg.pop_size)
        .map(|_| 2.0f32.powf(rng.random_range((log2_anchor - 1.0)..(log2_anchor + 1.0))))
        .collect();
    let omegas: Vec<f32> = (0..cfg.pop_size)
        .map(|_| AGENT_OMEGA_MEAN * (1.0 + rng.random_range(-AGENT_JITTER..AGENT_JITTER)))
        .collect();
    let mut phases: Vec<f32> = (0..cfg.pop_size)
        .map(|_| rng.random_range(0.0f32..TAU))
        .collect();
    let mut prev_phases = phases.clone();
    let mut theta_canonical = 0.0f32;
    let mut scramble_offset = rng.random_range(0.0f32..TAU);
    let total_steps = (cfg.duration_sec / SIM_DT).ceil() as usize;
    let mut attacks = Vec::new();

    for step in 0..total_steps {
        let t = step as f32 * SIM_DT;
        if cfg.condition == E3AudioCondition::Scrambled && step > 0 && step % steps_per_cycle == 0 {
            scramble_offset = rng.random_range(0.0f32..TAU);
        }

        let theta_drive = match cfg.condition {
            E3AudioCondition::Shared => theta_canonical,
            E3AudioCondition::Scrambled => theta_canonical + scramble_offset,
            E3AudioCondition::Off => theta_canonical,
        };

        for i in 0..cfg.pop_size {
            prev_phases[i] = phases[i];
            let k_eff = match cfg.condition {
                E3AudioCondition::Off => 0.0,
                E3AudioCondition::Shared | E3AudioCondition::Scrambled => K_TIME,
            };
            phases[i] = kuramoto_phase_step(phases[i], omegas[i], theta_drive, k_eff, 0.0, SIM_DT);
        }

        for i in 0..cfg.pop_size {
            if (phases[i] / TAU).floor() > (prev_phases[i] / TAU).floor() {
                attacks.push(E3AudioAttack { time: t, agent: i });
            }
        }

        theta_canonical += kick_omega * SIM_DT;
    }

    Ok((pitches_hz, attacks))
}

pub fn render_e3_audio(cfg: &E3AudioConfig, output_path: &std::path::Path) -> std::io::Result<()> {
    use std::f32::consts::TAU;

    const NOTE_DUR_SEC: f32 = 0.12;
    const NOTE_ATTACK_SEC: f32 = 0.005;
    const NOTE_AMP: f32 = 0.16;

    let (pitches_hz, attacks) = simulate_e3_audio_attacks(cfg)?;
    let fs = E3_FS;
    let total_samples = (cfg.duration_sec * fs).ceil() as usize;
    let mut samples = vec![0.0f32; total_samples.max(1)];

    let note_len = (NOTE_DUR_SEC * fs).ceil() as usize;
    for attack in &attacks {
        let start = (attack.time * fs).round() as usize;
        let freq = pitches_hz[attack.agent];
        for local_idx in 0..note_len {
            let sample_idx = start + local_idx;
            if sample_idx >= samples.len() {
                break;
            }
            let local_t = local_idx as f32 / fs;
            let env = if local_t < NOTE_ATTACK_SEC {
                (local_t / NOTE_ATTACK_SEC).clamp(0.0, 1.0)
            } else {
                (1.0 - (local_t - NOTE_ATTACK_SEC) / (NOTE_DUR_SEC - NOTE_ATTACK_SEC))
                    .clamp(0.0, 1.0)
            };
            let phase = TAU * freq * local_t;
            let voice = phase.sin() + 0.35 * (2.0 * phase).sin() + 0.18 * (3.0 * phase).sin();
            samples[sample_idx] += NOTE_AMP * env * voice;
        }
    }

    let peak = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if peak > 0.01 {
        let scale = 0.9 / peak;
        for sample in &mut samples {
            *sample *= scale;
        }
    }

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: fs as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, spec)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    for &sample in &samples {
        let i16_val = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(i16_val)
            .map_err(|e| std::io::Error::other(e.to_string()))?;
    }
    writer
        .finalize()
        .map_err(|e| std::io::Error::other(e.to_string()))?;

    eprintln!(
        "E3 audio: {} ({:.1}s, attacks={}, condition={})",
        output_path.display(),
        samples.len() as f32 / fs,
        attacks.len(),
        cfg.condition.label()
    );
    Ok(())
}

pub fn generate_e3_rhai(cfg: &E3AudioConfig, output_path: &std::path::Path) -> std::io::Result<()> {
    use std::io::Write;

    let (pitches_hz, mut attacks) = simulate_e3_audio_attacks(cfg)?;
    let condition_label = cfg.condition.label();

    eprintln!(
        "  E3 rhai: {} attacks from {} agents over {:.1}s (condition={})",
        attacks.len(),
        cfg.pop_size,
        cfg.duration_sec,
        condition_label
    );

    let mut rhai = format!(
        "// 30_e3_{condition_label}.rhai — Temporal scaffold assay\n\
         // AUTO-GENERATED — DO NOT EDIT\n\
         // {n} agents, {kick:.1} Hz canonical beat, condition={condition_label}\n\
         // shared: continuous common scaffold\n\
         // scrambled: cycle-wise phase resets\n\
         // off: no scaffold coupling\n\
         // Each attack creates a short-lived seq voice at a static audibility pitch.\n\
         // No external drone or metronome is mixed into the render.\n\n",
        n = cfg.pop_size,
        kick = cfg.theta_freq_hz
    );

    for i in 0..cfg.pop_size {
        rhai.push_str(&format!(
            "let s{i} = derive(harmonic).brain(\"seq\").pitch_mode(\"lock\")\
             .amp(0.18).adsr(0.003, 0.08, 0.0, 0.03);\n"
        ));
    }
    rhai.push('\n');

    attacks.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    rhai.push_str(&format!("scene(\"e3_{condition_label}\", || {{\n"));
    let quant = 0.005f32;
    let mut time_cursor = 0.0f32;
    for attack in &attacks {
        let target_t = (attack.time / quant).round() * quant;
        if target_t > time_cursor + 0.0001 {
            let wait = target_t - time_cursor;
            rhai.push_str(&format!("    wait({wait:.4});\n"));
            time_cursor = target_t;
        }
        rhai.push_str(&format!(
            "    create(s{}, 1).freq({:.2}); flush();\n",
            attack.agent, pitches_hz[attack.agent]
        ));
    }
    let remaining = cfg.duration_sec - time_cursor;
    if remaining > 0.01 {
        rhai.push_str(&format!("    wait({remaining:.3});\n"));
    }
    rhai.push_str("});\n");

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(output_path)?;
    f.write_all(rhai.as_bytes())?;

    eprintln!("  wrote {} ({} bytes)", output_path.display(), rhai.len());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use conchordal::life::individual::AgentMetadata;

    #[test]
    fn e6_spawn_spec_uses_e2_crowding_and_e4_sigma() {
        let spec = e6_spawn_spec(E4_ANCHOR_HZ, 0.5, None, None, None, false, false);
        let pitch = spec.control.pitch;
        assert_eq!(pitch.core_kind, PitchCoreKind::HillClimb);
        assert_eq!(
            pitch.neighbor_step_cents,
            Some(E2_E6_HILL_NEIGHBOR_STEP_CENTS)
        );
        assert!((pitch.exploration - E2_E6_HILL_EXPLORATION).abs() <= 1e-6);
        assert!((pitch.persistence - E2_E6_HILL_PERSISTENCE).abs() <= 1e-6);
        assert!((pitch.move_cost_coeff - E2_E6_HILL_MOVE_COST_COEFF).abs() <= 1e-6);
        assert!((pitch.improvement_threshold - E2_E6_HILL_IMPROVEMENT_THRESHOLD).abs() <= 1e-6);
        assert!((pitch.crowding_strength - E2_MATCH_CROWDING_WEIGHT).abs() <= 1e-6);
        assert!((pitch.crowding_sigma_cents - E4_MATCH_SIGMA_CENTS).abs() <= 1e-6);
        assert!(!pitch.crowding_sigma_from_roughness);
        assert!(pitch.leave_self_out);
        assert_eq!(pitch.leave_self_out_mode, LeaveSelfOutMode::ApproxHarmonics);
        assert_eq!(pitch.global_peak_count, 0);
        assert_eq!(pitch.ratio_candidate_count, 0);
    }

    #[test]
    fn e6_lifecycle_is_neutral_turnover_with_external_anchor_recharge() {
        let e6_policy = MetabolismPolicy::from_lifecycle(&e6_lifecycle());
        assert!((e6_policy.basal_cost_per_sec - E6_METABOLISM_RATE).abs() <= 1e-6);
        assert_eq!(e6_policy.action_cost_per_attack, 0.0);
        assert_eq!(e6_policy.recharge_per_attack, 0.0);
    }

    #[test]
    fn e6_parent_fertility_weight_scales_with_energy() {
        let spec = e6_spawn_spec(E4_ANCHOR_HZ, 0.0, None, None, None, false, false);
        let meta = AgentMetadata {
            group_id: E3_GROUP_AGENTS,
            member_idx: 0,
        };
        let mut low = spec.clone().spawn(0, 0, meta.clone(), E3_FS, 0);
        let mut high = spec.spawn(1, 0, meta, E3_FS, 0);
        if let AnyArticulationCore::Entrain(ref mut core) = low.articulation.core {
            core.energy = 0.2;
        }
        if let AnyArticulationCore::Entrain(ref mut core) = high.articulation.core {
            core.energy = 0.8;
        }
        let low_w = e6_parent_fertility_weight(&low).expect("low fertility weight");
        let high_w = e6_parent_fertility_weight(&high).expect("high fertility weight");
        assert!(high_w > low_w);
    }

    #[test]
    fn parent_harmonicity_sampler_uses_local_window_and_parent_notch() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let parent_freq = 220.0f32;
        let mut parent_landscape = build_e6_parent_harmonicity_landscape(
            &space,
            &params,
            parent_freq,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        parent_landscape.harmonicity01.fill(1.0);
        let mut current_landscape = parent_landscape.clone();
        current_landscape.roughness01.fill(0.0);
        let weights = parent_window_and_notch_weights(&space, parent_freq, E4_MATCH_SIGMA_CENTS);
        let (tessitura_min_hz, tessitura_max_hz) = e3_tessitura_bounds(E4_ANCHOR_HZ, &space);
        let parent_idx = space.nearest_index(parent_freq);
        let notch_escape_idx = space.nearest_index(parent_freq * 2.0f32.powf(0.03));
        let near_idx = space.nearest_index(parent_freq * 2.0f32.powf(0.10));
        let far_idx = space.nearest_index(parent_freq * 2.0f32.powf(0.80));
        assert!(
            weights[parent_idx] < weights[notch_escape_idx],
            "parent notch should suppress the immediate parent vicinity"
        );
        assert!(
            weights[near_idx] > weights[far_idx],
            "gaussian window should prefer nearer bins over far bins"
        );
        let mut rng = SmallRng::seed_from_u64(0xE6AA_55CC);
        let mut mean_abs_delta_oct = 0.0f32;
        for _ in 0..128 {
            let sample = sample_from_parent_harmonicity(
                &parent_landscape,
                &current_landscape,
                &[],
                parent_freq,
                E4_MATCH_SIGMA_CENTS,
                tessitura_min_hz,
                tessitura_max_hz,
                &mut rng,
                E6_HEREDITY_FAMILY_OCCUPANCY_STRENGTH,
                E6FamilyNfdMode::Off,
                None,
            );
            let freq = sample.offspring_freq_hz;
            assert!(
                freq >= tessitura_min_hz && freq <= tessitura_max_hz,
                "tessitura-truncated heredity sampler should stay within the configured register"
            );
            assert!(
                sample.parent_azimuth_st.abs() <= E6_HEREDITY_AZIMUTH_CLIP_ST + 1e-6,
                "parent azimuth should stay clipped within the configured inheritance band"
            );
            assert!(
                sample.child_azimuth_st.abs() <= E6_HEREDITY_AZIMUTH_CLIP_ST + 1e-6,
                "child azimuth should stay clipped within the configured inheritance band"
            );
            mean_abs_delta_oct += (freq / parent_freq).log2().abs();
        }
        mean_abs_delta_oct /= 128.0;
        assert!(
            mean_abs_delta_oct < 0.60,
            "gaussian window should keep hereditary respawns local on average"
        );
    }

    #[test]
    fn e6_family_azimuth_sampler_keeps_children_near_parent_family() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let parent_freq = E4_ANCHOR_HZ * 2.0f32.powf(7.0 / 12.0);
        let parent_landscape = build_e6_parent_harmonicity_landscape(
            &space,
            &params,
            parent_freq,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let current_landscape = build_anchor_landscape(
            &space,
            &params,
            E4_ANCHOR_HZ,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let (min_hz, max_hz) = e3_tessitura_bounds(E4_ANCHOR_HZ, &space);
        let mut rng = SmallRng::seed_from_u64(0xE6AA_77DD);
        let mut inherited_count = 0usize;
        let mut mean_abs_child_azimuth = 0.0f32;
        for _ in 0..256 {
            let sample = sample_from_parent_harmonicity(
                &parent_landscape,
                &current_landscape,
                &[],
                parent_freq,
                E4_MATCH_SIGMA_CENTS,
                min_hz,
                max_hz,
                &mut rng,
                E6_HEREDITY_FAMILY_OCCUPANCY_STRENGTH,
                E6FamilyNfdMode::Off,
                None,
            );
            assert!(
                sample.offspring_freq_hz >= min_hz && sample.offspring_freq_hz <= max_hz,
                "family/azimuth heredity should stay within the configured tessitura"
            );
            if sample.family_inherited {
                inherited_count += 1;
            }
            mean_abs_child_azimuth += sample.child_azimuth_st.abs();
        }
        mean_abs_child_azimuth /= 256.0;
        assert!(
            inherited_count > 128,
            "same-family inheritance should be the dominant heredity mode"
        );
        assert!(
            mean_abs_child_azimuth < E6_HEREDITY_AZIMUTH_CLIP_ST,
            "azimuth inheritance should keep children close to the selected family center"
        );
    }

    #[test]
    fn e6_vacant_niches_respect_hard_anti_fusion() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let reference_landscape =
            e3_reference_landscape_with_partials(E4_ANCHOR_HZ, E4_ENV_PARTIALS_DEFAULT);
        let occupied = vec![
            E4_ANCHOR_HZ * 2.0f32.powf(4.0 / 12.0),
            E4_ANCHOR_HZ * 2.0f32.powf(7.0 / 12.0),
        ];
        let scene_landscape = build_e6_scene_landscape_with_anchor(
            &space,
            &params,
            &reference_landscape,
            &occupied,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let parent_landscape = build_e6_parent_harmonicity_landscape(
            &space,
            &params,
            occupied[1],
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let (min_hz, max_hz) = e3_tessitura_bounds(E4_ANCHOR_HZ, &space);
        let niches = e6_extract_vacant_niches(
            &space,
            &scene_landscape,
            &occupied,
            Some(&parent_landscape),
            min_hz,
            max_hz,
            E2_MATCH_CROWDING_WEIGHT,
            0.0,
        );
        assert!(
            !niches.is_empty(),
            "vacant niche extraction should expose at least one valid seat"
        );
        for niche in niches {
            assert!(
                niche.center_freq_hz >= min_hz && niche.center_freq_hz <= max_hz,
                "niche center should stay inside tessitura"
            );
            assert!(
                e6_respects_hard_anti_fusion(niche.center_freq_hz, &occupied),
                "vacant niche centers must satisfy the hard anti-fusion gate"
            );
        }
    }

    #[test]
    fn e6_polyphonic_selection_prefers_spread_over_duplicate_stack() {
        let spread = vec![
            E4_ANCHOR_HZ * (5.0 / 4.0),
            E4_ANCHOR_HZ * (3.0 / 2.0),
            E4_ANCHOR_HZ * (15.0 / 8.0),
        ];
        let duplicate = vec![
            E4_ANCHOR_HZ * (3.0 / 2.0),
            E4_ANCHOR_HZ * (3.0 / 2.0),
            E4_ANCHOR_HZ * (3.0 / 2.0),
        ];
        let spread_score = e6_mean_selection_score_for_freqs_mode(
            &spread,
            E4_ANCHOR_HZ,
            E6SelectionScoreMode::PolyphonicLooCrowding,
            0.0,
            None,
            None,
        );
        let duplicate_score = e6_mean_selection_score_for_freqs_mode(
            &duplicate,
            E4_ANCHOR_HZ,
            E6SelectionScoreMode::PolyphonicLooCrowding,
            0.0,
            None,
            None,
        );
        assert!(
            spread_score > duplicate_score,
            "polyphonic selection should reward spread harmonic scenes over duplicate stacks"
        );
    }

    #[test]
    fn run_e6_polyphonic_vacant_niche_mode_avoids_close_endpoint_collisions() {
        let cfg = E6RunConfig {
            seed: 0xE6AA_9911,
            steps_cap: 1500,
            min_deaths: 24,
            pop_size: 6,
            first_k: 10,
            condition: E6Condition::Heredity,
            snapshot_interval: 25,
            landscape_weight: 0.0,
            shuffle_landscape: false,
            env_partials: None,
            oracle_azimuth_radius_st: None,
            oracle_global_anchor_c: false,
            oracle_freeze_pitch_after_respawn: false,
            crowding_strength_override: Some(0.0),
            adaptation_enabled_override: Some(false),
            range_oct_override: Some(2.0),
            e2_aligned_exact_local_search_radius_st: None,
            disable_within_life_pitch_movement: true,
            selection_enabled: true,
            selection_contextual_mix_weight: 0.25,
            selection_score_mode: E6SelectionScoreMode::PolyphonicLooCrowding,
            polyphonic_crowding_weight_override: None,
            juvenile_contextual_settlement_enabled: false,
            juvenile_cull_enabled: false,
            record_life_diagnostics: true,
            family_occupancy_strength_override: None,
            family_nfd_mode: E6FamilyNfdMode::Off,
            respawn_mode: E6RespawnMode::VacantNicheByParentPrior,
            legacy_family_min_spacing_cents: None,
        };
        let result = run_e6(&cfg);
        assert!(
            result.total_deaths >= cfg.min_deaths,
            "polyphonic vacant-niche run should reach the requested turnover"
        );
        let final_snapshot = result
            .snapshots
            .last()
            .expect("polyphonic vacant-niche run should produce snapshots");
        for (idx, &freq_hz) in final_snapshot.freqs_hz.iter().enumerate() {
            let occupied = final_snapshot
                .freqs_hz
                .iter()
                .enumerate()
                .filter_map(|(other_idx, &other_freq_hz)| {
                    (other_idx != idx).then_some(other_freq_hz)
                })
                .collect::<Vec<_>>();
            assert!(
                e6_respects_hard_anti_fusion(freq_hz, &occupied),
                "polyphonic vacant-niche endpoint should not collapse into close collisions"
            );
        }
    }

    #[test]
    fn e6_peak_family_extractor_returns_local_maxima() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let mut weights = vec![0.0f32; space.n_bins()];
        let idx_a = space.nearest_index(E4_ANCHOR_HZ * 2.0f32.powf(4.0 / 12.0));
        let idx_b = space.nearest_index(E4_ANCHOR_HZ * 2.0f32.powf(7.0 / 12.0));
        weights[idx_a] = 1.0;
        weights[idx_b] = 0.8;
        if idx_a + 1 < weights.len() {
            weights[idx_a + 1] = 0.7;
        }
        let peaks = extract_peak_families(&space, &weights);
        assert!(
            !peaks.is_empty(),
            "peak extraction should find local maxima"
        );
        assert!(
            peaks.iter().any(|peak| peak.center_idx == idx_a)
                || peaks
                    .iter()
                    .any(|peak| semitone_distance(&space, peak.center_idx, idx_a)
                        <= E6_HEREDITY_PEAK_MERGE_ST),
            "peak extraction should retain the strongest local family"
        );
    }

    #[test]
    fn e6b_run_reaches_turnover_without_near_unisons() {
        let cfg = E6bRunConfig {
            seed: 0xE6B0_1234,
            steps_cap: 1800,
            min_deaths: 24,
            pop_size: 8,
            first_k: 10,
            condition: E6Condition::Heredity,
            snapshot_interval: 20,
            selection_enabled: true,
            shuffle_landscape: false,
        };
        let result = run_e6b(&cfg);
        assert!(
            result.total_deaths >= cfg.min_deaths,
            "E6b run should reach the requested turnover"
        );
        let final_snapshot = result
            .snapshots
            .last()
            .expect("E6b run should produce snapshots");
        let mut min_spacing_cents = f32::INFINITY;
        for (idx, &freq_hz) in final_snapshot.freqs_hz.iter().enumerate() {
            let occupied = final_snapshot
                .freqs_hz
                .iter()
                .enumerate()
                .filter_map(|(other_idx, &other_freq_hz)| {
                    (other_idx != idx).then_some(other_freq_hz)
                })
                .collect::<Vec<_>>();
            for &other_freq_hz in &occupied {
                min_spacing_cents =
                    min_spacing_cents.min(1200.0 * (freq_hz / other_freq_hz).log2().abs());
            }
            assert!(
                e6_respects_min_spacing_cents(freq_hz, &occupied, E6B_FUSION_MIN_SEPARATION_CENTS,),
                "E6b endpoint should avoid near-unison collisions while allowing soft fusion; min spacing {:.3} cents; freqs {:?}",
                min_spacing_cents,
                final_snapshot.freqs_hz
            );
        }
    }

    #[test]
    fn e4_mirror_weight_changes_consonance_landscape() {
        let cfg = E4SimConfig::test_defaults();
        let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
        let mut params_m0 = make_landscape_params(&space, cfg.fs, 1.0);
        let mut params_m1 = make_landscape_params(&space, cfg.fs, 1.0);
        params_m0.harmonicity_kernel.params.mirror_weight = 0.0;
        params_m1.harmonicity_kernel.params.mirror_weight = 1.0;

        let landscape_m0 = build_anchor_landscape(
            &space,
            &params_m0,
            cfg.anchor_hz,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        let landscape_m1 = build_anchor_landscape(
            &space,
            &params_m1,
            cfg.anchor_hz,
            cfg.env_partials,
            cfg.env_partial_decay,
        );

        let diff_sum: f32 = landscape_m0
            .consonance_field_level
            .iter()
            .zip(landscape_m1.consonance_field_level.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff_sum > 1e-4,
            "expected mirror weight to change consonance landscape, diff_sum={diff_sum:.6}"
        );
    }

    #[test]
    fn e4_asymmetric_env_mirror_changes_harmonicity_shape() {
        let cfg = E4SimConfig::test_defaults();
        let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
        let mut params_m0 = make_landscape_params(&space, cfg.fs, 1.0);
        let mut params_m1 = make_landscape_params(&space, cfg.fs, 1.0);
        params_m0.harmonicity_kernel.params.mirror_weight = 0.0;
        params_m1.harmonicity_kernel.params.mirror_weight = 1.0;

        let mut env_scan = vec![0.0f32; space.n_bins()];
        add_harmonic_partials_to_env(
            &space,
            &mut env_scan,
            cfg.anchor_hz,
            1.0,
            cfg.env_partials,
            cfg.env_partial_decay,
        );
        add_harmonic_partials_to_env(
            &space,
            &mut env_scan,
            cfg.anchor_hz * 1.5,
            0.7,
            cfg.env_partials,
            cfg.env_partial_decay,
        );

        let h_m0 = params_m0
            .harmonicity_kernel
            .potential_h_dual_from_log2_spectrum(&env_scan, &space)
            .blended;
        let h_m1 = params_m1
            .harmonicity_kernel
            .potential_h_dual_from_log2_spectrum(&env_scan, &space)
            .blended;

        let dot = h_m0
            .iter()
            .zip(h_m1.iter())
            .map(|(a, b)| *a * *b)
            .sum::<f32>();
        let n0 = h_m0.iter().map(|x| x * x).sum::<f32>().sqrt();
        let n1 = h_m1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine = dot / (n0 * n1 + 1e-9);

        assert!(
            cosine.is_finite(),
            "cosine similarity must be finite: {cosine}"
        );
        assert!(
            cosine < 0.99999,
            "mirror_weight regression: harmonicity shapes are too similar (cos={cosine})"
        );
    }

    #[test]
    fn e4_tail_samples_landscape_metrics_are_aligned_and_finite() {
        let tail_window = 16u32;
        let samples = run_e4_condition_tail_samples(0.5, 0xE4AA_55CC, tail_window);
        assert_eq!(samples.tail_window, tail_window);
        assert_eq!(
            samples.landscape_metrics_by_step.len(),
            samples.agent_freqs_by_step.len(),
            "landscape/agent tail lengths must match"
        );
        assert_eq!(
            samples.landscape_metrics_by_step.len(),
            tail_window as usize,
            "tail landscape length must equal tail_window"
        );
        for (i, m) in samples.landscape_metrics_by_step.iter().enumerate() {
            assert!(
                m.root_affinity.is_finite(),
                "root_affinity not finite at tail index {i}"
            );
            assert!(
                m.overtone_affinity.is_finite(),
                "overtone_affinity not finite at tail index {i}"
            );
            assert!(
                m.binding_strength.is_finite(),
                "binding_strength not finite at tail index {i}"
            );
            assert!(
                m.harmonic_tilt.is_finite(),
                "harmonic_tilt not finite at tail index {i}"
            );
        }
    }

    #[test]
    fn harmonic_env_produces_positive_tilt_vs_pure_tone() {
        let cfg = E4SimConfig::test_defaults();
        let space = Log2Space::new(cfg.fmin, cfg.fmax, cfg.bins_per_oct);
        let params = make_landscape_params(&space, cfg.fs, 1.0);

        let mut pure_env = vec![0.0f32; space.n_bins()];
        let pure_idx = space
            .index_of_freq(cfg.anchor_hz)
            .unwrap_or(space.n_bins() / 2);
        pure_env[pure_idx] = 1.0;

        let mut harmonic_env = vec![0.0f32; space.n_bins()];
        add_harmonic_partials_to_env(
            &space,
            &mut harmonic_env,
            cfg.anchor_hz,
            1.0,
            cfg.env_partials,
            cfg.env_partial_decay,
        );

        let pure_tilt = params
            .harmonicity_kernel
            .potential_h_dual_from_log2_spectrum(&pure_env, &space)
            .metrics
            .harmonic_tilt;
        let harmonic_tilt = params
            .harmonicity_kernel
            .potential_h_dual_from_log2_spectrum(&harmonic_env, &space)
            .metrics
            .harmonic_tilt;

        assert!(
            pure_tilt.is_finite() && harmonic_tilt.is_finite(),
            "tilt must stay finite (pure={pure_tilt}, harmonic={harmonic_tilt})"
        );
        assert!(
            pure_tilt.abs() < 0.05,
            "pure-tone tilt should stay near zero, got {pure_tilt}"
        );
        assert!(
            harmonic_tilt > 0.02,
            "harmonic env should show positive tilt, got {harmonic_tilt}"
        );
    }

    #[test]
    fn e3_audio_render_smoke_test() {
        let tmp = std::env::temp_dir().join("conc_e3_smoke.wav");
        let cfg = E3AudioConfig {
            seed: 99,
            pop_size: 4,
            duration_sec: 0.5,
            condition: E3AudioCondition::Shared,
            anchor_hz: E4_ANCHOR_HZ,
            theta_freq_hz: 2.0,
        };
        render_e3_audio(&cfg, &tmp).expect("e3 audio render should succeed");
        let meta = std::fs::metadata(&tmp).expect("wav should exist");
        assert!(meta.len() > 1000, "wav should not be trivially small");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn e3_audio_conditions_differ() {
        let tmp_shared = std::env::temp_dir().join("conc_e3_diff_shared.wav");
        let tmp_off = std::env::temp_dir().join("conc_e3_diff_off.wav");
        let shared = E3AudioConfig {
            seed: 7,
            pop_size: 6,
            duration_sec: 2.0,
            condition: E3AudioCondition::Shared,
            anchor_hz: E4_ANCHOR_HZ,
            theta_freq_hz: 2.0,
        };
        let off = E3AudioConfig {
            condition: E3AudioCondition::Off,
            ..shared.clone()
        };
        render_e3_audio(&shared, &tmp_shared).unwrap();
        render_e3_audio(&off, &tmp_off).unwrap();
        let bytes_shared = std::fs::read(&tmp_shared).unwrap();
        let bytes_off = std::fs::read(&tmp_off).unwrap();
        assert_eq!(
            bytes_shared.len(),
            bytes_off.len(),
            "same duration should produce same file length"
        );
        assert_ne!(
            bytes_shared, bytes_off,
            "shared and off scaffold conditions should produce different audio"
        );
        let _ = std::fs::remove_file(&tmp_shared);
        let _ = std::fs::remove_file(&tmp_off);
    }

    #[test]
    fn e3_rhai_zero_pop_returns_invalid_input() {
        let tmp = std::env::temp_dir().join("conc_e3_zero_pop.rhai");
        let cfg = E3AudioConfig {
            seed: 1,
            pop_size: 0,
            duration_sec: 1.0,
            condition: E3AudioCondition::Shared,
            anchor_hz: E4_ANCHOR_HZ,
            theta_freq_hz: 2.0,
        };
        let err = generate_e3_rhai(&cfg, &tmp).expect_err("zero-pop should be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn e3_rhai_uses_temporal_scaffold_labels() {
        let tmp = std::env::temp_dir().join("conc_e3_scrambled.rhai");
        let cfg = E3AudioConfig {
            seed: 11,
            pop_size: 5,
            duration_sec: 1.0,
            condition: E3AudioCondition::Scrambled,
            anchor_hz: E4_ANCHOR_HZ,
            theta_freq_hz: 2.0,
        };
        generate_e3_rhai(&cfg, &tmp).expect("scrambled rhai generation should succeed");
        let script = std::fs::read_to_string(&tmp).expect("rhai script should exist");
        assert!(script.contains("Temporal scaffold assay"));
        assert!(script.contains("condition=scrambled"));
        assert!(script.contains("cycle-wise phase resets"));
        assert!(!script.contains("Consonance-gated entrainment"));
        let _ = std::fs::remove_file(&tmp);
    }
}
