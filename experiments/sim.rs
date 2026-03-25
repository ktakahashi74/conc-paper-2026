use conchordal::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use conchordal::core::erb::hz_to_erb;
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{Landscape, LandscapeParams, RoughnessScalarMode};
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
pub const E4_ANCHOR_HZ: f32 = 220.0;
#[allow(dead_code)]
pub const E4_WINDOW_CENTS: f32 = 50.0;

pub(crate) const E4_ENV_PARTIALS_DEFAULT: u32 = 6;
pub(crate) const E4_ENV_PARTIAL_DECAY_DEFAULT: f32 = 1.0;

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
#[cfg(test)]
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
pub const E6B_DEFAULT_RANGE_OCT: f32 = 4.0;
pub const E6B_DEFAULT_POP_SIZE: usize = 16;
const E6B_FUSION_MIN_SEPARATION_CENTS: f32 = 10.0;
const E6B_SELECTION_CROWDING_WEIGHT: f32 = 0.005;
const E6B_LOCAL_CAPACITY_RADIUS_CENTS: f32 = 35.0;
const E6B_LOCAL_CAPACITY_FREE_VOICES: usize = 3;
const E6B_LOCAL_CAPACITY_WEIGHT: f32 = 0.07;
const E6B_PARENT_SHARE_WEIGHT: f32 = 1.0;
const E6B_PARENT_ENERGY_WEIGHT: f32 = 0.25;
const E6B_SURVIVAL_SCORE_LOW: f32 = 0.30;
const E6B_SURVIVAL_SCORE_HIGH: f32 = 0.80;
const E6B_SURVIVAL_RECHARGE_PER_SEC: f32 = 0.20;
const E6B_BACKGROUND_DEATH_RATE_PER_SEC: f32 = 0.03;
const E6B_PARENT_SELECTION_ENERGY_CAP: f32 = 1.0;
const E6B_RESPAWN_PARENT_PRIOR_MIX: f32 = 0.15;
const E6B_RESPAWN_SAME_BAND_DISCOUNT: f32 = 0.08;
const E6B_RESPAWN_OCTAVE_DISCOUNT: f32 = 0.20;
const E6B_RESPAWN_OCTAVE_WINDOW_CENTS: f32 = 35.0;
const E6B_PARENT_PROPOSAL_KIND: E6ParentProposalKind = E6ParentProposalKind::ContextualC;
const E6B_PARENT_PROPOSAL_SIGMA_ST: f32 = 9.0;
const E6B_PARENT_PROPOSAL_UNISON_NOTCH_GAIN: f32 = 0.95;
const E6B_PARENT_PROPOSAL_UNISON_NOTCH_SIGMA_ST: f32 = 0.35;
const E6B_PARENT_PROPOSAL_CANDIDATE_COUNT: usize = 16;
const E6B_PARENT_PEAK_SCENE_SCORE_EXP: f32 = 0.35;
const E6B_JUVENILE_TUNING_RADIUS_ST: f32 = 0.20;
const E6B_JUVENILE_TUNING_GRID_ST: f32 = 0.10;
const E6B_JUVENILE_TUNING_TICKS: u32 = 2;
const E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST: f32 = 0.50;
const E6B_AZIMUTH_LOCAL_SEARCH_GRID_ST: f32 = 0.05;

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
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6SelectionScoreMode {
    PolyphonicLooCrowding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6RespawnMode {
    LegacyFamilyAzimuth,
    ParentPeakProposal,
    ScenePeakMatchedRandom,
    LogRandomFilteredCandidates,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6ParentProposalKind {
    ContextualC,
    HOnly,
}

impl E6ParentProposalKind {
    pub fn label(self) -> &'static str {
        match self {
            E6ParentProposalKind::ContextualC => "contextual_c",
            E6ParentProposalKind::HOnly => "h_only",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6bRandomBaselineMode {
    LogRandomFiltered,
    MatchedScenePeaks,
    HardRandomLog,
}

impl E6bRandomBaselineMode {
    pub fn label(self) -> &'static str {
        match self {
            E6bRandomBaselineMode::LogRandomFiltered => "log_random_filtered",
            E6bRandomBaselineMode::MatchedScenePeaks => "matched_scene_peaks",
            E6bRandomBaselineMode::HardRandomLog => "hard_random_log",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6bAzimuthMode {
    Inherit,
    LocalSearch,
}

impl E6bAzimuthMode {
    pub fn label(self) -> &'static str {
        match self {
            E6bAzimuthMode::Inherit => "inherit",
            E6bAzimuthMode::LocalSearch => "search",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6DeathCause {
    EnergyExhaustion,
    BackgroundTurnover,
    JuvenileCull,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum E6StopReason {
    MinDeaths,
    StepsCap,
}

impl E6StopReason {
    pub fn label(self) -> &'static str {
        match self {
            E6StopReason::MinDeaths => "min_deaths",
            E6StopReason::StepsCap => "steps_cap",
        }
    }
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
    pub polyphonic_overcapacity_weight_override: Option<f32>,
    pub polyphonic_capacity_radius_cents_override: Option<f32>,
    pub polyphonic_capacity_free_voices_override: Option<usize>,
    pub polyphonic_parent_share_weight_override: Option<f32>,
    pub polyphonic_parent_energy_weight_override: Option<f32>,
    pub juvenile_contextual_tuning_ticks_override: Option<u32>,
    pub juvenile_contextual_settlement_enabled: bool,
    pub juvenile_cull_enabled: bool,
    pub record_life_diagnostics: bool,
    pub survival_score_low_override: Option<f32>,
    pub survival_score_high_override: Option<f32>,
    pub survival_recharge_per_sec_override: Option<f32>,
    pub background_death_rate_per_sec_override: Option<f32>,
    pub respawn_parent_prior_mix_override: Option<f32>,
    pub respawn_same_band_discount_override: Option<f32>,
    pub respawn_octave_discount_override: Option<f32>,
    pub parent_proposal_kind_override: Option<E6ParentProposalKind>,
    pub parent_proposal_sigma_st_override: Option<f32>,
    pub parent_proposal_unison_notch_gain_override: Option<f32>,
    pub parent_proposal_unison_notch_sigma_st_override: Option<f32>,
    pub parent_proposal_candidate_count_override: Option<usize>,
    pub azimuth_mode_override: Option<E6bAzimuthMode>,
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
}

#[derive(Clone, Debug)]
pub struct E6RunResult {
    pub deaths: Vec<E3DeathRecord>,
    pub life_checks: Vec<E6LifeCheckRecord>,
    pub snapshots: Vec<E6PitchSnapshot>,
    pub respawns: Vec<E6RespawnRecord>,
    pub total_deaths: usize,
    pub stop_reason: E6StopReason,
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
    pub polyphonic_crowding_weight_override: Option<f32>,
    pub polyphonic_overcapacity_weight_override: Option<f32>,
    pub polyphonic_capacity_radius_cents_override: Option<f32>,
    pub polyphonic_capacity_free_voices_override: Option<usize>,
    pub polyphonic_parent_share_weight_override: Option<f32>,
    pub polyphonic_parent_energy_weight_override: Option<f32>,
    pub juvenile_contextual_tuning_ticks_override: Option<u32>,
    pub juvenile_contextual_settlement_enabled_override: Option<bool>,
    pub survival_score_low_override: Option<f32>,
    pub survival_score_high_override: Option<f32>,
    pub survival_recharge_per_sec_override: Option<f32>,
    pub background_death_rate_per_sec_override: Option<f32>,
    pub respawn_parent_prior_mix_override: Option<f32>,
    pub respawn_same_band_discount_override: Option<f32>,
    pub respawn_octave_discount_override: Option<f32>,
    pub parent_proposal_kind_override: Option<E6ParentProposalKind>,
    pub parent_proposal_sigma_st_override: Option<f32>,
    pub parent_proposal_unison_notch_gain_override: Option<f32>,
    pub parent_proposal_unison_notch_sigma_st_override: Option<f32>,
    pub parent_proposal_candidate_count_override: Option<usize>,
    pub azimuth_mode_override: Option<E6bAzimuthMode>,
    pub random_baseline_mode: E6bRandomBaselineMode,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E6bRunResult {
    pub deaths: Vec<E3DeathRecord>,
    pub life_checks: Vec<E6LifeCheckRecord>,
    pub snapshots: Vec<E6PitchSnapshot>,
    pub respawns: Vec<E6RespawnRecord>,
    pub total_deaths: usize,
    pub stop_reason: E6StopReason,
}

impl From<E6RunResult> for E6bRunResult {
    fn from(value: E6RunResult) -> Self {
        Self {
            deaths: value.deaths,
            life_checks: value.life_checks,
            snapshots: value.snapshots,
            respawns: value.respawns,
            total_deaths: value.total_deaths,
            stop_reason: value.stop_reason,
        }
    }
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct E6RespawnRecord {
    pub step: usize,
    pub dead_agent_id: usize,
    pub dead_freq_hz: Option<f32>,
    pub death_cause: Option<E6DeathCause>,
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
    pub chosen_band_occupancy: Option<usize>,
    pub offspring_freq_hz: f32,
    pub spawn_freq_hz: f32,
    pub spawn_c_level: f32,
    pub parent_family_center_hz: Option<f32>,
    pub parent_azimuth_st: Option<f32>,
    pub child_family_center_hz: Option<f32>,
    pub child_azimuth_st: Option<f32>,
    pub family_inherited: Option<bool>,
    pub family_mutated: Option<bool>,
    pub proposal_rank: Option<usize>,
    pub proposal_mass: Option<f32>,
    pub proposal_filter_gain: Option<f32>,
    pub local_opt_delta_st: Option<f32>,
    pub local_opt_score_gap: Option<f32>,
    pub oracle_applied: bool,
    pub oracle_freq_hz: Option<f32>,
    pub oracle_c_score: Option<f32>,
    pub oracle_c_level: Option<f32>,
    pub oracle_delta_st: Option<f32>,
}

#[derive(Clone, Debug)]
#[allow(dead_code)]
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
    pending_death_cause: Option<E6DeathCause>,
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
            pending_death_cause: None,
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
        self.pending_death_cause = None;
        self.was_alive = false;
    }
}

#[derive(Clone, Copy, Debug)]
struct E6PendingRespawn {
    id: u64,
    cause: E6DeathCause,
    dead_freq_hz: Option<f32>,
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
        stop_reason: E6StopReason::StepsCap,
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
    let (
        polyphonic_overcapacity_weight,
        polyphonic_overcapacity_radius_cents,
        polyphonic_overcapacity_free_voices,
    ) = e6_resolved_polyphonic_capacity(cfg);
    let polyphonic_parent_share_weight = cfg
        .polyphonic_parent_share_weight_override
        .unwrap_or(E6B_PARENT_SHARE_WEIGHT)
        .clamp(0.0, 1.0);
    let polyphonic_parent_energy_weight = cfg
        .polyphonic_parent_energy_weight_override
        .unwrap_or(E6B_PARENT_ENERGY_WEIGHT)
        .clamp(0.0, 1.0);
    let juvenile_contextual_tuning_ticks = cfg
        .juvenile_contextual_tuning_ticks_override
        .unwrap_or(E6_JUVENILE_CONTEXTUAL_TUNING_TICKS)
        .max(1);
    let parent_proposal_kind = cfg
        .parent_proposal_kind_override
        .unwrap_or(E6B_PARENT_PROPOSAL_KIND);
    let parent_proposal_sigma_st = cfg
        .parent_proposal_sigma_st_override
        .unwrap_or(E6B_PARENT_PROPOSAL_SIGMA_ST)
        .max(0.25);
    let parent_proposal_unison_notch_gain = cfg
        .parent_proposal_unison_notch_gain_override
        .unwrap_or(E6B_PARENT_PROPOSAL_UNISON_NOTCH_GAIN)
        .clamp(0.0, 1.0);
    let parent_proposal_unison_notch_sigma_st = cfg
        .parent_proposal_unison_notch_sigma_st_override
        .unwrap_or(E6B_PARENT_PROPOSAL_UNISON_NOTCH_SIGMA_ST)
        .max(0.05);
    let parent_proposal_candidate_count = cfg
        .parent_proposal_candidate_count_override
        .unwrap_or(E6B_PARENT_PROPOSAL_CANDIDATE_COUNT)
        .max(1);
    let azimuth_mode = cfg.azimuth_mode_override.unwrap_or(E6bAzimuthMode::Inherit);
    let (juvenile_polyphonic_tuning_radius_st, juvenile_polyphonic_tuning_grid_st) = if matches!(
        respawn_mode,
        E6RespawnMode::ParentPeakProposal | E6RespawnMode::ScenePeakMatchedRandom
    ) {
        (E6B_JUVENILE_TUNING_RADIUS_ST, E6B_JUVENILE_TUNING_GRID_ST)
    } else {
        (
            E6_JUVENILE_CONTEXTUAL_TUNING_RADIUS_ST,
            E6_JUVENILE_CONTEXTUAL_TUNING_GRID_ST,
        )
    };
    let survival_score_low = cfg.survival_score_low_override;
    let survival_score_high = cfg.survival_score_high_override;
    let survival_recharge_rate_per_sec = cfg
        .survival_recharge_per_sec_override
        .unwrap_or(E6_SELECTION_RECHARGE_PER_SEC)
        .max(0.0);
    let background_death_rate_per_sec = cfg
        .background_death_rate_per_sec_override
        .unwrap_or(0.0)
        .max(0.0);
    let _respawn_parent_prior_mix = cfg
        .respawn_parent_prior_mix_override
        .unwrap_or(E6B_RESPAWN_PARENT_PRIOR_MIX)
        .clamp(0.0, 1.0);
    let respawn_same_band_discount = cfg
        .respawn_same_band_discount_override
        .unwrap_or(E6B_RESPAWN_SAME_BAND_DISCOUNT)
        .clamp(0.0, 1.0);
    let respawn_octave_discount = cfg
        .respawn_octave_discount_override
        .unwrap_or(E6B_RESPAWN_OCTAVE_DISCOUNT)
        .clamp(0.0, 1.0);
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
                juvenile_contextual_tuning_ticks,
                juvenile_polyphonic_tuning_radius_st,
                juvenile_polyphonic_tuning_grid_st,
                polyphonic_crowding_weight,
                polyphonic_overcapacity_weight,
                polyphonic_overcapacity_radius_cents,
                polyphonic_overcapacity_free_voices,
            );
        }
        for agent_idx in 0..pop.individuals.len() {
            if !cfg.selection_enabled || !pop.individuals[agent_idx].is_alive() {
                continue;
            }
            let freq_hz = pop.individuals[agent_idx].body.base_freq_hz();
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
            let c_score = e6_polyphonic_selection_score(
                &loo_scene_landscape,
                freq_hz,
                &other_freqs_hz,
                polyphonic_crowding_weight,
                polyphonic_overcapacity_weight,
                polyphonic_overcapacity_radius_cents,
                polyphonic_overcapacity_free_voices,
            );
            let agent = &mut pop.individuals[agent_idx];
            let recharge_score =
                e6_survival_signal(c_score, survival_score_low, survival_score_high);
            apply_selection_recharge(agent, recharge_score, survival_recharge_rate_per_sec, dt);
        }
        if background_death_rate_per_sec > 0.0 {
            let hazard = (background_death_rate_per_sec * dt.max(0.0)).clamp(0.0, 1.0);
            if hazard > 0.0 {
                for agent in pop.individuals.iter_mut() {
                    if !agent.is_alive() {
                        continue;
                    }
                    let idx = agent.id() as usize;
                    if idx >= states.len() {
                        continue;
                    }
                    let state = &mut states[idx];
                    if state.pending_death_cause.is_some()
                        || state.ticks < juvenile_contextual_tuning_ticks
                    {
                        continue;
                    }
                    if rng.random::<f32>() < hazard {
                        state.pending_death_cause = Some(E6DeathCause::BackgroundTurnover);
                        agent.start_remove_fade(0.0);
                    }
                }
            }
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
                    let score = e6_polyphonic_selection_score(
                        &loo_scene_landscape,
                        freq_hz,
                        &other_freqs_hz,
                        polyphonic_crowding_weight,
                        polyphonic_overcapacity_weight,
                        polyphonic_overcapacity_radius_cents,
                        polyphonic_overcapacity_free_voices,
                    );
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
        let mut respawn_events: Vec<E6PendingRespawn> = Vec::new();
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
                        state.pending_death_cause = Some(E6DeathCause::JuvenileCull);
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

                let cause = state
                    .pending_death_cause
                    .take()
                    .unwrap_or(E6DeathCause::EnergyExhaustion);
                let dead_freq_hz = {
                    let freq_hz = agent.body.base_freq_hz();
                    (freq_hz.is_finite() && freq_hz > 0.0).then_some(freq_hz)
                };
                respawn_events.push(E6PendingRespawn {
                    id,
                    cause,
                    dead_freq_hz,
                });
                state.reset_for_new_life(next_life_id);
                next_life_id += 1;
            } else {
                state.was_alive = alive;
            }
        }

        let respawn_set: HashSet<u64> = respawn_events.iter().map(|event| event.id).collect();
        for respawn_event in respawn_events {
            let id = respawn_event.id;
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
                || matches!(
                    respawn_mode,
                    E6RespawnMode::ParentPeakProposal
                        | E6RespawnMode::ScenePeakMatchedRandom
                        | E6RespawnMode::LogRandomFilteredCandidates
                ) {
                None
            } else {
                Some(E6_POST_RESPAWN_AZIMUTH_TUNING_RADIUS_ST)
            };

            let offspring_strategy = match cfg.condition {
                E6Condition::Random => {
                    let random_structured_sample = if matches!(
                        respawn_mode,
                        E6RespawnMode::ScenePeakMatchedRandom
                            | E6RespawnMode::LogRandomFilteredCandidates
                    ) {
                        let alive_freqs_hz = collect_alive_freqs(&pop);
                        if matches!(respawn_mode, E6RespawnMode::ScenePeakMatchedRandom) {
                            let current_landscape = build_e6_scene_landscape_with_anchor(
                                &space,
                                &params,
                                &selection_reference_landscape,
                                &alive_freqs_hz,
                                partials,
                                E4_ENV_PARTIAL_DECAY_DEFAULT,
                            );
                            Some(sample_from_scene_peak_matched_random(
                                &current_landscape,
                                &alive_freqs_hz,
                                tessitura_min_hz,
                                tessitura_max_hz,
                                &mut rng,
                                polyphonic_crowding_weight,
                                polyphonic_overcapacity_weight,
                                polyphonic_overcapacity_radius_cents,
                                polyphonic_overcapacity_free_voices,
                                parent_proposal_candidate_count,
                                cfg.legacy_family_min_spacing_cents,
                            ))
                        } else {
                            let current_landscape = build_e6_scene_landscape_with_anchor(
                                &space,
                                &params,
                                &selection_reference_landscape,
                                &alive_freqs_hz,
                                partials,
                                E4_ENV_PARTIAL_DECAY_DEFAULT,
                            );
                            Some(sample_from_log_random_filtered_candidates(
                                &current_landscape,
                                &alive_freqs_hz,
                                tessitura_min_hz,
                                tessitura_max_hz,
                                &mut rng,
                                polyphonic_crowding_weight,
                                polyphonic_overcapacity_weight,
                                polyphonic_overcapacity_radius_cents,
                                polyphonic_overcapacity_free_voices,
                                parent_proposal_candidate_count,
                                cfg.legacy_family_min_spacing_cents,
                            ))
                        }
                    } else {
                        None
                    };
                    let offspring_freq = random_structured_sample
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
                            dead_freq_hz: respawn_event.dead_freq_hz,
                            death_cause: Some(respawn_event.cause),
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
                            chosen_band_occupancy: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level: oracle_c_level,
                            parent_family_center_hz: random_structured_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: random_structured_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: random_structured_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: random_structured_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            proposal_rank: random_structured_sample
                                .and_then(|sample| sample.proposal_rank),
                            proposal_mass: random_structured_sample
                                .and_then(|sample| sample.proposal_mass),
                            proposal_filter_gain: random_structured_sample
                                .and_then(|sample| sample.proposal_filter_gain),
                            local_opt_delta_st: None,
                            local_opt_score_gap: None,
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
                            dead_freq_hz: respawn_event.dead_freq_hz,
                            death_cause: Some(respawn_event.cause),
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
                            chosen_band_occupancy: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level: oracle_c_level,
                            parent_family_center_hz: random_structured_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: random_structured_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: random_structured_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: random_structured_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            proposal_rank: random_structured_sample
                                .and_then(|sample| sample.proposal_rank),
                            proposal_mass: random_structured_sample
                                .and_then(|sample| sample.proposal_mass),
                            proposal_filter_gain: random_structured_sample
                                .and_then(|sample| sample.proposal_filter_gain),
                            local_opt_delta_st: None,
                            local_opt_score_gap: None,
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
                            dead_freq_hz: respawn_event.dead_freq_hz,
                            death_cause: Some(respawn_event.cause),
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
                            chosen_band_occupancy: None,
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level,
                            parent_family_center_hz: random_structured_sample
                                .map(|sample| sample.parent_family_center_hz),
                            parent_azimuth_st: random_structured_sample
                                .map(|sample| sample.parent_azimuth_st),
                            child_family_center_hz: random_structured_sample
                                .map(|sample| sample.child_family_center_hz),
                            child_azimuth_st: random_structured_sample
                                .map(|sample| sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            proposal_rank: random_structured_sample
                                .and_then(|sample| sample.proposal_rank),
                            proposal_mass: random_structured_sample
                                .and_then(|sample| sample.proposal_mass),
                            proposal_filter_gain: random_structured_sample
                                .and_then(|sample| sample.proposal_filter_gain),
                            local_opt_delta_st: None,
                            local_opt_score_gap: None,
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
                    if let Some(sample) = random_structured_sample {
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
                            dead_freq_hz: respawn_event.dead_freq_hz,
                            death_cause: Some(respawn_event.cause),
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
                            chosen_band_occupancy: Some(sample.chosen_band_occupancy),
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: offspring_freq,
                            spawn_c_level: selection_reference_landscape
                                .evaluate_pitch_level(offspring_freq)
                                .clamp(0.0, 1.0),
                            parent_family_center_hz: Some(sample.parent_family_center_hz),
                            parent_azimuth_st: Some(sample.parent_azimuth_st),
                            child_family_center_hz: Some(sample.child_family_center_hz),
                            child_azimuth_st: Some(sample.child_azimuth_st),
                            family_inherited: None,
                            family_mutated: None,
                            proposal_rank: sample.proposal_rank,
                            proposal_mass: sample.proposal_mass,
                            proposal_filter_gain: sample.proposal_filter_gain,
                            local_opt_delta_st: None,
                            local_opt_score_gap: None,
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
                    let alive_freqs_hz = collect_alive_freqs(&pop);
                    let parent_selection_scene_landscape = if cfg.selection_enabled
                        && matches!(
                            selection_score_mode,
                            E6SelectionScoreMode::PolyphonicLooCrowding
                        ) {
                        Some(build_e6_scene_landscape_with_anchor(
                            &space,
                            &params,
                            &selection_reference_landscape,
                            &alive_freqs_hz,
                            partials,
                            E4_ENV_PARTIAL_DECAY_DEFAULT,
                        ))
                    } else {
                        None
                    };
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
                            let fertility_energy =
                                if survival_score_low.is_some() || survival_score_high.is_some() {
                                    energy.min(E6B_PARENT_SELECTION_ENERGY_CAP)
                                } else {
                                    energy
                                };
                            let parent_idx = a.id() as usize;
                            let weight = if !cfg.selection_enabled {
                                1.0
                            } else {
                                let other_freqs_hz = pop
                                    .individuals
                                    .iter()
                                    .filter(|other| {
                                        other.is_alive()
                                            && other.id() != a.id()
                                            && !respawn_set.contains(&other.id())
                                    })
                                    .filter_map(|other| {
                                        let other_freq = other.body.base_freq_hz();
                                        (other_freq.is_finite() && other_freq > 0.0)
                                            .then_some(other_freq)
                                    })
                                    .collect::<Vec<_>>();
                                let loo_scene_landscape = build_e6_contextual_loo_landscape(
                                    &space,
                                    &params,
                                    parent_selection_scene_landscape
                                        .as_ref()
                                        .expect("polyphonic parent scene landscape"),
                                    freq,
                                    partials,
                                    E4_ENV_PARTIAL_DECAY_DEFAULT,
                                );
                                let score = e6_polyphonic_selection_score(
                                    &loo_scene_landscape,
                                    freq,
                                    &other_freqs_hz,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                    polyphonic_overcapacity_radius_cents,
                                    polyphonic_overcapacity_free_voices,
                                );
                                let local_occupancy = e6_local_band_occupancy(
                                    freq,
                                    &other_freqs_hz,
                                    polyphonic_overcapacity_radius_cents,
                                );
                                e6_parent_selection_weight_from_polyphonic_score(
                                    fertility_energy,
                                    score,
                                    local_occupancy,
                                    polyphonic_overcapacity_free_voices,
                                    polyphonic_parent_share_weight,
                                    polyphonic_parent_energy_weight,
                                )
                            };
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
                            E6RespawnMode::ParentPeakProposal => {
                                sample_from_scene_peak_parent_bias(
                                    &parent_landscape,
                                    &landscape,
                                    &alive_freqs_hz,
                                    parent_freq,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                    &mut rng,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                    polyphonic_overcapacity_radius_cents,
                                    polyphonic_overcapacity_free_voices,
                                    parent_proposal_kind,
                                    parent_proposal_sigma_st,
                                    parent_proposal_unison_notch_gain,
                                    parent_proposal_unison_notch_sigma_st,
                                    parent_proposal_candidate_count,
                                    respawn_same_band_discount,
                                    respawn_octave_discount,
                                    cfg.legacy_family_min_spacing_cents,
                                    azimuth_mode,
                                )
                            }
                            E6RespawnMode::ScenePeakMatchedRandom => {
                                sample_from_scene_peak_matched_random(
                                    &landscape,
                                    &alive_freqs_hz,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                    &mut rng,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                    polyphonic_overcapacity_radius_cents,
                                    polyphonic_overcapacity_free_voices,
                                    parent_proposal_candidate_count,
                                    cfg.legacy_family_min_spacing_cents,
                                )
                            }
                            E6RespawnMode::LogRandomFilteredCandidates => {
                                sample_from_log_random_filtered_candidates(
                                    &landscape,
                                    &alive_freqs_hz,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                    &mut rng,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                    polyphonic_overcapacity_radius_cents,
                                    polyphonic_overcapacity_free_voices,
                                    parent_proposal_candidate_count,
                                    cfg.legacy_family_min_spacing_cents,
                                )
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
                        let (local_opt_delta_st, local_opt_score_gap) =
                            if matches!(respawn_mode, E6RespawnMode::ParentPeakProposal) {
                                let (delta_st, score_gap) = e6_family_local_opt_diagnostics(
                                    &landscape,
                                    heredity_sample.child_family_center_hz,
                                    spawn_freq,
                                    &alive_freqs_hz,
                                    tessitura_min_hz,
                                    tessitura_max_hz,
                                    polyphonic_crowding_weight,
                                    polyphonic_overcapacity_weight,
                                    polyphonic_overcapacity_radius_cents,
                                    polyphonic_overcapacity_free_voices,
                                    cfg.legacy_family_min_spacing_cents,
                                );
                                (Some(delta_st), Some(score_gap))
                            } else {
                                (None, None)
                            };
                        let oracle_delta_st = oracle_c_level
                            .map(|_| 12.0 * (spawn_freq / offspring_freq.max(1e-6)).log2());
                        out.respawns.push(E6RespawnRecord {
                            step,
                            dead_agent_id: id as usize,
                            dead_freq_hz: respawn_event.dead_freq_hz,
                            death_cause: Some(respawn_event.cause),
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
                            chosen_band_occupancy: Some(heredity_sample.chosen_band_occupancy),
                            offspring_freq_hz: offspring_freq,
                            spawn_freq_hz: spawn_freq,
                            spawn_c_level,
                            parent_family_center_hz: Some(heredity_sample.parent_family_center_hz),
                            parent_azimuth_st: Some(heredity_sample.parent_azimuth_st),
                            child_family_center_hz: Some(heredity_sample.child_family_center_hz),
                            child_azimuth_st: Some(heredity_sample.child_azimuth_st),
                            family_inherited: Some(heredity_sample.family_inherited),
                            family_mutated: Some(heredity_sample.family_mutated),
                            proposal_rank: heredity_sample.proposal_rank,
                            proposal_mass: heredity_sample.proposal_mass,
                            proposal_filter_gain: heredity_sample.proposal_filter_gain,
                            local_opt_delta_st,
                            local_opt_score_gap,
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
                dead_freq_hz: respawn_event.dead_freq_hz,
                death_cause: Some(respawn_event.cause),
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
                chosen_band_occupancy: None,
                offspring_freq_hz: spawn_freq,
                spawn_freq_hz: spawn_freq,
                spawn_c_level,
                parent_family_center_hz: None,
                parent_azimuth_st: None,
                child_family_center_hz: None,
                child_azimuth_st: None,
                family_inherited: None,
                family_mutated: None,
                proposal_rank: None,
                proposal_mass: None,
                proposal_filter_gain: None,
                local_opt_delta_st: None,
                local_opt_score_gap: None,
                oracle_applied: false,
                oracle_freq_hz: None,
                oracle_c_score: None,
                oracle_c_level: None,
                oracle_delta_st: None,
            });
        }

        landscape.rhythm.advance_in_place(dt);

        if out.deaths.len() >= cfg.min_deaths {
            out.stop_reason = E6StopReason::MinDeaths;
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
                        u32::MAX,
                        juvenile_polyphonic_tuning_radius_st,
                        juvenile_polyphonic_tuning_grid_st,
                        polyphonic_crowding_weight,
                        polyphonic_overcapacity_weight,
                        polyphonic_overcapacity_radius_cents,
                        polyphonic_overcapacity_free_voices,
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
    let respawn_mode = match cfg.condition {
        E6Condition::Heredity => E6RespawnMode::ParentPeakProposal,
        E6Condition::Random => match cfg.random_baseline_mode {
            E6bRandomBaselineMode::LogRandomFiltered => E6RespawnMode::LogRandomFilteredCandidates,
            E6bRandomBaselineMode::MatchedScenePeaks => E6RespawnMode::ScenePeakMatchedRandom,
            E6bRandomBaselineMode::HardRandomLog => E6RespawnMode::LegacyFamilyAzimuth,
        },
    };
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
        polyphonic_crowding_weight_override: cfg
            .polyphonic_crowding_weight_override
            .or(Some(E6B_SELECTION_CROWDING_WEIGHT)),
        polyphonic_overcapacity_weight_override: cfg
            .polyphonic_overcapacity_weight_override
            .or(Some(E6B_LOCAL_CAPACITY_WEIGHT)),
        polyphonic_capacity_radius_cents_override: cfg
            .polyphonic_capacity_radius_cents_override
            .or(Some(E6B_LOCAL_CAPACITY_RADIUS_CENTS)),
        polyphonic_capacity_free_voices_override: cfg
            .polyphonic_capacity_free_voices_override
            .or(Some(E6B_LOCAL_CAPACITY_FREE_VOICES)),
        polyphonic_parent_share_weight_override: cfg
            .polyphonic_parent_share_weight_override
            .or(Some(E6B_PARENT_SHARE_WEIGHT)),
        polyphonic_parent_energy_weight_override: cfg
            .polyphonic_parent_energy_weight_override
            .or(Some(E6B_PARENT_ENERGY_WEIGHT)),
        juvenile_contextual_tuning_ticks_override: cfg
            .juvenile_contextual_tuning_ticks_override
            .or(Some(E6B_JUVENILE_TUNING_TICKS)),
        juvenile_contextual_settlement_enabled: cfg
            .juvenile_contextual_settlement_enabled_override
            .unwrap_or(true),
        juvenile_cull_enabled: false,
        record_life_diagnostics: false,
        survival_score_low_override: cfg
            .survival_score_low_override
            .or(Some(E6B_SURVIVAL_SCORE_LOW)),
        survival_score_high_override: cfg
            .survival_score_high_override
            .or(Some(E6B_SURVIVAL_SCORE_HIGH)),
        survival_recharge_per_sec_override: cfg
            .survival_recharge_per_sec_override
            .or(Some(E6B_SURVIVAL_RECHARGE_PER_SEC)),
        background_death_rate_per_sec_override: cfg.background_death_rate_per_sec_override.or(
            Some(if cfg.selection_enabled {
                0.0
            } else {
                E6B_BACKGROUND_DEATH_RATE_PER_SEC
            }),
        ),
        respawn_parent_prior_mix_override: cfg
            .respawn_parent_prior_mix_override
            .or(Some(E6B_RESPAWN_PARENT_PRIOR_MIX)),
        respawn_same_band_discount_override: cfg
            .respawn_same_band_discount_override
            .or(Some(E6B_RESPAWN_SAME_BAND_DISCOUNT)),
        respawn_octave_discount_override: cfg
            .respawn_octave_discount_override
            .or(Some(E6B_RESPAWN_OCTAVE_DISCOUNT)),
        parent_proposal_kind_override: cfg
            .parent_proposal_kind_override
            .or(Some(E6B_PARENT_PROPOSAL_KIND)),
        parent_proposal_sigma_st_override: cfg
            .parent_proposal_sigma_st_override
            .or(Some(E6B_PARENT_PROPOSAL_SIGMA_ST)),
        parent_proposal_unison_notch_gain_override: cfg
            .parent_proposal_unison_notch_gain_override
            .or(Some(E6B_PARENT_PROPOSAL_UNISON_NOTCH_GAIN)),
        parent_proposal_unison_notch_sigma_st_override: cfg
            .parent_proposal_unison_notch_sigma_st_override
            .or(Some(E6B_PARENT_PROPOSAL_UNISON_NOTCH_SIGMA_ST)),
        parent_proposal_candidate_count_override: cfg
            .parent_proposal_candidate_count_override
            .or(Some(E6B_PARENT_PROPOSAL_CANDIDATE_COUNT)),
        azimuth_mode_override: cfg
            .azimuth_mode_override
            .or(Some(E6bAzimuthMode::LocalSearch)),
        family_occupancy_strength_override: None,
        family_nfd_mode: E6FamilyNfdMode::Off,
        respawn_mode,
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
    _nfd_mode: E6FamilyNfdMode,
    apply_nfd: bool,
) -> f32 {
    let occupancy_weight = if apply_nfd {
        family_occupancy_weight(center_freq_hz, alive_freqs_hz, occupancy_strength)
    } else {
        1.0
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
    chosen_band_occupancy: usize,
    proposal_rank: Option<usize>,
    proposal_mass: Option<f32>,
    proposal_filter_gain: Option<f32>,
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

fn e6_local_band_occupancy(freq_hz: f32, other_freqs_hz: &[f32], radius_cents: f32) -> usize {
    if !freq_hz.is_finite() || freq_hz <= 0.0 {
        return 0;
    }
    let radius_cents = radius_cents.max(0.0);
    1 + other_freqs_hz
        .iter()
        .copied()
        .filter(|other_freq_hz| other_freq_hz.is_finite() && *other_freq_hz > 0.0)
        .filter(|&other_freq_hz| {
            1200.0 * (freq_hz / other_freq_hz).log2().abs() <= radius_cents + 1e-6
        })
        .count()
}

fn e6_local_band_share(local_occupancy: usize, free_voices: usize) -> f32 {
    let local_occupancy = local_occupancy.max(1) as f32;
    let free_voices = free_voices.max(1) as f32;
    (free_voices / (free_voices + (local_occupancy - 1.0))).clamp(0.0, 1.0)
}

fn e6_same_band_distance_cents(freq_a_hz: f32, freq_b_hz: f32) -> f32 {
    if !freq_a_hz.is_finite() || freq_a_hz <= 0.0 || !freq_b_hz.is_finite() || freq_b_hz <= 0.0 {
        return f32::INFINITY;
    }
    (1200.0 * (freq_a_hz / freq_b_hz).log2()).abs()
}

fn e6_is_same_band_respawn(freq_a_hz: f32, freq_b_hz: f32, radius_cents: f32) -> bool {
    e6_same_band_distance_cents(freq_a_hz, freq_b_hz) <= radius_cents.max(0.0) + 1e-6
}

fn e6_is_parent_octave_respawn(freq_a_hz: f32, freq_b_hz: f32, window_cents: f32) -> bool {
    if !freq_a_hz.is_finite() || freq_a_hz <= 0.0 || !freq_b_hz.is_finite() || freq_b_hz <= 0.0 {
        return false;
    }
    let delta_cents = 1200.0 * (freq_a_hz / freq_b_hz).log2();
    let nearest_octave = (delta_cents / 1200.0).round();
    if nearest_octave.abs() < 0.5 {
        return false;
    }
    (delta_cents - nearest_octave * 1200.0).abs() <= window_cents.max(0.0) + 1e-6
}

fn e6_runtime_overcapacity(
    freq_hz: f32,
    other_freqs_hz: &[f32],
    radius_cents: f32,
    free_voices: usize,
) -> f32 {
    let local_occupancy = e6_local_band_occupancy(freq_hz, other_freqs_hz, radius_cents);
    local_occupancy.saturating_sub(free_voices.max(1)) as f32
}

fn e6_resolved_polyphonic_capacity(cfg: &E6RunConfig) -> (f32, f32, usize) {
    let weight = cfg
        .polyphonic_overcapacity_weight_override
        .unwrap_or(if cfg.legacy_family_min_spacing_cents.is_some() {
            E6B_LOCAL_CAPACITY_WEIGHT
        } else {
            0.0
        })
        .max(0.0);
    let radius_cents = cfg
        .polyphonic_capacity_radius_cents_override
        .unwrap_or(E6B_LOCAL_CAPACITY_RADIUS_CENTS)
        .max(0.0);
    let free_voices = cfg
        .polyphonic_capacity_free_voices_override
        .unwrap_or(E6B_LOCAL_CAPACITY_FREE_VOICES)
        .max(1);
    (weight, radius_cents, free_voices)
}

fn e6_polyphonic_selection_score(
    loo_scene_landscape: &Landscape,
    freq_hz: f32,
    other_freqs_hz: &[f32],
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
) -> f32 {
    let base_score = loo_scene_landscape.evaluate_pitch_score(freq_hz);
    base_score
        - crowding_weight.max(0.0) * e6_runtime_crowding(freq_hz, other_freqs_hz)
        - overcapacity_weight.max(0.0)
            * e6_runtime_overcapacity(
                freq_hz,
                other_freqs_hz,
                overcapacity_radius_cents,
                overcapacity_free_voices,
            )
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
        if let Some(last) = merged.last_mut()
            && semitone_distance(space, last.center_idx, peak.center_idx)
                <= E6_HEREDITY_PEAK_MERGE_ST
        {
            if peak.mass > last.mass {
                last.center_idx = peak.center_idx;
            }
            last.mass += peak.mass;
            continue;
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

fn azimuth_from_family_and_freq(center_freq_hz: f32, freq_hz: f32) -> f32 {
    if !center_freq_hz.is_finite()
        || center_freq_hz <= 0.0
        || !freq_hz.is_finite()
        || freq_hz <= 0.0
    {
        0.0
    } else {
        12.0 * (freq_hz / center_freq_hz).log2()
    }
}

#[allow(clippy::too_many_arguments)]
fn e6_family_local_azimuth_optimum(
    loo_scene_landscape: &Landscape,
    child_family_center_hz: f32,
    alive_freqs_hz: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    min_spacing_cents: Option<f32>,
) -> (f32, f32, f32) {
    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let step_count = ((2.0 * E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST) / E6B_AZIMUTH_LOCAL_SEARCH_GRID_ST)
        .round() as i32;
    let mut best: Option<(f32, f32, f32)> = None;
    for step_idx in 0..=step_count {
        let azimuth_st = -E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST
            + step_idx as f32 * E6B_AZIMUTH_LOCAL_SEARCH_GRID_ST;
        let freq_hz =
            freq_from_family_and_azimuth(child_family_center_hz, azimuth_st).clamp(min_hz, max_hz);
        if min_spacing_cents.is_some_and(|min_spacing| {
            !e6_respects_min_spacing_cents(freq_hz, alive_freqs_hz, min_spacing)
        }) {
            continue;
        }
        let score = e6_polyphonic_selection_score(
            loo_scene_landscape,
            freq_hz,
            alive_freqs_hz,
            crowding_weight,
            overcapacity_weight,
            overcapacity_radius_cents,
            overcapacity_free_voices,
        );
        if !score.is_finite() {
            continue;
        }
        let replace = match best {
            Some((_, _, best_score)) => score > best_score + 1e-6,
            None => true,
        };
        if replace {
            best = Some((freq_hz, azimuth_st, score));
        }
    }
    if let Some(best) = best {
        return best;
    }

    let fallback_freq_hz = if let Some(min_spacing) = min_spacing_cents {
        e6_find_spaced_family_frequency(
            child_family_center_hz,
            0.0,
            min_hz,
            max_hz,
            alive_freqs_hz,
            min_spacing,
        )
        .unwrap_or(child_family_center_hz.clamp(min_hz, max_hz))
    } else {
        child_family_center_hz.clamp(min_hz, max_hz)
    };
    let fallback_score = e6_polyphonic_selection_score(
        loo_scene_landscape,
        fallback_freq_hz,
        alive_freqs_hz,
        crowding_weight,
        overcapacity_weight,
        overcapacity_radius_cents,
        overcapacity_free_voices,
    );
    (
        fallback_freq_hz,
        azimuth_from_family_and_freq(child_family_center_hz, fallback_freq_hz),
        fallback_score,
    )
}

#[allow(clippy::too_many_arguments)]
fn e6_family_local_opt_diagnostics(
    loo_scene_landscape: &Landscape,
    child_family_center_hz: f32,
    spawn_freq_hz: f32,
    alive_freqs_hz: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    min_spacing_cents: Option<f32>,
) -> (f32, f32) {
    let (_, best_azimuth_st, best_score) = e6_family_local_azimuth_optimum(
        loo_scene_landscape,
        child_family_center_hz,
        alive_freqs_hz,
        tessitura_min_hz,
        tessitura_max_hz,
        crowding_weight,
        overcapacity_weight,
        overcapacity_radius_cents,
        overcapacity_free_voices,
        min_spacing_cents,
    );
    let actual_score = e6_polyphonic_selection_score(
        loo_scene_landscape,
        spawn_freq_hz,
        alive_freqs_hz,
        crowding_weight,
        overcapacity_weight,
        overcapacity_radius_cents,
        overcapacity_free_voices,
    );
    let actual_azimuth_st = azimuth_from_family_and_freq(child_family_center_hz, spawn_freq_hz);
    (
        (actual_azimuth_st - best_azimuth_st).abs(),
        (best_score - actual_score).max(0.0),
    )
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

fn e6_parent_proposal_source_score(
    proposal_kind: E6ParentProposalKind,
    parent_landscape: &Landscape,
    idx: usize,
    freq_hz: f32,
) -> f32 {
    match proposal_kind {
        E6ParentProposalKind::ContextualC => {
            parent_landscape.evaluate_pitch_score(freq_hz).max(0.0)
        }
        E6ParentProposalKind::HOnly => parent_landscape
            .harmonicity01
            .get(idx)
            .copied()
            .unwrap_or(0.0)
            .max(0.0),
    }
}

fn e6_parent_distance_gaussian(delta_st: f32, sigma_st: f32) -> f32 {
    let sigma_st = sigma_st.max(1e-3);
    (-0.5 * (delta_st / sigma_st).powi(2)).exp()
}

fn e6_parent_unison_notch(delta_st: f32, notch_gain: f32, notch_sigma_st: f32) -> f32 {
    let notch_gain = notch_gain.clamp(0.0, 1.0);
    let notch_sigma_st = notch_sigma_st.max(1e-3);
    (1.0 - notch_gain * (-0.5 * (delta_st / notch_sigma_st).powi(2)).exp()).clamp(0.0, 1.0)
}

fn e6_collect_scene_peak_candidates(
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    candidate_count: usize,
) -> Vec<(usize, f32)> {
    let space = &current_landscape.space;
    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let mut base_weights = vec![0.0f32; space.n_bins()];
    for (idx, &freq_hz) in space.centers_hz.iter().enumerate() {
        if !freq_hz.is_finite() || freq_hz < min_hz || freq_hz > max_hz {
            continue;
        }
        let local_occupancy =
            e6_local_band_occupancy(freq_hz, alive_freqs_hz, overcapacity_radius_cents);
        if local_occupancy > overcapacity_free_voices.saturating_add(1) {
            continue;
        }
        let score = e6_polyphonic_selection_score(
            current_landscape,
            freq_hz,
            alive_freqs_hz,
            crowding_weight,
            overcapacity_weight,
            overcapacity_radius_cents,
            overcapacity_free_voices,
        );
        if score > 0.0 && score.is_finite() {
            base_weights[idx] = score.powf(E6B_PARENT_PEAK_SCENE_SCORE_EXP);
        }
    }
    let peaks = extract_peak_families(space, &base_weights);
    let mut ranked = peaks
        .iter()
        .filter_map(|peak| {
            let center_hz = space.centers_hz[peak.center_idx];
            let local_occupancy =
                e6_local_band_occupancy(center_hz, alive_freqs_hz, overcapacity_radius_cents);
            if local_occupancy > overcapacity_free_voices.saturating_add(1) {
                return None;
            }
            let score = e6_polyphonic_selection_score(
                current_landscape,
                center_hz,
                alive_freqs_hz,
                crowding_weight,
                overcapacity_weight,
                overcapacity_radius_cents,
                overcapacity_free_voices,
            );
            (score > 0.0 && score.is_finite())
                .then_some((peak.center_idx, score.powf(E6B_PARENT_PEAK_SCENE_SCORE_EXP)))
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    ranked.truncate(candidate_count.max(1));
    ranked
}

fn sample_from_scene_peak_parent_bias(
    parent_landscape: &Landscape,
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    parent_freq_hz: f32,
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    rng: &mut impl Rng,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    proposal_kind: E6ParentProposalKind,
    proposal_sigma_st: f32,
    proposal_unison_notch_gain: f32,
    proposal_unison_notch_sigma_st: f32,
    proposal_candidate_count: usize,
    same_band_discount: f32,
    octave_discount: f32,
    min_spacing_cents: Option<f32>,
    azimuth_mode: E6bAzimuthMode,
) -> E6HereditySample {
    let space = &current_landscape.space;
    let candidates = e6_collect_scene_peak_candidates(
        current_landscape,
        alive_freqs_hz,
        tessitura_min_hz,
        tessitura_max_hz,
        crowding_weight,
        overcapacity_weight,
        overcapacity_radius_cents,
        overcapacity_free_voices,
        proposal_candidate_count,
    );
    if candidates.is_empty() {
        return sample_from_parent_harmonicity(
            parent_landscape,
            current_landscape,
            alive_freqs_hz,
            parent_freq_hz,
            overcapacity_radius_cents,
            tessitura_min_hz,
            tessitura_max_hz,
            rng,
            E6_HEREDITY_FAMILY_OCCUPANCY_STRENGTH,
            E6FamilyNfdMode::Off,
            min_spacing_cents,
        );
    }

    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let parent_peak_idx = candidates
        .iter()
        .min_by(|(idx_a, _), (idx_b, _)| {
            let da = semitone_distance(space, *idx_a, space.nearest_index(parent_freq_hz));
            let db = semitone_distance(space, *idx_b, space.nearest_index(parent_freq_hz));
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| *idx)
        .unwrap_or(space.nearest_index(parent_freq_hz));
    let parent_family_center_hz = space.centers_hz[parent_peak_idx];
    let parent_azimuth_st = (12.0 * (parent_freq_hz / parent_family_center_hz.max(1e-6)).log2())
        .clamp(-E6_HEREDITY_AZIMUTH_CLIP_ST, E6_HEREDITY_AZIMUTH_CLIP_ST);

    let mut parent_weights = Vec::with_capacity(candidates.len());
    let mut final_weights = Vec::with_capacity(candidates.len());
    let mut scene_weights = Vec::with_capacity(candidates.len());
    for &(center_idx, scene_weight) in &candidates {
        let center_hz = space.centers_hz[center_idx];
        let delta_st = 12.0 * (center_hz / parent_freq_hz.max(1e-6)).log2();
        let mut parent_weight =
            e6_parent_proposal_source_score(proposal_kind, parent_landscape, center_idx, center_hz)
                * e6_parent_distance_gaussian(delta_st, proposal_sigma_st)
                * e6_parent_unison_notch(
                    delta_st,
                    proposal_unison_notch_gain,
                    proposal_unison_notch_sigma_st,
                );
        if e6_is_same_band_respawn(parent_freq_hz, center_hz, overcapacity_radius_cents) {
            parent_weight *= same_band_discount;
        }
        if e6_is_parent_octave_respawn(parent_freq_hz, center_hz, E6B_RESPAWN_OCTAVE_WINDOW_CENTS) {
            parent_weight *= octave_discount;
        }
        parent_weight = parent_weight.max(0.0);
        parent_weights.push(parent_weight);
        scene_weights.push(scene_weight);
        final_weights.push(0.0);
    }

    for idx in 0..candidates.len() {
        final_weights[idx] = (scene_weights[idx] * parent_weights[idx]).max(0.0);
    }
    if final_weights
        .iter()
        .all(|weight| *weight <= 0.0 || !weight.is_finite())
    {
        final_weights.clone_from(&scene_weights);
    }

    let chosen_local_idx = if let Ok(dist) = WeightedIndex::new(&final_weights) {
        dist.sample(rng)
    } else {
        0
    };
    let (child_center_idx, _) = candidates[chosen_local_idx];
    let child_family_center_hz = space.centers_hz[child_center_idx];
    let (offspring_freq_hz, child_azimuth_st) = match azimuth_mode {
        E6bAzimuthMode::Inherit => {
            let child_azimuth_st = (parent_azimuth_st
                + sample_standard_normal(rng) * E6_HEREDITY_AZIMUTH_SIGMA_ST)
                .clamp(-0.35, 0.35);
            let offspring_freq_hz = if let Some(min_spacing) = min_spacing_cents {
                e6_find_spaced_family_frequency(
                    child_family_center_hz,
                    child_azimuth_st,
                    min_hz,
                    max_hz,
                    alive_freqs_hz,
                    min_spacing,
                )
                .unwrap_or_else(|| {
                    freq_from_family_and_azimuth(child_family_center_hz, child_azimuth_st)
                })
            } else {
                freq_from_family_and_azimuth(child_family_center_hz, child_azimuth_st)
            };
            (
                offspring_freq_hz,
                azimuth_from_family_and_freq(child_family_center_hz, offspring_freq_hz).clamp(
                    -E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST,
                    E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST,
                ),
            )
        }
        E6bAzimuthMode::LocalSearch => {
            let (best_freq_hz, best_azimuth_st, _) = e6_family_local_azimuth_optimum(
                current_landscape,
                child_family_center_hz,
                alive_freqs_hz,
                min_hz,
                max_hz,
                crowding_weight,
                overcapacity_weight,
                overcapacity_radius_cents,
                overcapacity_free_voices,
                min_spacing_cents,
            );
            (best_freq_hz, best_azimuth_st)
        }
    };
    let chosen_band_occupancy =
        e6_local_band_occupancy(offspring_freq_hz, alive_freqs_hz, overcapacity_radius_cents);
    let chosen_parent_weight = parent_weights[chosen_local_idx];
    let total_parent_weight: f32 = parent_weights.iter().copied().sum();
    let top_parent_idx = parent_weights
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(chosen_local_idx);
    E6HereditySample {
        offspring_freq_hz,
        parent_family_center_hz,
        parent_azimuth_st,
        child_family_center_hz,
        child_azimuth_st,
        family_inherited: child_center_idx == parent_peak_idx,
        family_mutated: child_center_idx != parent_peak_idx,
        chosen_band_occupancy,
        proposal_rank: Some(
            1 + parent_weights
                .iter()
                .filter(|&&weight| weight > chosen_parent_weight + 1e-6)
                .count(),
        ),
        proposal_mass: Some((chosen_parent_weight / total_parent_weight.max(1e-6)).clamp(0.0, 1.0)),
        proposal_filter_gain: Some(scene_weights[chosen_local_idx] - scene_weights[top_parent_idx]),
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_from_scene_peak_matched_random(
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    rng: &mut impl Rng,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    proposal_candidate_count: usize,
    min_spacing_cents: Option<f32>,
) -> E6HereditySample {
    let space = &current_landscape.space;
    let candidates = e6_collect_scene_peak_candidates(
        current_landscape,
        alive_freqs_hz,
        tessitura_min_hz,
        tessitura_max_hz,
        crowding_weight,
        overcapacity_weight,
        overcapacity_radius_cents,
        overcapacity_free_voices,
        proposal_candidate_count,
    );
    if candidates.is_empty() {
        let freq_hz = sample_random_log_freq(rng, tessitura_min_hz, tessitura_max_hz);
        return E6HereditySample {
            offspring_freq_hz: freq_hz,
            parent_family_center_hz: freq_hz,
            parent_azimuth_st: 0.0,
            child_family_center_hz: freq_hz,
            child_azimuth_st: 0.0,
            family_inherited: false,
            family_mutated: false,
            chosen_band_occupancy: e6_local_band_occupancy(
                freq_hz,
                alive_freqs_hz,
                overcapacity_radius_cents,
            ),
            proposal_rank: None,
            proposal_mass: None,
            proposal_filter_gain: None,
        };
    }

    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let scene_weights = candidates
        .iter()
        .map(|(_, weight)| *weight)
        .collect::<Vec<_>>();
    let chosen_local_idx = if let Ok(dist) = WeightedIndex::new(&scene_weights) {
        dist.sample(rng)
    } else {
        0
    };
    let (child_center_idx, chosen_scene_weight) = candidates[chosen_local_idx];
    let child_family_center_hz = space.centers_hz[child_center_idx];
    let child_azimuth_st =
        (sample_standard_normal(rng) * E6_HEREDITY_AZIMUTH_SIGMA_ST).clamp(-0.35, 0.35);
    let offspring_freq_hz = if let Some(min_spacing) = min_spacing_cents {
        e6_find_spaced_family_frequency(
            child_family_center_hz,
            child_azimuth_st,
            min_hz,
            max_hz,
            alive_freqs_hz,
            min_spacing,
        )
        .unwrap_or_else(|| freq_from_family_and_azimuth(child_family_center_hz, child_azimuth_st))
    } else {
        freq_from_family_and_azimuth(child_family_center_hz, child_azimuth_st)
    };
    let chosen_band_occupancy =
        e6_local_band_occupancy(offspring_freq_hz, alive_freqs_hz, overcapacity_radius_cents);
    let total_scene_weight: f32 = scene_weights.iter().copied().sum();
    let top_scene_idx = scene_weights
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(chosen_local_idx);
    E6HereditySample {
        offspring_freq_hz,
        parent_family_center_hz: child_family_center_hz,
        parent_azimuth_st: 0.0,
        child_family_center_hz,
        child_azimuth_st,
        family_inherited: false,
        family_mutated: false,
        chosen_band_occupancy,
        proposal_rank: Some(
            1 + scene_weights
                .iter()
                .filter(|&&weight| weight > chosen_scene_weight + 1e-6)
                .count(),
        ),
        proposal_mass: Some((chosen_scene_weight / total_scene_weight.max(1e-6)).clamp(0.0, 1.0)),
        proposal_filter_gain: Some(scene_weights[chosen_local_idx] - scene_weights[top_scene_idx]),
    }
}

#[allow(clippy::too_many_arguments)]
fn sample_from_log_random_filtered_candidates(
    current_landscape: &Landscape,
    alive_freqs_hz: &[f32],
    tessitura_min_hz: f32,
    tessitura_max_hz: f32,
    rng: &mut impl Rng,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
    proposal_candidate_count: usize,
    min_spacing_cents: Option<f32>,
) -> E6HereditySample {
    let min_hz = tessitura_min_hz.min(tessitura_max_hz).max(1e-6);
    let max_hz = tessitura_max_hz.max(tessitura_min_hz).max(min_hz);
    let candidate_count = proposal_candidate_count.max(1);
    let candidate_separation_cents = overcapacity_radius_cents.max(0.0);
    let mut candidates: Vec<f32> = Vec::with_capacity(candidate_count);
    let max_attempts = candidate_count.saturating_mul(32).max(32);
    for _ in 0..max_attempts {
        if candidates.len() >= candidate_count {
            break;
        }
        let candidate_hz = sample_random_log_freq(rng, min_hz, max_hz);
        if candidate_separation_cents > 0.0
            && !e6_respects_min_spacing_cents(candidate_hz, &candidates, candidate_separation_cents)
        {
            continue;
        }
        candidates.push(candidate_hz);
    }
    if candidates.is_empty() {
        candidates.push(sample_random_log_freq(rng, min_hz, max_hz));
    }

    let mut scene_scores: Vec<f32> = Vec::with_capacity(candidates.len());
    let mut selection_weights: Vec<f32> = Vec::with_capacity(candidates.len());
    for &candidate_hz in &candidates {
        let scene_score = e6_polyphonic_selection_score(
            current_landscape,
            candidate_hz,
            alive_freqs_hz,
            crowding_weight,
            overcapacity_weight,
            overcapacity_radius_cents,
            overcapacity_free_voices,
        );
        scene_scores.push(scene_score);
        selection_weights.push(if scene_score.is_finite() {
            scene_score.max(0.0)
        } else {
            0.0
        });
    }

    let chosen_idx = if selection_weights
        .iter()
        .any(|weight| *weight > 0.0 && weight.is_finite())
    {
        if let Ok(dist) = WeightedIndex::new(&selection_weights) {
            dist.sample(rng)
        } else {
            selection_weights
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    } else {
        scene_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    };

    let chosen_freq_hz = candidates[chosen_idx];
    let offspring_freq_hz = if let Some(min_spacing) = min_spacing_cents {
        e6_find_spaced_family_frequency(
            chosen_freq_hz,
            0.0,
            min_hz,
            max_hz,
            alive_freqs_hz,
            min_spacing,
        )
        .unwrap_or(chosen_freq_hz)
    } else {
        chosen_freq_hz
    };
    let chosen_band_occupancy =
        e6_local_band_occupancy(offspring_freq_hz, alive_freqs_hz, overcapacity_radius_cents);
    let chosen_weight = selection_weights[chosen_idx];
    let total_weight: f32 = selection_weights.iter().copied().sum();
    E6HereditySample {
        offspring_freq_hz,
        parent_family_center_hz: chosen_freq_hz,
        parent_azimuth_st: 0.0,
        child_family_center_hz: chosen_freq_hz,
        child_azimuth_st: 0.0,
        family_inherited: false,
        family_mutated: false,
        chosen_band_occupancy,
        proposal_rank: Some(
            1 + scene_scores
                .iter()
                .filter(|&&score| score > scene_scores[chosen_idx] + 1e-6)
                .count(),
        ),
        proposal_mass: (total_weight > 0.0)
            .then_some((chosen_weight / total_weight.max(1e-6)).clamp(0.0, 1.0)),
        proposal_filter_gain: Some(0.0),
    }
}

#[allow(clippy::too_many_arguments)]
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
    _theta_phase: f32,
) {
    let mut freqs_hz: Vec<f32> = Vec::new();
    let mut agents: Vec<E6AgentSnapshot> = Vec::new();
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
    }
    agents.sort_by_key(|agent| agent.life_id);
    out.snapshots.push(E6PitchSnapshot {
        step,
        freqs_hz,
        agents,
    });
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
            chosen_band_occupancy: 1,
            proposal_rank: None,
            proposal_mass: None,
            proposal_filter_gain: None,
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
            chosen_band_occupancy: e6_local_band_occupancy(
                offspring_freq_hz,
                alive_freqs_hz,
                crowding_sigma_cents,
            ),
            proposal_rank: None,
            proposal_mass: None,
            proposal_filter_gain: None,
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
                chosen_band_occupancy: e6_local_band_occupancy(
                    offspring_freq_hz,
                    alive_freqs_hz,
                    crowding_sigma_cents,
                ),
                proposal_rank: None,
                proposal_mass: None,
                proposal_filter_gain: None,
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
        chosen_band_occupancy: e6_local_band_occupancy(
            offspring_freq_hz,
            alive_freqs_hz,
            crowding_sigma_cents,
        ),
        proposal_rank: None,
        proposal_mass: None,
        proposal_filter_gain: None,
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
    juvenile_tuning_ticks: u32,
    juvenile_tuning_radius_st: f32,
    juvenile_tuning_grid_st: f32,
    crowding_weight: f32,
    overcapacity_weight: f32,
    overcapacity_radius_cents: f32,
    overcapacity_free_voices: usize,
) {
    let mut eligible_ids: Vec<usize> = pop
        .individuals
        .iter()
        .filter(|agent| agent.is_alive())
        .filter_map(|agent| {
            let idx = agent.id() as usize;
            let state = states.get(idx)?;
            (state.ticks < juvenile_tuning_ticks).then_some(idx)
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
            overcapacity_radius_cents,
            overcapacity_free_voices,
        ) - if current_spacing_ok { 0.0 } else { 1.0 };

        let grid_st = juvenile_tuning_grid_st.max(0.05);
        let radius_steps = (juvenile_tuning_radius_st.max(0.0) / grid_st).round() as i32;
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
                overcapacity_radius_cents,
                overcapacity_free_voices,
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

fn e6_survival_signal(score: f32, low: Option<f32>, high: Option<f32>) -> f32 {
    let (Some(low), Some(high)) = (low, high) else {
        return score.max(0.0);
    };
    if !score.is_finite() {
        return 0.0;
    }
    let hi = if high > low { high } else { low + 1e-6 };
    ((score - low) / (hi - low)).clamp(0.0, 1.0)
}

#[cfg(test)]
pub fn e6_mean_selection_score_for_freqs_mode(
    freqs_hz: &[f32],
    anchor_hz: f32,
    _selection_score_mode: E6SelectionScoreMode,
    _contextual_mix_weight: f32,
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
                .filter_map(|(other_idx, &other_freq)| (other_idx != idx).then_some(other_freq))
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
                E6B_LOCAL_CAPACITY_RADIUS_CENTS,
                E6B_LOCAL_CAPACITY_FREE_VOICES,
            )
        })
        .sum::<f32>()
        / clean_freqs.len() as f32
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

#[cfg(test)]
fn e6_parent_fertility_weight(agent: &Individual) -> Option<f32> {
    let energy = e6_agent_energy(agent)?;
    let weight = energy.powf(E6_FERTILITY_ENERGY_EXPONENT);
    Some(weight.max(1e-6))
}

fn e6_parent_selection_weight_from_score_with_energy(
    energy: f32,
    score: f32,
    energy_weight: f32,
) -> f32 {
    if !score.is_finite() || !energy.is_finite() {
        return 1e-6;
    }
    let energy_weight = energy_weight.clamp(0.0, 1.0);
    let energy_factor = 1.0 - energy_weight + energy_weight * energy.max(1e-3).sqrt();
    let score_factor = 0.2 + score.max(0.0);
    (energy_factor * score_factor).max(1e-6)
}

fn e6_parent_selection_weight_from_polyphonic_score(
    energy: f32,
    score: f32,
    local_occupancy: usize,
    free_voices: usize,
    share_weight: f32,
    energy_weight: f32,
) -> f32 {
    let base_weight =
        e6_parent_selection_weight_from_score_with_energy(energy, score, energy_weight);
    let share = e6_local_band_share(local_occupancy, free_voices);
    let niche_factor = 1.0 - share_weight.clamp(0.0, 1.0) + share_weight.clamp(0.0, 1.0) * share;
    (base_weight * niche_factor.max(1e-3)).max(1e-6)
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

fn crowding_sigma_erb_from_hz(freq_hz: f32, sigma_cents: f32) -> f32 {
    let sigma_log2 = (sigma_cents.max(1e-3)) / 1200.0;
    let base_hz = freq_hz.max(1e-6);
    let plus_hz = base_hz * 2.0f32.powf(sigma_log2);
    (hz_to_erb(plus_hz) - hz_to_erb(base_hz)).abs().max(1e-6)
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

#[cfg(test)]
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
            anchor_hz: 440.0,
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

// Keep the temporal-scaffold render percussive: the note-off happens quickly,
// and the audible tail comes from the release rather than a held tone.
// Use the neutral drone render modulator for replay so Conchordal's SeqGate
// does not hard-gate the release tail after its fixed internal duration.
const E3_AUDIO_NOTE_DUR_SEC: f32 = 0.035;
const E3_AUDIO_NOTE_ATTACK_SEC: f32 = 0.0030;
const E3_AUDIO_NOTE_AMP: f32 = 0.04;
const E3_AUDIO_RHAI_DECAY_SEC: f32 = 0.420;
const E3_AUDIO_RHAI_SUSTAIN_LEVEL: f32 = 0.08;
const E3_AUDIO_RHAI_RELEASE_SEC: f32 = 0.200;

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

fn sample_e3_audio_pitch_hz(
    rng: &mut impl Rng,
    space: &Log2Space,
    landscape: &Landscape,
    anchor_hz: f32,
) -> f32 {
    let (min_freq, max_freq) = e3_tessitura_bounds_for_range(anchor_hz, space, E3_RANGE_OCT);
    let log2_pitch = sample_e4_initial_pitch_log2(rng, space, landscape, min_freq, max_freq);
    2.0f32.powf(log2_pitch)
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

    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let landscape = build_anchor_landscape(
        &space,
        &params,
        cfg.anchor_hz,
        E4_ENV_PARTIALS_DEFAULT,
        E4_ENV_PARTIAL_DECAY_DEFAULT,
    );
    let pitches_hz: Vec<f32> = (0..cfg.pop_size)
        .map(|_| sample_e3_audio_pitch_hz(&mut rng, &space, &landscape, cfg.anchor_hz))
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

    // Historical E3 supplementary audio used the internal renderer path with
    // longer, brighter notes than the Rhai monitor. Keep that timbre here so
    // the pulse remains legible in the dense shared/scaffold conditions.
    const NOTE_DUR_SEC: f32 = 0.16;
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
            let voice = phase.sin() + 0.18 * (2.0 * phase).sin() + 0.08 * (3.0 * phase).sin();
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

    #[derive(Clone, Copy)]
    enum EventKind {
        Release { attack_idx: usize },
        Attack { attack_idx: usize, agent: usize },
    }

    #[derive(Clone, Copy)]
    struct ScheduledEvent {
        time: f32,
        kind: EventKind,
    }

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
         // Each attack emits a short one-shot pulse at a static audibility pitch.\n\
         // No external drone or metronome is mixed into the render.\n\n",
        n = cfg.pop_size,
        kick = cfg.theta_freq_hz
    );

    for i in 0..cfg.pop_size {
        rhai.push_str(&format!(
            "let s{i} = derive(sine).brain(\"drone\").pitch_mode(\"lock\")\
             .amp({:.3}).adsr({:.4}, {:.3}, {:.3}, {:.3});\n",
            E3_AUDIO_NOTE_AMP,
            E3_AUDIO_NOTE_ATTACK_SEC,
            E3_AUDIO_RHAI_DECAY_SEC,
            E3_AUDIO_RHAI_SUSTAIN_LEVEL,
            E3_AUDIO_RHAI_RELEASE_SEC
        ));
    }
    rhai.push('\n');

    attacks.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    let quant = 0.001f32;
    let mut events = Vec::with_capacity(attacks.len() * 2);
    for (attack_idx, attack) in attacks.iter().enumerate() {
        let attack_t = (attack.time / quant).round() * quant;
        let release_t = (attack_t + E3_AUDIO_NOTE_DUR_SEC)
            .min(cfg.duration_sec)
            .max(attack_t);
        events.push(ScheduledEvent {
            time: attack_t,
            kind: EventKind::Attack {
                attack_idx,
                agent: attack.agent,
            },
        });
        if release_t > attack_t + 1e-6 {
            events.push(ScheduledEvent {
                time: release_t,
                kind: EventKind::Release { attack_idx },
            });
        }
    }
    events.sort_by(|a, b| {
        a.time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| match (a.kind, b.kind) {
                (EventKind::Release { .. }, EventKind::Attack { .. }) => std::cmp::Ordering::Less,
                (EventKind::Attack { .. }, EventKind::Release { .. }) => {
                    std::cmp::Ordering::Greater
                }
                _ => std::cmp::Ordering::Equal,
            })
    });

    rhai.push_str(&format!("scene(\"e3_{condition_label}\", || {{\n"));
    let mut time_cursor = 0.0f32;
    for event in &events {
        if event.time > time_cursor + 0.0001 {
            let wait = event.time - time_cursor;
            rhai.push_str(&format!("    wait({wait:.4});\n"));
            time_cursor = event.time;
        }
        match event.kind {
            EventKind::Attack { attack_idx, agent } => {
                rhai.push_str(&format!(
                    "    let g{attack_idx} = create(s{agent}, 1).freq({:.2}); flush();\n",
                    pitches_hz[agent]
                ));
            }
            EventKind::Release { attack_idx } => {
                rhai.push_str(&format!("    release(g{attack_idx}); flush();\n"));
            }
        }
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

    fn test_mean(values: &[f32]) -> f32 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f32>() / values.len() as f32
        }
    }

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
    fn e6_family_local_azimuth_search_improves_or_matches_inherited_spawn_score() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let current_landscape = build_anchor_landscape(
            &space,
            &params,
            E4_ANCHOR_HZ,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let (min_hz, max_hz) =
            e3_tessitura_bounds_for_range(E4_ANCHOR_HZ, &space, E6B_DEFAULT_RANGE_OCT);
        let child_family_center_hz = E4_ANCHOR_HZ * (3.0 / 2.0);
        let inherited_freq_hz = freq_from_family_and_azimuth(child_family_center_hz, 0.35);
        let inherited_score = e6_polyphonic_selection_score(
            &current_landscape,
            inherited_freq_hz,
            &[],
            E6B_SELECTION_CROWDING_WEIGHT,
            E6B_LOCAL_CAPACITY_WEIGHT,
            E6B_LOCAL_CAPACITY_RADIUS_CENTS,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
        );
        let (best_freq_hz, best_azimuth_st, best_score) = e6_family_local_azimuth_optimum(
            &current_landscape,
            child_family_center_hz,
            &[],
            min_hz,
            max_hz,
            E6B_SELECTION_CROWDING_WEIGHT,
            E6B_LOCAL_CAPACITY_WEIGHT,
            E6B_LOCAL_CAPACITY_RADIUS_CENTS,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
            None,
        );
        assert!(
            best_freq_hz >= min_hz && best_freq_hz <= max_hz,
            "family-local azimuth search should stay within tessitura"
        );
        assert!(
            best_azimuth_st.abs() <= E6B_AZIMUTH_LOCAL_SEARCH_RADIUS_ST + 1e-6,
            "family-local azimuth search should stay inside the configured family window"
        );
        assert!(
            best_score + 1e-6 >= inherited_score,
            "family-local azimuth search should not underperform a fixed inherited azimuth"
        );
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
    fn e6b_local_search_azimuth_mode_reduces_spawn_local_opt_gap() {
        let base_cfg = E6bRunConfig {
            seed: 0xE6B0_4321,
            steps_cap: 900,
            min_deaths: 8,
            pop_size: 8,
            first_k: 10,
            condition: E6Condition::Heredity,
            snapshot_interval: 20,
            selection_enabled: true,
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
            azimuth_mode_override: Some(E6bAzimuthMode::Inherit),
            random_baseline_mode: E6bRandomBaselineMode::LogRandomFiltered,
        };
        let inherit_result = run_e6b(&base_cfg);
        let search_result = run_e6b(&E6bRunConfig {
            azimuth_mode_override: Some(E6bAzimuthMode::LocalSearch),
            ..base_cfg.clone()
        });

        let inherit_gap: Vec<f32> = inherit_result
            .respawns
            .iter()
            .filter_map(|respawn| respawn.local_opt_score_gap)
            .collect();
        let search_gap: Vec<f32> = search_result
            .respawns
            .iter()
            .filter_map(|respawn| respawn.local_opt_score_gap)
            .collect();
        let inherit_delta: Vec<f32> = inherit_result
            .respawns
            .iter()
            .filter_map(|respawn| respawn.local_opt_delta_st)
            .collect();
        let search_delta: Vec<f32> = search_result
            .respawns
            .iter()
            .filter_map(|respawn| respawn.local_opt_delta_st)
            .collect();

        assert!(
            !inherit_gap.is_empty() && !search_gap.is_empty(),
            "expected parent-peak respawns to record local-opt diagnostics"
        );
        assert!(
            test_mean(&search_gap) <= test_mean(&inherit_gap) + 1e-6,
            "local-search azimuth mode should reduce the mean spawn-to-local-opt score gap"
        );
        assert!(
            test_mean(&search_delta) <= test_mean(&inherit_delta) + 1e-6,
            "local-search azimuth mode should reduce the mean spawn-to-local-opt azimuth delta"
        );
    }

    #[test]
    fn e6_polyphonic_capacity_penalty_reduces_overcrowded_band_score() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let anchor = e3_reference_landscape(E4_ANCHOR_HZ);
        let target = E4_ANCHOR_HZ * 2.0f32.powf(7.0 / 12.0);
        let crowded_freqs = vec![
            target,
            target * 2.0f32.powf(2.0 / 1200.0),
            target * 2.0f32.powf(-2.0 / 1200.0),
            target * 2.0f32.powf(4.0 / 1200.0),
        ];
        let scene_landscape = build_e6_scene_landscape_with_anchor(
            &space,
            &params,
            &anchor,
            &crowded_freqs,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let other_freqs_hz = crowded_freqs.iter().copied().skip(1).collect::<Vec<_>>();
        let loo_scene_landscape = build_e6_contextual_loo_landscape(
            &space,
            &params,
            &scene_landscape,
            target,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let score_without_capacity = e6_polyphonic_selection_score(
            &loo_scene_landscape,
            target,
            &other_freqs_hz,
            E6B_SELECTION_CROWDING_WEIGHT,
            0.0,
            E6B_LOCAL_CAPACITY_RADIUS_CENTS,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
        );
        let score_with_capacity = e6_polyphonic_selection_score(
            &loo_scene_landscape,
            target,
            &other_freqs_hz,
            E6B_SELECTION_CROWDING_WEIGHT,
            E6B_LOCAL_CAPACITY_WEIGHT,
            E6B_LOCAL_CAPACITY_RADIUS_CENTS,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
        );
        assert!(
            e6_runtime_overcapacity(
                target,
                &other_freqs_hz,
                E6B_LOCAL_CAPACITY_RADIUS_CENTS,
                E6B_LOCAL_CAPACITY_FREE_VOICES,
            ) >= 1.0
        );
        assert!(
            score_with_capacity < score_without_capacity,
            "capacity penalty should reduce overcrowded band score"
        );
    }

    #[test]
    fn e6_local_band_share_decreases_with_occupancy() {
        let free = E6B_LOCAL_CAPACITY_FREE_VOICES;
        let share1 = e6_local_band_share(1, free);
        let share2 = e6_local_band_share(2, free);
        let share3 = e6_local_band_share(3, free);
        let share5 = e6_local_band_share(5, free);
        assert!((share1 - 1.0).abs() <= 1e-6);
        assert!(share2 < share1);
        assert!(share3 < share2);
        assert!(share5 < share3);
        assert!(share3 > 0.5, "2-3 shared voices should remain viable");
    }

    #[test]
    fn e6_parent_selection_weight_penalizes_overfilled_bands() {
        let sparse = e6_parent_selection_weight_from_polyphonic_score(
            1.0,
            0.8,
            1,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
            1.0,
            E6B_PARENT_ENERGY_WEIGHT,
        );
        let moderate = e6_parent_selection_weight_from_polyphonic_score(
            1.0,
            0.8,
            3,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
            1.0,
            E6B_PARENT_ENERGY_WEIGHT,
        );
        let crowded = e6_parent_selection_weight_from_polyphonic_score(
            1.0,
            0.8,
            6,
            E6B_LOCAL_CAPACITY_FREE_VOICES,
            1.0,
            E6B_PARENT_ENERGY_WEIGHT,
        );
        assert!(sparse > moderate);
        assert!(moderate > crowded);
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
    fn e6_parent_distance_gaussian_keeps_substantial_mass_at_one_octave() {
        let center = e6_parent_distance_gaussian(0.0, E6B_PARENT_PROPOSAL_SIGMA_ST);
        let octave = e6_parent_distance_gaussian(12.0, E6B_PARENT_PROPOSAL_SIGMA_ST);
        assert!((center - 1.0).abs() < 1e-6);
        assert!(
            (octave - 0.41).abs() < 0.03,
            "expected octave decay near 41%, got {octave:.4}"
        );
    }

    #[test]
    fn e6_parent_unison_notch_strongly_suppresses_exact_parent_pitch() {
        let at_unison = e6_parent_unison_notch(
            0.0,
            E6B_PARENT_PROPOSAL_UNISON_NOTCH_GAIN,
            E6B_PARENT_PROPOSAL_UNISON_NOTCH_SIGMA_ST,
        );
        let away = e6_parent_unison_notch(
            2.0,
            E6B_PARENT_PROPOSAL_UNISON_NOTCH_GAIN,
            E6B_PARENT_PROPOSAL_UNISON_NOTCH_SIGMA_ST,
        );
        assert!(
            (at_unison - 0.05).abs() < 0.01,
            "expected exact unison to retain about 5% mass, got {at_unison:.4}"
        );
        assert!(
            away > 0.99,
            "expected notch to mostly disappear away from unison, got {away:.4}"
        );
    }

    #[test]
    fn e6b_run_reaches_turnover_without_near_unisons() {
        let cfg = E6bRunConfig {
            seed: 0xE6B0_1234,
            steps_cap: 900,
            min_deaths: 4,
            pop_size: 8,
            first_k: 10,
            condition: E6Condition::Heredity,
            snapshot_interval: 20,
            selection_enabled: true,
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
            random_baseline_mode: E6bRandomBaselineMode::MatchedScenePeaks,
        };
        let result = run_e6b(&cfg);
        assert!(
            result.total_deaths >= cfg.min_deaths,
            "E6b run should still produce some turnover under survival-gated selection"
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
    fn e6_survival_signal_clamps_and_ramps() {
        assert_eq!(e6_survival_signal(-1.0, Some(0.15), Some(0.55)), 0.0);
        assert_eq!(e6_survival_signal(0.15, Some(0.15), Some(0.55)), 0.0);
        assert!((e6_survival_signal(0.35, Some(0.15), Some(0.55)) - 0.5).abs() < 1e-6);
        assert_eq!(e6_survival_signal(0.55, Some(0.15), Some(0.55)), 1.0);
        assert_eq!(e6_survival_signal(1.0, Some(0.15), Some(0.55)), 1.0);
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
