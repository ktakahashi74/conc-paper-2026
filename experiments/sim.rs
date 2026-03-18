use conchordal::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
use conchordal::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use conchordal::core::landscape::{
    Landscape, LandscapeParams, LandscapeUpdate, RoughnessScalarMode,
};
use conchordal::core::log2space::Log2Space;
use conchordal::core::modulation::{NeuralRhythms, RhythmBand};
use conchordal::core::psycho_state::{compute_roughness_reference, r_pot_scan_to_r_state01_scan};
use conchordal::core::roughness_kernel::{KernelParams, RoughnessKernel, erb_grid};
use conchordal::core::timebase::Timebase;
use conchordal::life::articulation_core::{
    AnyArticulationCore, ArticulationState, kuramoto_phase_step,
};
use conchordal::life::control::{AgentControl, LeaveSelfOutMode, PitchCoreKind, PitchMode};
use conchordal::life::individual::{PitchHillClimbPitchCore, SoundBody};
use conchordal::life::lifecycle::LifecycleConfig;
use conchordal::life::metabolism_policy::MetabolismPolicy;
use conchordal::life::population::Population;
use conchordal::life::scenario::{
    Action, ArticulationCoreConfig, EnvelopeConfig, PhonationSpec, RhythmCouplingMode,
    SpawnSpec, SpawnStrategy,
};
use rand::distr::Distribution;
use rand::distr::weighted::WeightedIndex;
use rand::prelude::SliceRandom;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
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
const E3_BINS_PER_OCT: u32 = 96;
const E3_FMIN: f32 = 80.0;
const E3_FMAX: f32 = 2000.0;
const E3_RANGE_OCT: f32 = 2.0; // +/- 1 octave around anchor
const E3_THETA_FREQ_HZ: f32 = 1.0;
const E3_METABOLISM_RATE: f32 = 0.5;
const E2_MATCH_CROWDING_WEIGHT: f32 = 0.15;
const E4_MATCH_SIGMA_CENTS: f32 = 15.0;
const E2_E6_HILL_NEIGHBOR_STEP_CENTS: f32 = 25.0;
const E2_E6_HILL_EXPLORATION: f32 = 0.0;
const E2_E6_HILL_PERSISTENCE: f32 = 0.5;
const E2_E6_HILL_MOVE_COST_COEFF: f32 = 0.5;
const E2_E6_HILL_IMPROVEMENT_THRESHOLD: f32 = 0.02;

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
    pub snapshots: Vec<E6PitchSnapshot>,
    pub total_deaths: usize,
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
}

#[derive(Clone, Copy, Debug)]
struct E3LifeState {
    life_id: u64,
    birth_step: u32,
    ticks: u32,
    sum_c_level_tick: f32,
    sum_c_level_tick_sq: f32,
    sum_c_level_firstk: f32,
    firstk_count: u32,
    c_level_birth: f32,
    sum_c_level_attack: f32,
    attack_tick_count: u32,
    pending_birth: bool,
    was_alive: bool,
}

impl E3LifeState {
    fn new(life_id: u64) -> Self {
        Self {
            life_id,
            birth_step: 0,
            ticks: 0,
            sum_c_level_tick: 0.0,
            sum_c_level_tick_sq: 0.0,
            sum_c_level_firstk: 0.0,
            firstk_count: 0,
            c_level_birth: 0.0,
            sum_c_level_attack: 0.0,
            attack_tick_count: 0,
            pending_birth: true,
            was_alive: true,
        }
    }

    fn reset_for_new_life(&mut self, life_id: u64) {
        self.life_id = life_id;
        self.birth_step = 0;
        self.ticks = 0;
        self.sum_c_level_tick = 0.0;
        self.sum_c_level_tick_sq = 0.0;
        self.sum_c_level_firstk = 0.0;
        self.firstk_count = 0;
        self.c_level_birth = 0.0;
        self.sum_c_level_attack = 0.0;
        self.attack_tick_count = 0;
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
                let c_level = landscape.evaluate_pitch_level(agent.body.base_freq_hz());
                if state.pending_birth {
                    state.birth_step = step as u32;
                    state.c_level_birth = c_level;
                    state.pending_birth = false;
                }
                state.ticks = state.ticks.saturating_add(1);
                state.sum_c_level_tick += c_level;
                state.sum_c_level_tick_sq += c_level * c_level;
                if state.firstk_count < first_k {
                    state.sum_c_level_firstk += c_level;
                    state.firstk_count += 1;
                }

                // Attack-time consonance
                if let AnyArticulationCore::Entrain(ref core) = agent.articulation.core {
                    if core.state == ArticulationState::Attack {
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
                let c_level_std_over_life = if state.ticks > 0 {
                    let mean_sq = state.sum_c_level_tick_sq / state.ticks as f32;
                    let var = (mean_sq - avg_c_level_tick * avg_c_level_tick).max(0.0);
                    var.sqrt()
                } else {
                    0.0
                };
                let c_level_firstk = if state.firstk_count > 0 {
                    state.sum_c_level_firstk / state.firstk_count as f32
                } else {
                    0.0
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
        snapshots: Vec::new(),
        total_deaths: 0,
    };
    if cfg.pop_size == 0 || cfg.steps_cap == 0 {
        return out;
    }

    let anchor_hz = E4_ANCHOR_HZ;
    let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
    let params = make_landscape_params(&space, E3_FS, 1.0);
    let partials = cfg.env_partials.unwrap_or(E4_ENV_PARTIALS_DEFAULT);
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

    let spec = e6_spawn_spec(anchor_hz, cfg.landscape_weight);
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
    let snapshot_interval = cfg.snapshot_interval.max(1);
    let mut rng = SmallRng::seed_from_u64(cfg.seed ^ 0xE600_5EED_u64);
    let half = 0.5 * E3_RANGE_OCT;
    let min_spawn = (anchor_hz * 2.0f32.powf(-half))
        .clamp(space.fmin, space.fmax)
        .min((anchor_hz * 2.0f32.powf(half)).clamp(space.fmin, space.fmax));
    let max_spawn = (anchor_hz * 2.0f32.powf(half))
        .clamp(space.fmin, space.fmax)
        .max((anchor_hz * 2.0f32.powf(-half)).clamp(space.fmin, space.fmax));

    const LANDSCAPE_REFRESH_INTERVAL: usize = 100;
    for step in 0..cfg.steps_cap {
        // Periodically recompute landscape from all alive agents (includes roughness)
        if step > 0 && step % LANDSCAPE_REFRESH_INTERVAL == 0 {
            update_e4_landscape_from_population(
                &space, &params, &pop, &mut landscape, partials, E4_ENV_PARTIAL_DECAY_DEFAULT,
            );
            if let Some(ref perm) = shuffle_perm {
                apply_e6_shuffle_permutation(&mut landscape, perm);
            }
        }
        pop.advance(E3_HOP, E3_FS, step as u64, dt, &landscape);

        if step % snapshot_interval == 0 {
            let mut freqs_hz: Vec<f32> = Vec::new();
            let mut agents: Vec<E6AgentSnapshot> = Vec::new();
            let mut phase_sin_sum = 0.0f32;
            let mut phase_cos_sum = 0.0f32;
            let mut phase_count = 0u32;
            let theta_phase = landscape.rhythm.theta.phase;
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
                let c_level = landscape.evaluate_pitch_level(agent.body.base_freq_hz());
                if state.pending_birth {
                    state.birth_step = step as u32;
                    state.c_level_birth = c_level;
                    state.pending_birth = false;
                }
                state.ticks = state.ticks.saturating_add(1);
                state.sum_c_level_tick += c_level;
                state.sum_c_level_tick_sq += c_level * c_level;
                if state.firstk_count < first_k {
                    state.sum_c_level_firstk += c_level;
                    state.firstk_count += 1;
                }

                // Attack-time consonance
                if let AnyArticulationCore::Entrain(ref core) = agent.articulation.core {
                    if core.state == ArticulationState::Attack {
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
                let c_level_std_over_life = if state.ticks > 0 {
                    let mean_sq = state.sum_c_level_tick_sq / state.ticks as f32;
                    let var = (mean_sq - avg_c_level_tick * avg_c_level_tick).max(0.0);
                    var.sqrt()
                } else {
                    0.0
                };
                let c_level_firstk = if state.firstk_count > 0 {
                    state.sum_c_level_firstk / state.firstk_count as f32
                } else {
                    0.0
                };
                let avg_c_level_attack = if state.attack_tick_count > 0 {
                    state.sum_c_level_attack / state.attack_tick_count as f32
                } else {
                    avg_c_level_tick // fallback
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

            let offspring_strategy = match cfg.condition {
                E6Condition::Random => strategy.clone(),
                E6Condition::Heredity => {
                    let alive_parents: Vec<f32> = pop
                        .individuals
                        .iter()
                        .filter(|a| a.is_alive() && a.id() != id)
                        .map(|a| a.body.base_freq_hz())
                        .filter(|f| f.is_finite() && *f > 0.0)
                        .collect();
                    if alive_parents.is_empty() {
                        strategy.clone()
                    } else {
                        let parent_idx = rng.random_range(0..alive_parents.len());
                        let parent_freq = alive_parents[parent_idx];
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
                        let offspring_freq =
                            sample_from_parent_harmonicity(&parent_landscape, parent_freq, &mut rng);
                        let spawn_freq = offspring_freq.clamp(min_spawn, max_spawn);
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
        }

        landscape.rhythm.advance_in_place(dt);

        if out.deaths.len() >= cfg.min_deaths {
            break;
        }
    }

    out.total_deaths = out.deaths.len();
    out
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
    LandscapeParams {
        fs,
        max_hist_cols: 1,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
        harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
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

fn compute_roughness_for_landscape(
    space: &Log2Space,
    params: &LandscapeParams,
    landscape: &mut Landscape,
) {
    let (_erb, du) = erb_grid(space);
    let (density, _mass) = conchordal::core::psycho_state::normalize_density(
        &landscape.subjective_intensity, &du, 1e-12,
    );
    let (r_pot, _r_total) = params
        .roughness_kernel
        .potential_r_from_log2_spectrum_density(&density, space);
    let r_ref = compute_roughness_reference(params, space);
    landscape.roughness = r_pot;
    let mut r01 = vec![0.0f32; space.n_bins()];
    r_pot_scan_to_r_state01_scan(&landscape.roughness, r_ref.peak, params.roughness_k, &mut r01);
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

/// Sample from the parent's harmonicity profile within the open interval (-1 oct, +1 oct),
/// excluding unison with the parent. Falls back to uniform under the same constraints.
fn sample_from_parent_harmonicity(landscape: &Landscape, parent_freq_hz: f32, rng: &mut impl Rng) -> f32 {
    let space = &landscape.space;
    let n = space.n_bins();
    let parent_log2 = parent_freq_hz.max(1.0).log2();
    if n == 0 || landscape.harmonicity01.len() != n {
        return parent_freq_hz;
    }

    let parent_idx = space
        .index_of_log2(parent_log2)
        .unwrap_or_else(|| space.nearest_index(parent_freq_hz));
    let mut indices = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n {
        let log2 = space.centers_log2[i];
        let delta_oct = log2 - parent_log2;
        if delta_oct <= -1.0 || delta_oct >= 1.0 || i == parent_idx {
            continue;
        }
        let w = landscape.harmonicity01[i].max(0.0);
        indices.push(i);
        weights.push(w);
    }
    if indices.is_empty() {
        return sample_uniform_parent_interval(space, parent_log2, parent_idx, rng);
    }
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return sample_uniform_parent_interval(space, parent_log2, parent_idx, rng);
    }
    let r: f32 = rng.random::<f32>() * total;
    let mut cumulative = 0.0f32;
    for (j, &w) in weights.iter().enumerate() {
        cumulative += w;
        if r <= cumulative {
            return space.centers_hz[indices[j]];
        }
    }
    space.centers_hz[*indices.last().unwrap()]
}

fn sample_uniform_parent_interval(
    space: &Log2Space,
    parent_log2: f32,
    parent_idx: usize,
    rng: &mut impl Rng,
) -> f32 {
    let indices: Vec<usize> = space
        .centers_log2
        .iter()
        .enumerate()
        .filter_map(|(i, &log2)| {
            let delta_oct = log2 - parent_log2;
            if delta_oct <= -1.0 || delta_oct >= 1.0 || i == parent_idx {
                None
            } else {
                Some(i)
            }
        })
        .collect();
    if indices.is_empty() {
        return parent_log2.exp2();
    }
    let pick = rng.random_range(0..indices.len());
    space.centers_hz[indices[pick]]
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
fn e6_spawn_spec(anchor_hz: f32, landscape_weight: f32) -> SpawnSpec {
    let mut control = AgentControl::default();
    configure_shared_hillclimb_control(&mut control, landscape_weight);
    control.pitch.freq = anchor_hz.max(1.0);
    control.pitch.range_oct = E3_RANGE_OCT;
    // perceptual repulsion: enabled=true (default), novelty_bias=1.0 (default)
    control.phonation.spec = PhonationSpec::default();

    let lifecycle = e3_lifecycle(E3Condition::Baseline);
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
    let half = 0.5 * E3_RANGE_OCT;
    let min_freq = anchor_hz * 2.0f32.powf(-half);
    let max_freq = anchor_hz * 2.0f32.powf(half);
    let min_freq = min_freq.clamp(space.fmin, space.fmax);
    let max_freq = max_freq.clamp(space.fmin, space.fmax);
    SpawnStrategy::RandomLog {
        min_freq: min_freq.min(max_freq),
        max_freq: max_freq.max(min_freq),
    }
}

fn e3_lifecycle(condition: E3Condition) -> LifecycleConfig {
    let recharge_rate = match condition {
        E3Condition::Baseline => None,
        E3Condition::NoRecharge => Some(0.0),
    };
    LifecycleConfig::Sustain {
        initial_energy: 1.0,
        metabolism_rate: E3_METABOLISM_RATE,
        recharge_rate,
        action_cost: None,
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
                (1.0 - (local_t - NOTE_ATTACK_SEC) / (NOTE_DUR_SEC - NOTE_ATTACK_SEC)).clamp(0.0, 1.0)
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

pub fn generate_e3_rhai(
    cfg: &E3AudioConfig,
    output_path: &std::path::Path,
) -> std::io::Result<()> {
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

    #[test]
    fn e6_spawn_spec_uses_e2_crowding_and_e4_sigma() {
        let spec = e6_spawn_spec(E4_ANCHOR_HZ, 0.5);
        let pitch = spec.control.pitch;
        assert_eq!(pitch.core_kind, PitchCoreKind::HillClimb);
        assert_eq!(pitch.neighbor_step_cents, Some(E2_E6_HILL_NEIGHBOR_STEP_CENTS));
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
    fn parent_harmonicity_sampler_excludes_unison_and_exact_octaves() {
        let space = Log2Space::new(E3_FMIN, E3_FMAX, E3_BINS_PER_OCT);
        let params = make_landscape_params(&space, E3_FS, 1.0);
        let parent_freq = 220.0f32;
        let landscape = build_e6_parent_harmonicity_landscape(
            &space,
            &params,
            parent_freq,
            E4_ENV_PARTIALS_DEFAULT,
            E4_ENV_PARTIAL_DECAY_DEFAULT,
        );
        let mut rng = SmallRng::seed_from_u64(0xE6AA_55CC);
        for _ in 0..128 {
            let freq = sample_from_parent_harmonicity(&landscape, parent_freq, &mut rng);
            let delta_oct = (freq / parent_freq).log2();
            assert!(delta_oct > -1.0 && delta_oct < 1.0, "sample escaped open octave interval");
            assert!(delta_oct.abs() > 1e-6, "unison must be excluded");
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
