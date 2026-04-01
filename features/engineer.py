"""
Feature Engineering for ML Models (v2)

Defines the feature names list used by the training pipeline.
The actual feature computation is now done in dataset.py's build_dataset().
"""

from __future__ import annotations

# Feature names list for model training — must match columns in ml_dataset.csv
FEATURE_NAMES = [
    # Race context (7)
    "distance", "venue_enc", "condition_enc", "field_size",
    "gate", "draw_weight_lb", "horse_weight_kg",
    # Timing (1)
    "days_since_last",
    # Horse performance history (6)
    "hist_avg_finish_pos", "hist_best_finish_pos",
    "hist_win_rate", "hist_place_rate",
    "last_finish_pos", "last_asr",
    # Base profile metrics (14)
    "hist_asr_mean", "hist_asr_std", "hist_asr_trend",
    "hist_speed_mean", "hist_speed_std", "hist_speed_trend",
    "hist_fap_mean", "hist_fap_std", "hist_fap_trend",
    "hist_fi_mean", "hist_fi_std",
    "hist_edi_mean", "hist_pa_mean",
    "racing_style_enc",
    # Advanced metrics (15)
    "peak_speed_ms", "finishing_burst", "speed_decay_rate",
    "max_acceleration", "kinetic_energy_index", "weight_efficiency",
    "power_output_watts", "turn_penalty_ms", "draw_extra_distance_m",
    "positioning_cost", "drafting_factor", "form_trend",
    "freshness", "distance_aptitude_ms", "track_affinity_ms",
    # Jockey features (3)
    "jockey_win_rate", "jockey_place_rate", "jockey_n_rides",
    # Venue/distance affinity (6)
    "n_races_same_venue", "venue_avg_finish", "venue_place_rate",
    "n_races_similar_dist", "dist_avg_finish", "dist_place_rate",
    # History depth (1)
    "n_history_races",
    # Relative features (1)
    "weight_vs_field_avg",
    # Changes (3)
    "dist_change", "weight_change", "last_speed_ms",
    # Section time metadata (2)
    "has_section_times", "n_sections",
    # Interactions (2)
    "gate_x_venue", "weight_x_distance",
    # Last race FAP (1)
    "last_fap",
    # ========== PHASE 1 FEATURES (28) — A/B Consensus-Based Signals ==========
    # Consensus Module (4) — Encodes A & B model agreement signals
    "consensus_agreement_signal",
    "consensus_agreement_strength",
    "consensus_divergence_factor",
    "consensus_is_consensus_pick",
    # Confidence Module (5) — Encodes selection position confidence
    "conf_a_position",
    "conf_b_position",
    "conf_combined_confidence",
    "conf_agreement",
    "conf_banker_signal",
    # Scenario Module (3) — Venue & field-level context
    "scenario_venue_alignment",
    "scenario_field_strength",
    "scenario_expected_uncertainty",
    # Recency Module (5) — Recent form & momentum
    "recency_3races_win_rate",
    "recency_6races_trend",
    "recency_form_momentum",
    "recency_layoff_penalty",
    "recency_strength",
    # Pairing Module (4) — Jockey-horse combination effects
    "pairing_jockey_horse_affinity",
    "pairing_jockey_recent_form",
    "pairing_distance_affinity",
    "pairing_confidence",
    # Relative Strength Module (5) — Field-relative performance
    "strength_vs_field_speed",
    "strength_vs_field_winrate",
    "strength_field_dominance",
    "strength_upset_potential",
    "strength_favorite_indicator",
]

# Phase 1 feature names (for enricher.py reference)
PHASE1_FEATURE_NAMES = [
    "consensus_agreement_signal",
    "consensus_agreement_strength",
    "consensus_divergence_factor",
    "consensus_is_consensus_pick",
    "conf_a_position",
    "conf_b_position",
    "conf_combined_confidence",
    "conf_agreement",
    "conf_banker_signal",
    "scenario_venue_alignment",
    "scenario_field_strength",
    "scenario_expected_uncertainty",
    "recency_3races_win_rate",
    "recency_6races_trend",
    "recency_form_momentum",
    "recency_layoff_penalty",
    "recency_strength",
    "pairing_jockey_horse_affinity",
    "pairing_jockey_recent_form",
    "pairing_distance_affinity",
    "pairing_confidence",
    "strength_vs_field_speed",
    "strength_vs_field_winrate",
    "strength_field_dominance",
    "strength_upset_potential",
    "strength_favorite_indicator",
]

# Encoding maps (used by backtest runner)
VENUE_ENC = {"SHA_TIN": 0, "HAPPY_VALLEY": 1}
CONDITION_ENC = {
    "FIRM": 0, "GOOD_TO_FIRM": 1, "GOOD": 2,
    "GOOD_TO_YIELDING": 3, "YIELDING": 4, "SLOW": 5, "HEAVY": 6,
}
TREND_ENC = {"improving": 1, "stable": 0, "declining": -1}
STYLE_ENC = {"front-runner": 0, "stalker": 1, "closer": 2, "unknown": -1}
