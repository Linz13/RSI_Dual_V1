from __future__ import annotations

FRAMEWORK_VERSION = "DualRSI-Train-V6-AudioOnly4Attr-0.6"
CYCLE_SFT_SELECTION = "semantic_top1"
SCHEMA_VERSION = "dual_isl_train.psyn.v1"
TRAJECTORY_VERSION = "dual_isl_train.trajectory.v1"
RUN_STATE_VERSION = 1
CHECKPOINT_METADATA = "dual_isl_train_training.json"
WRAPPER_KEY = "Target_JSON_Schema"

CLOSED_FIELDS = {
    "semantic_content.language", "semantic_content.intent",
    "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre", "speaker_profile.accent",
    "paralinguistic.speaking_rate", "paralinguistic.pitch_level", "paralinguistic.volume_level",
    "paralinguistic.emotion", "paralinguistic.emotion_intensity", "paralinguistic.emphasis.level",
    "environment.recording_quality",
}
OPEN_FIELDS = {
    "semantic_content.transcript", "semantic_content.topic",
    "paralinguistic.emphasis.emphasized_text", "paralinguistic.prosody", "paralinguistic.pause",
    "environment.background_sound_events", "environment.acoustic_scene",
}
LIST_FIELDS = {
    "paralinguistic.emphasis.emphasized_text",
    "paralinguistic.nonverbal_vocalization",
    "environment.background_sound_events",
}
ALL_FIELDS = (
    "semantic_content.language", "semantic_content.transcript", "semantic_content.topic", "semantic_content.intent",
    "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre", "speaker_profile.accent",
    "paralinguistic.speaking_rate", "paralinguistic.pitch_level", "paralinguistic.volume_level",
    "paralinguistic.emotion", "paralinguistic.emotion_intensity",
    "paralinguistic.emphasis.level", "paralinguistic.emphasis.emphasized_text",
    "paralinguistic.prosody", "paralinguistic.pause", "paralinguistic.nonverbal_vocalization",
    "environment.background_sound_events", "environment.recording_quality", "environment.acoustic_scene",
)
ENVIRONMENT_FIELDS = {field for field in ALL_FIELDS if field.startswith("environment.")}
SOURCE_FIELDS = tuple(field for field in ALL_FIELDS if field not in ENVIRONMENT_FIELDS)
SYNTHESIZABLE_FIELDS = (
    "semantic_content.language", "semantic_content.transcript",
    "speaker_profile.gender", "speaker_profile.age", "speaker_profile.timbre", "speaker_profile.accent",
    "paralinguistic.speaking_rate", "paralinguistic.pitch_level", "paralinguistic.volume_level",
    "paralinguistic.emotion", "paralinguistic.emotion_intensity",
    "paralinguistic.emphasis.level", "paralinguistic.emphasis.emphasized_text",
    "paralinguistic.prosody", "paralinguistic.pause", "paralinguistic.nonverbal_vocalization",
)
SYNTHESIZABLE_FIELD_SET = frozenset(SYNTHESIZABLE_FIELDS)
TTS_CONTROL_FIELDS = frozenset(
    field for field in SYNTHESIZABLE_FIELDS
    if field.startswith("speaker_profile.") or field.startswith("paralinguistic.")
)

SOURCE_DOMAIN_TARGET = "source_domain"
GENERATED_INTERMEDIATE = "generated_intermediate"
