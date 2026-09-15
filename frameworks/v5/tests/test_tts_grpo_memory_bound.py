from __future__ import annotations

import json

import pytest

from dual_isl_train.config import load_config
from dual_isl_train.distributed import cost_bucketed_distributed_schedule
from dual_isl_train.telemetry import ProgressLogger
from dual_isl_train.workers.qwen_voice_design import candidate_backward_scale, heartbeat_frame_due


def _schedules(count: int, world_size: int):
    rows = [{"id": f"group-{index}", "cost": index + 1} for index in range(count)]
    return rows, [
        cost_bucketed_distributed_schedule(
            rows, {"epochs": 1, "shuffle": True}, 42, rank, world_size,
            lambda row: row["cost"],
        )
        for rank in range(world_size)
    ]


@pytest.mark.parametrize(("count", "world_size", "steps", "padding"), [
    (184, 7, 27, 5),
    (196, 8, 25, 4),
    (223, 8, 28, 1),
])
def test_cost_bucketed_schedule_preserves_every_row(count, world_size, steps, padding):
    rows, schedules = _schedules(count, world_size)
    assert {len(schedule) for schedule in schedules} == {steps}
    active = [row["id"] for schedule in schedules for _, _, row, padded in schedule if not padded]
    assert sorted(active) == sorted(row["id"] for row in rows)
    assert len(active) == len(set(active)) == count
    assert sum(padded for schedule in schedules for _, _, _, padded in schedule) == padding


def test_cost_bucketed_schedule_clusters_similar_work_and_rotates_heaviest_rank():
    _, schedules = _schedules(196, 8)
    heaviest_ranks = set()
    for step in range(25):
        assigned = [schedules[rank][step][2] for rank in range(8)]
        costs = [row["cost"] for row in assigned]
        assert max(costs) - min(costs) <= 7
        heaviest_ranks.add(costs.index(max(costs)))
    assert len(heaviest_ranks) > 1


def test_cost_bucketed_schedule_is_seed_deterministic():
    rows = [{"id": f"group-{index}", "cost": index % 3} for index in range(31)]
    first = cost_bucketed_distributed_schedule(
        rows, {"epochs": 1, "shuffle": True}, 17, 2, 8, lambda row: row["cost"],
    )
    second = cost_bucketed_distributed_schedule(
        rows, {"epochs": 1, "shuffle": True}, 17, 2, 8, lambda row: row["cost"],
    )
    assert first == second


def test_candidate_backward_scale_matches_group_mean_and_active_rank_scaling():
    candidate_gradients = [2.0, -1.0, 5.0, 4.0]
    scale = candidate_backward_scale(len(candidate_gradients), world_size=8, active_count=7, padded=False)
    sequential = sum(gradient * scale for gradient in candidate_gradients)
    old_group_backward = sum(candidate_gradients) / len(candidate_gradients) * 8 / 7
    assert sequential == pytest.approx(old_group_backward)
    assert candidate_backward_scale(4, world_size=8, active_count=7, padded=True) == 0.0


@pytest.mark.parametrize("processed", [32, 64, 95])
def test_frame_heartbeat_fires_on_interval_and_final_frame(processed):
    assert heartbeat_frame_due(processed, 95, 32) is (processed in {32, 64, 95})


def test_progress_logger_flushes_rank_scoped_jsonl(tmp_path):
    logger = ProgressLogger(tmp_path / "progress", rank=5)
    logger.log("frame_progress", step=2, processed_frames=64, total_frames=471)
    record = json.loads(logger.path.read_text().strip())
    assert record["event"] == "frame_progress"
    assert record["rank"] == 5
    assert record["step"] == 2
    assert record["processed_frames"] == 64
    assert isinstance(record["timestamp"], float)


@pytest.mark.parametrize("config_path", [
    "configs/gpu_smoke_7gpu_h100.yaml",
    "configs/gpu_smoke_8gpu_h100.yaml",
    "configs/train_7gpu_h100.yaml",
    "configs/train_8gpu_h100.yaml",
])
def test_h100_configs_require_memory_bounded_tts_grpo_schedule(config_path):
    config = load_config(config_path)
    phase = config["tts"]["training"]["phases"]["grpo"]
    assert phase["schedule"] == "length_bucketed"
    assert phase["heartbeat_frames"] == 32
