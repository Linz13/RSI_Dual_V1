from __future__ import annotations

from dual_isl_train.distributed import distributed_schedule, shard_indexed_rows


def test_seven_rank_sharding_preserves_groups_and_builds_balanced_schedule():
    grouped_rows = [
        {"id": f"group-{group}::candidate-{candidate}", "group_id": f"group-{group}"}
        for group in range(10)
        for candidate in range(4)
    ]
    shards = [shard_indexed_rows(grouped_rows, rank, 7) for rank in range(7)]

    assigned_indices = [index for shard in shards for index, _ in shard]
    assert sorted(assigned_indices) == list(range(len(grouped_rows)))
    for group in range(10):
        owners = {
            rank
            for rank, shard in enumerate(shards)
            if any(row["group_id"] == f"group-{group}" for _, row in shard)
        }
        assert len(owners) == 1

    training_rows = [{"id": f"group-{group}"} for group in range(184)]
    schedules = [
        distributed_schedule(training_rows, {"epochs": 1, "shuffle": False}, 42, rank, 7)
        for rank in range(7)
    ]
    assert {len(schedule) for schedule in schedules} == {27}
    active_ids = [row["id"] for schedule in schedules for _, _, row, padded in schedule if not padded]
    assert len(active_ids) == 184
    assert len(set(active_ids)) == 184
    assert sum(padded for schedule in schedules for _, _, _, padded in schedule) == 5
