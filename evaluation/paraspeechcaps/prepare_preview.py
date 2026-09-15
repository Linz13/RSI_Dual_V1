#!/usr/bin/env python3
"""Export the ParaSpeechCaps paper test split and build a local HTML preview."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
PARQUET_PATH = ROOT / "data" / "test-00000-of-00001.parquet"
AUDIO_ROOTS = {
    "ears": ROOT / "audio" / "ears",
    "expresso": ROOT / "audio" / "expresso",
    "voxceleb": ROOT / "audio" / "voxceleb",
}
LIST_COLUMNS = {
    "text_description",
    "intrinsic_tags",
    "situational_tags",
    "basic_tags",
    "all_tags",
}


def enrich_rows() -> list[dict]:
    rows = pq.read_table(PARQUET_PATH).to_pylist()
    for index, row in enumerate(rows):
        row["benchmark_index"] = index
        candidate = AUDIO_ROOTS[row["source"]] / row["relative_audio_path"]
        row["local_audio_path"] = (
            candidate.relative_to(ROOT).as_posix() if candidate.is_file() else None
        )
    return rows


def write_jsonl(rows: list[dict]) -> None:
    output = ROOT / "data" / "test.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(rows: list[dict]) -> None:
    output = ROOT / "data" / "test.csv"
    columns = list(rows[0])
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            serialized = {
                key: json.dumps(value, ensure_ascii=False)
                if key in LIST_COLUMNS and value is not None
                else value
                for key, value in row.items()
            }
            writer.writerow(serialized)


def build_summary(rows: list[dict]) -> dict:
    source_counts = Counter(row["source"] for row in rows)
    tag_counts = Counter(row["tag_of_interest"] for row in rows)
    return {
        "examples": len(rows),
        "duration_seconds": round(sum(row["duration"] for row in rows), 3),
        "sources": dict(sorted(source_counts.items())),
        "unique_tags_of_interest": len(tag_counts),
        "tag_counts": dict(sorted(tag_counts.items())),
        "locally_playable_audio": sum(row["local_audio_path"] is not None for row in rows),
        "official_dataset": "https://huggingface.co/datasets/ajd12342/paraspeechcaps",
        "paper": "https://arxiv.org/abs/2503.04713",
        "license": "CC BY-NC-SA 4.0",
    }


def tags_html(tags: list[str] | None) -> str:
    if not tags:
        return '<span class="muted">—</span>'
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)


def row_html(row: dict) -> str:
    prompts = "<br>".join(html.escape(text.strip()) for text in row["text_description"])
    if row["local_audio_path"]:
        audio_url = quote(row["local_audio_path"], safe="/")
        audio = f'<audio controls preload="none" src="{audio_url}"></audio>'
        has_audio = "yes"
    else:
        audio = '<span class="muted">仅元数据</span>'
        has_audio = "no"
    return f"""
      <tr data-source="{html.escape(row['source'])}" data-audio="{has_audio}"
          data-search="{html.escape(' '.join([
              row['tag_of_interest'], row['transcription'], *row['text_description'],
              *(row['all_tags'] or []), row['relative_audio_path']
          ]).lower(), quote=True)}">
        <td>{row['benchmark_index']}</td>
        <td><strong>{html.escape(row['tag_of_interest'])}</strong><br>{tags_html(row['all_tags'])}</td>
        <td><span class="source">{html.escape(row['source'])}</span><br>{audio}</td>
        <td>{prompts}</td>
        <td>{html.escape(row['transcription'].strip())}</td>
        <td class="path">{html.escape(row['relative_audio_path'])}</td>
      </tr>"""


def write_html(rows: list[dict], summary: dict) -> None:
    source_options = "".join(
        f'<option value="{source}">{source} ({count})</option>'
        for source, count in summary["sources"].items()
    )
    table_rows = "\n".join(row_html(row) for row in rows)
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ParaSpeechCaps Test Benchmark Preview</title>
  <style>
    :root {{ color-scheme: light dark; --line: #8a8a8a55; --accent: #6d5dfc; }}
    body {{ font: 15px/1.5 system-ui, sans-serif; margin: 24px; }}
    h1 {{ margin-bottom: 4px; }}
    .note {{ max-width: 1000px; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid var(--line); border-radius: 10px; padding: 12px 16px; }}
    .card strong {{ display: block; font-size: 22px; }}
    .filters {{ position: sticky; top: 0; z-index: 2; padding: 12px 0; background: Canvas; }}
    input, select {{ font: inherit; padding: 8px; margin-right: 8px; }}
    input[type=search] {{ width: min(520px, 55vw); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 61px; background: Canvas; z-index: 1; }}
    tr:nth-child(even) {{ background: color-mix(in srgb, CanvasText 4%, Canvas); }}
    .tag {{ display: inline-block; border: 1px solid var(--line); border-radius: 999px; padding: 1px 6px; margin: 2px; font-size: 12px; }}
    .source {{ text-transform: uppercase; font-size: 12px; letter-spacing: .05em; }}
    .muted, .path {{ color: GrayText; }}
    .path {{ font: 12px/1.35 ui-monospace, monospace; word-break: break-all; }}
    audio {{ width: 230px; max-width: 25vw; margin-top: 6px; }}
  </style>
</head>
<body>
  <h1>ParaSpeechCaps 论文主测试集</h1>
  <p class="note">这是论文中从 PSC-Base holdout 构造的 246 条 tag-balanced 主测试 benchmark。
  官方发布包仅包含 caption、标签、转写和源音频相对路径，不包含音频文件。
  本机已有的 EARS 数据使其中 26 条可以直接播放；VoxCeleb 和 Expresso 行目前显示“仅元数据”。</p>
  <div class="cards">
    <div class="card"><strong>{summary['examples']}</strong>测试样本</div>
    <div class="card"><strong>{summary['unique_tags_of_interest']}</strong>目标 rich tags</div>
    <div class="card"><strong>{summary['duration_seconds'] / 60:.1f} 分钟</strong>参考音频总时长</div>
    <div class="card"><strong>{summary['locally_playable_audio']}</strong>本地可播放</div>
  </div>
  <div class="filters">
    <input id="query" type="search" placeholder="搜索 tag、prompt、transcript 或路径…">
    <select id="source"><option value="">所有来源</option>{source_options}</select>
    <label><input id="audioOnly" type="checkbox">只看可播放音频</label>
    <span id="visible"></span>
  </div>
  <table>
    <thead><tr><th>#</th><th>目标标签 / 全部标签</th><th>来源 / 音频</th><th>Style prompt</th><th>Transcript</th><th>官方相对路径</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
  <script>
    const rows = [...document.querySelectorAll('tbody tr')];
    const query = document.querySelector('#query');
    const source = document.querySelector('#source');
    const audioOnly = document.querySelector('#audioOnly');
    const visible = document.querySelector('#visible');
    function filterRows() {{
      const q = query.value.trim().toLowerCase();
      let count = 0;
      for (const row of rows) {{
        const show = (!q || row.dataset.search.includes(q)) &&
          (!source.value || row.dataset.source === source.value) &&
          (!audioOnly.checked || row.dataset.audio === 'yes');
        row.hidden = !show;
        if (show) count++;
      }}
      visible.textContent = `显示 ${{count}} / ${{rows.length}} 条`;
    }}
    query.addEventListener('input', filterRows);
    source.addEventListener('change', filterRows);
    audioOnly.addEventListener('change', filterRows);
    filterRows();
  </script>
</body>
</html>
"""
    (ROOT / "preview.html").write_text(document, encoding="utf-8")


def main() -> None:
    rows = enrich_rows()
    summary = build_summary(rows)
    write_jsonl(rows)
    write_csv(rows)
    (ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_html(rows, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
