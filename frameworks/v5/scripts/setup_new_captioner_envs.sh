#!/usr/bin/env bash
set -euo pipefail

# Prepare isolated environments for the two new RewardV2 Captioner branches.
# This script only clones already validated environments; it never upgrades or
# mutates the source environments and it never downloads model weights.
umask 000

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHARED_ROOT="$(cd "${PROJECT_ROOT}/../.." && pwd)"
CONDA_BIN="${SHARED_ROOT}/miniconda3/bin/conda"
REPORT_DIR="${PROJECT_ROOT}/reports/new_captioner_envs"

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda executable is missing: ${CONDA_BIN}" >&2
  exit 2
fi
mkdir -p "${REPORT_DIR}"

clone_or_refuse() {
  local name="$1" source="$2" target="$3"
  local marker="${target}/.dualisl_captioner_env.json"
  if [[ -e "${target}" ]]; then
    if [[ ! -f "${marker}" ]]; then
      echo "Refusing to overwrite existing unmarked environment: ${target}" >&2
      exit 3
    fi
    "${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python" - "${marker}" "${name}" "${source}" "${target}" <<'PY'
import json, pathlib, sys
marker, name, source, target = map(pathlib.Path, sys.argv[1:])
value = json.loads(marker.read_text(encoding="utf-8"))
if value.get("name") != str(name) or value.get("source") != str(source) or value.get("target") != str(target):
    raise SystemExit(f"Existing environment marker does not match requested identity: {marker}")
PY
    echo "Reusing verified environment: ${target}"
  else
    echo "Cloning ${source} -> ${target}"
    # --override-channels avoids an unnecessary Anaconda ToS lookup.  A clone
    # only copies the source prefix; it does not need network package solving.
    "${CONDA_BIN}" create --prefix "${target}" --clone "${source}" --yes --offline \
      --override-channels -c "file://${SHARED_ROOT}/miniconda3/pkgs"
  fi
  "${SHARED_ROOT}/miniconda3/envs/qwen3-tts/bin/python" - "${marker}" "${name}" "${source}" "${target}" <<'PY'
import json, pathlib, sys
marker = pathlib.Path(sys.argv[1])
marker.write_text(json.dumps({
    "name": sys.argv[2], "source": sys.argv[3], "target": sys.argv[4],
    "purpose": "DualISL-Train RewardV2 isolated Captioner environment",
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

clone_or_refuse \
  "midasheng-0p6b-captioner" \
  "${SHARED_ROOT}/miniconda3/envs/midasheng-captioner" \
  "${SHARED_ROOT}/miniconda3/envs/midasheng-0p6b-captioner"
clone_or_refuse \
  "qwen2_5-omni-3b-captioner" \
  "${SHARED_ROOT}/miniconda3/envs/qwen3-captioner" \
  "${SHARED_ROOT}/miniconda3/envs/qwen2_5-omni-3b-captioner"

validate_env() {
  local name="$1" env_prefix="$2"
  local report="${REPORT_DIR}/${name}.json"
  local python="${env_prefix}/bin/python"
  if [[ ! -x "${python}" ]]; then
    echo "Cloned environment has no executable Python: ${python}" >&2
    exit 4
  fi
  "${python}" - "${name}" "${env_prefix}" "${report}" <<'PY'
import json, pathlib, subprocess, sys
name, prefix, report = sys.argv[1:]
packages = ["torch", "transformers", "peft", "accelerate", "flash_attn", "librosa", "soundfile"]
expected_versions = {
    "python": "3.10.20",
    "torch": "2.6.0+cu124",
    "transformers": "4.57.3",
    "peft": "0.18.1",
    "accelerate": "1.13.0",
    "flash_attn": "2.7.4.post1",
    "librosa": "0.11.0",
    "soundfile": "0.14.0",
}
if name.startswith("qwen2"):
    packages += ["qwen_omni_utils", "Qwen2_5OmniThinkerForConditionalGeneration", "Qwen2_5OmniProcessor"]
else:
    packages += ["AutoModelForCausalLM", "AutoProcessor"]
import torch
versions = {"python": sys.version.split()[0], "torch_cuda": torch.version.cuda, "cuda_available": bool(torch.cuda.is_available())}
for package in packages:
    if package.startswith("Qwen2") or package.startswith("Auto"):
        continue
    try:
        module = __import__(package)
        versions[package] = getattr(module, "__version__", "imported")
    except Exception as exc:
        raise RuntimeError(f"Cannot import {package}: {exc}") from exc
version_mismatches = {
    package: {"expected": expected, "actual": versions.get(package)}
    for package, expected in expected_versions.items()
    if versions.get(package) != expected
}
if version_mismatches:
    raise RuntimeError(
        "Existing Captioner environment has incompatible versions; refusing to reuse it: "
        + json.dumps(version_mismatches, sort_keys=True)
    )
from transformers import AutoModelForCausalLM, AutoProcessor
if name.startswith("qwen2"):
    from transformers import Qwen2_5OmniThinkerForConditionalGeneration, Qwen2_5OmniProcessor
    versions["qwen2_classes"] = True
else:
    versions["midasheng_auto_classes"] = True
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for the H100 Captioner environments")
left = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
right = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
result = left @ right
if result.dtype != torch.bfloat16 or not bool(torch.isfinite(result).all()):
    raise RuntimeError("BF16 CUDA kernel validation failed")
pip_check = subprocess.run([str(pathlib.Path(prefix) / "bin" / "pip"), "check"], text=True, capture_output=True)
if pip_check.returncode:
    raise RuntimeError("pip check failed: " + pip_check.stdout + pip_check.stderr)
payload = {"status": "ok", "name": name, "prefix": prefix, "versions": versions,
           "pip_check": (pip_check.stdout or "").strip(), "bf16_cuda_kernel": "ok"}
pathlib.Path(report).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  "${CONDA_BIN}" list --prefix "${env_prefix}" --explicit > "${REPORT_DIR}/${name}.conda-explicit.txt"
  "${python}" -m pip freeze > "${REPORT_DIR}/${name}.pip-freeze.txt"
  echo "Validated ${name}; report: ${report}"
}

validate_env "midasheng-0p6b-captioner" "${SHARED_ROOT}/miniconda3/envs/midasheng-0p6b-captioner"
validate_env "qwen2_5-omni-3b-captioner" "${SHARED_ROOT}/miniconda3/envs/qwen2_5-omni-3b-captioner"
echo "New Captioner environments are ready. No model files were changed."
