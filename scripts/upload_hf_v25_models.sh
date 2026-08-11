#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/outputs/hf_upload_logs/v25-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

MODELS=(
  "vanch007/mlx-indextts2-2.5-fp16|models/mlx-IndexTTS-2.5"
  "vanch007/mlx-indextts2-2.5-8bit|models/mlx-IndexTTS-2.5-8bit"
  "vanch007/mlx-indextts2-2.5-fp32|models/mlx-IndexTTS-2.5-fp32"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

validate_local() {
  local directory="$1"
  "$ROOT_DIR/.venv/bin/python" - "$directory" <<'PY'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
report = json.loads((directory / "conversion_report.json").read_text())
manifest = json.loads((directory / "model_manifest.json").read_text())

if report.get("status") != "pass" or report.get("model_version") != "2.5":
    raise SystemExit(f"invalid conversion report: {directory}")
if manifest.get("model_version") != "2.5":
    raise SystemExit(f"invalid model manifest: {directory}")

required = {
    "README.md",
    "LICENSE",
    "gpt.safetensors",
    "codec.safetensors",
    "s2mel.safetensors",
    "bigvgan.safetensors",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "config.yaml",
    "config.json",
    "model_manifest.json",
    "conversion_report.json",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
}
missing = sorted(name for name in required if not (directory / name).is_file())
if missing:
    raise SystemExit(f"{directory} missing files: {missing}")

for name, expected_size in report["artifacts"].items():
    actual_size = (directory / name).stat().st_size
    if actual_size != expected_size:
        raise SystemExit(
            f"{directory / name}: expected {expected_size} bytes, got {actual_size}"
        )

card = (directory / "README.md").read_text()
if "https://github.com/vanch007/mlx-indextts2" not in card:
    raise SystemExit(f"model card does not link to the GitHub project: {directory}")

print(f"local validation passed: {directory}")
PY
}

verify_remote() {
  local repo="$1"
  "$ROOT_DIR/.venv/bin/python" - "$repo" <<'PY'
import json
import sys

from huggingface_hub import HfApi, hf_hub_download

repo_id = sys.argv[1]
api = HfApi()
names = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
required = {
    "README.md",
    "LICENSE",
    "gpt.safetensors",
    "codec.safetensors",
    "s2mel.safetensors",
    "bigvgan.safetensors",
    "multilingual_zh_ja_yue_char_del.tiktoken",
    "config.yaml",
    "config.json",
    "model_manifest.json",
    "conversion_report.json",
    "feat1.pt",
    "feat2.pt",
    "wav2vec2bert_stats.pt",
}
missing = sorted(required - names)
if missing:
    raise SystemExit(f"{repo_id} missing remote files: {missing}")

manifest_path = hf_hub_download(repo_id, "model_manifest.json")
report_path = hf_hub_download(repo_id, "conversion_report.json")
readme_path = hf_hub_download(repo_id, "README.md")
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
with open(report_path, encoding="utf-8") as handle:
    report = json.load(handle)
with open(readme_path, encoding="utf-8") as handle:
    readme = handle.read()

if manifest.get("model_version") != "2.5" or report.get("status") != "pass":
    raise SystemExit(f"{repo_id} remote manifest/report validation failed")
if "https://github.com/vanch007/mlx-indextts2" not in readme:
    raise SystemExit(f"{repo_id} model card has the wrong project URL")

info = api.repo_info(repo_id=repo_id, repo_type="model")
print(f"remote validation passed: {repo_id}@{info.sha} ({len(names)} files)")
PY
}

log "Preparing model cards and licenses"
"$ROOT_DIR/.venv/bin/python" scripts/prepare_hf_model_cards.py

log "Hugging Face account: $(hf auth whoami | head -n 1)"
log "Upload logs: $LOG_DIR"

for item in "${MODELS[@]}"; do
  repo="${item%%|*}"
  directory="${item#*|}"
  model_log="$LOG_DIR/${repo#vanch007/}.log"

  validate_local "$directory" | tee "$model_log"
  hf repo create "$repo" --repo-type model --exist-ok | tee -a "$model_log"

  log "Uploading $directory to $repo"
  hf upload-large-folder "$repo" "$directory" \
    --repo-type model \
    --num-workers "${HF_UPLOAD_WORKERS:-4}" \
    --no-bars \
    2>&1 | tee -a "$model_log"

  verify_remote "$repo" | tee -a "$model_log"
done

log "All IndexTTS 2.5 MLX model repositories uploaded and verified"
