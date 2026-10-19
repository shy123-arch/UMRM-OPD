# sourced by every shell entry point: repo config + interpreter defaults
VHO_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$VHO_REPO/configs/paths.env" ] && source "$VHO_REPO/configs/paths.env"
Q=${VHO_PY_QWEN:-python}; K=${VHO_PY_KIMODO:-python}; H=${VHO_PY_HERO:-python}; HL=${VHO_HERO_TORCHLIB:-}
C=${VHO_CACHE:-/tmp/vho_repo_cache}
hero() { env -u PYTORCH_CUDA_ALLOC_CONF CUDA_VISIBLE_DEVICES=$1 LD_LIBRARY_PATH=$HL:/usr/local/cuda/lib64 $H -u "${@:2}"; }  # torch 1.12 rejects expandable_segments
RUNSDIR=${VHO_RUNS:-$VHO_REPO/runs}; mkdir -p "$RUNSDIR/logs"
