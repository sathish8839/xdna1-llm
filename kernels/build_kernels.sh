#!/usr/bin/env bash
# Build the npu1 (XDNA1) kernels this project uses, from mlir-aie's programming examples, inside WSL/Linux.
#
#   MLIR_AIE=~/mlir-aie XRT_SDK_WIN=/mnt/c/.../xrt_sdk/xrt ./kernels/build_kernels.sh [out_dir] [set]
#
# sets: engine (default: the two chunk GEMMs the engine needs), gemv / qgemv (bf16 / Q4_K + Q6_K decode GEMVs),
#       bw (streaming probes),
#       shapes (per-shape GEMM benchmark kernels), smoke, all
# Requires: an mlir-aie checkout with its ironenv (utils/env_install.sh) and xclbinutil on PATH
# (on WSL use kernels/xclbinutil-wsl, which this script puts on PATH automatically).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$(realpath -m "${1:-$HERE/build}")"
SET="${2:-engine}"
MLIR_AIE="${MLIR_AIE:-$HOME/mlir-aie}"
mkdir -p "$OUT" "$HERE/.bin"

# xclbinutil: prefer a native one, else the WSL shim
if ! command -v xclbinutil >/dev/null; then
  install -m 755 "$HERE/xclbinutil-wsl" "$HERE/.bin/xclbinutil"
  export PATH="$HERE/.bin:$PATH"
fi

set --  # mlir-aie's env_setup.sh reads positional args
cd "$MLIR_AIE" || { echo "MLIR_AIE=$MLIR_AIE not found"; exit 1; }
set +u  # mlir-aie's setup scripts reference unset variables
# shellcheck disable=SC1091
source ironenv/bin/activate || { echo "no ironenv in $MLIR_AIE (run utils/env_install.sh)"; exit 1; }
# shellcheck disable=SC1091
source utils/env_setup.sh > /dev/null 2>&1
set -u
export PATH="$HERE/.bin:$PATH"
GEMM="$MLIR_AIE/programming_examples/basic/matrix_multiplication/whole_array"

gemm() {  # out_name M K N m k n dtype_in dtype_out
  local name=$1 M=$2 K=$3 N=$4 m=$5 k=$6 n=$7 di=$8 do=$9
  local sfx="${M}x${K}x${N}_${m}x${k}x${n}_4c"
  [[ -f "$OUT/$name.xclbin" ]] && { echo "skip  $name"; return; }
  (cd "$GEMM" && rm -rf build && NPU2=0 make devicename=npu M=$M K=$K N=$N m=$m k=$k n=$n \
     dtype_in=$di dtype_out=$do "build/final_$sfx.xclbin" "build/insts_$sfx.bin" > "$OUT/$name.log" 2>&1 < /dev/null)
  if [[ -f "$GEMM/build/final_$sfx.xclbin" ]]; then
    cp "$GEMM/build/final_$sfx.xclbin" "$OUT/$name.xclbin"; cp "$GEMM/build/insts_$sfx.bin" "$OUT/$name.bin"
    rm -f "$OUT/$name.log"; echo "built $name"
  else
    echo "FAIL  $name (see $OUT/$name.log)"
  fi
}

if [[ "$SET" == engine || "$SET" == all ]]; then
  # fp32 output needs n=32 tiles: a 64x64 fp32 C tile plus double-buffered A/B overflows the 64 KB tile memory
  gemm gemm_256x2048x1024_bf16_f32 256 2048 1024 64 64 32 bf16 f32
  gemm gemm_256x7680x1024_bf16_f32 256 7680 1024 64 64 32 bf16 f32
fi
if [[ "$SET" == shapes || "$SET" == all ]]; then
  # LFM3-1.7B projection shapes: pf_ = prefill (M=256), dc_ = decode stand-in (M=64)
  while read -r name K N; do
    gemm "pf_$name" 256 "$K" "$N" 64 64 64 bf16 bf16
    gemm "dc_$name" 64 "$K" "$N" 16 64 64 bf16 bf16
  done <<'EOF'
attn_qo 2048 2048
kv_fused 2048 1024
conv_in 2048 6144
ffn_gateup 2048 15360
ffn_down 7680 2048
lm_head 2048 12800
EOF
  gemm dc_conv_in_i8 64 2048 6144 16 64 64 i8 i32
fi
if [[ "$SET" == smoke || "$SET" == all ]]; then
  VSM="$MLIR_AIE/programming_examples/basic/vector_scalar_mul"
  if [[ ! -f "$OUT/smoke_vector_scalar_mul.xclbin" ]]; then
    (cd "$VSM" && rm -rf build && NPU2=0 make devicename=npu build/final_8192.xclbin build/insts_8192.bin \
       > "$OUT/smoke.log" 2>&1 < /dev/null) \
      && cp "$VSM/build/final_8192.xclbin" "$OUT/smoke_vector_scalar_mul.xclbin" \
      && cp "$VSM/build/insts_8192.bin" "$OUT/smoke_vector_scalar_mul.bin" && rm -f "$OUT/smoke.log" \
      && echo "built smoke_vector_scalar_mul" || echo "FAIL  smoke (see $OUT/smoke.log)"
  fi
fi

design() {  # out_name script.py [design args...]: an IRON design from kernels/designs
  local name=$1 py=$2; shift 2
  [[ -f "$OUT/$name.xclbin" ]] && { echo "skip  $name"; return; }
  if (cd "$OUT" && python3 "$HERE/designs/$py" --dev npu "$@" --xclbin-path="$OUT/$name.xclbin"         --insts-path="$OUT/$name.bin" > "$OUT/$name.log" 2>&1 < /dev/null); then
    rm -rf "$OUT/$name.log" "$OUT/$name.prj"; echo "built $name"
  else
    echo "FAIL  $name (see $OUT/$name.log)"
  fi
}

if [[ "$SET" == gemv || "$SET" == all ]]; then
  for shape in 1024x2048 2048x2048 6144x2048 15360x2048 2048x7680; do  # LFM3-1.7B projections (N x K)
    design "gemv_$shape" gemv.py -N "${shape%x*}" -K "${shape#*x}"
  done
fi
if [[ "$SET" == qgemv || "$SET" == all ]]; then
  # every build has the same cores (one context runs them all); each matrix shape needs its instruction stream
  for shape in 512x2048 2048x2048 2560x2048 3072x2048 6144x2048 15360x2048 2048x7680; do  # Q4_K tensors
    design "q4_$shape" qgemv.py -Q 4 -N "${shape%x*}" -K "${shape#*x}"
  done
  # Q6_K: attn_v, LM head and ffn_down in Q4_K_M; every tensor of an all-Q6_K model ("quality")
  for shape in 512x2048 2048x2048 3072x2048 6144x2048 15360x2048 128000x2048 2048x7680; do
    design "q6_$shape" qgemv.py -Q 6 -N "${shape%x*}" -K "${shape#*x}"
  done
fi
if [[ "$SET" == bw || "$SET" == all ]]; then
  for lanes in 1 2 4 8; do
    design "bw_l${lanes}_mb16" stream_bw.py --lanes "$lanes" --tile-kb 16 --total-mb 16
  done
fi
