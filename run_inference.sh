#!/bin/bash

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

NUM_GPUS=4

init_gpu_pool() {
    local n=$1
    local fifo
    fifo="$(mktemp -u "${TMPDIR:-/tmp}/gpu_pool.XXXXXX")"
    mkfifo "$fifo"
    exec 3<>"$fifo"
    rm -f "$fifo"
    local i
    for ((i = 0; i < n; i++)); do
        echo "$i" >&3
    done
}

close_gpu_pool() {
    exec 3>&-
}

init_gpu_pool "$NUM_GPUS"

jobs=(
    'MVTecAD|general|visa_2_mvtec|n1a1|outputs'
    'VisA|general|mvtec_2_visa|n1a1|outputs'
    'BTAD|general|mvtec_2_visa|n1a1|outputs'
    'MPDD|general|mvtec_2_visa|n1a1|outputs'
    'AITEX|general|mvtec_2_visa|n1a1|outputs'
    'BraTS2021|general|mvtec_2_visa|n1a1|outputs'
    'Liver|general|mvtec_2_visa|n1a1|outputs'
    'RESC|general|mvtec_2_visa|n1a1|outputs'
)

N_SHOT=1
A_SHOT=1

for spec in "${jobs[@]}"; do
    IFS='|' read -r target_set test_ano_setting model_weight fs_set save_root <<<"$spec"
    read -r g <&3
    exp_name="${target_set}"
    log_file="${save_root}/${exp_name}_${fs_set}_${test_ano_setting}_infer.log"
    echo "################# Queue ${target_set} (${test_ano_setting}) -> GPU ${g} ..."

    (
        export CUDA_VISIBLE_DEVICES=0
        trap 'echo "$g" >&3' EXIT
        python3 -u infer_ours.py \
            --data_root ./dataset/data \
            --data_target "${target_set}" \
            --test_ano_setting "${test_ano_setting}" \
            --checkpoint_path "./outputs/${fs_set}_${test_ano_setting}_${model_weight}.pth" \
            --backbone_name dinov2_vits14 \
            --gpu_id 0 --n_shot "${N_SHOT}" --a_shot "${A_SHOT}" \
            --save_path "${save_root}/${fs_set}_${test_ano_setting}/${target_set}"
    ) >"${log_file}" 2>&1 &

done

wait
close_gpu_pool
echo "The experiments finished."