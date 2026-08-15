export CUDA_VISIBLE_DEVICES=0
nohup python3 -u train.py \
        --data_root './dataset/data' \
        --data_target "VisA" \
        --test_ano_setting "general" \
        --backbone_name 'dinov2_vits14' \
        --epochs 20 \
        --batch_size 16 \
        --worker 8 \
        --train_choice 600 \
        --learning_rate 1e-3 \
        --image_size 448 \
        --weight_decay 1e-4 \
        --scheduler_type 'cosine' \
        --warmup_epochs 2 \
        --grad_clip 0.0 \
        --n_shot 1 --a_shot 1 \
        --nheads 8 \
        --topk 12 \
        --topr 4 \
        --proj_alpha 0.8 \
        --num_learnable_vectors 45 \
        --g_loss_w 1.0 \
        --or_loss_w 0.8 \
        --gpu_id 0 \
        --seed 17 \
        --save_path ./outputs \
        > train_mvtec_test_visa.log 2>&1 & 