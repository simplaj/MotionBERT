python flops.py \
        --config configs/action/MB_ft_NTU60_xsub.yaml \
        --pretrained checkpoint/pretrain/ \
        --checkpoint checkpoint/action/PD_foot_$datanum/$kidx \
        --selection best_epoch.bin \
        --datanum 1 \
        --print_freq 50 \
        --kidx 1