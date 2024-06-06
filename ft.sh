for datanum in 121; do
    for kidx in 0 1 2 3 4; do
        torchrun --nproc_per_node=2 --master-port 23496 train_action.py \
        --config configs/action/MB_ft_NTU60_xsub.yaml \
        --pretrained checkpoint/pretrain/ \
        --checkpoint checkpoint/action/PD_foot_$datanum/$kidx \
        --selection best_epoch.bin \
        --datanum $datanum \
        --print_freq 50 \
        --kidx $kidx
    done
done