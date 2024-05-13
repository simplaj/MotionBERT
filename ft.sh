for datanum in 109 121 128 ; do
    for kidx in 0 1 2 3 4; do
        python train_action.py \
        --config configs/action/MB_ft_NTU60_xsub.yaml \
        --pretrained checkpoint/pretrain/ \
        --checkpoint checkpoint/action/PD_$datanum/$kidx \
        --selection best_epoch.bin \
        --datanum $datanum \
        --print_freq 50 \
        --kidx $kidx
    done
done