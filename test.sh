for datanum in 109 121 128 ; do
    for kidx in 0 1 2 3 4; do
        python train_action.py \
        --config configs/action/MB_ft_NTU60_xsub.yaml \
        --datanum $datanum \
        --kidx $kidx \
        --evaluate checkpoint/action/PD_$datanum/$kidx/best_epoch.bin
        # --test checkpoint/action/PD_$datanum/$kidx/best_epoch.bin \
        mv temp.txt checkpoint/action/PD_$datanum/$kidx/val.txt
    done
done