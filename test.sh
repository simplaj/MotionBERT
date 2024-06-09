for datanum in 120 ; do
    for kidx in 0 1 2 3 4; do
        python train_action.py \
        --config configs/action/MB_ft_NTU60_xsub.yaml \
        --datanum $datanum \
        --kidx $kidx \
        --evaluate checkpoint/action/0608_PD_foot_$datanum/$kidx/best_epoch.bin \
        --test checkpoint/action/0608_PD_foot_$datanum/$kidx/best_epoch.bin
        mv temp.txt checkpoint/action/0608_PD_foot_$datanum/$kidx/test.txt
    done
done