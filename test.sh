for datanum in 120_2 ; do
    for kidx in 0 1 2 3 4; do
        for date in 0606; do
            python train_action.py \
            --config configs/action/MB_ft_NTU60_xsub.yaml \
            --datanum $datanum \
            --kidx $kidx \
            --evaluate checkpoint/action/${date}_PD_foot_121/$kidx/best_epoch.bin \
            --test checkpoint/action/${date}_PD_foot_$datanum/$kidx/best_epoch.bin
            mv temp.txt checkpoint/action/${date}_PD_foot_121/$kidx/test.txt
        done
    done
done