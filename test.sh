for datanum in 120_2 ; do
    for kidx in 0 1 2 3 4; do
        for date in 0613; do
            python train_action.py \
            --config configs/action/MB_ft_NTU60_xsub.yaml \
            --datanum $datanum \
            --kidx $kidx \
            --evaluate checkpoint/action/${date}_PD_$datanum/$kidx/best_epoch.bin
            # --test checkpoint/action/${date}_PD_$datanum/$kidx/best_epoch.bin
            mv temp.txt checkpoint/action/${date}_PD_$datanum/$kidx/val.txt
        done
    done
done