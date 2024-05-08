python train_action.py \
--config configs/action/MB_ft_NTU60_xsub.yaml \
--pretrained checkpoint/pretrain/ \
--checkpoint checkpoint/action/PD_121 \
--selection best_epoch.bin \
--datanum 121