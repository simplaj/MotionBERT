from lib.data.dataset_action import PoseTorchDataset
import os 


def merge(dir_path, mode):
    lines = []
    for subname in os.listdir(dir_path):
        subdir = os.path.join(dir_path, subname)
        if subname.isdigit():
            with open(f'{subdir}/{mode}.txt', 'r') as file:
                data = file.readlines()
            lines += data
    with open(f'{dir_path}/{mode}.txt', 'w') as file:
        file.writelines(lines)

if __name__ == '__main__':
    # for datanum in [109, 121, 128]:
    for mode in ['val', 'test']:
        merge(f'/home/tzh/Project/MotionBERT/checkpoint/action/0608_PD_foot_120', mode)
    # ds = PoseTorchDataset(mode='train', mask=None, random_move=False, scale_range=[2,2])
    # for d in ds:
    #     print(d['attr'])