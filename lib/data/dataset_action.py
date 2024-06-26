import torch
import numpy as np
import os
import pickle
import random
import copy
from torch.utils.data import Dataset, DataLoader
from lib.utils.utils_data import crop_scale, resample
from lib.utils.tools import read_pkl
    
def get_action_names(file_path = "data/action/ntu_actions.txt"):
    f = open(file_path, "r")
    s = f.read()
    actions = s.split('\n')
    action_names = []
    for a in actions:
        action_names.append(a.split('.')[1][1:])
    return action_names

def make_cam(x, img_shape):
    '''
        Input: x (M x T x V x C)
               img_shape (height, width)
    '''
    h, w = img_shape
    if w >= h:
        x_cam = x / w * 2 - 1
    else:
        x_cam = x / h * 2 - 1
    return x_cam

def coco2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        COCO: {0-nose 1-Leye 2-Reye 3-Lear 4Rear 5-Lsho 6-Rsho 7-Lelb 8-Relb 9-Lwri 10-Rwri 11-Lhip 12-Rhip 13-Lkne 14-Rkne 15-Lank 16-Rank}
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,11,:] + x[:,:,12,:]) * 0.5
    y[:,:,1,:] = x[:,:,12,:]
    y[:,:,2,:] = x[:,:,14,:]
    y[:,:,3,:] = x[:,:,16,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,13,:]
    y[:,:,6,:] = x[:,:,15,:]
    y[:,:,8,:] = (x[:,:,5,:] + x[:,:,6,:]) * 0.5
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,1,:] + x[:,:,2,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,7,:]
    y[:,:,13,:] = x[:,:,9,:]
    y[:,:,14,:] = x[:,:,6,:]
    y[:,:,15,:] = x[:,:,8,:]
    y[:,:,16,:] = x[:,:,10,:]
    return y
    
def dwpose2h36m(x):
    '''
        Input: x (M x T x V x C)
        
        dwpose: 
        0 - nose
        1 - neck
        2 - right shoulder
        3 - right elbow
        4 - right wrist
        5 - left shoulder
        6 - left elbow
        7 - left wrist
        8 - right hip
        9 - right knee
        10 - right ankle
        11 - left hip
        12 - left knee
        13 - left ankle
        14 - right eye
        15 - left eye
        16 - right ear
        17 - left ear
        0 - left big toe
        1 - left small toe
        2 - left heel
        3 - right big toe
        4 - right small toe
        5 - right heel 
        
        H36M:
        0: 'root',
        1: 'rhip',
        2: 'rkne',
        3: 'rank',
        4: 'lhip',
        5: 'lkne',
        6: 'lank',
        7: 'belly',
        8: 'neck',
        9: 'nose',
        10: 'head',
        11: 'lsho',
        12: 'lelb',
        13: 'lwri',
        14: 'rsho',
        15: 'relb',
        16: 'rwri'
    '''
    y = np.zeros(x.shape)
    y[:,:,0,:] = (x[:,:,8,:] + x[:,:,11,:]) * 0.5
    y[:,:,1,:] = x[:,:,8,:]
    y[:,:,2,:] = x[:,:,9,:]
    y[:,:,3,:] = x[:,:,10,:]
    y[:,:,4,:] = x[:,:,11,:]
    y[:,:,5,:] = x[:,:,12,:]
    y[:,:,6,:] = x[:,:,13,:]
    y[:,:,8,:] = x[:,:,1,:]
    y[:,:,7,:] = (y[:,:,0,:] + y[:,:,8,:]) * 0.5
    y[:,:,9,:] = x[:,:,0,:]
    y[:,:,10,:] = (x[:,:,14,:] + x[:,:,15,:]) * 0.5
    y[:,:,11,:] = x[:,:,5,:]
    y[:,:,12,:] = x[:,:,6,:]
    y[:,:,13,:] = x[:,:,7,:]
    y[:,:,14,:] = x[:,:,3,:]
    y[:,:,15,:] = x[:,:,4,:]
    y[:,:,16,:] = x[:,:,5,:]
    return y[:,:,:17,:]
    
def random_move(data_numpy,
                angle_range=[-10., 10.],
                scale_range=[0.9, 1.1],
                transform_range=[-0.1, 0.1],
                move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    A = np.random.uniform(angle_range[0], angle_range[1], num_node)
    S = np.random.uniform(scale_range[0], scale_range[1], num_node)
    T_x = np.random.uniform(transform_range[0], transform_range[1], num_node)
    T_y = np.random.uniform(transform_range[0], transform_range[1], num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)
    data_numpy = np.transpose(data_numpy, (3,1,2,0)) # C,T,V,M -> M,T,V,C
    return data_numpy  


def random_move_(data_numpy, angle_range=[-6., 6.], scale_range=[0.7, 1.3], transform_range=[-0.2, 0.2], move_time_candidate=[1]):
    data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))  # M,T,V,C-> C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    angle = np.random.uniform(angle_range[0], angle_range[1], 1)
    scale = np.random.uniform(scale_range[0], scale_range[1], 1)
    trans = np.random.uniform(transform_range[0], transform_range[1], 1)
    A = np.random.uniform(angle, angle, num_node)
    S = np.random.uniform(scale, scale, num_node)
    T_x = np.random.uniform(trans, trans, num_node)
    T_y = np.random.uniform(trans, trans, num_node)
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)
    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])
    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])
    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        mask = (xy != 0)  # Create a mask for non-padding values
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        # Only update non-padding values
        data_numpy[0:2, i_frame, :, :] = np.where(mask, new_xy.reshape(2, V, M), xy)
    
    data_numpy = np.transpose(data_numpy, (3, 1, 2, 0))  # C,T,V,M -> M,T,V,C
    return data_numpy
  

def human_tracking(x):
    M, T = x.shape[:2]
    if M==1:
        return x
    else:
        diff0 = np.sum(np.linalg.norm(x[0,1:] - x[0,:-1], axis=-1), axis=-1)        # (T-1, V, C) -> (T-1)
        diff1 = np.sum(np.linalg.norm(x[0,1:] - x[1,:-1], axis=-1), axis=-1)
        x_new = np.zeros(x.shape)
        sel = np.cumsum(diff0 > diff1) % 2
        sel = sel[:,None,None]
        x_new[0][0] = x[0][0]
        x_new[1][0] = x[1][0]
        x_new[0,1:] = x[1,1:] * sel + x[0,1:] * (1-sel)
        x_new[1,1:] = x[0,1:] * sel + x[1,1:] * (1-sel)
        return x_new

class PoseTorchDataset(torch.utils.data.Dataset):
    """Some Information about PoseTrain"""
    def __init__(self, mode, mask, random_move, scale_range, prefix='fix_pickles_01', dual=True, datanum=109):
        super(PoseTorchDataset, self).__init__()
        self.random_move = random_move
        self.scale_range = scale_range
        self.dual = dual
        self.attrs = []
        self.labels = []
        self.names = []
        self.file = []
        self.prefix = prefix
        self.input_tensors = []
        self.mode = mode 
        self.path = f'../PD/Gait_without_dgl/train_subset{datanum}_vote.npy' if not mode == 'test' else f'../PD/Gait_without_dgl/test_subset{datanum}_vote.npy'
        # self.path = 'PD_43.npy' if mode == 'train' else 'sub.npy'
        self.flag = ''
        self.model = 'dwpose'
        self.foot = False #True
        self.mask = mask
        self.process()
        
    def process(self):
        ksplit = 6
        model = self.model
        flag = self.flag
        properties = np.load(self.path, allow_pickle=True)
        label_dict = {}
        for label, name in properties:
            label_dict[name] = label
        # print(label_dict)
        for j, name in enumerate(label_dict.keys()):
            for i in range(6):
                if isinstance(name, str) and name.startswith('HY'):
                    pickle_path = f'../PD/Pose/split_{model}_res/{self.prefix}{flag}_health/{int(name[2:])}/{int(name[2:])}_{i}.pickle'
                    with open(pickle_path, 'rb') as file:
                        attr_dict = pickle.load(file)
                else:
                    pickle_path = f'../PD/Pose/split_{model}_res/{self.prefix}{flag}/{name}/{name}_{i}.pickle'
                    with open(pickle_path, 'rb') as file:
                        attr_dict = pickle.load(file)
                if self.model == 'dwpose' and not self.foot: 
                    attr = [np.concatenate([data['bodies']['candidate'], data['foot'][0]], axis=0) for data in attr_dict \
                        if not data == {} and data['bodies']['candidate'].shape[0] == 18]
                    attr = np.concatenate(attr, axis=0).reshape((-1, 24, 2))
                    confi = [x['bodies']['confi'] for x in attr_dict if not x == {} and x['bodies']['candidate'].shape[0] == 18]
                    confi = np.concatenate(confi, axis=0).reshape((-1, 18, 1))
                    # attr = attr[:, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -6, -5, -4, -3, -2, -1]), :]
                elif self.model == 'dwpose' and self.foot:
                    attr = [np.concatenate([data['bodies']['candidate'], data['foot'][0]], axis=0) for data in attr_dict \
                        if not data == {} and data['bodies']['candidate'].shape[0] == 18]
                    attr = np.concatenate(attr, axis=0).reshape((-1, 24, 2))
                    confi = [x['bodies']['confi'] for x in attr_dict if not x == {} and x['bodies']['candidate'].shape[0] == 18]
                    confi = np.concatenate(confi, axis=0).reshape((-1, 18, 1))
                    attr = attr[:, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -6, -5, -4, -3, -2, -1]), :]
                    confi = confi[:, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]), :]
                else:
                    attr = [np.stack([data['x'], data['y'], data['visibility'], data['presence']], axis=-1) for data in attr_dict if not data == {}]
                    attr = np.stack(attr, axis=0)
                    attr = attr[:, np.array([x for x in range(11, 33)]), :]
                    
                # index_ab = check_abnormal(attr)
                # if index_ab is not False:
                #     # print(index_ab)
                #     if min(index_ab) < 15:
                #         attr = attr[15:, :, :]
                #     if max(index_ab) > attr.shape[0] - 15:
                #         attr = attr[:attr.shape[0] - 15, :, :]
                
                # for i in range(2):
                #     min_val = np.min(attr[:, :, i], axis=0)
                #     max_val = np.max(attr[:, :, i], axis=0)
                #     attr[:, :, i] = (attr[:, :, i] - min_val) / (max_val - min_val)    
                f, n, c = attr.shape
                frame_nums = 64

                if f <= frame_nums:
                    pad_frames = frame_nums - f
                    attr = np.pad(attr, ((0, pad_frames), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
                    confi = np.pad(confi, ((0, pad_frames), (0, 0), (0, 0)), 'constant', constant_values=(0, 0))
                else:
                    # start_index = np.random.randint(0, f - frame_nums + 1)
                    start_index = 0
                    attr = attr[start_index : start_index + frame_nums, :, :]
                    confi = confi[start_index : start_index + frame_nums, :, :]
                
                label = int(label_dict[name])
                
                # mark no info as 0
                attr[attr == -1] = 0
                
                # aug
                # if self.mode == 'train':
                #     attr = random_move_(attr[None, :, :, :])[0]
                    
                # attr = normalize_attr_skip_minus_one(attr)
                # attr = normalize_attr_skip_minus_one(attr)
                
                mask = self.mask
                # print('mask', mask)
                neck_idx = [0, 1]
                arm_idx = [2, 3, 4, 5, 6, 7]
                leg_idx = [8, 9, 10, 11, 12, 13]
                foot_idx = [14, 15, 16, 17, 18, 19]
                if mask == 'arm_leg':
                    attr[:, np.array(neck_idx + foot_idx), :] = 0
                elif mask == 'leg_foot':
                    attr[:, np.array(neck_idx + arm_idx), :] = 0
                elif mask == 'neck':
                    attr[:, np.array(arm_idx + leg_idx + foot_idx), :] = 0
                elif mask == 'arm_foot':
                    attr[:, np.array(neck_idx + leg_idx), :] = 0
                elif mask == 'leg':
                    attr[:, np.array(neck_idx + arm_idx + foot_idx), :] = 0
                elif mask == 'arm':
                    attr[:, np.array(neck_idx + leg_idx + foot_idx), :] = 0
                elif mask == 'foot':
                    attr[:, np.array(neck_idx + leg_idx + arm_idx), :] = 0
                elif mask == 'neck_foot':
                    attr[:, np.array(leg_idx + arm_idx), :] = 0
                elif mask == 'neck_arm':
                    attr[:, np.array(leg_idx + foot_idx), :] = 0
                elif mask == 'neck_leg':
                    attr[:, np.array(arm_idx + foot_idx), :] = 0
                elif mask == 'neck_arm_leg':
                    attr[:, np.array(foot_idx), :] = 0
                elif mask == 'neck_arm_foot':
                    attr[:, np.array(leg_idx), :] = 0
                elif mask == 'neck_leg_foot':
                    attr[:, np.array(arm_idx), :] = 0
                elif mask == 'arm_leg_foot':
                    attr[:, np.array(neck_idx), :] = 0
                elif mask == 'two_point':
                    attr[:, np.array([i for i in range(10)] + [11, 12] + [i for i in range(14, 20)]), :] = 0
                # else:
                #     print('mask not in define')
                #     return
                    
                # add left/right token
                # if attr[:, np.array([2,3,4])].all() == 0 or sum(np.sign(np.diff(attr[:, 1, 0]))) < 0:
                #     attr = np.pad(attr, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=(0, 1))
                #     # print('from right to left')
                # elif attr[:, np.array([5, 6, 7])].all() == 0 or sum(np.sign(np.diff(attr[:, 1, 0]))) > 0:
                #     attr = np.pad(attr, ((0, 0), (0, 1), (0, 0)), 'constant', constant_values=(0, 2))
                #     # print('from left to right')
                # else:
                #     print('error')
                #     return
                
                # filename = pickle_path.replace('.pickle', '_gei.jpg')
                # input_image = Image.open(filename).convert('L').convert('RGB')
                # input_tensor = preprocess(input_image)
                if not self.foot:
                    attr = make_cam(attr[None,:,:,:], (1,1))
                    attr = dwpose2h36m(attr)
                    confi = dwpose2h36m(confi[None,:,:,:])
                    attr = np.concatenate((attr, confi), axis=-1)
                else:
                    attr = attr[None,:,:,:]
                    confi = np.concatenate([confi, np.ones((frame_nums, 6, 1))*0.5], axis=1)[None,:,:,:]
                    attr = np.concatenate((attr, confi), axis=-1)
                    
                
                self.attrs.append(attr)
                self.labels.append(label)
                self.names.append(j * ksplit + i)
                self.file.append(f'{str(name).strip()}_{i}')
                # self.input_tensors.append(input_tensor)
                
        self.labels = torch.LongTensor(self.labels)

    def shuffle(self, attrs):
        indices = torch.randperm(attrs.size(0))
        return attrs[indices]

    def __getitem__(self, index):
        # print(motion, label)
        # if self.random_move:
        #     motion = random_move(motion)
        # print(motion)
        # if self.scale_range:
        #     result = crop_scale(motion, scale_range=self.scale_range)
        # else:
        #     result = motion
        # print(motion, label)
        # print(motion.shape)
        return {'attr': self.attrs[index].astype(np.float32),
                'label': self.labels[index],
                'name': self.names[index],
                'file': self.file[index]
                # 'im': self.input_tensors[index]
                }

    def __len__(self):
        return len(self.labels)
    
class ActionDataset(Dataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=True):   # data_split: train/test etc.
        np.random.seed(0)
        dataset = read_pkl(data_path)
        if check_split:
            assert data_split in dataset['split'].keys()
            self.split = dataset['split'][data_split]
        annotations = dataset['annotations']
        self.random_move = random_move
        self.is_train = "train" in data_split or (check_split==False)
        if "oneshot" in data_split:
            self.is_train = False
        self.scale_range = scale_range
        motions = []
        labels = []
        for sample in annotations:
            if check_split and (not sample['frame_dir'] in self.split):
                continue
            resample_id = resample(ori_len=sample['total_frames'], target_len=n_frames, randomness=self.is_train)
            motion_cam = make_cam(x=sample['keypoint'], img_shape=sample['img_shape'])
            motion_cam = human_tracking(motion_cam)
            motion_cam = coco2h36m(motion_cam)
            motion_conf = sample['keypoint_score'][..., None]
            motion = np.concatenate((motion_cam[:,resample_id], motion_conf[:,resample_id]), axis=-1)
            if motion.shape[0]==1:                                  # Single person, make a fake zero person
                fake = np.zeros(motion.shape)
                motion = np.concatenate((motion, fake), axis=0)
            motions.append(motion.astype(np.float32)) 
            labels.append(sample['label'])
        self.motions = np.array(motions)
        self.labels = np.array(labels)
        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.motions)

    def __getitem__(self, index):
        raise NotImplementedError 

class NTURGBD(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1]):
        super(NTURGBD, self).__init__(data_path, data_split, n_frames, random_move, scale_range)
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label
    
class NTURGBD1Shot(ActionDataset):
    def __init__(self, data_path, data_split, n_frames=243, random_move=True, scale_range=[1,1], check_split=False):
        super(NTURGBD1Shot, self).__init__(data_path, data_split, n_frames, random_move, scale_range, check_split)
        oneshot_classes = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114]
        new_classes = set(range(120)) - set(oneshot_classes)
        old2new = {}
        for i, cid in enumerate(new_classes):
            old2new[cid] = i
        filtered = [not (x in oneshot_classes) for x in self.labels]
        self.motions = self.motions[filtered]
        filtered_labels = self.labels[filtered]
        self.labels = [old2new[x] for x in filtered_labels]
        
    def __getitem__(self, idx):
        'Generates one sample of data'
        motion, label = self.motions[idx], self.labels[idx] # (M,T,J,C)
        if self.random_move:
            motion = random_move(motion)
        if self.scale_range:
            result = crop_scale(motion, scale_range=self.scale_range)
        else:
            result = motion
        return result.astype(np.float32), label