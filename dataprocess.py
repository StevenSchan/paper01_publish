import os

import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader, random_split


class NestedDataset(Dataset):
    def __init__(self, input_data):
        self.input_data = input_data
        self.length = self._determine_length(self.input_data)
        if self.length is None:
            raise ValueError("无法确定数据集样本数量，请检查 input_data 结构。")
    
    def _determine_length(self, data):
        if isinstance(data, torch.Tensor):
            return data.size(0)
        elif isinstance(data, dict):
            for sub in data.values():
                length = self._determine_length(sub)
                if length is not None:
                    return length
        return None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        def recursive_index(item, idx):
            if isinstance(item, torch.Tensor):
                return item[idx]
            elif isinstance(item, dict):
                return {k: recursive_index(v, idx) for k, v in item.items()}
            else:
                return item
        return {k: recursive_index(v, idx) for k, v in self.input_data.items()}


def raw_data_load(folder_path):
    ''''Load raw data from dataset.'''
    out_npz = os.path.join(folder_path, 'example_data.npz')
    data = np.load(out_npz)
    # 获得 NumPy 数组：
    x_up_raw = data['x_up']              # (B,1440,3)
    x_mid_C_raw = data['x_mid_C']        # (B,1080,3)
    x_mid_D_raw = data['x_mid_D']        # (B,2160,3)
    z_raw = data['z']                    # (B,)
    s_raw = data['s']                    # (B,...)

    print('x_up_raw.shape', x_up_raw.shape)
    print('x_mid_C_raw.shape', x_mid_C_raw.shape)
    print('x_mid_D_raw.shape', x_mid_D_raw.shape)
    print('z_raw.shape', z_raw.shape)
    print('s_raw.shape', s_raw.shape)

    x_mid = {}
    x_up = torch.from_numpy(x_up_raw).float()
    x_mid['C'] = torch.from_numpy(x_mid_C_raw).float()
    x_mid['D'] = torch.from_numpy(x_mid_D_raw).float()
    z = torch.from_numpy(z_raw).float()
    s = torch.from_numpy(s_raw).float()
    unsplit_data = {'x_up': x_up, 'x_mid': x_mid, 'z': z, 's': s}

    return unsplit_data

def produce_dataset_total(config_template, batch_size):
    folder_path = config_template['folder_path']
    train_ratio = config_template['train_ratio']

    # 读取原始数据并转化为torch float tensor
    unsplit_data = raw_data_load(folder_path)
    # 创建 Dataset 对象
    dataset = NestedDataset(unsplit_data)
    print("Total samples:", len(dataset))

    # 使用 random_split 进行划分
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    print(f'train : test = {train_size} : {test_size}')
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建 DataLoader
    unsplit_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_data = {
        'unsplit_loader':unsplit_loader,
        'train_loader':train_loader,
        'test_loader':test_loader,
    }

    return input_data


if __name__ == "__main__":
    raw_data_load('./publish')