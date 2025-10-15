import torch
import matplotlib
matplotlib.use('Agg')


from generate_config import read_configs
from upstream import UpLevelModel
from midstream import MidLevelModel
from downstream import DownLevelModel


def check_keys(missing_keys, unexpected_keys):
    if missing_keys:
        print("警告：以下参数在 checkpoint 里没给到，依然保留了初始化值：")
        for k in missing_keys:
            print("  ", k)
    elif unexpected_keys:
        print("警告：以下参数在 新模型 里没有, checkpoint 中有:")
        for k in unexpected_keys:
            print("  ", k)
    else:
        print("所有模型参数都匹配，已从 checkpoint 加载。")

if __name__ == "__main__":
    # 1. 默认配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")    
    torch.set_default_dtype(torch.float32)
    config = read_configs()

    # 2. 配置
    upstream_para = {
        'input_dim' : config['up_input_dim'],
        'input_len' : config['up_input_len'],
        'd_model' : config['up_d_model'],
        'nhead' : config['up_nhead'],
        'num_layers' : config['up_num_layers'],
        'dropout' : config['up_dropout'],
    }
    midstream_para = {
        'mid_mode': config['mid_mode'],
        'model_key': config['mid_model_key'],
        'fusion_methods' : config['mid_fusion_methods'],
        'ts_in_dim' : config['mid_input_dim'],
        'd_model' : config['mid_d_model'],
        'transformer_layers' : config['mid_num_layers'],
        'nhead' : config['mid_nhead'],
        'num_features' : config['mid_num_features'],
        'dropout' : config['mid_dropout'],
    }

    # 2. 模型
    up_model = UpLevelModel(input_dim=upstream_para['input_dim'], input_len=upstream_para['input_len'], d_model=upstream_para['d_model'], nhead=upstream_para['nhead'], num_layers=upstream_para['num_layers'], dropout=upstream_para['dropout']).to(device)
    mid_model = MidLevelModel(fusion_method=midstream_para['fusion_methods'], ts_in_dim=midstream_para['ts_in_dim'], d_model=midstream_para['d_model'], transformer_layers=midstream_para['transformer_layers'], nhead=midstream_para['nhead'], out_dim=midstream_para['num_features'], dropout=midstream_para['dropout'], mode=midstream_para['mid_mode'], model_key=midstream_para['model_key']).to(device)
    down_model = DownLevelModel(num_features=midstream_para['num_features'], mid_mode=midstream_para['mid_mode']).to(device)

    # 3. 模型加载
    ckpt_example = torch.load('./publish/example_model.pth', map_location=device, weights_only=True)
    missing_keys, unexpected_keys = up_model.load_state_dict(ckpt_example['up_model'], strict=False)
    check_keys(missing_keys, unexpected_keys)
    missing_keys, unexpected_keys = mid_model.load_state_dict(ckpt_example['mid_model'], strict=False)
    check_keys(missing_keys, unexpected_keys)
    missing_keys, unexpected_keys = down_model.load_state_dict(ckpt_example['down_model'], strict=False)
    check_keys(missing_keys, unexpected_keys)
    # 4. 测试
    up_model.eval()
    mid_model.eval()
    down_model.eval()
    print("模型加载完成，开始测试...")