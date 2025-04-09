import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import random
from model import *
from utils.cross_data_processor import *
from utils.multiGAN_trainer import  *

def triple_gan_all_pipeline(
        data_path,
        output_dir,
        feature_columns,
        target_columns,
        window_size1,
        window_size2,
        window_size3,
        distill,
        num_epoch,
        batch_size,
        train_split,
        random_seed,
        device=0
):


    # Set random seed if provided
    if random_seed is not None:
        print("Random seed set:", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Device configuration
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print('Device {}'.format(device))
    # Create output_triple_cross directory
    os.makedirs(output_dir, exist_ok=True)
    print("output_triple_cross dir: {}".format(output_dir))
    print('Loading data...')

    # 1. Data Loading & Preprocessing
    train_x, test_x,train_y,test_y,y_scaler= load_data_all(data_path,target_columns,feature_columns,train_split)



    # 3. Sliding Window Processing
    train_x1, train_y1, train_y_gan1 = create_sequences_combine(train_x, train_y, window_size1,window_size3)  # 生成训练集序列数据1
    train_x2, train_y2, train_y_gan2 = create_sequences_combine(train_x, train_y, window_size2,window_size3)  # 生成训练集序列数据2
    train_x3, train_y3, train_y_gan3 = create_sequences_combine(train_x, train_y, window_size3,window_size3)  # 生成训练集序列数据3
    #assert torch.equal(train_y1, train_y2) and torch.equal(train_y2, train_y3), "train_y1, train_y2, train_y3 is not the same"


    test_x1, test_y1, test_y_gan1 = create_sequences_combine(test_x, test_y, window_size1,window_size3)  # 生成测试集序列数据1
    test_x2, test_y2, test_y_gan2 = create_sequences_combine(test_x, test_y, window_size2,window_size3)  # 生成测试集序列数据2
    test_x3, test_y3, test_y_gan3 = create_sequences_combine(test_x, test_y, window_size3,window_size3)  # 生成测试集序列数据3
    print('initializing model...')
    # 4. Model Training
    modelG1 = Generator_gru(train_x1.shape[-1],train_y1.shape[-1]).to(device)
    modelG2 = Generator_lstm(train_x2.shape[-1],train_y2.shape[-1]).to(device)
    modelG3 = Generator_transformer(train_x.shape[-1],output_len=train_y.shape[-1]).to(device)

    modelD1 = Discriminator3(window_size1,out_size=train_y.shape[-1]).to(device)
    modelD2 = Discriminator3(window_size2,out_size=train_y.shape[-1]).to(device)
    modelD3 = Discriminator3(window_size3,out_size=train_y.shape[-1]).to(device)



    trainDataloader1 = DataLoader(TensorDataset(train_x1, train_y_gan1), batch_size=batch_size, shuffle=False,
                                  generator=torch.manual_seed(random_seed))
    trainDataloader2 = DataLoader(TensorDataset(train_x2, train_y_gan2), batch_size=batch_size, shuffle=False,
                                  generator=torch.manual_seed(random_seed))
    trainDataloader3 = DataLoader(TensorDataset(train_x3, train_y_gan3), batch_size=batch_size, shuffle=True,
                                  generator=torch.manual_seed(random_seed))


    print("test size:",test_x1.shape[0])

    train_x1 = train_x1.to(device)
    train_x2 = train_x2.to(device)
    train_x3 = train_x3.to(device)

    test_x1 = test_x1.to(device)
    test_x2 = test_x2.to(device)
    test_x3 = test_x3.to(device)

    results = train_multi_gan([modelG1, modelG2, modelG3],
                              [modelD1, modelD2, modelD3],
                              [trainDataloader1, trainDataloader2, trainDataloader3],
                              [window_size1, window_size2, window_size3],
                              y_scaler,
                              [train_x1, train_x2, train_x3],
                              train_y1,
                              [test_x1, test_x2, test_x3],
                              test_y1,
                              distill,
                              num_epoch,
                              output_dir,
                              device)

    return results

