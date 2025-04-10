import os
from torch.utils.data import DataLoader, TensorDataset
import random
from models.model1 import *
from utils.cross_data_processor import *
from utils.cross_GAN_trainer import train_dual_gan

def dual_gan_pipeline(
        data_path,
        output_dir,
        feature_columns,
        target_columns,
        window_size1,
        window_size2,
        model_shorter_term,
        model_longer_term,
        distill,
        num_epoch,
        batch_size,
        train_split,
        random_seed,
        device=0):

    if random_seed is not None:
        print("Random seed set:", random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(output_dir, exist_ok=True)
    print("Output directory:", output_dir)
    print("Loading data...")

    train_x, test_x, train_y, test_y, y_scaler = load_data_all(
        data_path, target_columns, feature_columns, train_split)

    # sliding window
    train_x1, train_y1, train_y_gan1 = create_sequences_combine(train_x, train_y, window_size1, window_size2)
    train_x2, train_y2, train_y_gan2 = create_sequences_combine(train_x, train_y, window_size2, window_size2)

    test_x1, test_y1, test_y_gan1 = create_sequences_combine(test_x, test_y, window_size1, window_size2)
    test_x2, test_y2, test_y_gan2 = create_sequences_combine(test_x, test_y, window_size2, window_size2)

    print("Initializing models...")

    GRU=Generator_gru(train_x1.shape[-1])
    LSTM=Generator_lstm(train_x2.shape[-1])
    TRANSFORMER=Generator_transformer(train_x1.shape[-1])
    models={"gru":GRU, "lstm":LSTM, "transformer":TRANSFORMER}


    modelG1 = models[model_shorter_term].to(device)
    modelG2 = models[model_longer_term].to(device)

    modelD1 = Discriminator1(window_size1).to(device)
    modelD2 = Discriminator2(window_size2).to(device)

    trainDataloader1 = DataLoader(TensorDataset(train_x1, train_y_gan1), batch_size=batch_size, shuffle=True,
                                  generator=torch.manual_seed(random_seed))
    trainDataloader2 = DataLoader(TensorDataset(train_x2, train_y_gan2), batch_size=batch_size, shuffle=True,
                                  generator=torch.manual_seed(random_seed))

    train_x1 = train_x1.to(device)
    train_x2 = train_x2.to(device)
    test_x1 = test_x1.to(device)
    test_x2 = test_x2.to(device)

    results = train_dual_gan(modelG1, modelG2, modelD1, modelD2,
                             trainDataloader1, trainDataloader2,
                             window_size1, window_size2,
                             y_scaler,
                             train_x1, train_x2, train_y1,
                             test_x1, test_x2, test_y1,
                             distill, num_epoch, output_dir, device)

    return results
