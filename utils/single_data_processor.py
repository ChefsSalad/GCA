from sklearn.preprocessing import MinMaxScaler
import math
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import copy
import matplotlib.pyplot as plt
import numpy as np
import os




def load_data(data_path,target_column,feature_columns,train_split):
    data = pd.read_csv(data_path)

    y = data.iloc[:,target_column].values  # 获取多个目标字段

    # Feature selection
    x = data.iloc[:, feature_columns].values
    #y = data['y'].values.reshape(-1, 1)

    # Data splitting
    train_size = int(data.shape[0] * train_split)

    target_name = data.columns[target_column]
    print(target_name)
    feature_names = data.columns[feature_columns].tolist()
    print(f"Selected features: {feature_names}")


    train_x, test_x = x[:train_size], x[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    # Normalization
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y)
    test_y = y_scaler.transform(test_y)

    return train_x,  test_x, train_y, test_y ,y_scaler


def load_data_combine(data_path, target_columns, feature_columns, train_split, val_split):
    data = pd.read_csv(data_path)

    # 选择多个目标列
    y = data[target_columns].values  # 获取多个目标字段

    # Feature selection
    x = data[feature_columns].values

    # Data splitting
    train_size = int(data.shape[0] * train_split)
    val_size = int(data.shape[0] * val_split)

    train_x, val_x, test_x = x[:train_size], x[train_size:val_size], x[val_size:]
    train_y, val_y, test_y = y[:train_size], y[train_size:val_size], y[val_size:]

    # Normalization
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_x = x_scaler.fit_transform(train_x)
    val_x = x_scaler.transform(val_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y)
    val_y = y_scaler.transform(val_y)
    test_y = y_scaler.transform(test_y)

    return train_x, val_x, test_x, train_y, val_y, test_y, y_scaler


def create_sequences(x, y, window_size,start):
    x_ = []
    y_ = []
    y_gan = []
    for i in range(start, x.shape[0]):
        tmp_x = x[i - window_size: i, :]
        tmp_y = y[i]
        tmp_y_gan = y[i - window_size: i + 1]
        x_.append(tmp_x)
        y_.append(tmp_y)
        y_gan.append(tmp_y_gan)
    x_ = torch.from_numpy(np.array(x_)).float()
    y_ = torch.from_numpy(np.array(y_)).float()
    y_gan = torch.from_numpy(np.array(y_gan)).float()
    return x_, y_, y_gan




def predict(model, data,device):
    with torch.no_grad():
        return model(torch.FloatTensor(data).to(device)).cpu().numpy()

# Inverse scaling



def plot_G_training_loss(histG,  hist_val_loss, output_dir, filename='training_loss.png'):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    epochs = np.arange(len(histG))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, histG, label='Generator Loss', color='b')
    plt.plot(epochs, hist_val_loss, label='Validation Loss', color='g')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    # 将图像保存到指定目录
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# 可视化训练过程中的损失并保存为图片
def plot_training_loss(histG, histD, hist_val_loss, output_dir, filename='training_loss.png'):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    epochs = np.arange(len(histG))

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, histG, label='Generator Loss', color='b')
    plt.plot(epochs, histD, label='Discriminator Loss', color='r')
    plt.plot(epochs, hist_val_loss, label='Validation Loss', color='g')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()

    # 将图像保存到指定目录
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


# 可视化预测结果与真实值并保存为图片
def plot_predictions(true_values, pred_values,output_dir, title='Predictions vs True Values', filename='predictions.png'):
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='True Values', color='b')
    plt.plot(pred_values, label='Predicted Values', color='r')
    plt.xlabel('Time Steps')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()

    # 将图像保存到指定目录
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()




def calculate_metrics(true_values, predicted_values):
    """
    计算 MSE, MAE, RMSE, MAPE
    """
    mse = mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

    return mse, mae, rmse, mape




def compute_shap_values(model, train_x, device):

    # 使用SHAP的DeepExplainer来解释深度学习模型
    #explainer = shap.DeepExplainer(model, train_x.clone().detach().to(device))  # 使用clone().detach()
    explainer = shap.GradientExplainer(model, train_x.clone().detach().to(device))
    model.train()
    shap_values = explainer.shap_values(train_x.clone().detach().to(device))  # 使用clone().detach()

    # 取出生成器的SHAP值
    shap_values = shap_values[0] # 假设我们关注生成器的输出
    return shap_values




def plot_shap_summary(shap_values, feature_columns, output_dir):
    # 确保 feature_columns 中的所有元素都是字符串类型
    feature_columns = [str(f) for f in feature_columns]

    # 创建 SHAP summary plot
    shap.summary_plot(shap_values, feature_names=feature_columns, show=False)  # 不显示图像

    # 保存图像
    plt.savefig(f"{output_dir}/shap_summary_plot.png", bbox_inches="tight")  # 保存为 PNG 文件
    plt.close()  # 关闭图像，避免内存泄漏