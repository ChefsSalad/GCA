from models.model1 import *
from utils.single_data_processor import *
from utils.baseframe_trainer import *
import random


def run_baseframe_pipeline(
        data_path,
        feature_columns,
        target_column,
        window_size,
        model,
        batch_size,
        num_epochs,
        learning_rate,
        train_split,
        random_seed,
        output_dir,
        device
):
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Device configuration

    print("using:", device)
    print("using model:", model)
    print("using window size:", window_size)

    # Create output_triple_cross directory
    # os.makedirs(output_dir, exist_ok=True)
    # 1. Data Loading & Preprocessing
    print(feature_columns)
    train_x, test_x, train_y, test_y, y_scaler = load_data(data_path, target_column, feature_columns, train_split)

    # 2. Sliding Window Processing
    train_x, train_y, train_y_gan = create_sequences(train_x, train_y, window_size,20)
    test_x, test_y, test_y_gan = create_sequences(test_x, test_y, window_size,20)

    # 4. Model Training
    if model == "gru":
        generator = Generator_gru(train_x.shape[-1], train_y.shape[-1]).to(device)
    elif model == "lstm":
        generator = Generator_lstm(train_x.shape[-1], train_y.shape[-1]).to(device)
    elif model == "transformer":
        generator = Generator_transformer(train_x.shape[-1], output_len=train_y.shape[-1]).to(device)

    else:
        print("model not supported")


    best_model_state, histG, hist_val_loss=train_generator(train_x, train_y, test_x, test_y, y_scaler,
                        learning_rate, batch_size, num_epochs, generator, device, patience=50)

    # 5. Evaluation & Saving Results
    generator.load_state_dict(best_model_state)
    generator.eval()

    # Generate predictions
    train_pred = predict(generator, train_x, device)
    test_pred = predict(generator, test_x, device)

    # 计算评估指标
    results = {
        'train_true': inverse_transform(train_y, y_scaler),
        'train_pred': inverse_transform(train_pred, y_scaler),
        'test_true': inverse_transform(test_y, y_scaler),
        'test_pred': inverse_transform(test_pred, y_scaler)
    }

    # 计算训练集、验证集和测试集的 MSE, MAE, RMSE, MAPE
    train_mse, train_mae, train_rmse, train_mape = calculate_metrics(results['train_true'], results['train_pred'])
    test_mse, test_mae, test_rmse, test_mape = calculate_metrics(results['test_true'], results['test_pred'])

    # 打印训练集和测试集的评估指标
    print(f"Train MSE: {train_mse:.4f}, MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}")
    print(f"Test MSE: {test_mse:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}")

    # # 可视化训练损失并保存为文件
    plot_G_training_loss(histG,  hist_val_loss, output_dir, filename='training_loss.png')

    # 可视化训练和测试集预测结果并保存到 output_dir 目录
    plot_predictions(results['train_true'], results['train_pred'], title='Train Predictions vs True Values',
                     output_dir=output_dir, filename='train_predictions.png')
    plot_predictions(results['test_true'], results['test_pred'], title='Test Predictions vs True Values',
                     output_dir=output_dir, filename='test_predictions.png')
    #
    # # Save model states
    # # torch.save(best_model, os.path.join(output_dir, 'best_generator.pth'))
    # # torch.save(discriminator.state_dict(), os.path.join(output_dir, 'discriminator.pth'))
    #
    # print(f"Training completed. Results saved to {output_dir}")
    return {
        'train_mse': train_mse,
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_mape': train_mape,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_mape': test_mape
    }



