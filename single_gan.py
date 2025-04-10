from utils.evaluate_visualization import *
from models.model import *
from utils.single_data_processor import *
import random
from utils.single_GAN_trainer import *
def single_gan_pipeline(
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
        max_window,
        device
):
    
    # Set random seed if provided
    if random_seed is not None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    # Device configuration

    print("using:",device)
    print("using model:",model)
    print("using window size:",window_size)

    # Create output_triple_cross directory
    #os.makedirs(output_dir, exist_ok=True)
    output_dir='out_put/gan/gru'
    # 1. Data Loading & Preprocessing

    train_x, test_x, train_y, test_y,y_scaler=load_data(data_path,target_column,feature_columns,train_split)

    # 2. Sliding Window Processing
    train_x, train_y ,train_y_gan= create_sequences(train_x, train_y, window_size,max_window)
    test_x, test_y ,test_y_gan= create_sequences(test_x, test_y, window_size,max_window)

    # 4. Model Training
    if model=="gru":
        generator = Generator_gru(train_x.shape[-1],train_y.shape[-1]).to(device)
    elif model=="lstm":
        generator = Generator_lstm(train_x.shape[-1],train_y.shape[-1]).to(device)
    elif model=="transformer":
        generator = Generator_transformer(train_x.shape[-1],output_len=train_y.shape[-1]).to(device)
    else:
        print("model not supported")

    discriminator = Discriminator3(window_size,out_size=train_y.shape[-1]).to(device)
    best_model = None

    print("Training GAN...")
    best_model_state, histG, histD, hist_val_loss=train_gan(train_x, train_y_gan, test_x, test_y, y_scaler, window_size, learning_rate, batch_size,
              num_epochs, generator, discriminator, device=device, patience=50)

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
    print(results['train_true'].shape)
    # 计算训练集、验证集和测试集的 MSE, MAE, RMSE, MAPE
    train_metrics = compute_metrics(results['train_true'], results['train_pred'])
    test_metrics=compute_metrics(results['test_true'], results['test_pred'])

    # 打印训练集和测试集的评估指标
    print(f"Train Metrics for G1: MSE={train_metrics[0]:.4f}, MAE={train_metrics[1]:.4f}, RMSE={train_metrics[2]:.4f}, MAPE={train_metrics[3]:.4f}")
    print(f"Test  Metrics for G1: MSE={test_metrics[0]:.4f}, MAE={test_metrics[1]:.4f}, RMSE={test_metrics[2]:.4f}, MAPE={test_metrics[3]:.4f}")

    # # 可视化训练损失并保存为文件
    plot_training_loss(histG, histD, hist_val_loss, output_dir, filename='training_loss.png')

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
        "train_mse": train_metrics[0],
        "train_mae": train_metrics[1],
        "train_rmse":  train_metrics[2],
        "train_mape":  train_metrics[3],
        "train_mse_per_target": train_metrics[4],
        "test_mse": test_metrics[0],
        "test_mae": test_metrics[1],
        "test_rmse": test_metrics[2],
        "test_mape": test_metrics[3],
        "test_mse_per_target": test_metrics[4]
    }



