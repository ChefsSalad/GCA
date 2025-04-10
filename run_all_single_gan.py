import pandas as pd
import argparse
from single_gan import *

def run_multiple_experiments(data_path, output_dir,  window_size, model,batch_size,
                             num_epochs, learning_rate, train_split, random_seed,max_window, device):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        print(f"Output directory already exists: {output_dir}")
    device = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    output_path = f'{output_dir}/{model}_{window_size}_GT_NPDC_market.csv'

    print(f"Running experiment on device={device}...")

    #feature_columns = list(range(92, 105))
    feature_columns=[]
    #feature_columns = list(range(35,36))
    #target_columns = [[i] for i in range(1, 22)]
    target_columns = [list(range(1, 92))]

    # 逐个组合进行实验
    for target in target_columns:
        # for target,feature in zip(target_columns,feature_columns):
        # 运行实验，获取结果
        target_feature_columns=feature_columns
        # target_feature_columns = feature_columns
        # target_feature_columns=target_feature_columns.extend(target)
        target_feature_columns.extend(target)
        # target_feature_columns.append(target)
        print("using features:", target_feature_columns)

        results = single_gan_pipeline(
            data_path=data_path,
            window_size=window_size,
            model=model,
            feature_columns=target_feature_columns,
            target_column=target,
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            train_split=train_split,
            random_seed=random_seed,
            max_window=max_window,
            device=device
        )

        result_row = {
            'target_column': target,
            'train_mse': results['train_mse'],
            'train_mae': results['train_mae'],
            'train_rmse': results['train_rmse'],
            'train_mape': results['train_mape'],
            "train_mse_per_target": results['train_mse_per_target'],
            'test_mse': results['test_mse'],
            'test_mae': results['test_mae'],
            'test_rmse': results['test_rmse'],
            'test_mape': results['test_mape'],
            "test_mse_per_target": results['test_mse_per_target'],
        }

        # Append to CSV after each run
        pd.DataFrame([result_row]).to_csv(output_path,header=not pd.io.common.file_exists(output_path), mode='a',  index=False)

    print(f"All experiments completed. Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run multiple experiments for GAN training.')
    parser.add_argument('--data_path', type=str, default="database/cleaned_data.csv")
    parser.add_argument('--output_dir', type=str, default='out_put/4_all')
    parser.add_argument('--window_size', type=int, default=5)
    parser.add_argument('--max_window', type=int, default=15)
    parser.add_argument('--model', type=str, default='gru')
    parser.add_argument('--device', type=int, default=1, help="GPU index (i for GPU i)")
    

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=3407)
    args = parser.parse_args()


    run_multiple_experiments(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size=args.window_size,
        model=args.model,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        train_split=args.train_split,
        random_seed=args.random_seed,
        max_window=args.max_window,
        device=args.device
    )
