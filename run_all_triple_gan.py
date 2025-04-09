import os
import pandas as pd
import argparse
from triple_gan import *

def run_experiments(data_path, output_dir, window_size1, window_size2, window_size3, distill,num_epoch,batch_size,
                    train_split, random_seed,device):
    # 创建保存结果的CSV文件
    results_file = os.path.join(output_dir, "gca_GT_NPDC_market.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("Output directory created")

    # 定义特征和目标列的组合

    # feature_columns = list(range(2,56))
    feature_columns = []
    # feature_columns = list(range(35,36))
    # target_columns = [[i] for i in range(1, 22)]
    target_columns = [list(range(1, 2))]

    # 逐个组合进行实验
    for target in target_columns:
        # for target,feature in zip(target_columns,feature_columns):
        # 运行实验，获取结果
        target_feature_columns = feature_columns
        # target_feature_columns = feature_columns
        # target_feature_columns=target_feature_columns.extend(target)
        target_feature_columns.extend(target)
        # target_feature_columns.append(target)
        print("using features:", target_feature_columns)

        results = triple_gan_all_pipeline(
            data_path=data_path,
            output_dir=output_dir,
            feature_columns=target_feature_columns,
            target_columns=target,
            window_size1=window_size1,
            window_size2=window_size2,
            window_size3=window_size3,
            distill=distill,
            num_epoch=num_epoch,
            batch_size=batch_size,
            train_split=train_split,
            random_seed=random_seed,
            device=device
        )

        # 将结果保存到CSV
        result_row = {
            "feature_columns": feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "train_mse_per_target": results["train_mse_per_target"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"],
            "test_mse_per_target": results["test_mse_per_target"]
        }
        df = pd.DataFrame([result_row])
        df.to_csv(results_file, mode='a', header=not pd.io.common.file_exists(results_file), index=False)

if __name__ == "__main__":
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="Run experiments for triple GAN model")
    parser.add_argument('--data_path', type=str, required=False, help="Path to the input data file",default="database/cleaned_data.csv")
    parser.add_argument('--output_dir', type=str, required=False, help="Directory to save the output",default="out_put/4_all")
    parser.add_argument('--window_size1', type=int, help="Window size for first dimension", default=5)
    parser.add_argument('--window_size2', type=int, help="Window size for second dimension", default=10)
    parser.add_argument('--window_size3', type=int, help="Window size for third dimension", default=15)
    parser.add_argument('--distill', type=bool, help="Whether to do distillation", default=True)
    parser.add_argument('--device', type=int, help="Window size for third dimension", default=0)

    parser.add_argument('--num_epoch', type=int, help="epoch", default=50)
    parser.add_argument('--batch_size', type=int, help="Batch size for training", default=64)
    parser.add_argument('--train_split', type=float, help="Train-test split ratio", default=0.8)
    parser.add_argument('--random_seed', type=int, help="Random seed for reproducibility", default=3407)

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("===============================================")

    # 调用run_experiments函数
    run_experiments(
        data_path=args.data_path,
        output_dir=args.output_dir,
        window_size1=args.window_size1,
        window_size2=args.window_size2,
        window_size3=args.window_size3,
        distill=args.distill,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        train_split=args.train_split,
        random_seed=args.random_seed,
        device=args.device
    )
