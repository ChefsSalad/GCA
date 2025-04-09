import os
import pandas as pd
from multiprocessing import Pool, Semaphore
from itertools import cycle
from triple_gan import *  # 请确保 triple_gan_all_pipeline 函数在 triple_gan 模块中

# 控制并发任务数（最多3个）
semaphore = Semaphore(3)

def run_single_experiment(args):
    semaphore.acquire()
    try:
        target, device_id, common_args = args
        (data_path, output_dir, window_size1, window_size2, window_size3,
         distill, num_epoch, batch_size, train_split, random_seed) = common_args

        feature_columns = []
        target_feature_columns = feature_columns + target

        print(f"[GPU {device_id}] Running target {target} using features {target_feature_columns}")

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
            device=device_id
        )

        results_file = os.path.join(output_dir, "gca_results.csv")
        result_row = {
            "feature_columns": feature_columns,
            "target_columns": target,
            "train_mse": results["train_mse"],
            "train_mae": results["train_mae"],
            "train_rmse": results["train_rmse"],
            "train_mape": results["train_mape"],
            "test_mse": results["test_mse"],
            "test_mae": results["test_mae"],
            "test_rmse": results["test_rmse"],
            "test_mape": results["test_mape"]
        }

        df = pd.DataFrame([result_row])
        # 如果文件不存在，写 header；否则追加
        df.to_csv(results_file, mode='a', header=not os.path.exists(results_file), index=False)

    finally:
        semaphore.release()


def run_all(data_path, output_dir, window_size1, window_size2, window_size3, distill, num_epoch,
            batch_size, train_split, random_seed):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory '{output_dir}' created")

    results_file = os.path.join(output_dir, "gca_results.csv")
    if not os.path.exists(results_file):
        df = pd.DataFrame(
            columns=["feature_columns", "target_columns", "train_mse", "train_mae", "train_rmse", "train_mape",
                     "test_mse", "test_mae", "test_rmse", "test_mape"])
        df.to_csv(results_file, index=False)
        print("Initialized results file")

    # 所有的 target
    target_columns = [[i] for i in range(1, 22)]
    gpu_cycle = cycle([0])  # 交替分配 GPU: 0, 1

    common_args = (
        data_path, output_dir, window_size1, window_size2, window_size3,
        distill, num_epoch, batch_size, train_split, random_seed
    )

    task_args = [(target, next(gpu_cycle), common_args) for target in target_columns]

    # 开始并发运行任务（最多3个）
    print(f"Starting experiments with up to 3 concurrent processes using 2 GPUs...")
    with Pool(processes=3) as pool:
        pool.map(run_single_experiment, task_args)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Triple GAN experiments in parallel with multi-GPU support")
    parser.add_argument('--data_path', type=str, default="database/npdc_with_trend.csv")
    parser.add_argument('--output_dir', type=str, default="out_put/triple_gan")
    parser.add_argument('--window_size1', type=int, default=5)
    parser.add_argument('--window_size2', type=int, default=10)
    parser.add_argument('--window_size3', type=int, default=20)
    parser.add_argument('--distill', type=bool, default=False)
    parser.add_argument('--num_epoch', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--random_seed', type=int, default=3407)

    args = parser.parse_args()

    print("===== Running with the following arguments =====")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("================================================")

    run_all(
        args.data_path,
        args.output_dir,
        args.window_size1,
        args.window_size2,
        args.window_size3,
        args.distill,
        args.num_epoch,
        args.batch_size,
        args.train_split,
        args.random_seed
    )
