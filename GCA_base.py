from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os

class GCABase(ABC):
    """
    GCA 框架的虚基类，定义核心方法接口。
    所有子类必须实现以下方法。
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_path, output_path,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill: bool = False,
                 device = None,
                 seed=None):
        """
        初始化必备的超参数。

        :param N_pairs: 生成器or对抗器的个数
        :param batch_size: 小批次处理
        :param num_epochs: 预定训练轮数
        :param initial_learning_rate: 初始学习率
        :param generators: 建议是一个iterable object，包括了表示具有不同特征的生成器
        :param discriminators: 建议是一个iterable object，可以是相同的判别器
        :param ckpt_path: 各模型检查点
        :param output_path: 可视化、损失函数的log等输出路径
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_path = ckpt_path
        self.output_path = output_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill = do_distill
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # 初始化随机种子
        self.device = setup_device(device)
        print("Running Device:", self.device)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            print("Output directory created! ")

        if not os.path.exists(self.ckpt_path):
            os.makedirs(self.ckpt_path)
            print("Checkpoint directory created! ")

    def set_seed(self, seed):
        """
        设置随机种子以确保实验的可重复性。

        :param seed: 随机种子
        """
        print("Random seed set:", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = "0"

        torch.backends.cudnn.deterministic = True  # 启用确定性算法
        torch.backends.cudnn.benchmark = False  # 关闭自动优化

    def seed_worker(self, worker_id):
        base_seed = torch.initial_seed() % 2 ** 32
        worker_seed = base_seed  # 差异化种子
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    @abstractmethod
    def process_data(self):
        """数据预处理，包括读取、清洗、划分等"""
        pass

    @abstractmethod
    def init_model(self):
        """模型结构初始化"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """初始化用于训练与评估的数据加载器"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """初始化训练所需的超参数"""
        pass

    @abstractmethod
    def train(self):
        """执行训练过程"""
        pass

    @abstractmethod
    def distill(self):
        """执行知识蒸馏过程"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """评估模型性能并可视化结果"""
        pass

    @abstractmethod
    def init_history(self):
        """初始化训练过程中的指标记录结构"""
        pass
