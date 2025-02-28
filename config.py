import torch

class Config:
    def __init__(self):
        self.batch_size = 32
        self.block_size = 8
        self.max_iters = 5000
        self.eval_interval = 500
        self.learning_rate = 1e-3
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.eval_iters = 200
        self.n_embed = 32
        self.seed = 1337
        self.data_path = 'data.txt'
        self.split_ratio = 0.9

config = Config()
