import torch
from config import config

def get_batch(split, data):
    '''generate a small batch of data of input x and targets y'''
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    model.eval()
    losses = {split: torch.zeros(config.eval_iters) for split in ['train', 'val']}
    for split in ['train', 'val']:
        data = train_data if split == 'train' else val_data
        for k in range(config.eval_iters):
            X, Y = get_batch(split, data)
            _, loss = model(X, Y)
            losses[split][k] = loss.item()
    model.train()
    return {split: losses[split].mean().item() for split in ['train', 'val']}
