import os
import time
import torch
import logging
from src.config import config
from src.data_loader import prepare_data
from src.model import BigramLanguageModel
from src.utils import get_batch, estimate_loss
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {config.device}")

    torch.manual_seed(config.seed)
    
    logger.info("Loading data...")
    train_data, val_data, vocab_size, decode = prepare_data(config.data_path)
    logger.info("Data loaded successfully.")

    model = BigramLanguageModel(vocab_size, config.n_embed, config.num_head, config.n_layer, config.block_size, config.dropout).to(config.device)
    logger.info("Model initialized.")
    
    num_params = sum(p.numel() for p in model.parameters())/1e6

    logger.info(f"{num_params} M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scalar = GradScaler()
    writer = SummaryWriter(log_dir=os.path.join('logs', 'tensorboard'))

    best_val_loss = float('inf')

    for iter in range(config.max_iters):

        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            train_loss_tensor = torch.tensor(losses['train'])
            val_loss_tensor = torch.tensor(losses['val'])
            if torch.isnan(train_loss_tensor) or torch.isnan(val_loss_tensor):
                logger.warning('Nan loss detected, stopping training.')
                return float('inf'), float('inf')
            
            logger.info(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            writer.add_scalar('Loss/train', losses['train'], iter)
            writer.add_scalar('Loss/val', losses['val'], iter)

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save(model.state_dict(), os.path.join('checkpoints', 'best_model.pth'))

        xb, yb = get_batch('train', train_data)

        with autocast(device_type=config.device):
            logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        scalar.scale(loss).backward()
        scalar.step(optimizer)
        scalar.update()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_text = decode(model.generate(context, max_new_tokens=500, block_size=config.block_size)[0].tolist())
    logger.info("Generated text: %s", generated_text)

    writer.close()

    return losses['train'], losses['val']
