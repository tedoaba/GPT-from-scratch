import torch
import logging
from config import config
from data_loader import prepare_data
from model import BigramLanguageModel
from utils import get_batch, estimate_loss

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for iter in range(config.max_iters):

        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data)
            logger.info(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        xb, yb = get_batch('train', train_data)

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_text = decode(model.generate(context, max_new_tokens=500, block_size=config.block_size)[0].tolist())
    logger.info("Generated text: %s", generated_text)
    