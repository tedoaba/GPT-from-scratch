# GPT-from-scratch

This project implements a Bigram Language Model from scratch using PyTorch. The model is trained to generate text based on a given dataset.

## Project Structure

- **train.py**: Script to train the language model.
- **model.py**: Defines the architecture of the Bigram Language Model.
- **data_loader.py**: Handles data loading and preprocessing.
- **config.py**: Contains configuration settings and hyperparameters.
- **main.py**: Entry point to start the training process.
- **logs/**: Directory to store training logs and hyperparameter tuning results.
- **experiments/**: Directory to store experiment scripts and configurations.
- **checkpoints/**: Directory to save model checkpoints during training.

## Requirements

- Python 3.7+
- PyTorch
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/tedoaba/GPT-from-scratch.git
    cd GPT-from-scratch
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv .venv
    source .venv/Scripts/activate
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Prepare your dataset:
    - Place your text data in a file named `data.txt` in the project root directory.

2. Configure the model:
    - Adjust the hyperparameters and settings in `config.py` as needed.

3. Train the model:
    ```sh
    python main.py
    ```

4. Generate text:
    - After training, the model will generate text based on the learned patterns from the dataset.

5. Tune hyperparameters:
    - Run the hyperparameter tuning script to find the best parameters for training:
    ```sh
    python experiments/hyperparameter_tuning.py
    ```
    - The best hyperparameters will be saved in `logs/hyperparameters/best_hyperparameters.txt`

## Configuration

The `config.py` file contains various settings and hyperparameters for the model:

- `batch_size`: Number of samples per batch.
- `block_size`: Length of the context window for training.
- `max_iters`: Maximum number of training iterations.
- `eval_interval`: Interval for evaluating the model on validation data.
- `learning_rate`: Learning rate for the optimizer.
- `device`: Device to run the model on (`cuda` or `cpu`).
- `eval_iters`: Number of iterations for evaluation.
- `n_embed`: Size of the embedding vectors.
- `num_head`: Number of attention heads.
- `n_layer`: Number of layers in the model.
- `dropout`: Dropout rate for regularization.
- `seed`: Random seed for reproducibility.
- `data_path`: Dataset path for training.

## Model Architecture

The model consists of the following components:

- **Embedding Layers**: Token and position embeddings.
- **Attention Mechanism**: Multi-head self-attention.
- **Feedforward Network**: Two-layer feedforward network.
- **Stacked Blocks**: Multiple layers of attention and feedforward blocks.
- **Output Layer**: Linear layer to predict the next token.

## Training

The training process involves:

1. Loading and preprocessing the data.
2. Initializing the model and optimizer.
3. Iteratively training the model and evaluating its performance.
4. Generating text from the trained model.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the GPT architecture and various tutorials on language modeling with PyTorch.