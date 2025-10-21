# config.py

class Config:
    # Data
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    max_seq_len: int = 512  # truncate long files for memory safety
    min_pairs: int = 1      # require at least N input/output matches

    # Training
    batch_size: int = 8
    num_epochs: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_steps: int = 200

    # Model (Transformer)
    vocab_size: int = 258  # 256 bytes + BOS + EOS
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 512
    dropout: float = 0.1

    # Misc
    seed: int = 42
    device: str = "cuda"  # "cuda" if available; train.py will auto-fallback to "cpu"
    save_path: str = "checkpoint.pt"