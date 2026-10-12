from src.lm_training.data_helpers import load_checkpoint
from src.lm_training.decoder import Decoder
from src.tokenizer.tokenizer import Tokenizer
from src.transformer.attention import create_mask
from src.transformer.rope import RotaryPositionalEmbedding
from src.transformer.transformer_lm import TransformerLM


vocab_data_file_prefix = "TinyStoriesV2-GPT4-train"
context_length = 256
d_model = 512
num_heads = 16
device = "cpu"

rope = RotaryPositionalEmbedding(10000, d_model // num_heads, context_length, device=device)
mask = create_mask(context_length, device=device)

model = TransformerLM(
    vocab_size=10000,
    context_length=context_length,
    d_model=d_model,
    num_layers=4,
    num_heads=num_heads,
    rope=rope,
    mask=mask,
    device=device,
)

load_checkpoint("./experiments/1_TinyStories_Base_H100/checkpoint-set2-4.pt", model)

tokenizer = Tokenizer.from_files(
    f"./data/{vocab_data_file_prefix}-vocab.pkl",
    f"./data/{vocab_data_file_prefix}-merges.pkl",
    lru_cache_capacity=10000,
)

decoder = Decoder(model, tokenizer)

output = decoder.decode(
    "Once upon a time, there was a",
    256,
    0.4,
    0.9,
)
print(output)
