from huggingface_hub import snapshot_download
from tokenization_hy import *
from tokenizers import normalizers
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

snapshot_download(
    "mlx-community/Hunyuan-7B-Instruct-3bit",
    local_dir=".",
    allow_patterns=["hy.tiktoken", "tokenization_hy.py", "tokenizer_config.json", "special_tokens_map.json"]
)

original = HYTokenizer.from_pretrained(".")

converter = TikTokenConverter(
    vocab_file="hy.tiktoken",
    pattern=PAT_STR,
    additional_special_tokens=[t[1] for t in SPECIAL_TOKENS],
)
converted = converter.converted()
converted.normalizer = normalizers.NFC()

t_fast = PreTrainedTokenizerFast(
    tokenizer_object=converted,
    model_input_names=original.model_input_names,
    model_max_length=256*1024,
    clean_up_tokenization_spaces=False,
)
t_fast.chat_template = original.chat_template
t_fast.push_to_hub("Hunyuan-7B-Instruct-tokenizer")

