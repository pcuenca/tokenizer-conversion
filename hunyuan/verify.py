from datasets import load_dataset
from tqdm import tqdm
from tokenization_hy import HYTokenizer
from transformers import AutoTokenizer

original = HYTokenizer.from_pretrained("mlx-community/Hunyuan-7B-Instruct-3bit")
t_fast = AutoTokenizer.from_pretrained("pcuenq/Hunyuan-7B-Instruct-tokenizer")

# Testing on XNLI

xnli = load_dataset("xnli", "all_languages", split="validation")

def verify(lang, text):
    encoded_original = original.encode(text)
    encoded_fast = t_fast.encode(text)
    assert encoded_fast == encoded_original, f"Fast encode error: {lang} - {text}"
    decoded = original.decode(encoded_original)
    decoded_fast = t_fast.decode(encoded_fast, skip_special_tokens=True)
    assert decoded_fast == decoded, f"Fast decode error: {lang} - {text}"

for p in tqdm(xnli["premise"]):
    for lang, text in p.items():
        verify(lang, text)


# Testing on codeparrot subset

ds = load_dataset("codeparrot/github-code", streaming=True, trust_remote_code=True, split="train")

iterator = iter(ds)
for _ in tqdm(range(1000)):
    item = next(iterator)
    code = item["code"]
    lang = item["language"]
    verify(lang, code)

