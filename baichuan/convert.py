from tokenization_baichuan import BaichuanTokenizer

original = BaichuanTokenizer.from_pretrained(".")

from transformers.convert_slow_tokenizer import SpmConverter, LlamaConverter, GemmaConverter, _get_prepend_scheme
from tokenizers import decoders, normalizers, pre_tokenizers, processors, Tokenizer, AddedToken
from tokenizers.models import BPE

class BaichuanConverter(SpmConverter):
    handle_byte_fallback = True

    def vocab(self, proto):
        vocab = [
            (self.original_tokenizer.convert_ids_to_tokens(0), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(1), 0.0),
            (self.original_tokenizer.convert_ids_to_tokens(2), 0.0),
        ]
        vocab += [(piece.piece, piece.score) for piece in proto.pieces[3:]]
        return vocab

    def unk_id(self, proto):
        unk_id = 0
        return unk_id

    def decoder(self, replacement, add_prefix_space):
        sequence = [
            decoders.Replace("▁", " "),
            decoders.ByteFallback(),
            decoders.Fuse(),
        ]
        return decoders.Sequence(sequence)

    def normalizer(self, proto):
        return normalizers.Replace(pattern=" ", content="▁")

    def pre_tokenizer(self, replacement, add_prefix_space):
        return None

    def post_processor(self):
        return None

    def tokenizer(self, proto):
        vocab_scores = self.vocab(proto)
        _, merges = self.SpmExtractor(self.original_tokenizer.vocab_file).extract(vocab_scores)
        bpe_vocab = {word: i for i, (word, score) in enumerate(vocab_scores)}
        tokenizer = Tokenizer(
            BPE(
                bpe_vocab,
                merges,
                unk_token=proto.trainer_spec.unk_piece,
                fuse_unk=True,
                byte_fallback=self.handle_byte_fallback,
                dropout=None,
            )
        )

        # control tokens are special
        # user defined symbols are not
        # both user and control tokens are AddedTokens
        # Add user defined symbols (type == 4) from sentencepiece (https://github.com/google/sentencepiece/blob/6225e08edb2577757163b3f5dbba4c0b670ef445/src/sentencepiece_model.proto#L299C29-L299C33)
        spm_added_tokens = [
            (id, p.piece, p.type == 3 or p.piece in self.special_tokens)
            for id, p in enumerate(proto.pieces)
            if p.type in [3, 4]
        ]

        # Reproduce weird behaviour in original tokenizer
        # only add tokens that did not originally exist
        bad_added_tokens = set()
        for _, token, _ in spm_added_tokens:
            encoded = self.original_tokenizer.encode(token)
            if len(encoded) != 1:
                bad_added_tokens.add(token)

        tokenizer.add_tokens(
            [
                AddedToken(token, normalized=True, special=special)
                for id, token, special in sorted(spm_added_tokens, key=lambda x: x[0])
                if token not in bad_added_tokens
            ]
        )

        return tokenizer

converter = BaichuanConverter(original)
converted = converter.converted()

from transformers import PreTrainedTokenizerFast

t_fast = PreTrainedTokenizerFast(
    tokenizer_object=converted,
    model_input_names=original.model_input_names,
    model_max_length=32768,
    clean_up_tokenization_spaces=False,
)

test_strings = [
    " {\n",
    "  {\n",
    "x  {\n",
    "----------------------------------------------------------------------------\n",
    "\n \n",
    "\n  \n",
    '// -----------------------------------------------------------------------\n',
    '-----------------------------------------------------------------------\n',
]
for test_string in test_strings:
    print("Original:", original.encode(test_string))
    print("Fast:    ", t_fast.encode(test_string))


# Testing on xnli

from datasets import load_dataset
from tqdm import tqdm

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

# Testing on codeparrot

ds = load_dataset("codeparrot/github-code", streaming=True, trust_remote_code=True, split="train")

iterator = iter(ds)
for _ in tqdm(range(1000)):
    item = next(iterator)
    code = item["code"]
    lang = item["language"]
    verify(lang, code)

#t_fast.push_to_hub("Baichuan-M1-14B-Instruct-tokenizer")




