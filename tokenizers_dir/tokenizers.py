from transformers import AutoTokenizer

def load_tokenizers(model_name, source_tokenizer_langs, target_tokenizer_lang):
    source_tokenizers = []
    for lang in source_tokenizer_langs:
        source_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=lang)
        source_tokenizers.append(source_tokenizer)
    target_tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=target_tokenizer_lang, tgt_lang=target_tokenizer_lang)
    return source_tokenizers, target_tokenizer