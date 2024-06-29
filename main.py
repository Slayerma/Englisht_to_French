from Englisht_to_French.data_loader import load_data, preprocess, pad_sequences_with_max_length
from Englisht_to_French.model_builder import build_simple_rnn_model, build_bidirectional_rnn_model, build_bidirectional_embed_model
from Englisht_to_French.model_train import train_model, save_model_and_tokenizers
from Englisht_to_French.model_eval import evaluate_model

# Load data
english_sentences = load_data('/Users/src_dir/dir/english')
french_sentences = load_data('/Users/src_dir/dir/french')

# Preprocess data
preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)

max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index) + 1  # +1 for padding token
french_vocab_size = len(french_tokenizer.word_index) + 1  # +1 for padding token

print('Data Preprocessed')
print(f"Max English sentence length: {max_english_sequence_length}")
print(f"Max French sentence length: {max_french_sequence_length}")
print(f"English vocabulary size: {english_vocab_size}")
print(f"French vocabulary size: {french_vocab_size}")

# Train and evaluate models
tmp_x = pad_sequences_with_max_length(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

# Simple RNN model
simple_rnn_model = build_simple_rnn_model(tmp_x.shape, english_vocab_size, french_vocab_size)
train_model(simple_rnn_model, tmp_x, preproc_french_sentences)
evaluate_model(simple_rnn_model, tmp_x, french_sentences, french_tokenizer)

# Bidirectional RNN model
bd_rnn_model = build_bidirectional_rnn_model(tmp_x.shape, english_vocab_size, french_vocab_size)
train_model(bd_rnn_model, tmp_x, preproc_french_sentences)
evaluate_model(bd_rnn_model, tmp_x, french_sentences, french_tokenizer)

# Bidirectional RNN model with embedding
tmp_x_embed = tmp_x.reshape((-1, preproc_french_sentences.shape[-2]))
embed_rnn_model = build_bidirectional_embed_model(tmp_x_embed.shape, english_vocab_size, french_vocab_size)
train_model(embed_rnn_model, tmp_x_embed, preproc_french_sentences)
evaluate_model(embed_rnn_model, tmp_x_embed, french_sentences, french_tokenizer)

# Save models and tokenizers
save_model_and_tokenizers(embed_rnn_model, english_tokenizer, french_tokenizer, max_french_sequence_length)
