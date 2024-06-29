import numpy as np

def logits_to_text(logits, tokenizer):
    index_to_words = {id: word for word, id in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join(index_to_words[prediction] for prediction in np.argmax(logits, 1))

def evaluate_model(model, x, y, tokenizer):
    prediction = logits_to_text(model.predict(x[:1])[0], tokenizer)
    print("Prediction:", prediction)
    print("Correct Translation:", y[:1])
