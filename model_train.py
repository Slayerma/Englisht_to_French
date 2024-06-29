import json

from keras.callbacks import EarlyStopping

def train_model(model, x, y, batch_size=1024, epochs=10, validation_split=0.2):
    # Ensure y is of integer type
    y = y.astype(int)
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(x, y, batch_size=batch_size, epochs=epochs, validation_split=validation_split, callbacks=[early_stopping])


def save_model_and_tokenizers(model, english_tokenizer, french_tokenizer, max_french_sequence_length):
    model.save('english_to_french_model')
    
    with open('english_tokenizer.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(english_tokenizer.to_json(), ensure_ascii=False))
    
    with open('french_tokenizer.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(french_tokenizer.to_json(), ensure_ascii=False))
    
    with open('sequence_length.json', 'w', encoding='utf8') as f:
        f.write(json.dumps(max_french_sequence_length, ensure_ascii=False))
