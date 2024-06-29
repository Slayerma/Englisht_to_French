import streamlit as st
import subprocess
import json

def load_file(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        return f.read()

def main():
    st.title('English to French Translation Model')

    st.header('Running main.py')
    result = subprocess.run(['python', 'main.py'], capture_output=True, text=True)
    
    if result.returncode == 0:
        st.success('main.py executed successfully.')
    else:
        st.error('Error executing main.py.')
        st.text(result.stderr)
        return

    st.header('Model Output')

    # Displaying model and tokenizers information
    st.subheader('English to French Model')
    st.text(load_file('english_to_french_model'))

    st.subheader('English Tokenizer')
    st.json(json.loads(load_file('english_tokenizer.json')))

    st.subheader('French Tokenizer')
    st.json(json.loads(load_file('french_tokenizer.json')))

    st.subheader('Max French Sequence Length')
    st.json(load_file('sequence_length.json'))

    st.subheader('Console Output')
    st.text(result.stdout)

if __name__ == '__main__':
    main()
