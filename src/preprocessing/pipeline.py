from cleaner import clean_text

from normalizer import normalize_text

from tokenizer import tokenize

from noise_handler import normalize_noise




def preprocessing_pipeline(text):


    # Unicode normalization

    text=normalize_text(text)



    # Cleaning

    text=clean_text(text)



    # Tokenization

    tokens=tokenize(text)



    # Noise correction

    tokens=normalize_noise(tokens)



    return tokens