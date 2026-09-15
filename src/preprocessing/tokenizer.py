from nltk.tokenize import word_tokenize



def tokenize(text):


    return word_tokenize(
        text
    )



def remove_stopwords(tokens):


    from nltk.corpus import stopwords


    stop_words=set(
        stopwords.words("english")
    )


    return [
        word
        for word in tokens
        if word not in stop_words
    ]