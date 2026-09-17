from sklearn.feature_extraction.text import CountVectorizer



def create_ngram_features(texts):


    vectorizer=CountVectorizer(

        ngram_range=(1,3),

        analyzer="word"

    )


    X=vectorizer.fit_transform(
        texts
    )


    return X,vectorizer





def create_character_ngrams(texts):


    vectorizer=CountVectorizer(

        analyzer="char",

        ngram_range=(3,5),

        max_features=3000

    )


    X=vectorizer.fit_transform(
        texts
    )


    return X,vectorizer