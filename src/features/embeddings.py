from gensim.models import Word2Vec

import pandas as pd

import pickle

import os




def train_word2vec(sentences):


    model=Word2Vec(

        sentences,

        vector_size=100,

        window=5,

        min_count=1,

        workers=4,

        sg=1

    )


    os.makedirs(

        "../../models_saved",

        exist_ok=True

    )


    model.save(

        "../../models_saved/word2vec.model"

    )


    return model





def sentence_embedding(
        tokens,
        model
):


    vectors=[]


    for word in tokens:


        if word in model.wv:

            vectors.append(
                model.wv[word]
            )



    if len(vectors)==0:

        return [0]*100



    return sum(vectors)/len(vectors)