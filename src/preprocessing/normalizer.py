import unicodedata
import re



def unicode_normalize(text):


    return unicodedata.normalize(
        "NFKC",
        text
    )




def reduce_repeated_characters(text):


    return re.sub(
        r"(.)\1{2,}",
        r"\1\1",
        text
    )




def normalize_text(text):


    text=unicode_normalize(text)


    text=reduce_repeated_characters(
        text
    )


    return text