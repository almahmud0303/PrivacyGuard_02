import re



def remove_html(text):

    """
    Remove HTML tags
    """

    return re.sub(
        r"<[^>]+>",
        "",
        text
    )



def remove_urls(text):

    """
    Remove web URLs
    """

    return re.sub(
        r"http\S+|www\S+",
        "",
        text
    )



def remove_special_characters(text):

    """
    Keep letters numbers spaces
    """

    return re.sub(
        r"[^a-zA-Z0-9\s@._]",
        "",
        text
    )



def normalize_spaces(text):

    """
    Remove extra spaces
    """

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()



def clean_text(text):


    text=remove_html(text)


    text=remove_urls(text)


    text=remove_special_characters(text)


    text=normalize_spaces(text)



    return text.lower()