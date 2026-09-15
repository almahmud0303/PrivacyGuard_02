NOISE_MAP={


"phn":"phone",

"phon":"phone",

"nmbr":"number",

"num":"number",

"mail":"email",

"eml":"email",

"addr":"address",

"amar":"my",

"holo":"is"

}




def normalize_noise(tokens):


    cleaned=[]


    for token in tokens:


        if token in NOISE_MAP:

            cleaned.append(
                NOISE_MAP[token]
            )


        else:

            cleaned.append(token)



    return cleaned