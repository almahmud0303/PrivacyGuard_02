import re



def detect_entities(text):


    entities=[]



    # Email

    emails=re.findall(

        r"[\w\.-]+@[\w\.-]+",

        text

    )


    for email in emails:


        entities.append({

            "entity":email,

            "type":"EMAIL"

        })



    # Phone

    phones=re.findall(

        r"01\d{9}",

        text

    )


    for phone in phones:


        entities.append({

            "entity":phone,

            "type":"PHONE"

        })



    # NID

    nids=re.findall(

        r"\b\d{10}\b",

        text

    )


    for nid in nids:


        entities.append({

            "entity":nid,

            "type":"NID"

        })



    return entities