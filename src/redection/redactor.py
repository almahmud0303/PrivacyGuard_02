REPLACEMENT_MAP = {


    "PERSON":
    "[PERSON]",


    "EMAIL":
    "[EMAIL]",


    "PHONE":
    "[PHONE_NUMBER]",


    "NID":
    "[NID]",


    "LOCATION":
    "[LOCATION]",


    "ACCOUNT":
    "[ACCOUNT_NUMBER]",


    "CREDIT_CARD":
    "[CREDIT_CARD]"


}

def replace_entity(
        text,
        entity,
        entity_type
):


    placeholder = REPLACEMENT_MAP.get(

        entity_type,

        "[PRIVATE_DATA]"

    )


    return text.replace(

        entity,

        placeholder

    )
def redact_text(
        text,
        entities
):


    priority={

        "CREDIT_CARD":4,

        "ACCOUNT":4,

        "NID":4,

        "PHONE":3,

        "EMAIL":3,

        "PERSON":2,

        "LOCATION":1

    }



    entities=sorted(

        entities,

        key=lambda x:
        priority.get(
            x["type"],
            0
        ),

        reverse=True

    )



    redacted=text



    for item in entities:


        redacted=replace_entity(

            redacted,

            item["entity"],

            item["type"]

        )


    return redacted

def should_redact(
        confidence,
        threshold=0.80
):


    return confidence >= threshold