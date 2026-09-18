from .entity_detector import detect_entities

from .risk_score import calculate_risk

from .redactor import redact_text




def privacy_guard(text):


    entities=detect_entities(text)



    for entity in entities:


        risk=calculate_risk(

            entity["type"]

        )


        entity["risk"]=risk["level"]



    safe_text=redact_text(

        text,

        entities

    )



    return {


        "original":

        text,


        "entities":

        entities,


        "safe_text":

        safe_text

    }