from .entity_detector import detect_entities

from .risk_score import calculate_risk





def analyze_privacy(text):


    entities=detect_entities(text)



    results=[]



    for item in entities:


        risk=calculate_risk(

            item["type"]

        )


        results.append({


            "value":

            item["entity"],


            "type":

            item["type"],


            "risk":

            risk["level"],


            "score":

            risk["score"]


        })


    return results