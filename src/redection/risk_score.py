RISK_LEVELS = {


    "PERSON": {

        "level":"MEDIUM",

        "score":2

    },


    "LOCATION": {

        "level":"LOW",

        "score":1

    },


    "DATE": {

        "level":"MEDIUM",

        "score":2

    },


    "PHONE": {

        "level":"HIGH",

        "score":3

    },


    "EMAIL": {

        "level":"HIGH",

        "score":3

    },


    "ACCOUNT": {

        "level":"CRITICAL",

        "score":4

    },


    "NID": {

        "level":"CRITICAL",

        "score":4

    },


    "CREDIT_CARD": {

        "level":"CRITICAL",

        "score":4

    }

}
def calculate_risk(entity_type):


    if entity_type in RISK_LEVELS:


        return RISK_LEVELS[entity_type]


    else:


        return {


            "level":"UNKNOWN",

            "score":0

        }

def calculate_total_risk(results):


    total=0


    for item in results:


        total += item["score"]



    if total>=8:

        return "VERY HIGH"



    elif total>=5:

        return "HIGH"



    elif total>=3:

        return "MEDIUM"



    else:

        return "LOW"