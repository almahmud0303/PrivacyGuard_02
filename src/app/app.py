import streamlit as st


import sys

import os



# Add project root to Python path

ROOT_DIR=os.path.abspath(

    os.path.join(

        os.path.dirname(__file__),

        "../.."

    )
)

SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)


from redection.privacy_firewall import privacy_guard  # pyright: ignore[reportMissingImports]




# --------------------------
# Page Configuration
# --------------------------


st.set_page_config(

    page_title="PrivacyGuard",

    page_icon="🔒",

    layout="wide"

)



# --------------------------
# Title
# --------------------------


st.title(
    "🔒 PrivacyGuard"
)


st.subheader(
    "Context-Aware PII Detection and Redaction System"
)



st.write(

"""
PrivacyGuard detects Personally Identifiable Information (PII),
evaluates privacy risk, and automatically protects sensitive data.
"""

)



# --------------------------
# Input Area
# --------------------------


text=st.text_area(

    "Enter your text",

    height=200,

    placeholder=
    """
Example:

My name is John.
My phone number is 01712345678.
My email is john@gmail.com

"""

)



# --------------------------
# Analyze Button
# --------------------------


if st.button("Analyze Privacy"):


    if text.strip()=="":


        st.warning(
            "Please enter some text"
        )


    else:


        result=privacy_guard(text)



        st.success(
            "Analysis Completed"
        )



        # -------------------
        # Entities
        # -------------------


        st.header(
            "Detected Information"
        )



        if len(result["entities"])==0:


            st.info(
                "No sensitive information detected"
            )


        else:


            for entity in result["entities"]:


                st.write(

                f"""
                **Entity:** {entity['entity']}

                **Type:** {entity['type']}

                **Risk:** {entity['risk']}

                """

                )



        # -------------------
        # Redacted Text
        # -------------------


        st.header(
            "Protected Text"
        )


        st.code(

            result["safe_text"]

        )
