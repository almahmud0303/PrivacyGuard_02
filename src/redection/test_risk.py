from privacy_firewall import privacy_guard




text="""

My name is John Smith.

My phone number is 01712345678.

My email is john@gmail.com.

My NID is 1234567890.

"""



result=privacy_guard(text)



print(
"Detected Entities:"
)



for e in result["entities"]:

    print(e)



print("\nSafe Text:")


print(
result["safe_text"]
)