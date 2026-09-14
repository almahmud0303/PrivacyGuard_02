import pandas as pd
import random


# -------------------------
# PII DATA
# -------------------------


names = [
    "John Smith",
    "David Miller",
    "Sarah Ahmed",
    "Michael Brown",
    "Rahim Hasan"
]


emails = [
    "john@gmail.com",
    "david@yahoo.com",
    "sarah123@gmail.com",
    "rahim@hotmail.com"
]


phones = [
    "01712345678",
    "01855555555",
    "01999999999",
    "01677777777"
]


addresses = [
    "Dhaka Bangladesh",
    "New York USA",
    "London UK",
    "Chittagong Bangladesh"
]


accounts = [
    "123456789",
    "987654321",
    "456789123"
]


# -------------------------
# SENTENCE TEMPLATES
# -------------------------


templates = [

"My name is {name} and my email is {email}",


"Please contact me at {phone}",


"My phone number is {phone} and my name is {name}",


"My account number is {account}",


"I live at {address}",


"Name: {name}, Email: {email}, Phone: {phone}",


"Send the document to {email}",


"My address is {address}"

]

noisy_templates=[


"my phn no is {phone}",


"my ph0ne number {phone}",


"email me {email}",


"my nm is {name}",


"addr {address}"

]

# -------------------------
# GENERATOR
# -------------------------


dataset=[]

for i in range(3000):


    template=random.choice(
        noisy_templates
    )


    text=template.format(

        phone=random.choice(phones),

        email=random.choice(emails),

        name=random.choice(names),

        address=random.choice(addresses)

    )


    dataset.append(
        {
        "text":text
        }
    )

for i in range(10000):


    template=random.choice(
        templates
    )


    text=template.format(

        name=random.choice(names),

        email=random.choice(emails),

        phone=random.choice(phones),

        address=random.choice(addresses),

        account=random.choice(accounts)

    )


    dataset.append(
        {
        "text":text
        }
    )



df=pd.DataFrame(dataset)



df.to_csv(
"data/raw/pii_dataset.csv",
index=False
)


print(
"Dataset created successfully"
)


print(
df.head()
)