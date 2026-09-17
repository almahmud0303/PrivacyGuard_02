import pandas as pd
import random
import os


DATA_PATH = "../../data/raw/pii_dataset.csv"



names = [
    "Rahim Ahmed",
    "Karim Hasan",
    "Nusrat Jahan",
    "John Smith",
    "David Miller"
]


phones = [
    "01712345678",
    "01898765432",
    "01955667788"
]


emails = [
    "rahim@gmail.com",
    "john@yahoo.com",
    "david@gmail.com"
]


addresses = [
    "Dhaka Bangladesh",
    "Chittagong Bangladesh",
    "New York USA"
]


nids = [
    "1234567890",
    "9876543210"
]


templates = [

"My name is {name} and my phone number is {phone}",


"Contact me at {email}",


"My NID number is {nid}",


"I live at {address}",


"{name} lives in {address} and email is {email}",


"amar nam {name} amar phone {phone}",


"amar email holo {email}",


"ami thaki {address}",


]


data=[]


for i in range(5000):

    template=random.choice(templates)


    text=template.format(
        name=random.choice(names),
        phone=random.choice(phones),
        email=random.choice(emails),
        address=random.choice(addresses),
        nid=random.choice(nids)
    )


    data.append(
        {
            "id":i,
            "text":text
        }
    )


normal_sentences=[

    "Today is a beautiful day",

    "I like learning NLP",

    "The university campus is large",

    "Machine learning is interesting",

    "I am studying computer science"

]

for i in range(1000):


    data.append(
        {
            "id":5000+i,

            "text":random.choice(
                normal_sentences
            )
        }
    )

df=pd.DataFrame(data)



os.makedirs(
    "../../data/raw",
    exist_ok=True
)


df.to_csv(
    DATA_PATH,
    index=False
)



print("Dataset Created")
print(df.head())
print("Total samples:",len(df))