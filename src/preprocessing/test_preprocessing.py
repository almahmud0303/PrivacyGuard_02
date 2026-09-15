from pipeline import preprocessing_pipeline



texts=[


"My phone number is 01712345678!!!",


"my phn nmbr is 01712345678",


"amar email holo test@gmail.com"


]


for text in texts:


    print("\nOriginal:")

    print(text)


    print("Processed:")

    print(
        preprocessing_pipeline(text)
    )