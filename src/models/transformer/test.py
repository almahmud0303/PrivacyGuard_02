from bert_entity_detector import BERTEntityDetector



detector=BERTEntityDetector()



text="""

My phone number is 01712345678

My email is john@gmail.com

"""



result=detector.predict(text)



for entity in result:

    print(entity)