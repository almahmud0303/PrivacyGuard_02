from bert_entity_detector import BERTEntityDetector


def predict(text):
    detector = BERTEntityDetector()
    return detector.predict(text)


if __name__ == "__main__":
    print(predict("My phone number is 01712345678"))
