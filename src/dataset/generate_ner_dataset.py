from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "processed"

FIRST = ["Rysul", "Al", "Nirob", "Sarah", "Rahman", "Ayesha", "Nusrat", "Farhan", "Sadia", "Tanvir", "Mehedi",
         "Arif", "Jannat", "Tasnim", "John", "Maria", "David", "Sophia", "রাইসুল", "আল",
         "নিরব", "আয়েশা", "নুসরাত", "ফারহান", "সাদিয়া", "তানভীর"]
MIDDLE = ["Aman", "Mahmud", "Hasan", "Ahmed", "Jahan", "Islam", "Khan", "আমান", "মাহমুদ",
          "হাসান", "আহমেদ", "জাহান", "ইসলাম"]
LAST = ["Nirob", "Rahman", "Chowdhury", "Hossain", "Akter", "Kabir", "Smith", "Miller",
        "নিরব", "রহমান", "চৌধুরী", "হোসেন", "আক্তার", "কবির"]
LOCATIONS = ["Dhaka", "Chattogram", "Rajshahi", "Khulna", "Sylhet", "Barishal", "Rangpur",
             "Mymensingh", "Cox's Bazar", "New York", "London", "Toronto", "ঢাকা", "চট্টগ্রাম",
             "রাজশাহী", "খুলনা", "সিলেট", "বরিশাল", "রংপুর", "কক্সবাজার"]
ORGANIZATIONS = ["TechNova Solution Ltd", "TechNova Solutions Ltd", "City Care Hospital", "Dhaka University",
                 "North South University", "BRAC Bank", "Grameen Digital Services",
                 "Square Hospital", "Bangladesh University of Engineering and Technology"]
ADDRESSES = ["House 25 , Road 7 , Dhanmondi Dhaka Bangladesh", "House 25 Road 7 Dhanmondi Dhaka Bangladesh", "Flat 4B House 12 Banani Dhaka",
             "Road 3 Uttara Sector 7 Dhaka", "Village Sonapur Post Office Begumganj Noakhali",
             "42 Park Street London", "বাড়ি ২৫ রোড ৭ ধানমন্ডি ঢাকা বাংলাদেশ"]
CONDITIONS = ["diabetes", "diabetis", "high blood pressure", "asthma", "heart disease", "kidney disease",
              "migraine", "arthritis", "ডায়াবেটিস", "উচ্চ রক্তচাপ", "হাঁপানি"]
OCCUPATIONS = ["software engineer", "software enginer", "data scientist", "teacher", "doctor", "bank officer",
               "civil engineer", "research assistant", "ব্যাংক কর্মকর্তা", "সফটওয়্যার প্রকৌশলী"]
EDUCATION_FIELDS = ["Computer Science", "Electrical Engineering", "Business Administration",
                    "English Literature", "Economics", "Medicine", "কম্পিউটার বিজ্ঞান"]


def entity(kind: str, value: str) -> tuple[str, str]:
    return kind, value


def render(parts: list[str | tuple[str, str]]) -> dict[str, list[str]]:
    tokens, labels = [], []
    for part in parts:
        if isinstance(part, str):
            values = part.split()
            tokens.extend(values); labels.extend(["O"] * len(values))
        else:
            kind, value = part
            values = value.split()
            tokens.extend(values)
            labels.extend([f"B-{kind}"] + [f"I-{kind}"] * (len(values) - 1))
    return {"tokens": tokens, "labels": labels}


def person(rng: random.Random) -> str:
    count = rng.choice([2, 2, 3])
    return " ".join([rng.choice(FIRST), rng.choice(MIDDLE)] + ([rng.choice(LAST)] if count == 3 else []))


def values(rng: random.Random) -> dict[str, str]:
    name = person(rng)
    latin = "".join(char for char in name.casefold().replace(" ", ".") if char.isascii() and (char.isalnum() or char == ".")) or "user"
    return {
        "PERSON": name,
        "LOCATION": rng.choice(LOCATIONS),
        "PHONE": f"01{rng.choice('3456789')}{rng.randrange(10**8):08d}",
        "EMAIL": f"{latin}{rng.randrange(10,999)}@{rng.choice(['gmail.com','outlook.com','example.org'])}",
        "NID": "".join(rng.choice("0123456789") for _ in range(rng.choice([10, 13, 17]))),
        "CREDIT_CARD": " ".join(f"{rng.randrange(10000):04d}" for _ in range(4)),
        "ACCOUNT": rng.choice([f"AC{rng.randrange(10**10):010d}", f"{rng.randrange(10**12):012d}"]),
        "IP_ADDRESS": ".".join(str(rng.randrange(1, 255)) for _ in range(4)),
        "DATE": rng.choice([
            f"{rng.randrange(1,29):02d}-{rng.randrange(1,13):02d}-{rng.randrange(1960,2027)}",
            f"{rng.randrange(1,29)} {rng.choice(['January','February','March','April','May','June','July','August','September','October','November','December'])} {rng.randrange(1960,2027)}",
        ]),
        "PASSPORT": f"{rng.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{rng.randrange(10**7):07d}",
        "ORGANIZATION": rng.choice(ORGANIZATIONS),
        "ADDRESS": rng.choice(ADDRESSES),
        "EMPLOYEE_ID": f"EMP-{rng.randrange(10000, 99999)}",
        "MEDICAL_ID": f"MED-{rng.randrange(10000, 99999)}",
        "STUDENT_ID": f"{rng.choice(['CSE','EEE','BBA','ENG'])}-{rng.randrange(2010,2027)}-{rng.randrange(1000,9999)}",
        "HEALTH_CONDITION": rng.choice(CONDITIONS),
        "OCCUPATION": rng.choice(OCCUPATIONS),
        "EDUCATION": rng.choice(EDUCATION_FIELDS),
    }


def make_record(rng: random.Random) -> dict[str, list[str]]:
    v = values(rng)
    if rng.random() < .15:
        v["EMAIL"] = v["EMAIL"].replace("@", r"\@")
    templates = [
        ["My colleague", entity("PERSON", v["PERSON"]), "is travelling to", entity("LOCATION", v["LOCATION"]), "."],
        [entity("PERSON", v["PERSON"]), "was talking to", entity("PERSON", person(rng)), "in", entity("LOCATION", v["LOCATION"]), "."],
        ["Contact", entity("PERSON", v["PERSON"]), "at", entity("PHONE", v["PHONE"]), "or", entity("EMAIL", v["EMAIL"]), "."],
        ["NID", entity("NID", v["NID"]), "belongs to", entity("PERSON", v["PERSON"]), "."],
        ["Payment card", entity("CREDIT_CARD", v["CREDIT_CARD"]), "account", entity("ACCOUNT", v["ACCOUNT"]), "."],
        ["Login from", entity("IP_ADDRESS", v["IP_ADDRESS"]), "on", entity("DATE", v["DATE"]), "."],
        ["Passport", entity("PASSPORT", v["PASSPORT"]), "is registered to", entity("PERSON", v["PERSON"]), "."],
        [entity("PERSON", v["PERSON"]), "works at", entity("ORGANIZATION", v["ORGANIZATION"]),
         "with employee ID", entity("EMPLOYEE_ID", v["EMPLOYEE_ID"]), "."],
        ["The home address of", entity("PERSON", v["PERSON"]), "is", entity("ADDRESS", v["ADDRESS"]), "."],
        [entity("PERSON", v["PERSON"]), "visited", entity("ORGANIZATION", v["ORGANIZATION"]), "on",
         entity("DATE", v["DATE"]), "for", entity("HEALTH_CONDITION", v["HEALTH_CONDITION"]),
         "medical ID", entity("MEDICAL_ID", v["MEDICAL_ID"]), "."],
        [entity("PERSON", v["PERSON"]), "studied at", entity("ORGANIZATION", v["ORGANIZATION"]),
         "and student ID was", entity("STUDENT_ID", v["STUDENT_ID"]), "."],
        ["Name", entity("PERSON", v["PERSON"]), "company", entity("ORGANIZATION", v["ORGANIZATION"]),
         "employee", entity("EMPLOYEE_ID", v["EMPLOYEE_ID"]), "email", entity("EMAIL", v["EMAIL"]),
         "phone", entity("PHONE", "+88" + v["PHONE"]), "address", entity("ADDRESS", v["ADDRESS"]),
         "NID", entity("NID", v["NID"]), "account", entity("ACCOUNT", v["ACCOUNT"]), "."],
        [rng.choice(["My name is", "name is"]), entity("PERSON", v["PERSON"]),
         rng.choice(["and I work as a", "and I am work as a"]), entity("OCCUPATION", v["OCCUPATION"]), "in",
         entity("ORGANIZATION", v["ORGANIZATION"]), ". My employee ID is",
         entity("EMPLOYEE_ID", v["EMPLOYEE_ID"]), ". Contact me by email", entity("EMAIL", v["EMAIL"]),
         "or phone", entity("PHONE", "+88" + v["PHONE"]), ". I live at", entity("ADDRESS", v["ADDRESS"]),
         ". My national ID is", entity("NID", v["NID"]), "and bank account is", entity("ACCOUNT", v["ACCOUNT"]),
         ". Last month I", rng.choice(["visited", "was go to"]), entity("ORGANIZATION", v["ORGANIZATION"]),
         "on", entity("DATE", v["DATE"]), "because I have", entity("HEALTH_CONDITION", v["HEALTH_CONDITION"]),
         ". Doctor", entity("PERSON", rng.choice(["Rahman", "Ahmed", person(rng)])),
         "gave me medicine. My medical ID is", entity("MEDICAL_ID", v["MEDICAL_ID"]),
         ". I studied at", entity("ORGANIZATION", v["ORGANIZATION"]), "in", entity("EDUCATION", v["EDUCATION"]),
         "and student ID was",
         entity("STUDENT_ID", v["STUDENT_ID"]), "."],
        ["amar nam", entity("PERSON", v["PERSON"]), "ami thaki", entity("LOCATION", v["LOCATION"]), "."],
        [entity("PERSON", v["PERSON"]), "er mobile", entity("PHONE", v["PHONE"]), "ebong NID", entity("NID", v["NID"]), "."],
        ["আমার নাম", entity("PERSON", v["PERSON"]), "আমি থাকি", entity("LOCATION", v["LOCATION"]), "।"],
        [entity("PERSON", v["PERSON"]), "এর ফোন", entity("PHONE", v["PHONE"]), "এবং ইমেইল", entity("EMAIL", v["EMAIL"]), "।"],
        ["আজ আবহাওয়া সুন্দর এবং আমরা ভাষা প্রযুক্তি শিখছি ।"],
        ["This sentence contains no private information ."],
        ["Machine learning systems require careful evaluation ."],
    ]
    return render(rng.choice(templates))


def write(path: Path, records: list[dict]) -> None:
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multilingual BIO PII data")
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.samples < 100:
        raise SystemExit("--samples must be at least 100")
    rng = random.Random(args.seed)
    records = [make_record(rng) for _ in range(args.samples)]
    rng.shuffle(records)
    train_end, validation_end = int(args.samples * .8), int(args.samples * .9)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write(OUTPUT / "ner_dataset.json", records)
    write(OUTPUT / "train.json", records[:train_end])
    write(OUTPUT / "validation.json", records[train_end:validation_end])
    write(OUTPUT / "test.json", records[validation_end:])
    print(f"Generated {args.samples} records: train={train_end}, validation={validation_end-train_end}, test={args.samples-validation_end}")


if __name__ == "__main__":
    main()
