"""
Generate synthetic noisy STT transcripts with labeled PII entities.

Produces data/train.jsonl, data/dev.jsonl, data/test.jsonl
"""
import json
import os
import random
import argparse

FIRST_NAMES = ["john", "jane", "mike", "susan", "alex", "linda", "david", "maya"]
LAST_NAMES = ["doe", "smith", "johnson", "williams", "patel", "garcia"]
CITIES = ["new york", "san francisco", "chicago", "boston", "mumbai", "london"]
LOCATIONS = ["main street", "central park", "airport terminal", "office building"]


def noisy_email(name=None):
    if name is None:
        name = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
    parts = name.split()
    user = ".".join(parts)
    domains = ["gmail", "yahoo", "hotmail"]
    dom = random.choice(domains)
    return f"{user} at {dom} dot com"


def noisy_phone():
    # produce spoken digits, maybe with country code
    digits = [str(random.randint(0, 9)) for _ in range(10)]
    spoken = " ".join(digits)
    # optionally group
    return spoken


def noisy_credit_card():
    digits = [str(random.randint(0, 9)) for _ in range(16)]
    # speak as groups of 4
    groups = [" ".join(digits[i:i+4]) for i in range(0, 16, 4)]
    return " ".join(groups)


def noisy_date():
    months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    month = random.choice(months)
    day = random.randint(1, 28)
    year = random.randint(1990, 2025)
    return f"{month} {day} {year}"


def gen_example(idx):
    # choose a template and insert entities
    templates = [
        "call {person} at {phone}",
        "email {person} at {email}",
        "my credit card is {cc}",
        "i live in {city} near {location}",
        "meet {person} on {date}",
        "send to {email} and cc {person}",
    ]
    template = random.choice(templates)
    person = random.choice(FIRST_NAMES) + " " + random.choice(LAST_NAMES)
    phone = noisy_phone()
    email = noisy_email(person)
    cc = noisy_credit_card()
    city = random.choice(CITIES)
    location = random.choice(LOCATIONS)
    date = noisy_date()

    text = template.format(person=person, phone=phone, email=email, cc=cc, city=city, location=location, date=date)

    # lower and remove punctuation to simulate STT
    text = text.lower()

    entities = []
    # find spans
    if "{person}" not in template:
        pass
    # for each possible entity, find substrings and add spans
    for label, substr in [("PERSON_NAME", person), ("PHONE", phone), ("EMAIL", email), ("CREDIT_CARD", cc), ("CITY", city), ("LOCATION", location), ("DATE", date)]:
        if substr and substr in text:
            s = text.index(substr)
            e = s + len(substr)
            entities.append({"start": s, "end": e, "label": label})

    return {"id": f"utt_{idx:06d}", "text": text, "entities": entities}


def generate(out_dir: str, n_train=700, n_dev=150, n_test=150):
    os.makedirs(out_dir, exist_ok=True)
    train = [gen_example(i) for i in range(n_train)]
    dev = [gen_example(i + n_train) for i in range(n_dev)]
    test = [ {"id": f"test_{i}", "text": gen_example(i)["text"]} for i in range(n_test)]

    with open(os.path.join(out_dir, "train.jsonl"), "w", encoding="utf-8") as f:
        for j in train:
            f.write(json.dumps(j) + "\n")
    with open(os.path.join(out_dir, "dev.jsonl"), "w", encoding="utf-8") as f:
        for j in dev:
            f.write(json.dumps(j) + "\n")
    with open(os.path.join(out_dir, "test.jsonl"), "w", encoding="utf-8") as f:
        for j in test:
            f.write(json.dumps(j) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="data")
    parser.add_argument("--train", type=int, default=700)
    parser.add_argument("--dev", type=int, default=150)
    parser.add_argument("--test", type=int, default=150)
    args = parser.parse_args()
    generate(args.out_dir, n_train=args.train, n_dev=args.dev, n_test=args.test)
