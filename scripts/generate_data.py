import json
import random
import re
from pathlib import Path

# Simple noisy STT generator for PII entity recognition
# Produces JSONL with fields: id, text, entities (start, end, label)

random.seed(42)

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Entity pools
names = [
    "Ramesh Sharma", "Anita", "John Doe", "Maria Garcia", "Li Wei", "Emily", "Raj Kumar",
    "Sarah", "Mohammed", "Luisa", "Oliver", "Chris",
]

cities = [
    "New York", "Mumbai", "Delhi", "Bengaluru", "San Francisco", "Chicago", "Hyderabad",
]

locations = ["office", "branch", "warehouse", "hotel", "airport"]

dates = [
    "January 5th, 2021", "March 3rd", "10/10/2020", "next Monday", "the twenty-third of April",
]

emails = [
    "ramesh@gmail.com", "john.doe@example.com", "sales@company.org", "maria@yahoo.com",
]

# phone variants: numeric grouped and spoken
phones = [
    "9876543210", "987-654-3210", "(987) 654 3210", "nine eight seven six five four three two one",
    "one two three four five six seven eight nine zero",
]

# credit cards common patterns (digits grouped) and spoken word variants
credit_cards = [
    "4242 4242 4242 4242",
    "4333 5555 1222 7777",
    "4444-4444-4444-4444",
    "four two four two four two four two four two four two four two four two",
]

# Map simpler labels
LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

# Template patterns with placeholders
templates = [
    "My credit card number is {cc}.",
    "Call me at {phone}.",
    "Email is {email}.",
    "This is {name} from {city}.",
    "Met {name} on {date}.",
    "Went to the {location} in {city}.",
    "Reach out to {name} at {email}.",
    "Card ending in {last4}.",
    "My number is {phone} and email {email}.",
    "I live in {city} near the {location}.",
    # more realistic conversational variants
    "Hey, it's {name}. You can reach me at {phone} or {email}.",
    "I gave the card 4242 4242 4242 4242 to the agent.",
    "The booking is under {name} for {date} at the {location}.",
    "Send invoice to {email} and call {phone} if any issues.",
]

# Noise injection functions

def inject_noise(text):
    # lowercasing, occasional repeated words, filler words, missing punctuation
    text = text.lower()
    # replace digits with spoken forms sometimes
    text = re.sub(r"\b0\b", "oh", text)
    if random.random() < 0.2:
        # repeat a small word
        words = text.split()
        i = random.randrange(len(words))
        words.insert(i+1, words[i])
        text = " ".join(words)
    if random.random() < 0.15:
        text = "um " + text
    if random.random() < 0.1:
        text = text.replace("@", " at ").replace(".", " dot ")
    # spoken numbers sometimes grouped
    return text


def spoken_last4_from_cc(cc_text):
    # take last 4 numeric words
    digits = re.findall(r"\d+", cc_text)
    if digits:
        s = digits[-1]
        return " ".join(list(s[-4:]))
    # fallback
    parts = cc_text.split()
    return " ".join(parts[-4:])


def sample_entity(label):
    if label == "PERSON_NAME":
        return random.choice(names)
    if label == "CITY":
        return random.choice(cities)
    if label == "LOCATION":
        return random.choice(locations)
    if label == "DATE":
        return random.choice(dates)
    if label == "EMAIL":
        return random.choice(emails)
    if label == "PHONE":
        return random.choice(phones)
    if label == "CREDIT_CARD":
        return random.choice(credit_cards)
    return ""


def build_text_and_entities(template):
    # Replace placeholders and record entity spans
    text = template
    entities = []
    # handle last4 separately
    if "{cc}" in text:
        val = sample_entity("CREDIT_CARD")
        text = text.replace("{cc}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "CREDIT_CARD"})
    if "{phone}" in text:
        val = sample_entity("PHONE")
        text = text.replace("{phone}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "PHONE"})
    if "{email}" in text:
        val = sample_entity("EMAIL")
        text = text.replace("{email}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "EMAIL"})
    if "{name}" in text:
        val = sample_entity("PERSON_NAME")
        text = text.replace("{name}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "PERSON_NAME"})
    if "{city}" in text:
        val = sample_entity("CITY")
        text = text.replace("{city}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "CITY"})
    if "{location}" in text:
        val = sample_entity("LOCATION")
        text = text.replace("{location}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "LOCATION"})
    if "{date}" in text:
        val = sample_entity("DATE")
        text = text.replace("{date}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "DATE"})
    if "{last4}" in text:
        # sample cc then take last4
        cc = sample_entity("CREDIT_CARD")
        last4 = " ".join(cc.split()[-4:])
        val = last4
        text = text.replace("{last4}", val)
        start = text.index(val)
        end = start + len(val)
        entities.append({"start": start, "end": end, "label": "CREDIT_CARD"})

    # Inject noise
    noisy = inject_noise(text)
    # If noise changed length or words, we need to re-locate entities conservatively by searching the entity strings
    new_entities = []
    for ent in entities:
        label = ent["label"]
        ent_text = text[ent["start"]:ent["end"]]
        # find first occurrence in noisy text
        idx = noisy.find(ent_text)
        if idx >= 0:
            new_entities.append({"start": idx, "end": idx + len(ent_text), "label": label})
        else:
            # try simplified search: collapse multiple spaces
            simplified = re.sub(r"\s+", " ", noisy)
            idx2 = simplified.find(ent_text)
            if idx2 >= 0:
                new_entities.append({"start": idx2, "end": idx2 + len(ent_text), "label": label})
            else:
                # as fallback, skip this entity (rare)
                pass

    return noisy, new_entities


def generate_dataset(num_examples, out_path):
    with open(out_path, "w", encoding="utf-8") as fh:
        for i in range(num_examples):
            tmpl = random.choice(templates)
            text, entities = build_text_and_entities(tmpl)
            # sometimes compose multiple templates into one utterance
            if random.random() < 0.3:
                tmpl2 = random.choice(templates)
                text2, ent2 = build_text_and_entities(tmpl2)
                sep = random.choice([" and ", ", ", " then "])
                base = text + sep + text2
                # merge entities adjusting offsets
                entities = entities + [{"start": e["start"] + len(text) + len(sep), "end": e["end"] + len(text) + len(sep), "label": e["label"]} for e in ent2]
                text = base
            # add filler
            if random.random() < 0.25:
                text = "um " + text
            rec = {"id": f"gen_{i:05d}", "text": text, "entities": entities}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Generate ~600 train and ~150 dev by default
    generate_dataset(600, OUTPUT_DIR / "gen_train.jsonl")
    generate_dataset(150, OUTPUT_DIR / "gen_dev.jsonl")
    print("Wrote gen_train.jsonl and gen_dev.jsonl to data/")
