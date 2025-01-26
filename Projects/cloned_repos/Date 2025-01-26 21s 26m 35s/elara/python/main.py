import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from gliner import GLiNER
import json


class Msg(BaseModel):
    text: str


model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

with open("../labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/")
async def root(msg: Msg):
    entities = model.predict_entities(msg.text, labels=labels, threshold=0.2)
    entity_map = {}
    print(json.dumps(entities, indent=2))

    anonymized_text = ""
    next_start = -1

    for entity in entities:
        label = entity['label'].replace(" ", "_").upper()

        entity_idx = -1

        if label not in entity_map:
            entity_map[label] = [entity['text']]
            entity_idx = 1
        else:
            if entity['text'] not in entity_map[label]:
                entity_map[label].append(entity['text'])
                entity_idx = len(entity_map[label])
            else:
                entity_idx = entity_map[label].index(entity['text']) + 1

        print(entity_idx, entity['text'])

        if next_start == -1:
            anonymized_text += msg.text[:entity['start']] + \
                f"<PII_{label}_{entity_idx}>"
        else:
            anonymized_text += msg.text[next_start:entity['start']
                                        ] + f"<PII_{label}_{entity_idx}>"

        next_start = entity['end']

    if next_start != -1:
        anonymized_text += msg.text[next_start:]

    return {
        "entities": entity_map,
        "anonymized_text": anonymized_text
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
