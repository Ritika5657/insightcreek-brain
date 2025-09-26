# chatbot_brain.py
import random
from analysis import sentiment_score, top_keywords
from predict_today import load_model, predict_probability

# load model once
model = load_model()

INTENTS = [
    {"tag":"greeting","patterns":["hi","hello","hey"],"responses":["Hi! How can I help?"]},
    {"tag":"ask_keywords","patterns":["keywords","top keywords","important topics","extract keywords"],"responses":["Keywords: {keywords}"]},
    {"tag":"ask_sentiment","patterns":["sentiment","tone","feeling"],"responses":["Sentiment polarity: {polarity}"]},
    {"tag":"predict_lead","patterns":["predict","likelihood","probability","will they buy"],"responses":["Predicted probability: {prob:.2f}"]},
    {"tag":"thanks","patterns":["thanks","thank you"],"responses":["You're welcome!"]}
]

FALLBACK = ["Sorry, I didn't understand. Try asking about sentiment, keywords, or predict likelihood."]

def find_intent(message):
    if not message:
        return None
    m = message.lower()
    for it in INTENTS:
        for p in it['patterns']:
            if p in m:
                return it['tag']
    return None

def handle(session_id, message, note_text=None):
    intent = find_intent(message)
    if intent == "ask_keywords":
        if not note_text:
            return {"reply":"Give me the meeting note text to extract keywords.", "intent":intent, "score":0.0}
        kws = top_keywords(note_text, top_n=5)
        return {"reply": INTENTS[1]["responses"][0].format(keywords=", ".join(kws)), "intent":intent, "score":0.95, "meta":{"keywords":kws}}

    if intent == "ask_sentiment":
        if not note_text:
            return {"reply":"Provide the meeting notes text to analyze sentiment.", "intent":intent, "score":0.9}
        sent = sentiment_score(note_text)
        return {"reply": INTENTS[2]["responses"][0].format(polarity=sent["polarity"]), "intent":intent, "score":0.95, "meta":{"sentiment":sent}}

    if intent == "predict_lead":
        if model is None:
            return {"reply":"Model not available. Train the model and place it at models/lead_pipeline.joblib", "intent":intent, "score":0.0}
        if not note_text:
            return {"reply":"Provide meeting notes for prediction.", "intent":intent, "score":0.4}
        prob = predict_probability(model, note_text)
        label = "High" if prob >= 0.7 else ("Medium" if prob >= 0.45 else "Low")
        return {"reply": INTENTS[3]["responses"][0].format(prob=prob), "intent":intent, "score":float(prob), "meta":{"probability":prob, "label":label}}

    if intent == "greeting":
        return {"reply": random.choice(INTENTS[0]["responses"]), "intent":intent, "score":0.9}

    if intent == "thanks":
        return {"reply": random.choice(INTENTS[4]["responses"]), "intent":intent, "score":0.9}

    return {"reply": random.choice(FALLBACK), "intent":None, "score":0.0}
