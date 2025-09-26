# test_calls.py
from chatbot_brain import handle

note = "Client liked the demo and asked about pricing and integrations; wants a follow-up next week."
print("1) Greeting:")
print(handle("s1","hello", note))
print("\n2) Ask keywords:")
print(handle("s1","show keywords", note))
print("\n3) Ask sentiment:")
print(handle("s1","what's the sentiment", note))
print("\n4) Predict lead:")
print(handle("s1","predict likelihood", note))
