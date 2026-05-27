from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Your Azure details
KEY = ""
ENDPOINT = ""

# Connect to Azure AI
client = TextAnalyticsClient(ENDPOINT, AzureKeyCredential(KEY))
# These are the sentences we want to analyze
documents = [
    "I love learning Azure AI, it is amazing!",
    "This is very difficult and confusing.",
    "The weather today is okay."
]

# Send to Azure AI and get results
result = client.analyze_sentiment(documents)

# Print results
for i, doc in enumerate(result):
    print(f"Sentence : {documents[i]}")
    print(f"Sentiment: {doc.sentiment}")
    print(f"Positive : {doc.confidence_scores.positive:.2f}")
    print(f"Negative : {doc.confidence_scores.negative:.2f}")
    print(f"Neutral  : {doc.confidence_scores.neutral:.2f}")
    print("------------------------")