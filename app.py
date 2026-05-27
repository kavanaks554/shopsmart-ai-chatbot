import os
import httpx
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from groq import Groq
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Groq AI Setup - keys from .env file
groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
    http_client=httpx.Client(timeout=30.0)
)

# Azure Language AI Setup - keys from .env file
AZURE_KEY = os.getenv("AZURE_KEY")
AZURE_ENDPOINT = "https://my-ailanguage-service.cognitiveservices.azure.com/"
azure_client = TextAnalyticsClient(
    AZURE_ENDPOINT,
    AzureKeyCredential(AZURE_KEY)
)

def analyze_sentiment(text):
    try:
        result = azure_client.analyze_sentiment([text])
        sentiment = result[0].sentiment
        score = result[0].confidence_scores
        return {
            "sentiment": sentiment,
            "positive": round(score.positive, 2),
            "negative": round(score.negative, 2),
            "neutral": round(score.neutral, 2)
        }
    except Exception as e:
        print(f"Azure error: {e}")
        return {
            "sentiment": "neutral",
            "positive": 0,
            "negative": 0,
            "neutral": 1
        }

system_prompt = """You are a helpful e-commerce
assistant for ShopSmart India.
You help customers with:
- Product recommendations
- Order tracking
- Complaints and refunds
- Product comparisons

FAKE ORDER DATABASE:
Order #SS1001 - Samsung Galaxy M34 - Shipped - Arriving May 28
Order #SS1002 - boAt Earphones - Out for Delivery - Arriving Today
Order #SS1003 - Nike Shoes - Delivered - May 24
Order #SS1004 - Laptop Stand - Processing - Arriving May 30
Order #SS1005 - iPhone 15 - Packed - Arriving May 29

Always be friendly, helpful and professional!"""

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat_response():
    try:
        user_message = request.json.get("message")

        # Azure Sentiment Analysis
        sentiment_result = analyze_sentiment(user_message)
        sentiment = sentiment_result["sentiment"]

        # Tone based on sentiment
        if sentiment == "negative":
            tone = """IMPORTANT: Customer is ANGRY or UPSET!
            - Start with sincere apology
            - Be extra careful and empathetic
            - Offer immediate solution
            - Say issue escalated to priority team"""
        elif sentiment == "positive":
            tone = """Customer is HAPPY!
            - Be enthusiastic and friendly
            - Match their positive energy!"""
        else:
            tone = "Be helpful and professional!"

        full_prompt = f"""{system_prompt}

AZURE AI DETECTED SENTIMENT: {sentiment.upper()}
Positive: {sentiment_result['positive']}
Negative: {sentiment_result['negative']}
Tone Instruction: {tone}"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        bot_reply = response.choices[0].message.content

        return jsonify({
            "response": bot_reply,
            "sentiment": sentiment,
            "scores": sentiment_result
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            "response": "Sorry something went wrong! Please try again.",
            "sentiment": "neutral",
            "scores": {}
        })

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)