from dotenv import load_dotenv
import os

load_dotenv()  # loads .env variables into environment
api_key = os.getenv("WEATHER_API_KEY")

print("Your API key is:", api_key[:10] + "...")  # never print full keys in logs