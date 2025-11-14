from dotenv import load_dotenv
import os
import requests
import json
import datetime
# Access API key
load_dotenv()  # loads .env variables into environment
api_key = os.getenv("WEATHER_API_KEY")
print("Your API key is:", api_key[:10] + "...")  # never print full keys in logs

# Form basis for API calls to the weather API
url = "https://api.weatherapi.com/v1/history.json"
parameters = {"key": api_key, "q": "iata:BOS", "dt": "", "end_dt": "", "tp": "15"}

# Get the start and end dates for API calls
start_date_list = []
end_date_list = []
with open("dates.txt","r") as date_file:
    for line in date_file.readlines():
        l = line.strip()
        l = l.split("|")
        start_date = l[2]
        start_date = start_date.strip(" ")
        start_date_list.append(start_date)
        end_date = l[3]
        end_date = end_date.strip(" ")
        end_date_list.append(end_date)
date_file.close()    
print(start_date_list)

for i in range(0,len(start_date_list)):
    start_date = start_date_list[i]
    end_date = end_date_list[i]
    parameters["dt"] = start_date
    parameters["end_dt"] = end_date
    response = requests.get(url = url, params = parameters)
    print(response.url)
    if response.status_code == 200:
        filename = parameters["dt"] + "_to_" + parameters["end_dt"] + ".json"
        with open(filename, "w") as f:
            json.dump(response.json(), f, indent=4)
        print(f"✅ Saved {filename}")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json)
        print(response.text)