import json
import pandas as pd
from pathlib import Path

# Read the list of JSON files based on dates.txt
file_paths = []
with open("dates.txt","r") as date_file:
    for line in date_file.readlines():
        l = line.strip()
        l = l.split("|")
        start_date = l[2]
        start_date = start_date.strip(" ")
        end_date = l[3]
        end_date = end_date.strip(" ")
        file_path = start_date + "_to_" + end_date + ".json"
        file_paths.append(file_path)
date_file.close()  

# Initialize lists to hold data
dates = []
time = []
time_epoch = []
temp_c = []
temp_f = []
is_day = []
condition_text = []
wind_mph = []
wind_kph = []
wind_degree = []
wind_dir = []
pressure_mb = []
pressure_in = []
precip_mm = []
precip_in = []
snow_cm = []
humidity = []
cloud = []
feelslike_c = []
feelslike_f = []
windchill_c = []
windchill_f = []
heatindex_c = []
heatindex_f = []
dewpoint_c = []
dewpoint_f = []
will_it_rain = []
chance_of_rain = []
will_it_snow = []
chance_of_snow = []
vis_km = []
vis_miles = []
gust_mph = []
gust_kph = []
uv = []

# Process each JSON file and extract hourly data
for file_path in file_paths:
    with open(file_path, "r") as f:
        data = json.load(f)
        for day in data["forecast"]["forecastday"]:
            date = day["date"]
            for hour in day["hour"]:
                dates.append(date)
                time.append(hour["time"])
                time_epoch.append(hour["time_epoch"])
                temp_c.append(hour["temp_c"])
                temp_f.append(hour["temp_f"])
                is_day.append(hour["is_day"])
                condition_text.append(hour["condition"]["text"])
                wind_mph.append(hour["wind_mph"])
                wind_kph.append(hour["wind_kph"])
                wind_degree.append(hour["wind_degree"])
                wind_dir.append(hour["wind_dir"])
                pressure_mb.append(hour["pressure_mb"])
                pressure_in.append(hour["pressure_in"])
                precip_mm.append(hour["precip_mm"])
                precip_in.append(hour["precip_in"])
                snow_cm.append(hour["snow_cm"])
                humidity.append(hour["humidity"])
                cloud.append(hour["cloud"])
                feelslike_c.append(hour["feelslike_c"])
                feelslike_f.append(hour["feelslike_f"])
                windchill_c.append(hour["windchill_c"])
                windchill_f.append(hour["windchill_f"])
                heatindex_c.append(hour["heatindex_c"])
                heatindex_f.append(hour["heatindex_f"])
                dewpoint_c.append(hour["dewpoint_c"])
                dewpoint_f.append(hour["dewpoint_f"])
                will_it_rain.append(hour["will_it_rain"])
                chance_of_rain.append(hour["chance_of_rain"])
                will_it_snow.append(hour["will_it_snow"])
                chance_of_snow.append(hour["chance_of_snow"])
                vis_km.append(hour["vis_km"])
                vis_miles.append(hour["vis_miles"])
                gust_mph.append(hour["gust_mph"])
                gust_kph.append(hour["gust_kph"])
                uv.append(hour["uv"])
    f.close()
print(f"Processed {len(file_paths)} files.")
print(file_paths)
# Create a DataFrame and save to Excel
df = pd.DataFrame({
    "date": dates,
    "time": time,
    "time_epoch": time_epoch,
    "temp_c": temp_c,
    "temp_f": temp_f,
    "is_day": is_day,
    "condition_text": condition_text,
    "wind_mph": wind_mph,
    "wind_kph": wind_kph,
    "wind_degree": wind_degree,
    "wind_dir": wind_dir,
    "pressure_mb": pressure_mb,
    "pressure_in": pressure_in,
    "precip_mm": precip_mm,
    "precip_in": precip_in,
    "snow_cm": snow_cm,
    "humidity": humidity,
    "cloud": cloud,
    "feelslike_c": feelslike_c,
    "feelslike_f": feelslike_f,
    "windchill_c": windchill_c,
    "windchill_f": windchill_f,
    "heatindex_c": heatindex_c,
    "heatindex_f": heatindex_f,
    "dewpoint_c": dewpoint_c,
    "dewpoint_f": dewpoint_f,
    "will_it_rain": will_it_rain,
    "chance_of_rain": chance_of_rain,
    "will_it_snow": will_it_snow,
    "chance_of_snow": chance_of_snow,
    "vis_km": vis_km,
    "vis_miles": vis_miles,
    "gust_mph": gust_mph,
    "gust_kph": gust_kph,
    "uv": uv
})

ROOT = Path(__file__).resolve().parent
CLEAN_DIR = ROOT / "data" / "cleaned"
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

df.to_excel(CLEAN_DIR / "weather_data.xlsx", index=False)

""""
"hour": [
                    {
                        "time_epoch": 1546318800,
                        "time": "2019-01-01 00:00",
                        "temp_c": 4.5,
                        "temp_f": 40.1,
                        "is_day": 0,
                        "condition": {
                            "text": "Moderate or heavy rain shower",
                            "icon": "//cdn.weatherapi.com/weather/64x64/night/356.png",
                            "code": 1243
                        },
                        "wind_mph": 12.0,
                        "wind_kph": 19.3,
                        "wind_degree": 161,
                        "wind_dir": "SSE",
                        "pressure_mb": 1014.0,
                        "pressure_in": 29.94,
                        "precip_mm": 4.48,
                        "precip_in": 0.18,
                        "snow_cm": 0.0,
                        "humidity": 94,
                        "cloud": 100,
                        "feelslike_c": 0.5,
                        "feelslike_f": 33.0,
                        "windchill_c": 0.5,
                        "windchill_f": 33.0,
                        "heatindex_c": 4.5,
                        "heatindex_f": 40.1,
                        "dewpoint_c": 3.6,
                        "dewpoint_f": 38.5,
                        "will_it_rain": 1,
                        "chance_of_rain": 100,
                        "will_it_snow": 0,
                        "chance_of_snow": 0,
                        "vis_km": 5.0,
                        "vis_miles": 3.0,
                        "gust_mph": 23.4,
                        "gust_kph": 37.6,
                        "uv": 0.0
                    },"""