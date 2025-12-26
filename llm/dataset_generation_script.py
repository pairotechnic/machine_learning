'''
    Write a script that will programatically generate entries in a training dataset
    Intended for SQL generation Fine Tuning of Smollm:360m
    For simplicity, I will train it on only a few specific cases
    If it is able to respond accurately using even the user queries in its dataset,
    Then the Fine Tuning has worked
'''

import json
from datetime import datetime, timedelta
import random

columns = ["id", "city", "temperature", "humidity", "recorded_at", "wind_speed", "cloudiness", "feels_like", "measured_at", "pressure"]

# What and How questions
weather_metrics = ["weather", "temperature", "humidity", "wind speed", "cloudiness", "feels like", "pressure"]

# Where and when questions 
metadata_columns = ["city", "measured_at"]
# columns_to_ignore = ["id", "recorded_at"]

cities = ["Barcelona", "London", "Paris", "Hong Kong", "New York", 
          "Istanbul", "Singapore", "Bangkok", "Tokyo", "Dubai"]

time_periods = {
    "today",
    "last 24 hours",
    "yesterday",
    "last 3 days", 
    "this week",
    "last 7 days"
    "last week", 
    "this month", 
    "last 30 days"
    "last month",
}

aggregates = {
    "" : "",
    "highest" : "MAX", 
    "maximum" : "MAX", 
    "average" : "AVG", 
    "minimum" : "MIN",
    "lowest" : "MIN"
}

comparisons = {
    "above", 
    "greater than", 
    "less than"
    "below", 
}

def parse_user_query(user_query):
    pass

def generate_training_examples():
    examples = []
    
    system_prompt = "You are a helpful SQL assistant that generates PostgreSQL queries for weather data. The table 'weather_weatherdata' has columns: id, city, temperature, humidity, recorded_at, wind_speed, cloudiness, feels_like, measured_at, pressure. Available cities: Barcelona, London, Paris, Hong Kong, New York, Istanbul, Singapore, Bangkok, Tokyo, Dubai."

    user_query = "Show me the weather" # Find the latest weather entries for each city, average the weather columns out, and return a response
    sql = """ 
        SELECT
            AVG(temperature) as "Temperature",
            AVG(humidity) as "Humidity",
            AVG(wind_speed) as "Wind Speed",
            AVG(cloudiness) as "Cloudiness",
            AVG(feels_like) as "Feels like",
            AVG(pressure) as "Pressure"
        FROM (
            SELECT DISTINCT ON (city) * 
            FROM weather_weatherdata 
            ORDER BY city, measured_at DESC 
        );
    """

    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": sql}
        ]
    }
    
    examples.append(example)
    
    for city in cities:
        for metric in weather_metrics :
            user_query_aliases = [
                f"Show me the {metric} in {city} ",
                f"What's the {metric} in {city}",
                f"How's the {metric} in {city}"
            ]
            retrieve_val = metric.replace(" ", "_") if metric != "weather" else "*"

            sql = f"SELECT {retrieve_val} FROM weather_weatherdata WHERE city = '{city}' ORDER BY measured_at DESC LIMIT 1 ;"

            for user_query in user_query_aliases:

                example = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_query},
                        {"role": "assistant", "content": sql}
                    ]
                }
                
                examples.append(example)

            
    
    return examples

def main():
    # Generate and save dataset
    examples = generate_training_examples()
    with open('llm/weather_sql_dataset.jsonl', 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Generated {len(examples)} training examples in weather_sql_dataset.jsonl")
    pass

if __name__ == "__main__":
    main()

sample_training_examples = [
    "Show me the weather",
    "Show me the weather in London", # Repeat for all cities
    "Show me the weather in London right now", # Repeat 
    "What's the temperature in London right now",
    "What's the highest temperature of today",
    "What's the lowest temperature of today",
    "What's the highest temperature in London today",
    "What's the lowest temperature in London today",

]

user_query_format = "Show me/ What's the" + "Weather/ temperature/ ..." + "in London/ New York" + ""

# if aggregate and retrieve_val != "*":
#     retrieve_val = f"{aggregate}({retrieve_val})"