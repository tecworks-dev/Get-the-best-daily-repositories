from geopy.geocoders import Nominatim
from requests import get 
from src.FUNCTION.get_env import load_variable

def get_lat_lng(name: str) -> tuple:
    geolocator = Nominatim(user_agent="your_app_name")
    location = geolocator.geocode(name)
    if location:
        latitude = round(location.latitude, 3)
        longitude = round(location.longitude, 3)
        return latitude, longitude
    return 0, 0



def weather_report(city:str) -> dict:
    """Get the current weather for a location."""
    report = {}
    api_key = load_variable("Weather_api")
    lati , long = get_lat_lng(city)
    
    url = "https://weatherapi-com.p.rapidapi.com/current.json"

    #querystring = {"q":"53.1,-0.13"}
    querystring = {"q":f"{lati} ,{long}"}
    headers = {
        "x-rapidapi-key": api_key ,
        "x-rapidapi-host": "weatherapi-com.p.rapidapi.com"
    }

    response = get(url, headers=headers, params=querystring)
    
    all_data = response.json()
    #print(all_data)
    report["datetime"] = all_data["current"]["last_updated"]
    report["temp"] = all_data["current"]["temp_c"]
    report["condition"] = all_data["current"]["condition"]["text"]
    report["wind"] = all_data["current"]["wind_kph"]
    report["humidity"] = all_data["current"]["humidity"]
    report["cloud"] = all_data["current"]["cloud"]
    report["feels_like"] = all_data["current"]["feelslike_c"]
    report["uv"] = all_data["current"]["uv"]
    return report



