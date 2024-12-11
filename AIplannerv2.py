import pandas as pd
import numpy as np
import pickle
import math
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify

# Load the dataset
data = pd.read_csv('Combined_10000_Mixed_Attractions_Responses.csv')

# Standardize column names by replacing spaces and dashes with underscores and converting to lowercase
data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.lower()

# Ensure the required columns are present
required_columns = [
    'places_visited_in_the_uae',
    'types_of_attractions',
    'city_you_resided_in',
    'season',
    'age',
    'budget_per_person_daily',
    'travel_group_composition'
]

missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

# Explode 'places_visited_in_the_uae' column for multiple places per respondent
data['places_visited_in_the_uae'] = data['places_visited_in_the_uae'].str.split(',')
data = data.explode('places_visited_in_the_uae')

# Handle 'types_of_attractions'
data['types_of_attractions'] = data['types_of_attractions'].str.split(',')
attraction_types_expanded = data['types_of_attractions'].apply(
    lambda x: pd.Series(1, index=[f'types_of_attractions_{i.strip().lower()}' for i in x])
)
data = pd.concat([data, attraction_types_expanded], axis=1)
data.fillna(0, inplace=True)

# One-hot encode 'city_you_resided_in' and 'season'
data = pd.get_dummies(data, columns=['city_you_resided_in', 'season'])

# Convert numerical columns to numeric
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data['budget_per_person_daily'] = pd.to_numeric(data['budget_per_person_daily'], errors='coerce')
data['travel_group_composition'] = pd.to_numeric(data['travel_group_composition'], errors='coerce')

# Aggregate place-based information
numeric_columns = data.select_dtypes(include=['number']).columns
place_features = data.groupby('places_visited_in_the_uae')[numeric_columns].mean().reset_index()

# Load the places coordinates
places_coords = pd.read_csv('places_coordinates.csv')
places_coords['Place'] = places_coords['Place'].str.strip().str.lower()

# Standardize place names in place_features
place_features['places_visited_in_the_uae'] = place_features['places_visited_in_the_uae'].str.strip().str.lower()

# Merge place_features with places_coords
place_features = pd.merge(
    place_features,
    places_coords,
    left_on='places_visited_in_the_uae',
    right_on='Place',
    how='left'
)
place_features.drop(columns=['Place'], inplace=True)

# Ensure Latitude and Longitude are numeric
place_features['Latitude'] = pd.to_numeric(place_features['Latitude'], errors='coerce')
place_features['Longitude'] = pd.to_numeric(place_features['Longitude'], errors='coerce')

# Drop rows with missing coordinates
place_features.dropna(subset=['Latitude', 'Longitude'], inplace=True)

# Save place features
with open('place_features.pkl', 'wb') as f:
    pickle.dump(place_features, f)

# Define function to create user profile
def create_user_profile(age, budget, group_size, city, attractions, season):
    user_profile_data = {
        'age': [age],
        'budget_per_person_daily': [budget],
        'travel_group_composition': [group_size]
    }

    # One-hot encode the city and season
    for col in place_features.columns:
        if col.startswith('city_you_resided_in_'):
            user_profile_data[col] = [1 if col == f'city_you_resided_in_{city.lower()}' else 0]
        elif col.startswith('season_'):
            user_profile_data[col] = [1 if col == f'season_{season.lower()}' else 0]

    # Define columns to drop
    columns_to_drop = ['places_visited_in_the_uae', 'Latitude', 'Longitude']
    attraction_cols = [col for col in place_features.columns if col.startswith('types_of_attractions_')]
    columns_to_drop.extend(attraction_cols)

    # Define columns to use
    columns_to_use = [col for col in place_features.columns if col not in columns_to_drop]

    # Create DataFrame and reindex
    user_profile_df = pd.DataFrame(user_profile_data)
    user_profile_df = user_profile_df.reindex(columns=columns_to_use, fill_value=0)

    return user_profile_df.values

# Haversine distance function
def haversine_distance(lat1, lon1, lat2, lon2):
    # Earth radius in kilometers
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# Transportation cost and carbon footprint function
def calculate_transportation_cost_and_carbon(distance_km, mode):
    if mode == 'bus':
        cost_per_km = 0.5
        emission_factor = 0.105
    else:
        cost_per_km = 2.0
        emission_factor = 0.192
    return (2 + cost_per_km * distance_km, distance_km * emission_factor)

# Define function to recommend an itinerary
def recommend_itinerary(user_profile, num_days, budget_per_day, places_per_day=2, excluded_places=None, user_location=None, transportation_mode='bus'):
    if excluded_places is None:
        excluded_places = []
    if user_location is None:
        raise ValueError("User location must be provided.")

    columns_to_drop = ['places_visited_in_the_uae', 'Latitude', 'Longitude'] + \
                      [col for col in place_features.columns if col.startswith('types_of_attractions_')]
    columns_to_use = [col for col in place_features.columns if col not in columns_to_drop]

    similarities = cosine_similarity(user_profile, place_features[columns_to_use].values)[0]
    place_features['similarity'] = similarities

    places_within_budget = place_features[~place_features['places_visited_in_the_uae'].isin(excluded_places)]
    places_within_budget['distance_km'] = places_within_budget.apply(
        lambda row: haversine_distance(user_location[0], user_location[1], row['Latitude'], row['Longitude']), axis=1)
    places_within_budget.dropna(subset=['distance_km'], inplace=True)
    places_within_budget[['transportation_cost', 'carbon_footprint']] = places_within_budget['distance_km'].apply(
        lambda x: pd.Series(calculate_transportation_cost_and_carbon(x, transportation_mode)))
    places_within_budget['total_estimated_cost'] = places_within_budget['budget_per_person_daily'] + places_within_budget['transportation_cost']
    places_within_budget = places_within_budget.sort_values(by='similarity', ascending=False)

    total_places_needed = num_days * places_per_day
    recommended_places = places_within_budget[places_within_budget['total_estimated_cost'] <= budget_per_day].head(total_places_needed)

    if len(recommended_places) < total_places_needed:
        additional_places_needed = total_places_needed - len(recommended_places)
        remaining_places = places_within_budget[~places_within_budget.index.isin(recommended_places.index)]
        additional_places = remaining_places.head(additional_places_needed)
        recommended_places = pd.concat([recommended_places, additional_places])

    itinerary = {}
    for day in range(1, num_days + 1):
        start_idx = (day - 1) * places_per_day
        end_idx = start_idx + places_per_day
        day_places = recommended_places.iloc[start_idx:end_idx]

        day_place_names = []
        day_estimated_costs = []
        day_carbon_footprints = []

        current_location = user_location
        for idx, place in day_places.iterrows():
            distance = haversine_distance(current_location[0], current_location[1], place['Latitude'], place['Longitude'])
            transportation_cost, carbon_footprint = calculate_transportation_cost_and_carbon(distance, transportation_mode)
            total_cost = place['budget_per_person_daily'] + transportation_cost

            day_place_names.append(place['places_visited_in_the_uae'])
            day_estimated_costs.append(total_cost)
            day_carbon_footprints.append(carbon_footprint)

            current_location = (place['Latitude'], place['Longitude'])

        itinerary[f'Day {day}'] = {
            'Places': day_place_names,
            'Estimated_Costs': day_estimated_costs,
            'Carbon_Footprints': day_carbon_footprints
        }

    return itinerary

# Flask App
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the AI Travel Planner API!"

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_data = request.json

        user_latitude = user_data.get('user_latitude')
        user_longitude = user_data.get('user_longitude')
        transportation_mode = user_data.get('transportation_mode', 'bus')

        if user_latitude is None or user_longitude is None:
            return jsonify({"error": "User's current latitude and longitude must be provided."}), 400

        user_location = (float(user_latitude), float(user_longitude))

        user_profile = create_user_profile(
            age=user_data['age'],
            budget=user_data['budget'],
            group_size=user_data['group_size'],
            city=user_data['city'],
            attractions=user_data['attractions'],
            season=user_data['season']
        )

        num_days = user_data.get('num_days', 1)
        budget_per_day = user_data.get('budget_per_day', user_data['budget'])
        places_per_day = user_data.get('places_per_day', 2)
        excluded_places = user_data.get('excluded_places', [])

        itinerary = recommend_itinerary(
            user_profile,
            num_days,
            budget_per_day,
            places_per_day,
            excluded_places,
            user_location,
            transportation_mode
        )

        response = {
            "itinerary": itinerary
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
