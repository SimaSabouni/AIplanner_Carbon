import unittest
import requests

class TestTravelPlannerAPI(unittest.TestCase):
    def setUp(self):
        # Ensure the API URL matches the one used by Flask
        self.base_url = "http://127.0.0.1:8080/recommend"

    def test_itinerary_one_day(self):
        # Test data for one day itinerary
        payload = {
            "age": 25,
            "budget": 300,
            "group_size": 2,
            "city": "abu dhabi",
            "attractions": {"shopping": 2, "outdoor": 3, "historical": 1, "landmark": 4, "kids": 0},
            "season": "winter",
            "transportation_mode": "bus",
            "user_latitude": 24.5,
            "user_longitude": 55.7,
            "num_days": 1
        }
        response = requests.post(self.base_url, json=payload)
        print("\nResponse for test_itinerary_one_day:")
        print(response.json())  # Print the JSON response
        self.assertEqual(response.status_code, 200)

    def test_itinerary_multiple_days(self):
        # Test data for multiple days itinerary
        payload = {
            "age": 35,
            "budget": 1500,
            "group_size": 4,
            "city": "dubai",
            "attractions": {"shopping": 5, "outdoor": 1, "historical": 3, "landmark": 5, "kids": 5},
            "season": "summer",
            "transportation_mode": "taxi",
            "user_latitude": 25.2,
            "user_longitude": 55.3,
            "num_days": 3
        }
        response = requests.post(self.base_url, json=payload)
        print("\nResponse for test_itinerary_multiple_days:")
        print(response.json())  # Print the JSON response
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
