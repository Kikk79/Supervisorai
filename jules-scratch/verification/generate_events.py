import requests
import json

def generate_events():
    print("Generating some test events via API...")
    base_url = "http://localhost:5000/api"

    events = [
        {
            "event_type": "test_event_1",
            "source_system": "api_test",
            "message": "This is a test message 1 from the API.",
            "data": {"value": 123}
        },
        {
            "event_type": "test_event_2",
            "source_system": "api_test",
            "message": "This is a test message 2 from the API.",
            "data": {"value": 456}
        },
        {
            "event_type": "test_event_3",
            "source_system": "api_test",
            "message": "This is a test message 3 from the API.",
            "data": {"value": 789}
        }
    ]

    for event in events:
        # The API endpoint expects 'source', not 'source_system' in the top-level dict
        api_payload = {
            "event_type": event["event_type"],
            "source": event["source_system"],
            "message": event["message"],
            "data": event["data"]
        }
        try:
            response = requests.post(
                f"{base_url}/log_event",
                headers={"Content-Type": "application/json"},
                data=json.dumps(api_payload)
            )
            response.raise_for_status()
            print(f"Sent event: {event['event_type']}, Response: {response.json()}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending event {event['event_type']}: {e}")

if __name__ == "__main__":
    generate_events()
