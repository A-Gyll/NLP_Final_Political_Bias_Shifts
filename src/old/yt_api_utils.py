import requests
import ssl
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from dotenv import load_dotenv
import os 
from google.oauth2 import service_account

def check_API_key(api_key):
    test_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "id": "Ks-_Mh1QhMc",  # A sample video ID
        "part": "snippet",
        "key": api_key,
    }
    response = requests.get(test_url, params=params)
    if response.status_code == 200:
        print("API key is valid.\n")
        print(response.json())
    else:
        print(f"API key test failed with status code: {response.status_code}")
        print(response.json())

def check_yt_API_client(api_key):
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        print("YouTube API client initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing YouTube API client: {e}")

def check_youtube_api_quota(service_account_file):
    """
    Checks the YouTube API quota usage for the current project.

    Parameters:
    - service_account_file (str): Path to the service account JSON file.

    Returns:
    - dict: A dictionary containing quota usage information.
    """
    try:
        # Authenticate using the service account
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

        # Build the Service Usage API client
        service = build('serviceusage', 'v1', credentials=credentials)

        # Replace 'your_project_id' with your actual project ID
        project_id = "youtubedata-440303"
        parent = f"projects/{project_id}"

        # Get quota metrics
        request = service.services().get(name=f"{parent}/services/youtube.googleapis.com")
        response = request.execute()

        # Process and return quota information
        quota_metrics = response.get("config", {}).get("quota", {}).get("metricRules", {})
        if quota_metrics:
            return quota_metrics
        else:
            print("No quota metrics found.")
            return {}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def check_ssl():
    try:
        ssl.get_default_verify_paths()
        print("SSL certificates are correctly configured.\n")
    except Exception as e:
        print(f"SSL configuration issue: {e}")

if __name__ == "__main__": 
    # Load environment variables from .env file
    load_dotenv()  

    # Load API key from environment variable
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API key not found. Please add your API key to the .env file.")
        exit(1)

    check_API_key(api_key)
    check_yt_API_client(api_key)
    # service_account_file_path = "path_to_your_service_account.json"
    # quota_info = check_youtube_api_quota(service_account_file_path)
    # print(quota_info)    
    check_ssl()