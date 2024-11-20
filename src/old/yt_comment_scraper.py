import os
import re
import pandas as pd
from googleapiclient.discovery import build
from dotenv import load_dotenv

class YouTubeCommentsScraper:
    def __init__(self, api_key, channel_name, last_video_id=None):
        """
        Initialize scraper w API key/channel name
        """
        self.api_key = api_key
        self.channel_name = channel_name
        self.last_video_id = last_video_id
        self.youtube = build('youtube', 'v3', developerKey=api_key) # build youtube api connection

    def filter_videos(self, input_df):
        """
        Filters the input DataFrame for videos within a specific time period.
        If last_video_id is None, return videos from all of 2024.
        Otherwise, return videos between the start of 2024 and the date associated with last_video_id.
        """
        input_df['Date Posted'] = pd.to_datetime(input_df['Date Posted'])  # Ensure datetime type

        if self.last_video_id is None:
            # Filter for all videos from 2024
            return input_df[input_df['Date Posted'].dt.year == 2024]
        else:
            # Find the date of the last_video_id
            last_video_date = input_df.loc[
                input_df['Link'].str.contains(self.last_video_id, na=False, regex=False),
                'Date Posted'
            ].max()

            # Filter for videos between the start of 2024 and the last video date
            if pd.notna(last_video_date):
                return input_df[
                    (input_df['Date Posted'] >= "2024-01-01") & 
                    (input_df['Date Posted'] <= last_video_date)
                ]
            else:
                print(f"Warning: last_video_id {self.last_video_id} not found in the data.")
                return input_df[input_df['Date Posted'].dt.year == 2024]


    def get_video_comments(self, video_url):
        """
        get all the comments from a video
        """
        video_id = re.search(r"v=([a-zA-Z0-9_-]{11})", video_url)
        if not video_id:
            raise ValueError("Invalid YouTube video URL.")
        video_id = video_id.group(1)

        comments_data = []

        try:
            # comment query
            comment_response = self.youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
            ).execute()

            # while there is a response fromt he query, process the comments
            while comment_response:
                for item in comment_response['items']:
                    comment_snippet = item['snippet']['topLevelComment']['snippet']
                    comments_data.append({
                        "video_id": video_id,
                        "comment": comment_snippet['textDisplay'],
                        "username": comment_snippet['authorDisplayName'],
                        "comment_date": comment_snippet['publishedAt'][:10],  # extract 'YYYY-MM-DD'
                    })

                # check for the next page of comments
                if 'nextPageToken' in comment_response:
                    comment_response = self.youtube.commentThreads().list(
                        part='snippet,replies',
                        videoId=video_id,
                        pageToken=comment_response['nextPageToken'],
                        maxResults=100
                    ).execute()
                else:
                    break
            return pd.DataFrame(comments_data)
        except Exception as e:
            error_content = e.content.decode('utf-8')  # Decode the error response
            if 'quotaExceeded' in error_content:
                print("Quota limit exceeded.")
                print("Last video_id: ", video_id)
                return pd.DataFrame([])
            else:
                print(f"Some other error occurred: {error_content}")
            return pd.DataFrame([])

    def process_videos(self, input_df):
        """
        Take the YouTube video links and filter them before calling get_video_comments to scrape them.
        """
        all_comments = []
        filtered_df = self.filter_videos(input_df)
        links = list(filtered_df['Link'])

        for count, link in enumerate(filtered_df['Link']):
            print(f'Link #{count} | Url: {link}')
            comments_df = self.get_video_comments(video_url=link)
            if len(comments_df) != 0:
                all_comments.append(comments_df)
        if not all_comments:
            return pd.DataFrame([])
        return pd.concat(all_comments, ignore_index=True)


def main():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        print("Error: API key not found. Please add your API key to the .env file.")
        exit(1)

    channel_name = 'CNN' # Replace with video of interest
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_data_path = os.path.join(script_dir, f"../data/Raw Data/{channel_name}_links.csv")
    output_file = os.path.join(script_dir, f"../data/Raw Data/{channel_name}_comments.csv")

    try:
        input_df = pd.read_csv(input_data_path)
        print(f"Loaded {len(input_df)} video links from {input_data_path}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_data_path}")
        exit(1)

    if os.path.exists(output_file):
        existing_data = pd.read_csv(output_file)
        print(f"Loaded existing data with {len(existing_data)} comments.")
        last_video_id = existing_data.iloc[-1, 0]
        print(f"Last video (partially?) scrapped: {last_video_id}")
    else:
        existing_data = pd.DataFrame()

    scraper = YouTubeCommentsScraper(api_key, channel_name, last_video_id=last_video_id)

    new_comments = scraper.process_videos(input_df)
    if len(new_comments) == 0: 
        print("No new comments added ... returning")
        return
    else:
        final_data = pd.concat([existing_data, new_comments], ignore_index=True)
        final_data.to_csv(output_file, index=False)
        print(f"{len(new_comments)} Comments saved to {output_file}")

if __name__ == "__main__":
    main()