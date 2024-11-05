import os
import csv
import datetime
import googleapiclient.discovery
import googleapiclient.errors

"""~
Scrapper to get youtube links of interest to the be fed into the yt_comment_scrapper

API_KEY: (see group drive for my key since this)
Channel ID: UCXIJgqnII2ZOINSWNOGFThA
Playlist ID: UUXIJgqnII2ZOINSWNOGFThA
"""

def main():
    api_service_name = "youtube"
    api_version = "v3"
    api_key = "AIzaSyAZzsTLiy-IQcFoZkqG2SssnQzs_qPCMWY"

    youtube = googleapiclient.discovery.build(
        api_service_name, api_version, developerKey=api_key)

    # Parameters for search
    channel_id = "UCXIJgqnII2ZOINSWNOGFThA"
    year = 2016

    # Define start and end date for the specified year in RFC 3339 format
    published_after = f"{year}-01-01T00:00:00Z"
    published_before = f"{year + 1}-01-01T00:00:00Z"  

     # Define politically related keywords
    keywords = ["politics", "election", "government", "policy", "congress", "president"]
    query_string = "|".join(keywords)  # Create OR-separated keyword string

    output_file = "../data/youtube_channel_political_videos.csv"

    # CSV file setup
    with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Title", "Keyword Match", "Description", "Published Date", "Video ID", "Video URL", "Comment Count"])

        # YouTube API request for searching videos with keywords
        request = youtube.search().list(
            part="snippet",
            channelId=channel_id,
            type="video",
            publishedAfter=published_after,
            publishedBefore=published_before,
            maxResults=50,
            order="date",
            q=query_string  # Add the query parameter for keyword search
        )

        # Pagination handling
        while request:
            response = request.execute()

            for item in response.get("items", []):
                video_title = item["snippet"]["title"]
                video_description = item["snippet"]["description"]
                published_at = item["snippet"]["publishedAt"]
                video_id = item["id"]["videoId"]
                video_url = f"https://www.youtube.com/watch?v={video_id}"

                # Detect which keyword matched
                keyword_match = next((kw for kw in keywords if kw.lower() in video_title.lower() or kw.lower() in video_description.lower()), "N/A")

                # Fetch comment count for each video using videos().list
                video_request = youtube.videos().list(
                    part="statistics",
                    id=video_id
                )
                video_response = video_request.execute()

                # Extract comment count (if available) and ensure it’s greater than 300
                comment_count = int(video_response["items"][0]["statistics"].get("commentCount", 0))
                if comment_count > 300:
                    # Write to CSV only if comment count is above 300
                    csvwriter.writerow([video_title, keyword_match, video_description, published_at, video_id, video_url, comment_count])

            # Get the next page of results, if there are more
            request = youtube.search().list_next(request, response)

    print(f"Data saved to {output_file}")

if __name__ == "__main__":
    main()
