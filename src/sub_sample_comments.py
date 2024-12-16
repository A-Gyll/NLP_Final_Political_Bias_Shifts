import pandas as pd
import random
import argparse
import pandas as pd
import numpy as np
import yaml
import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

def filter_and_sample_comments(file_path, timestamps, time_range, number_of_comments, output_file):
    try:
        if os.path.exists(output_file):
            os.remove(output_file)

        df = pd.read_csv(file_path)
        timestamp_column = 'comment_date' 

        if timestamp_column not in df.columns:
            raise ValueError("The CSV file must have a 'comment_date' column.")

        # Parse comment dates (with trailing 'Z' in the format)
        df['comment_date'] = pd.to_datetime(df['comment_date'], format="%Y-%m-%dT%H:%M:%SZ", utc=True)

        if df['comment_date'].isna().any():
            print("Warning: Some dates could not be parsed.")

        all_filtered_comments = pd.DataFrame()

        # Iterate through each timestamp, filter, and accumulate comments
        for ts in timestamps:
            ref_time = pd.to_datetime(ts, format="%Y-%m-%dT%H:%M:%SZ", utc=True)
            start_time = ref_time - relativedelta(months=time_range)

            filtered_df = df[
                (df['comment_date'] >= start_time) &
                (df['comment_date'] < ref_time)
            ]

            print(f"Filtered {len(filtered_df)} comments for interval ending at {ref_time}.")
            all_filtered_comments = pd.concat([all_filtered_comments, filtered_df], ignore_index=True)

        # Deduplicate comments in case of overlapping intervals
        before_dedup = len(all_filtered_comments)
        all_filtered_comments.drop_duplicates(inplace=True)
        after_dedup = len(all_filtered_comments)
        print(f"Removed {before_dedup - after_dedup} duplicate comments due to overlapping intervals.")

        # Randomly sample comments if we exceed the number_of_comments threshold
        if len(all_filtered_comments) > number_of_comments:
            sampled_df = all_filtered_comments.sample(n=number_of_comments, replace=False)
        else:
            sampled_df = all_filtered_comments

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        sampled_df.to_csv(output_file, index=False)
        print(f"Filtered and sampled data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter and sample comments based on time ranges.")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML configuration file.")

    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = yaml.safe_load(config_file)

    source = config["source"]
    test = config["test"]
    file_path = config["file_path"]
    output_dir = config["output_dir"]
    number_of_comments = config["number_of_comments"]
    timestamps = config["timestamps"]
    time_range = config["time_range"]

    output_file = os.path.join(output_dir, f"{source}_sampled", f"{test}.csv")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    filter_and_sample_comments(
        file_path=file_path,
        timestamps=timestamps,
        time_range=time_range,
        number_of_comments=number_of_comments,
        output_file=output_file
    )