#notebook code for AusReddit

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from tqdm import tqdm

class ExplorationAR:
    def __init__(self) -> None:
        pass


    def get_data(self, url, filename, force = False) -> str:
        cwd = os.getcwd()
        folder = os.path.join(cwd, "data")
        filepath = os.path.join(folder, filename)

        if os.path.exists(filepath) and not force:
            return filepath
        
        response = requests.get(url, stream=True)
        
        # Sizes in bytes.
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            with open(filepath, "wb") as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
        
        if total_size != 0 and progress_bar.n != total_size:
            raise RuntimeError("Could not download file")

        return filepath

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        """
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return pd.DataFrame()

    # ngrams

    def top_ngrams_in_date_range(self, data: pd.DataFrame, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get top n n-grams in a specified time frame by summing their counts.
        Assumes data has a 'month' column (YYYY-MM-01 format), 
        an 'ngram' column (containing single or multi-word n-grams as strings),
        and a 'count' column (frequency of that ngram for the month).
        """
        try:
            data_copy = data.copy() # Work on a copy to avoid SettingWithCopyWarning
            # Ensure 'month' column is in datetime format
            data_copy['month'] = pd.to_datetime(data_copy['month'])
            
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data based on the 'month' column and the provided date range
            mask = (data_copy['month'] >= start_dt) & (data_copy['month'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['ngram', 'total_count'])

            if 'ngram' not in filtered_data.columns:
                print("Error: 'ngram' column not found.")
                return pd.DataFrame(columns=['ngram', 'total_count'])
            if 'count' not in filtered_data.columns:
                print("Error: 'count' column not found.")
                return pd.DataFrame(columns=['ngram', 'total_count'])

            # Group by 'ngram' and sum the 'count' for the filtered period
            ngram_counts = filtered_data.groupby('ngram')['count'].sum()
            
            # Get the top N n-grams
            top_n_ngrams = ngram_counts.nlargest(n).reset_index(name='total_count')
            
            return top_n_ngrams
        except KeyError as e:
            print(f"Error processing n-grams: Missing expected column {e}. Ensure 'month', 'ngram', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing n-grams: {e}")
            return pd.DataFrame()
        
    def keyword_search_in_date_range(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (as a whole word, case-insensitive) within the 'ngram' column 
        in a specified time frame.
        Assumes data has a 'month' column (YYYY-MM-01 format), 
        an 'ngram' column (containing single or multi-word n-grams as strings),
        and a 'count' column (frequency of that ngram for the month).
        Returns matching rows with 'month', 'ngram', and 'count'.
        """
        try:
            data_copy = data.copy()
            # Ensure 'month' column is in datetime format
            data_copy['month'] = pd.to_datetime(data_copy['month'])
            
            # Convert start_date and end_date strings to datetime objects for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Filter data based on the 'month' column and the provided date range
            date_mask = (data_copy['month'] >= start_dt) & (data_copy['month'] <= end_dt)
            filtered_by_date = data_copy[date_mask]

            if filtered_by_date.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['month', 'ngram', 'count'])

            if 'ngram' not in filtered_by_date.columns:
                print("Error: 'ngram' column not found.")
                return pd.DataFrame(columns=['month', 'ngram', 'count'])
            if 'count' not in filtered_by_date.columns:
                print("Error: 'count' column not found.")
                return pd.DataFrame(columns=['month', 'ngram', 'count'])

            # Search for the keyword within the 'ngram' column (whole word, case-insensitive)
            # Ensure 'ngram' is string type for .str accessor
            keyword_data = filtered_by_date[filtered_by_date['ngram'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found as a whole word in any n-gram within the specified date range.")
                return pd.DataFrame(columns=['month', 'ngram', 'count'])
            
            return keyword_data[['month', 'ngram', 'count']]
        except KeyError as e:
            print(f"Error searching for keyword: Missing expected column {e}. Ensure 'month', 'ngram', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword: {e}")
            return pd.DataFrame()

        #domains
        
    def top_domains_in_date_range(self, data: pd.DataFrame, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get top domains in a specified time frame by summing their counts.
        Assumes data has a 'day' column for date, 'domain' column for domain strings,
        and a 'count' column for frequency.
        """
        try:
            data_copy = data.copy()
            # Ensure 'day' column is in datetime format
            data_copy['day'] = pd.to_datetime(data_copy['day'])
            
            # Convert start_date and end_date strings to datetime objects for comparison
            # Ensure they are timezone-naive if 'day' becomes timezone-naive after conversion, or match timezones.
            start_dt = pd.to_datetime(start_date).tz_localize(None) # Assuming start/end_date are naive
            end_dt = pd.to_datetime(end_date).tz_localize(None)   # Assuming start/end_date are naive
            
            # Make 'day' column timezone-naive for comparison if it's timezone-aware
            if data_copy['day'].dt.tz is not None:
                data_copy['day'] = data_copy['day'].dt.tz_localize(None)

            mask = (data_copy['day'] >= start_dt) & (data_copy['day'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['domain', 'total_count'])

            if 'domain' not in filtered_data.columns:
                print("Error: 'domain' column not found.")
                return pd.DataFrame(columns=['domain', 'total_count'])
            if 'count' not in filtered_data.columns:
                print("Error: 'count' column not found.")
                return pd.DataFrame(columns=['domain', 'total_count'])

            # Extract domain: handles http(s):// and www. variations
            extracted_domains = filtered_data['domain'].astype(str).str.extract(r'(?:https?://)?(?:www\.)?([^/]+)')[0].str.lower()
            
            # Assign to a temporary column to group
            temp_df = filtered_data.assign(extracted_domain=extracted_domains)
            
            domain_counts = temp_df.groupby('extracted_domain')['count'].sum()
            
            top_n_domains = domain_counts.nlargest(n).reset_index(name='total_count').rename(columns={'extracted_domain': 'domain'})
            
            return top_n_domains
        except KeyError as e:
            print(f"Error processing domains: Missing expected column {e}. Ensure 'day', 'domain', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing domains: {e}")
            return pd.DataFrame()

    def keyword_search_in_domains(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (case-insensitive) within extracted domains in a specified time frame.
        Assumes data has a 'day' column for date, 'domain' column for domain strings,
        and a 'count' column for frequency.
        Returns matching original rows with 'day', 'domain', and 'count'.
        """
        try:
            data_copy = data.copy()
            data_copy['day'] = pd.to_datetime(data_copy['day'])

            start_dt = pd.to_datetime(start_date).tz_localize(None)
            end_dt = pd.to_datetime(end_date).tz_localize(None)

            if data_copy['day'].dt.tz is not None:
                data_copy['day'] = data_copy['day'].dt.tz_localize(None)
            
            date_mask = (data_copy['day'] >= start_dt) & (data_copy['day'] <= end_dt)
            filtered_by_date = data_copy[date_mask]

            if filtered_by_date.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['day', 'domain', 'count'])

            if 'domain' not in filtered_by_date.columns:
                print("Error: 'domain' column not found.")
                return pd.DataFrame(columns=['day', 'domain', 'count'])
            if 'count' not in filtered_by_date.columns: # Check for count column
                print("Error: 'count' column not found.")
                return pd.DataFrame(columns=['day', 'domain', 'count'])

            # Extract domain for searching
            extracted_domains = filtered_by_date['domain'].astype(str).str.extract(r'(?:https?://)?(?:www\.)?([^/]+)')[0].str.lower()
            
            # Perform case-insensitive containment search on the extracted domain
            keyword_mask = extracted_domains.str.contains(keyword.lower(), case=False, na=False)
            
            keyword_data = filtered_by_date[keyword_mask]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found in any domain within the specified date range.")
                return pd.DataFrame(columns=['day', 'domain', 'count'])
            
            return keyword_data[['day', 'domain', 'count']] # Return relevant columns
        except KeyError as e:
            print(f"Error searching for keyword in domains: Missing expected column {e}. Ensure 'day', 'domain', and 'count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword in domains: {e}")
            return pd.DataFrame()
#emotions

    def plot_emotion_trends(self, data: pd.DataFrame, start_date: str, end_date: str) -> None:
        """
        Plot emotion trends over a specified time range.
        Assumes data has a 'date' column and columns for each emotion:
        "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust".
        The date range must be longer than three days.
        """
        try:
            data_copy = data.copy()
            
            emotion_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
            required_columns = ["date"] + emotion_columns

            for col in required_columns:
                if col not in data_copy.columns:
                    print(f"Error: Missing required column '{col}'.")
                    return

            # Convert date columns and inputs to datetime objects
            data_copy['date'] = pd.to_datetime(data_copy['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt >= end_dt:
                print("Error: Start date must be before end date.")
                return

            if (end_dt - start_dt).days <= 3:
                print("Error: Date range must be longer than three days.")
                return

            # Filter data for the specified date range
            mask = (data_copy['date'] >= start_dt) & (data_copy['date'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            # Set 'date' as index for plotting
            filtered_data = filtered_data.set_index('date')

            # Plotting
            plt.figure(figsize=(15, 8))
            for emotion in emotion_columns:
                plt.plot(filtered_data.index, filtered_data[emotion], label=emotion.capitalize())
            
            plt.title(f'Emotion Trends from {start_date} to {end_date}')
            plt.xlabel('Date')
            plt.ylabel('Emotion Score / Intensity')
            plt.legend()
            plt.grid(True)
            
            # Format x-axis to show dates nicely
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            plt.show()

        except KeyError as e:
            print(f"Error plotting emotion trends: Missing expected column {e}.")
        except Exception as e:
            print(f"Error plotting emotion trends: {e}")

    def plot_emotion_highlights(self, data: pd.DataFrame, start_date: str, end_date: str) -> None:
        """
        Plot emotion trends, highlighting maximum and minimum non-zero values
        for each emotion over a specified time range.
        Assumes data has a 'date' column and columns for each emotion.
        The date range must be longer than three days.
        """
        try:
            data_copy = data.copy()
            
            emotion_columns = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]
            required_columns = ["date"] + emotion_columns

            for col in required_columns:
                if col not in data_copy.columns:
                    print(f"Error: Missing required column '{col}'.")
                    return

            data_copy['date'] = pd.to_datetime(data_copy['date'])
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            if start_dt >= end_dt:
                print("Error: Start date must be before end date.")
                return

            if (end_dt - start_dt).days <= 3:
                print("Error: Date range must be longer than three days.")
                return

            mask = (data_copy['date'] >= start_dt) & (data_copy['date'] <= end_dt)
            filtered_data = data_copy[mask].copy() # Use .copy() to avoid SettingWithCopyWarning on set_index

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return

            filtered_data.set_index('date', inplace=True)

            plt.figure(figsize=(18, 10)) # Increased figure size for clarity
            
            line_handles = [] # For custom legend

            for emotion in emotion_columns:
                # Plot the emotion line
                line, = plt.plot(filtered_data.index, filtered_data[emotion], label=emotion.capitalize(), alpha=0.7)
                line_handles.append(line)
                line_color = line.get_color()

                # Highlight maximum value
                if not filtered_data[emotion].empty:
                    max_val = filtered_data[emotion].max()
                    # Check if max_val is NaN (can happen if all values are NaN for the emotion in range)
                    if pd.notna(max_val):
                        max_date = filtered_data[emotion].idxmax()
                        plt.scatter(max_date, max_val, marker='o', color=line_color, s=80, zorder=5, edgecolors='black')

                # Highlight minimum non-zero value
                non_zero_emotion_data = filtered_data[emotion][filtered_data[emotion] > 0]
                if not non_zero_emotion_data.empty:
                    min_nz_val = non_zero_emotion_data.min()
                    min_nz_date = non_zero_emotion_data.idxmin()
                    plt.scatter(min_nz_date, min_nz_val, marker='x', color=line_color, s=80, zorder=5)  # removed edgecolors
            
            # Create custom legend handles for markers
            # These are dummy plots just for the legend
            from matplotlib.lines import Line2D
            legend_elements = line_handles + [
                Line2D([0], [0], marker='o', color='w', label='Max Value', markerfacecolor='gray', markeredgecolor='black', markersize=10),
                Line2D([0], [0], marker='x', color='w', label='Min Non-Zero Value', markerfacecolor='gray', markeredgecolor='black', markersize=10)
            ]

            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1)) # Place legend outside
            
            plt.title(f'Emotion Highlights from {start_date} to {end_date}')
            plt.xlabel('Date')
            plt.ylabel('Emotion Score / Intensity')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=12)) # Adjust tick density
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend outside
            plt.show()

        except KeyError as e:
            print(f"Error plotting emotion highlights: Missing expected column {e}.")
        except Exception as e:
            print(f"Error plotting emotion highlights: {e}")

# topics

    def top_topics_per_day_in_date_range(self, data: pd.DataFrame, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the top n topics for each date within a specified date range.
        Assumes data has 'date', 'topic_name', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'topic_name', 'doc_count'.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Filter data for the date range
            mask = (data_copy['date'] >= start_dt) & (data_copy['date'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'topic_name', 'doc_count'])

            if 'topic_name' not in filtered_data.columns:
                print("Error: 'topic_name' column not found.")
                return pd.DataFrame(columns=['date', 'topic_name', 'doc_count'])
            if 'doc_count' not in filtered_data.columns:
                print("Error: 'doc_count' column not found.")
                return pd.DataFrame(columns=['date', 'topic_name', 'doc_count'])

            top_topics = (
                filtered_data
                .sort_values(['date', 'doc_count'], ascending=[True, False])
                .groupby('date')
                .head(n)
                .reset_index(drop=True)
            )

            return top_topics[['date', 'topic_name', 'doc_count']]
        except KeyError as e:
            print(f"Error processing topics: Missing expected column {e}. Ensure 'date', 'topic_name', and 'doc_count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error processing topics: {e}")
            return pd.DataFrame()
        
    def search_keyword_in_topics_by_date(self, data: pd.DataFrame, keyword: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Search for a keyword (case-insensitive, whole word) in the 'top_words' column and
        return topic clusters (by date) where that keyword occurred.
        Assumes data has 'date', 'topic_name', 'top_words', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'topic_name', 'top_words', 'doc_count'.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Filter by date range
            mask = (data_copy['date'] >= start_dt) & (data_copy['date'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            for col in ['topic_name', 'top_words', 'doc_count']:
                if col not in filtered_data.columns:
                    print(f"Error: '{col}' column not found.")
                    return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            keyword_mask = filtered_data['top_words'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            keyword_data = filtered_data[keyword_mask]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found in any topic cluster within the specified date range.")
                return pd.DataFrame(columns=['date', 'topic_name', 'top_words', 'doc_count'])

            return keyword_data[['date', 'topic_name', 'top_words', 'doc_count']]
        except KeyError as e:
            print(f"Error searching for keyword in topics: Missing expected column {e}. Ensure 'date', 'topic_name', 'top_words', and 'doc_count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword in topics: {e}")
            return pd.DataFrame()
        
    def top_subreddits_for_keyword_by_date(self, data: pd.DataFrame, keyword: str, n: int, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Return the top n subreddit_ids for a keyword search in 'top_words', sorted by date and doc_count.
        Assumes data has 'date', 'subreddit_id', 'top_words', and 'doc_count' columns.
        Returns a DataFrame with columns: 'date', 'subreddit_id', 'top_words', 'doc_count'.
        """
        try:
            data_copy = data.copy()
            data_copy['date'] = pd.to_datetime(data_copy['date'])

            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)

            # Filter by date range
            mask = (data_copy['date'] >= start_dt) & (data_copy['date'] <= end_dt)
            filtered_data = data_copy[mask]

            if filtered_data.empty:
                print(f"No data found in the date range {start_date} to {end_date}.")
                return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            for col in ['subreddit_id', 'top_words', 'doc_count']:
                if col not in filtered_data.columns:
                    print(f"Error: '{col}' column not found.")
                    return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            # Search for keyword in 'top_words' (whole word, case-insensitive)
            keyword_mask = filtered_data['top_words'].astype(str).str.contains(f'\\b{keyword}\\b', case=False, na=False, regex=True)
            keyword_data = filtered_data[keyword_mask]

            if keyword_data.empty:
                print(f"Keyword '{keyword}' not found in any subreddit within the specified date range.")
                return pd.DataFrame(columns=['date', 'subreddit_id', 'top_words', 'doc_count'])

            # For each date, get top n subreddit_ids by doc_count
            top_subreddits = (
                keyword_data
                .sort_values(['date', 'doc_count'], ascending=[True, False])
                .groupby('date')
                .head(n)
                .reset_index(drop=True)
            )

            return top_subreddits[['date', 'subreddit_id', 'top_words', 'doc_count']]
        except KeyError as e:
            print(f"Error searching for keyword in subreddits: Missing expected column {e}. Ensure 'date', 'subreddit_id', 'top_words', and 'doc_count' columns exist.")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error searching for keyword in subreddits: {e}")
            return pd.DataFrame()


