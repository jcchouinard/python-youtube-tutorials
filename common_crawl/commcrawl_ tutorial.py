# By: Jean-Christophe Chouinard
# Full tutorial on https://www.jcchouinard.com/python-commoncrawl-extraction/
# Linkedin: https://www.linkedin.com/in/jeanchristophechouinard/
# YouTube: https://www.youtube.com/@jcchouinard
# Twitter: https://x.com/ChouinardJC
# Website: https://www.jcchouinard.com/

import requests
import json
import os
import pandas as pd
# For parsing URLs:
from urllib.parse import quote_plus
from  bs4 import BeautifulSoup

# For parsing WARC records:
from warcio.archiveiterator import ArchiveIterator


# The URL you want to look up in the Common Crawl index
target_url = 'indeed.com/.*'  # Replace with your target URL

# list of indexes https://commoncrawl.org/get-started
indexes  = ['CC-MAIN-2024-33','CC-MAIN-2024-30','CC-MAIN-2024-26']


def search_cc_index(url, index_name):
    """
    Search the Common Crawl Index for a given URL.

    This function queries the Common Crawl Index API to find records related to the specified URL. 
    It uses the index specified by `index_name` to retrieve the data and returns a list of JSON objects, 
    each representing a record from the index.

    Arguments:
        url (str): The URL to search for in the Common Crawl Index.
        index_name (str): The name of the Common Crawl Index to search (e.g., "CC-MAIN-2024-10").

    Returns:
        list: A list of JSON objects representing records found in the Common Crawl Index. 
              Returns None if the request fails or no records are found.

    Example:
        >>> search_cc_index("example.com", "CC-MAIN-2024-10")
        [{...}, {...}, ...]
    """
    encoded_url = quote_plus(url)
    index_url = f'http://index.commoncrawl.org/{index_name}-index?url={encoded_url}&output=json'
    response = requests.get(index_url)

    if response.status_code == 200:
        records = response.text.strip().split('\n')
        return [json.loads(record) for record in records]
    else:
        return None



def fetch_single_record(warc_record_filename, offset, length):
    """
    Fetch a single WARC record from Common Crawl.

    Arguments:
        record {dict} -- A dictionary containing the WARC record details.

    Returns:
        bytes or None -- The raw content of the response if found, otherwise None.
    """
    
    s3_url = f'https://data.commoncrawl.org/{warc_record_filename}'

    # Define the byte range for the request
    byte_range = f'bytes={offset}-{offset + length - 1}'

    # Send the HTTP GET request to the S3 URL with the specified byte range
    response = requests.get(
        s3_url,
        headers={'Range': byte_range},
        stream=True
    )

    if response.status_code == 206:
        # Use `stream=True` in the call to `requests.get()` to get a raw byte stream,
        # because it's gzip compressed data
        stream = ArchiveIterator(response.raw)
        for warc_record in stream:
            if warc_record.rec_type == 'response':
                return warc_record.content_stream().read()
    else:
        print(f"Failed to fetch data: {response.status_code}")
    
    return None


def append_df_row_to_pickle(row, pickle_file):
    """
    Append a row to a DataFrame stored in a pickle file.
    
    Arguments:
        row {pd.Series} -- The row to be appended to the DataFrame.
        pickle_file {str} -- The path to the pickle file where the DataFrame is stored.
    """
    # Check if the pickle file exists
    if os.path.exists(pickle_file):
        # Load the existing DataFrame from the pickle file
        df = pd.read_pickle(pickle_file)
    else:
        # If the file doesn't exist, create a new DataFrame
        df = pd.DataFrame(columns=row.index)

    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    
    # Save the updated DataFrame back to the pickle file
    df.to_pickle(pickle_file)


def load_processed_indices(pickle_file):
    """
    Load processed indices from a pickle file to check previously processed records.

    Arguments:
        pickle_file {str} -- The path to the pickle file where the DataFrame is stored.
    
    Returns:
        Set of processed indices.
    """
    if os.path.exists(pickle_file):
        df = pd.read_pickle(pickle_file)
        # Assuming 'index' column is in the DataFrame and contains indices of processed records
        processed_indices = set(df['index'].unique())
        print(f"Loaded {len(processed_indices)} processed indices from {pickle_file}")
        return processed_indices
    else:
        print(f"No processed indices found. Pickle file '{pickle_file}' does not exist.")
        return set()


record_dfs = []

# Fetch each index and store into a datafram
for index_name in indexes:
    print('Running: ', index_name)
    records = search_cc_index(target_url,index_name)
    record_df = pd.DataFrame(records)
    record_df['index_name'] = index_name
    record_dfs.append(record_df)

# Combine individual dataframes
all_records_df = pd.concat(record_dfs)
all_records_df = all_records_df.sort_values(by='index_name', ascending=False)
all_records_df = all_records_df.reset_index()

# Create columns where to store data later
all_records_df['success_status'] = 'not processed'
all_records_df['html'] = ''

df = all_records_df[all_records_df['languages'] == 'eng']
df = df[df['url'].str.contains('/job/')]
df.head()

#all_records_df.to_csv('all_records_df.csv', index=False)
# all_records_df = pd.read_csv('all_expedia_records_df.csv')


# If pickle file exists, check for processed items
pickle_file = 'commcrawl_indeed.pkl'
processed_indices = load_processed_indices(pickle_file)
if processed_indices:
    # Remove processed items
    df = df[~df['index'].isin(processed_indices)]

# Create storage for later
successful = set()
results = {}

# Keep track of each row processed
i = 0 
perc = 0
n_records = len(df)
print(f"Found {n_records} records for {target_url}")
mod = int(n_records * 0.01)

# Reset index to help with looping
df.reset_index(drop=True,inplace=True)


for i in range(len(df)):
    # Print every 1% process
    if i % mod == 0: 
        print(f'{i} of {n_records}: {perc}%')
        perc += 1

    record_url = df.loc[i, 'url']

    # Fetch only URLs that were not processed
    # If it was already processed, skip URL 
    # (Helps speeding if you only need one version of the HTML, not its history)
    if not record_url in successful:
        length = int(df.loc[i, 'length'])
        offset = int(df.loc[i, 'offset'])
        warc_record_filename = df.loc[i, 'filename']
        result = fetch_single_record(warc_record_filename, offset, length)
        
        if not result:
            df.loc[i,'success_status'] = 'invalid warc'
        else:
            df.loc[i,'success_status'] = 'success'
            df.loc[i,'html'] = result
    else: 
        df.loc[i,'success_status'] = 'previously processed'

    # Add to pickle file
    append_df_row_to_pickle(df.loc[i, :], pickle_file)

commoncrawl_data = pd.read_pickle(pickle_file)
commoncrawl_data[
    ['url','filename','index_name','success_status','html']
    ].head()




# Select HTML from row 0
content = commoncrawl_data.loc[0, 'html']

# Parse in Beautiful soup
soup = BeautifulSoup(content, 'html.parser')
print(soup.find('title'))