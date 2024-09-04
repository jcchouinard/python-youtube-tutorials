import gc  
import json
import os
import time
import warnings
from urllib.parse import quote_plus

import pandas as pd
import psutil  # New import for memory usage monitoring
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
from warcio.archiveiterator import ArchiveIterator

warnings.simplefilter(action='ignore', category=FutureWarning)


# The URL you want to look up in the Common Crawl index
target_url = 'indeed.com/.*'  # Replace with your target URL

# List of indexes
with open('commoncrawl_index_names_2024-08-29.txt', 'r') as f:
    indexes = f.read().strip().split('\n')

indexes = indexes[:2]
def check_memory_and_pause(threshold=80, resume_threshold=60, max_wait_time=500):
    """
    Check the current memory usage and pause execution if it exceeds the defined threshold.
    Resume execution only when memory usage drops below the resume threshold.
    """
    memory = psutil.virtual_memory()
    
    # If memory usage is above the threshold, start pausing
    if memory.percent > threshold:
        print(f"Memory usage is at {memory.percent}%. Pausing execution to free up memory...")
        
        # Set up a counter for maximum wait time
        total_wait_time = 0
        
        # Loop until memory usage drops below the resume threshold or max wait time is reached
        while memory.percent > resume_threshold and total_wait_time <= max_wait_time:
            time.sleep(1)  # Wait for 1 second
            total_wait_time += 1
            memory = psutil.virtual_memory()  # Refresh memory usage stats

        # Check if maximum wait time was reached
        if total_wait_time > max_wait_time:
            print(f"Memory usage still high after waiting {max_wait_time} seconds.")
        else:
            print(f"Memory usage dropped to {memory.percent}%. Resuming execution...")
    else:
        pass
        #print(f"Memory usage is at {memory.percent}%. Continuing execution.")


def cc_records_to_pkl(df, pickle_file):
    """
    Processes records from a DataFrame and saves the processed records to a pickle file.

    Args:
        df (pd.DataFrame): The DataFrame containing records to process.
        pickle_file (str): The path to the pickle file where processed records will be stored.

    Returns:
        str: Returns 'Done' after processing all records.
    """
    processed_indices = load_processed_indices(pickle_file)
    if processed_indices:
        # Remove processed items
        df = df[~df['index'].isin(processed_indices)]

    # Create storage for later
    successful = set()

    # Keep track of each row processed
    i = 0 
    perc = 0
    n_records = len(df)
    mod = int(n_records * 0.01)

    # Reset index to help with looping
    df.reset_index(drop=True,inplace=True)
    
    for i in range(len(df)):
        # Check memory usage and pause if necessary
        # check_memory_and_pause()

        # Print progress every 1% of records processed
        if i % mod == 0: 
            perc += 1

        record_url = df.loc[i, 'url']

        # Fetch only URLs that were not processed
        if not record_url in successful:
            length = int(df.loc[i, 'length'])
            offset = int(df.loc[i, 'offset'])
            warc_record_filename = df.loc[i, 'filename']
            result = fetch_single_record(warc_record_filename, offset, length)
            
            if not result:
                df.loc[i, 'success_status'] = 'invalid warc'
            else:
                df.loc[i, 'success_status'] = 'success'
                title, body, description, meta_robots, canonical, head = get_html_tags(result)
                df.loc[i, 'title'] = str(title)
                df.loc[i, 'body'] = str(body)
                df.loc[i, 'description'] = str(description)
                df.loc[i, 'meta_robots'] = str(meta_robots)
                df.loc[i, 'canonical'] = str(canonical)
                df.loc[i, 'head'] = str(head)
        else: 
            df.loc[i, 'success_status'] = 'previously processed'

        # Add to pickle file
        append_df_row_to_pickle(df.loc[i, :], pickle_file)
    
    del df
    gc.collect()
    return 'Done'


def threaded_cc_records_to_pkl(df, max_workers=10):
    """
    Processes records from a DataFrame using a thread pool executor and saves the processed records to separate pickle files.

    Args:
        df (pd.DataFrame): The DataFrame containing records to process.
        max_workers (int): The maximum number of worker threads to use. Default is 10.

    Returns:
        None
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i in range(max_workers):
            pickle_file = f'data/commcrawl_th{i}.pkl'
            thread_df = df[df.index % max_workers == i]
            future = executor.submit(cc_records_to_pkl, thread_df, pickle_file)
            futures.append(future)

        # Wait for all threads to complete
        for future in futures:
            future.result()
    del df
    gc.collect()


def get_html_tags(html):
    """
    Extracts HTML tags such as title, body, description, meta robots, canonical link, and head from a given HTML string.

    Args:
        html (str): The HTML content to parse.

    Returns:
        tuple: A tuple containing extracted HTML tags (title, body, description, meta_robots, canonical, head).
    """
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup.find_all('script'):
        script.decompose()
    title = soup.find('title')
    body = soup.find('body')
    title = title.text if title else 'not_defined'
    body = body if body else 'not_defined'
    head = soup.find('head')
    head = head if head else 'not_defined'
    description = soup.find('meta', attrs={'name':'description'})
    meta_robots =  soup.find('meta', attrs={'name':'robots'})
    canonical = soup.find('link', {'rel': 'canonical'})
    description = description['content'] if description else 'not_defined'
    meta_robots =  meta_robots['content'] if meta_robots else 'not_defined'
    canonical = canonical['href'] if canonical else 'not_defined'
    return title, body, description, meta_robots, canonical, head

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



def append_df_row_to_pickle(row, pickle_file):
    """
    Append a row to a DataFrame stored in a pickle file using a checkpoint mechanism to prevent overwriting if it breaks.
    
    Arguments:
        row {pd.Series} -- The row to be appended to the DataFrame.
        pickle_file {str} -- The path to the pickle file where the DataFrame is stored.
    """
    checkpoint_file = pickle_file + '.checkpoint'  # Temporary checkpoint file

    # Load the existing DataFrame from the pickle file or create a new DataFrame if the file does not exist
    if os.path.exists(pickle_file):
        try:
            df = pd.read_pickle(pickle_file)
        except Exception as e:
            print(f"Error loading {pickle_file}: {e}")
            return  # Exit if there is an issue loading the original file
    else:
        df = pd.DataFrame(columns=row.index)

    # Append the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    # Save the updated DataFrame to a checkpoint file first
    try:
        df.to_pickle(checkpoint_file)
        # print(f"Checkpoint saved to {checkpoint_file}")

        # If checkpoint is saved successfully, rename the checkpoint to the original pickle file
        os.replace(checkpoint_file, pickle_file)
        # print(f"Data appended to {pickle_file} successfully.")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # If there is an error, leave the checkpoint file intact and do not overwrite the original file


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
        # print(f"Loaded {len(processed_indices)} processed indices from {pickle_file}")
        return processed_indices
    else:
        # print(f"No processed indices found. Pickle file '{pickle_file}' does not exist.")
        return set()

def loop_records(target_url, indexes):
    """
    Fetches records from multiple Common Crawl indexes for a given URL and combines them into a single DataFrame.

    Args:
        target_url (str): The URL to search for in the Common Crawl indexes.
        indexes (list): A list of index names to search.

    Returns:
        pd.DataFrame: A DataFrame containing all the fetched records from the specified indexes, sorted by index name.
    """
    record_dfs = []

    # Fetch each index and store into a dataframe
    for index_name in indexes:
        print('Running: ', index_name)
        records = search_cc_index(target_url, index_name)
        record_df = pd.DataFrame(records)
        record_df['index_name'] = index_name
        record_dfs.append(record_df)

    # Combine individual dataframes
    all_records_df = pd.concat(record_dfs)
    all_records_df = all_records_df.sort_values(by='index_name', ascending=False)
    all_records_df = all_records_df.reset_index()
    
    return all_records_df

current_dir = os.getcwd()
storage_dir = current_dir + '/data'
if not os.path.isdir(storage_dir):
    os.makedirs(storage_dir)

all_records_df = loop_records(target_url, indexes)
# Create columns where to store data later
all_records_df['success_status'] = 'not processed'
all_records_df['title'] = 'not_defined'
all_records_df['body'] = 'not_defined'
all_records_df['description'] = 'not_defined'
all_records_df['meta_robots'] = 'not_defined'
all_records_df['canonical'] = 'not_defined'
all_records_df['head'] = 'not_defined'


df = all_records_df[all_records_df['languages'] == 'eng']
df = df[df['url'].str.contains('/job/')]
df.head()


threaded_cc_records_to_pkl(df, max_workers=10)
