"""Functions for processing (big) datasets. Will be called from the notebook"""
import pandas as pd
import time


def run_simple_function_on_chunks(reader, fct, print_time = False) -> pd.DataFrame: 
    """
    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The reader is an iterator object that returns the chunks, you can get it for example by 
    calling pd.read_csv(...., chunksize=something).
    The function returns a single dataframe with the results from all chunks concatenated.
    """
    with reader:
        result = pd.DataFrame()
        if not print_time:
            for chunk in reader:
                result = pd.concat([result, fct(chunk)])
            return result
        else:
            time_start_global = time.time()
            for i, chunk in reader.iterrows():
                print(f"Going through video chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                time_end = time.time()
                print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")
            return result
        