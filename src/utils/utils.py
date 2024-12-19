"""Functions for processing (big) datasets. Will be called from the notebook"""
import time
import pandas as pd
from typing import Tuple, Optional, Dict, List
import gc
import os
from math import floor
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from sklearn.metrics import jaccard_score

def run_simple_function_on_chunks_concat(reader, fct, print_time: bool | Tuple = False, 
                                         save: Optional[str] = None, 
                                         save_every: Optional[int] = None,
                                         compress: str = "") -> pd.DataFrame:
    """
    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The function returns a single dataframe with the results from all chunks concatenated.

    Args:
        reader: iterator object that returns the chunks, you can get it for example by calling pd.read_csv(...., chunksize=something).
        print_time: If False (default), does not print time data. If True, prints the average time per chunk.
            If a tuple with two entries is given, where the fist is the chunk size used in the reader, 
            and the second is the total number of entries in the dataset,
            then additional data about estimated time left is printed.
        save: If a string (must be a valid path, WITHOUT .csv at the end!") is given here, the concatenated df is saved here.
        save_every: If a positive integer is given here, the concatenated df will be saved every <save_every> chunk. This
            is so that the resulting df does not become too big.
        compress: If a string is passed here, it will be appended to the filename and the compression would be inferred from that
            example: pass '.gz' 
    """
    
    with reader:
        
        result = pd.DataFrame()
        if not print_time:
            i = 0
            for chunk in reader:
                result = pd.concat([result, fct(chunk)])
                if not not save_every:
                    if (i+1) % save_every == 0:
                        result.to_csv(save +  f"__{i + 1 - save_every}_{i}.csv" + compress)  # save the df concatenated so far
                        result = pd.DataFrame()  # clear the df to avoid memory becoming too big
                        print(f"saved data under {save}__{i + 1 -save_every}_{i}.csv" + compress)
                        collector = gc.collect()
                        print(f"Collected {collector} garbages.")
                    

            if not not save and not save_every:
                result.to_csv(save + ".csv" + compress)
                print(f"saved data under {save}.csv" + compress)
            elif not not save_every:
                result.to_csv(save + f"__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)  # save the remaining data (if the number of chunks is not a multiple of save_every)
                print(f"saved data under {save}__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)

            return result
        
        elif print_time is True:
            time_start_global = time.time()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                if save_every and (i + 1) % save_every == 0:
                    chunk_file = f"{save}__{i + 1 - save_every}_{i}.csv" + (compress or '')
                    result.to_csv(chunk_file, index=False)
                    print(f"Saved data under {chunk_file}")
                    result = pd.DataFrame()  # Clear the DataFrame to save memory
                    
                    
                    gc.collect()
                    print(f"Garbage collected.")
                time_end = time.time()
                print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")
            
            if not not save and not save_every:
                result.to_csv(save + ".csv" + compress)
                print(f"saved data under {save}.csv" + compress)
            elif not not save_every:
                result.to_csv(save + f"__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)  # save the remaining data (if the number of chunks is not a multiple of save_every)
                print(f"saved data under {save}__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)


            return result
        
        else:
            time_start_global = time.time()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                if save_every:
                    if (i+1) % save_every == 0:
                        result.to_csv(save +  f"__{i + 1 - save_every}_{i}.csv" + compress)  # save the df concatenated so far
                        result = pd.DataFrame()  # clear the df to avoid memory becoming too big
                        print(f"saved data under {save}__{i + 1-save_every}_{i}.csv" + compress)
                        collector = gc.collect()
                        print(f"Collected {collector} garbages.")
                
                time_end = time.time()
                processed_entries = (i+1)*print_time[0]
                entries_left = print_time[1] - processed_entries
                avg_time_per_chunk = (time_end-time_start_global)/(i+1)
                print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk /(print_time[0]* 60):.3f} minutes left.")

            if  not not save and not save_every:
                result.to_csv(save + ".csv" + compress)
                print(f"saved data under {save}.csv" + compress)
            elif  not not save_every:
                result.to_csv(save + f"__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)  # save the remaining data (if the number of chunks is not a multiple of save_every)
                print(f"saved data under {save}__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)

            return result


'''

def run_simple_function_on_chunks_save_csv(reader, fct, filename: str, 
                                           index: bool, index_label = None,
                                           every: int = 1, overwrite: bool = False,
                                           print_time: bool | Tuple = False,
                                           strip_newlines: bool = False):
    """
    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The function returns nothing, but saves the results into a single csv.

    Args:
        reader: iterator object that returns the chunks, you can get it for example by calling pd.read_csv(...., chunksize=something).
        filename: file path and name, including the ending .csv, and optionally an ending for compression (such as .gz)
        index: bool, will be passed directly to the to_csv function. If True, also saves the index.
        index_label: can be string, list of strings, False or None (default). Is passed directly to the to_csv function.
            If False, doesn't save a label for the index column. If None, uses the index name from the df. Otherwise, 
            uses the given string. (Sequence is only used for multi index)
        overwrite: if True (default is False), will overwrite existing file. Otherwise raises error.
        every: will save every <every> chunks. Default is 1 (saves after every chunk)
        print_time: If False (default), does not print time data. If True, prints the average time per chunk.
            If a tuple with two entries is given, where the fist is the chunk size used in the reader, 
            and the second is the total number of entries in the dataset,
            then additional data about estimated time left is printed.
        
    """
    if os.path.isfile(filename) and overwrite is not True:
        raise ValueError("the given file already exists.\n" + 
                         "This function will not overwrite files for data safety reasons, unless overwrite=True is passed.")
    elif os.path.isfile(filename) and overwrite is True:
        os.remove(filename)
        print("Removed the existing file, because overwrite=True was passed.")
    
    with reader:
        
        # result = pd.DataFrame()
        if not print_time:
            header = True  # when writing to the csv the first time, include header
            result = pd.DataFrame()
            for i, chunk in enumerate(reader):
                result = pd.concat([result, fct(chunk)])
                if (i + 1) % every == 0:
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    header = False  # when appending new rows to the csv, don't include header
                    print("Appended new rows to the csv file")
                    del result
                    gc.collect()
                    result = pd.DataFrame()
            
            if not result.empty:  # if there is something left to apppend to the csv, do that now
                #result.description.apply(lambda x : x.replace('\n',''))
                result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                print("Appended last rows to the csv file")
            return
        
        elif print_time is True:
            time_start_global = time.time()
            header = True
            result = pd.DataFrame()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                if (i+1) % every == 0:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended new rows to the csv file")
                    print(f"Time spent saving: {time_save_end - time_save_start} secs")
                    header = False  # when appending new rows to the csv, don't include header
                    del result
                    gc.collect()
                    result = pd.DataFrame()
                time_end = time.time()
                print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")

            if not result.empty:  # if there is something left to apppend to the csv, do that now
                time_save_start = time.time()
                result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                time_save_end = time.time()
                print("Appended last rows to the csv file")                   
                print(f"Time spent saving: {time_save_end-time_save_start} secs")
            return
        
        else:
            time_start_global = time.time()
            header = True
            result = pd.DataFrame()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result,fct(chunk)])
                if (i+1) % every == 0:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended new rows to the csv file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs\n")
                    header = False  # when appending new rows to the csv, don't include header
                    del result
                    gc.collect()
                    result = pd.DataFrame()
                    result = result['cahnnekl_id', 'description']
                time_end = time.time()
                processed_entries = (i+1)*print_time[0]
                entries_left = print_time[1] - processed_entries
                avg_time_per_chunk = (time_end-time_start_global)/(i+1)
                print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk /(print_time[0]* 60):.3f} minutes left.\n")

            # if there is something left to apppend to the csv, do that now
            if not result.empty:
                time_save_start = time.time()
                result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                time_save_end = time.time()
                print("Appended last rows to the csv file")                   
                print(f"Time spent saving: {time_save_end-time_save_start} secs")
            return
'''


def run_simple_function_on_chunks_save_csv(reader, fct, filename: str, 
                                           index: bool, index_label = None,
                                           every: int = 1, overwrite: bool = False,
                                           print_time: bool | Tuple = False,
                                           video: bool = False):
    """
    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The function returns nothing, but saves the results into a single csv.

    Args:
        reader: iterator object that returns the chunks, you can get it for example by calling pd.read_csv(...., chunksize=something).
        filename: file path and name, including the ending .csv, and optionally an ending for compression (such as .gz)
        index: bool, will be passed directly to the to_csv function. If True, also saves the index.
        index_label: can be string, list of strings, False or None (default). Is passed directly to the to_csv function.
            If False, doesn't save a label for the index column. If None, uses the index name from the df. Otherwise, 
            uses the given string. (Sequence is only used for multi index)
        overwrite: if True (default is False), will overwrite existing file. Otherwise raises error.
        every: will save every <every> chunks. Default is 1 (saves after every chunk)
        print_time: If False (default), does not print time data. If True, prints the average time per chunk.
            If a tuple with two entries is given, where the first is the chunk size used in the reader, 
            and the second is the total number of entries in the dataset,
            then additional data about estimated time left is printed.
    """
    
    if os.path.isfile(filename) and overwrite is not True:
        raise ValueError("The given file already exists.\nThis function will not overwrite files for data safety reasons, unless overwrite=True is passed.")
    elif os.path.isfile(filename) and overwrite is True:
        os.remove(filename)
        print("Removed the existing file, because overwrite=True was passed.")
    
    # Predefined list of expected columns to ensure consistency
    expected_columns_video = ['categories', 'channel_id', 'crawl_date', 'description', 
                        'dislike_count', 'display_id', 'duration', 'like_count', 
                        'tags', 'title', 'upload_date', 'view_count']
    expected_columns_comments= ['author','video_id','likes','replies']
    with reader:
        header = True  # First chunk will have the header
        result = pd.DataFrame()

        if not print_time:
            # Default chunk processing mode (without time tracking)
            if video==True:
                for i, chunk in enumerate(reader):
                    chunk.columns = chunk.columns.str.strip()  # Remove extra spaces from column names
                    chunk = chunk[expected_columns_video]  # Reorder columns to match the expected order

                    # Apply the function to the chunk
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        header = False  # After the first chunk, don't write header again
                        print(f"Appended new rows to the CSV file (after {i+1} chunks)")
                        del result
                        gc.collect()
                        result = pd.DataFrame()

                # After all chunks have been processed, save the remaining result
                if not result.empty:
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    print("Appended last rows to the CSV file")
            else:
                for i, chunk in enumerate(reader):
                    chunk.columns = chunk.columns.str.strip()  # Remove extra spaces from column names
                    chunk = chunk[expected_columns_comments]  # Reorder columns to match the expected order

                    # Apply the function to the chunk
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        header = False  # After the first chunk, don't write header again
                        print(f"Appended new rows to the CSV file (after {i+1} chunks)")
                        del result
                        gc.collect()
                        result = pd.DataFrame()

                # After all chunks have been processed, save the remaining result
                if not result.empty:
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    print("Appended last rows to the CSV file")
        
        elif print_time is True:
            # Time logging with average time per chunk
            time_start_global = time.time()
            header = True
            result = pd.DataFrame()

            if video==True:
                for i, chunk in enumerate(reader):
                    print(f"Going through chunk {i}...")
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk[expected_columns_video]
                    
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        time_save_start = time.time()
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        time_save_end = time.time()
                        print("Appended new rows to the CSV file")
                        print(f"Time spent saving: {time_save_end - time_save_start} secs")
                        header = False
                        del result
                        gc.collect()
                        result = pd.DataFrame()
                    
                    time_end = time.time()
                    print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")

                if not result.empty:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended last rows to the CSV file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs")
            else :
                for i, chunk in enumerate(reader):
                    print(f"Going through chunk {i}...")
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk[expected_columns_comments]
                    
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        time_save_start = time.time()
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        time_save_end = time.time()
                        print("Appended new rows to the CSV file")
                        print(f"Time spent saving: {time_save_end - time_save_start} secs")
                        header = False
                        del result
                        gc.collect()
                        result = pd.DataFrame()
                    
                    time_end = time.time()
                    print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")

                if not result.empty:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended last rows to the CSV file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs")

        
        else:
            # Full time logging with remaining time estimation
            time_start_global = time.time()
            header = True
            result = pd.DataFrame()

            if video==True: 
                for i, chunk in enumerate(reader):
                    print(f"Going through chunk {i}...")
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk[expected_columns_video]
                    
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        time_save_start = time.time()
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        time_save_end = time.time()
                        print("Appended new rows to the CSV file")                   
                        print(f"Time spent saving: {time_save_end-time_save_start} secs\n")
                        header = False
                        del result
                        gc.collect()
                        result = pd.DataFrame()

                    time_end = time.time()
                    processed_entries = (i + 1) * print_time[0]
                    entries_left = print_time[1] - processed_entries
                    avg_time_per_chunk = (time_end - time_start_global) / (i + 1)
                    print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                    print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk / (print_time[0] * 60):.3f} minutes left.\n")

                if not result.empty:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended last rows to the CSV file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs")
            else: 
                for i, chunk in enumerate(reader):
                    print(f"Going through chunk {i}...")
                    chunk.columns = chunk.columns.str.strip()
                    chunk = chunk[expected_columns_comments]
                    
                    result = pd.concat([result, fct(chunk)], ignore_index=True)
                    
                    if (i + 1) % every == 0:
                        time_save_start = time.time()
                        result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                        time_save_end = time.time()
                        print("Appended new rows to the CSV file")                   
                        print(f"Time spent saving: {time_save_end-time_save_start} secs\n")
                        header = False
                        del result
                        gc.collect()
                        result = pd.DataFrame()

                    time_end = time.time()
                    processed_entries = (i + 1) * print_time[0]
                    entries_left = print_time[1] - processed_entries
                    avg_time_per_chunk = (time_end - time_start_global) / (i + 1)
                    print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                    print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk / (print_time[0] * 60):.3f} minutes left.\n")

                if not result.empty:
                    time_save_start = time.time()
                    result.to_csv(filename, header=header, mode='a', index=index, index_label=index_label)
                    time_save_end = time.time()
                    print("Appended last rows to the CSV file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs")

        
    return


def get_na_entries(data: pd.DataFrame, col: str = "any", reverse: bool = False) -> pd.DataFrame:
    """
    Filters a dataframe to return only rows with na elements (using pd.isna).
    Can be specified to either return rows where any entry is na, or rows where a specified column is na, or where all columns are na.
    
    Args:
        data: the dataframe to be filtered
        col: string defining the criterion. 
            'any' (default) returns rows where any entry is na. 
            '<col_name>' returns rows where entries in the column <col_name> are na.
            'all' returns rows where all entries are na.
        reverse: if True, instead returns the complement of the filtered dataframe, i.e., 
            all rows that do not have na entries (in any, all or a specific column, as defined by 'col')
            
    Returns:
        the filtered dataframe
    """

    if col == "any":
        if not reverse:
            return data[data.isna().any(axis=1)]
        else:
            return data[data.notna().all(axis=1)]
    elif col == "all":
        if not reverse:
            return data[data.isna().all(axis=1)]
        else:
            return data[data.notna().any(axis=1)]
    else:
        if not reverse:
            return data[data.isna()[col]]
        else:
            return data[data.notna()[col]]
        
def get_empty_entries(data: pd.DataFrame, col: str = "any", reverse: bool = False) -> pd.DataFrame:
    """
    Filters a dataframe to return only rows with empty elements (using specific condition).
    Can be specified to either return rows where any entry is empty, or rows where a specified column is empty, or where all columns are empty.
    
    Args:
        data: the dataframe to be filtered
        col: string defining the criterion. 
            'any' (default) returns rows where any entry is empty. 
            '<col_name>' returns rows where entries in the column <col_name> are empty.
            'all' returns rows where all entries are empty.
        reverse: if True, instead returns the complement of the filtered dataframe, i.e., 
            all rows that do not have na entries (in any, all or a specific column, as defined by 'col')
            
    Returns:
        the filtered dataframe
    """
    # # Check for empty strings ('' or ' ') by using .str.strip() to remove spaces
    # empty_check = data.applymap(lambda x: str(x).strip() == '')
    
    # Check for empty strings ('')
    empty_check = data.map(lambda x: str(x) == '')  # I removed the .strip() because I think it would return space rows as well otherwise

    if col == "any":
        if not reverse:
            return data[empty_check.any(axis=1)]
        else:
            return data[~empty_check.any(axis=1)]
    elif col == "all":
        if not reverse:
            return data[empty_check.all(axis=1)]
        else:
            return data[~empty_check.all(axis=1)]
    else:
        if not reverse:
            return data[empty_check[col]]
        else:
            return data[~empty_check[col]]
    
def get_space_entries(data: pd.DataFrame, col: str = "any", reverse: bool = False) -> pd.DataFrame:
    """
    Filters a dataframe to return only rows with space elements (using specific condition).
    Can be specified to either return rows where any entry is a space, or rows where a specified column is a space, or where all columns are composed of space.
    
    Args:
        data: the dataframe to be filtered
        col: string defining the criterion. 
            'any' (default) returns rows where any entry is empty. 
            '<col_name>' returns rows where entries in the column <col_name> are empty.
            'all' returns rows where all entries are empty.
        reverse: if True, instead returns the complement of the filtered dataframe, i.e., 
            all rows that do not have na entries (in any, all or a specific column, as defined by 'col')
            
    Returns:
        the filtered dataframe
    """
    # # Check for empty strings ('' or ' ') by using .str.strip() to remove spaces
    # space_check = data.applymap(lambda x: str(x).strip() == ' ')

    # Check for space strings (' ')
    space_check = data.map(lambda x: str(x) == ' ')  # I removed the .strip() because I think it would never return anything otherwise

    if col == "any":
        if not reverse:
            return data[space_check.any(axis=1)]
        else:
            return data[~space_check.any(axis=1)]
    elif col == "all":
        if not reverse:
            return data[space_check.all(axis=1)]
        else:
            return data[~space_check.all(axis=1)]
    else:
        if not reverse:
            return data[space_check[col]]
        else:
            return data[~space_check[col]]


def get_na_empty_space_entries(data: pd.DataFrame, col: str = "any", reverse: bool = False) -> pd.DataFrame:
    """
    Filters a dataframe to return only rows with na elements, empty strings or blank spaces.
    Adds a columns stating which of the characters lead to the filtering (na, empty or space)
    
    Args:
        data: the dataframe to be filtered
        col: string defining the criterion. 
            'any' (default) returns rows where any entry is na/empty string/space. 
            '<col_name>' returns rows where entries in the column <col_name> are na/ empty string/space.
            'all' returns rows where all entries are na.
        reverse: if True, instead returns the complement of the filtered dataframe, i.e., 
            all rows that do not have na/empty string/space entries (in any, all or a specific column, as defined by 'col')
            
    Returns:
        the filtered dataframe
    """
    na_entries = get_na_entries(data, col, reverse)
    na_entries['char'] = 'na'

    empty_entries = get_empty_entries(data, col, reverse)
    empty_entries['char'] = 'empty'

    space_entries = get_space_entries(data, col, reverse)
    space_entries['char'] = 'empty'

    return pd.concat([na_entries,
                      empty_entries,
                      space_entries])


def count_na_entries(data: pd.DataFrame, col: str = "any", reverse: bool = False) -> pd.DataFrame:
    """
    Goes through a dataframe to count the rows with na elements (using pd.isna).
    Can be specified to either count rows where any entry is na, 
    or rows where a specified column is na, or where all columns are na.
    
    Args:
        data: the dataframe to be filtered
        col: string defining the criterion. 
            'any' (default) counts rows where any entry is na. 
            '<col_name>' counts rows where entries in the column <col_name> are na.
            'all' counts rows where all entries are na.
        reverse: if True, instead counts the complement of the filtered dataframe, i.e., 
            all rows that do not have na entries (in any, all or a specific column, as defined by 'col')
            
    Returns:
        Dataframe with one row, two columns, first is the count of na rows (or not-na rows when reverse=True),
        the second is the total count of rows in the given dataframe
    """

    if col == "any":
        if not reverse:
            return pd.DataFrame({"na rows": len(data[data.isna().any(axis=1)]), 
                                 "total rows": len(data)}, index=[0])
        else:
            return pd.DataFrame({"non-na rows": len(data[data.notna().all(axis=1)]), 
                                 "total rows": len(data)}, index=[0])
    elif col == "all":
        if not reverse:
            return pd.DataFrame({"na rows": len(data[data.isna().all(axis=1)]), 
                                 "total rows": len(data)}, index=[0])
        else:
            return pd.DataFrame({"non-na rows": len(data[data.notna().any(axis=1)]), 
                                 "total rows": len(data)}, index=[0])
    else:
        if not reverse:
            return pd.DataFrame({"na rows": len(data[data.isna()[col]]), 
                                 "total rows": len(data)}, index=[0])
        else:
            return pd.DataFrame({"non-na rows": len(data[data.notna()[col]]), 
                                 "total rows": len(data)}, index=[0])

def clean (df, save):
    """
    Cleans a given df form empty and NaN values. Can be applied to a small df that we want to save in a clean version, as well as temporary chunks when reading a huge df.

    Args:
        df: df to clean
        save: if True, activates the 'inplace' parameter of replace() and dropna(), meaning that the given df will be modified (use this when applying on chunks).
              if False, the modified df has to be saved in a new parameter ! 

    """
  

    if save==True:
        # replace empty values with NaNs
        df.replace('', np.nan, inplace = save)
        # delete NaN columns
        df.dropna(inplace= save)
      

    if save==False:
        return df.replace('', np.nan).dropna()
    


def filter_channel_data(dataset_root_path, df_videos_news_pol_manually_selected, channel_id, channel_name, filename_suffix):
    """
    filter videos for a specific channel.
    
    Args:
    - dataset_root_path (str): Path to the dataset root directory.
    - df_videos_news_pol_manually_selected (DataFrame): The manually selected videos dataframe.
    - channel_id (str): The channel ID 
    - channel_name (str): The name of the channel to be used in the CSV filename.
    - filename_suffix (str): The suffix for the output file name.
    
    Returns:
    - DataFrame: The filtered videos dataframe for the specified channel.
    """
    file_path = dataset_root_path + f"../generated_data/videos_news_pol_{filename_suffix}.csv"
    
    try: # ... try to load the data from file
        df_channel = pd.read_csv(file_path)
        print(f"Data for {channel_name} read from file.")
    except FileNotFoundError: # otherwise, generate it and save
        df_channel = df_videos_news_pol_manually_selected.loc[
            df_videos_news_pol_manually_selected.channel_id == channel_id]
        df_channel.to_csv(file_path, index=False)
        print(f"Data for {channel_name} filtered and saved.")
    
    return df_channel


def plot_dist_comment(comment_counts, channel_name, color):
    """
    Plot distribution of the comments

    Args:
        comments_counts: number of comments
        channel_name: name of channel you're interested in 
        color: color of your histogram 

    """
    plt.figure(figsize=(15, 6))
    plt.plot(comment_counts.index, comment_counts, marker='o', color=color, alpha=0.7)  
    plt.title(f'Number of Comments per Author in {channel_name} channel')  
    plt.xlabel('Author id') 
    plt.ylabel('Number of Comments')  
    plt.xticks(rotation=45, ha='right') 
    plt.grid(True)  
    plt.tight_layout() 
    plt.show() 


def plot_histo(comment_counts, channel_name, color, ylim):
    """
    Plot histogram in normal scale

    Args:
        comments_counts: number of comments
        channel_name: name of channel you're interested in 
        color: color of your histogram 
        ylim: y limit you want to set

    """
    bins = range(0, comment_counts.max() + 10, 10)  # Create bins from 0 to max comments in steps of 10
    bin_labels = [f'{i}-{i+9}' for i in bins[:-1]]  # Labels like '0-10', '10-20', etc.

    # Use pd.cut to bin the comment counts 
    comment_counts_binned = pd.cut(comment_counts, bins=bins, labels=bin_labels, right=False)
    authors_per_bin = comment_counts_binned.value_counts().sort_index()
    authors_per_bin = authors_per_bin.reset_index()
    authors_per_bin.columns = ['Comment Range', 'Number of Authors']

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(comment_counts, bins=bins, alpha=0.7, color=color)
    plt.title(f'Distribution of Comments per Author ({channel_name})')
    plt.xlabel('Number of Comments')
    plt.ylim(0, ylim)
    plt.ylabel('Number of Authors')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_log_histo(comment_counts, channel_name, color):
    """
    Plot histogram in log-log scale

    Args:
        comments_counts: number of comments
        channel_name: name of channel you're interested in 
        color: color of your histogram 

    """
    log_bins = np.logspace(np.log10(1), np.log10(comment_counts.max()), num=20)    
    bin_labels = [f'{int(log_bins[i])}-{int(log_bins[i+1])}' for i in range(len(log_bins)-1)] 

    comment_counts_binned = pd.cut(comment_counts, bins=log_bins, labels=bin_labels, right=False)
    authors_per_bin = comment_counts_binned.value_counts().sort_index()
    authors_per_bin = authors_per_bin.reset_index()
    authors_per_bin.columns = ['Comment Range', 'Number of Authors']

    # Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(comment_counts, bins=log_bins, alpha=0.7, color=color)
    plt.xscale('log')  
    plt.yscale('log') 
    plt.title(f'Distribution of Comments per Author ({channel_name})')
    plt.xlabel('Number of Comments')
    plt.ylabel('Number of Authors')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def common_commented_videos(matrix, user_i: int, user_j: int):
    """
    Returns a boolean vector saying which videos both user i and user j have commented on.

    Args:
        matrix: The (sparse) matrix which contains the video-user data
        user_i: the id of the first user to consider
        user_j: the id of the second user to consider
    Returns:
        (sparse) vector with True values for those videos that both given users have commented on
    """

    return matrix.getcol(user_i).multiply(matrix.getcol(user_j))


def num_common_commented_videos(matrix, user_i: int, user_j: int) -> int:
    """
    Returns the number of videos that both user i and user j have commented on.

    Args:
        matrix: The (sparse) matrix which contains the video-user data
        user_i: the id of the first user to consider
        user_j: the id of the second user to consider
    Returns:
        number of videos both given users have commented on
    """

    return common_commented_videos(matrix, user_i, user_j).sum()


def normed_num_commented_videos(matrix, user_i: int, user_j: int) -> float:
    """
    Returns the number of videos that both user i and user j have commented on, normalized
    w.r.t. the mean number of videos both users have commented on.

    Args:
        matrix: The (sparse) matrix which contains the video-user data
        user_i: the id of the first user to consider
        user_j: the id of the second user to consider
    Returns:
        number of videos both given users have commented on, normalized as stated above
    """

    vid_vec_i = matrix.getcol(user_i)
    vid_vec_j = matrix.getcol(user_j)
    vid_vec_both = vid_vec_i.multiply(vid_vec_j)

    return vid_vec_both.sum() / np.mean([vid_vec_i.sum(), vid_vec_j.sum()])


def get_int_mapping(series: pd.Series) -> pd.Series:
    """
    Converts a unique series into integers from 0 to len(series)-1
    The resulting integers will thus correspond to the ids in the video-user matrix (i.e., 
    the first video will now have index 0 which fits as it is in row 0 of the matrix, etc.)

    Args:
        series: Series which must have unique values
    Returns:
        Series with the values of the given Series as index, and the new integer display id's as values
    Raises:
        ValueError: If the given Series does not have unique values
    """

    if not series.is_unique:
        raise ValueError('The given series is not unique.')
    
    current_name = series.name  # get the name of the current series, such as display_id or author

    return (series  # take the given series
            .drop_duplicates()  # remove any diplicates if there are some
            .reset_index(drop=True)  # create a new index which goes from 0 to number of authors (this is stored in the index) (we don't need the old index, so drop=True)
            .reset_index()  # reset index again, in order to move the just created index into a column (with name 'index')
            .set_index(current_name)['index'] # set the old id (e.g. display_id) to be the index, then select the new index column
            )


def get_inverse_int_mapping(map: pd.Series) -> pd.Series:
    """
    Inverts a mapping (i.e., inverts a Series), meaning that the old values are the new index,
    and the old index is now the values.
    Useful to convert int indices (that were created with the get_int_mapping fct) back to the original indices
    
    Args:
        map: a Series containing the mapping to be inverted. Both the index and the values must be unique
    Returns:
        A Series with the inverted mapping (index and values switched)
    Raises:
        ValueError: If the index or the values of the given map are not unique
    """

    return pd.Series({given_value:given_index for given_index, given_value in map.items()})


def get_video_user_entry_data_for_chunk(comment_data: pd.DataFrame, users: pd.DataFrame, print_time_every: int = 0)-> pd.DataFrame:
    """
    Creates a df with video ids as rows, user ids as columns, and the entries being booleans showing whether the user has commented.
    and one column with the list of videos that user has commented on.
    Args:
        comment_data: a df containing the comments to go through. 
            Video_id must be integers, corresponding to the assumed ordering of video ids in the matrix!
            (So for example, you would have to change video ids ['abc', 'xyz', 'pqr'] to [0, 1, 2], 
            and then change the video id column in the comment data to show, 0, 1, or 2 instead of 'abc', 'xyz' or 'pqr')
        users: the list of users to do this for. The user ids MUST be integers corresponding to their respective columns in the matrix
    Returns:
        df as described above
    """

    comments_by_given_users_grouped = comment_data.loc[comment_data.author.isin(users)].groupby("author")
    video_ids_of_comments_by_given_users_grouped = comments_by_given_users_grouped['video_id']
 
    rows = []
    cols = []
    time_duration_avg = 0
    for i, (user, vid_indices) in enumerate(video_ids_of_comments_by_given_users_grouped.groups.items()):
        time_start = time.time()

        # new row ids are the video ids of the videos the user has commented on
        new_row_ids = comment_data.loc[vid_indices].video_id.values
        rows.extend(new_row_ids)

        # new col ids are simply the column corresponding to the current user. That's why the user id's must correspond to their column ids!
        cols.extend([user] * len(new_row_ids))
        
        
        time_end = time.time()

        # Some timing stuff, not used anymore since the code has gotten fast enough
        if not not print_time_every:
            time_duration = time_end - time_start
            time_duration_avg = (time_duration_avg * i + time_duration) / (i+1)
            time_left_in_chunk_avg = (len(video_ids_of_comments_by_given_users_grouped.groups) - i) * (time_duration_avg)
            if i % print_time_every == 0:
                print(f"    User {i} of {len(video_ids_of_comments_by_given_users_grouped.groups)}: {time_duration:.3f} s ({time_duration_avg:.3f} s avg)   |   " 
                    + f"{time_left_in_chunk_avg:.3f} s ({time_left_in_chunk_avg / 60:.3f} min) left.")

    return rows, cols


def get_video_user_matrix(users_to_consider: pd.Series, 
                          comment_data_filepath: str, chunksize: int, 
                          video_id_int_mapping: pd.Series,
                          filename: str,
                          print_stats: bool = False):
    """
    Creates a video-user matrix, where the rows represent videos, the colums users,
    and each value (i,j) is a boolean which shows whether user j has commented under video i.
    The matrix is always saved, hence a filename must be given.

    Requires that a mapping is given, which maps the video ids in the comment data to their
    respective row in the matrix. (Such a mapping can be created using get_int_mapping() fct)

    A similar mapping for the given users (i.e., the cluster) will be created automatically
    (i.e., the user id's will be replaced by the column id of the user in the matrix)

    The core functionality (extracting video-user matrix for a single chunk) is done by
    calling the get_video_user_entry_data_for_chunk fct.

    Args:
        users_to_consider: Series containing all user id's of the cluster for which the matrix is to be created.
        comment_data_filepath: path to the csv file containing all comments to go through
        video_int_mapping: Series which maps the video_ids in the comment data to a sequence of integers starting from 0
        filename: path to where the resulting matrix will be stored (must end with .npz)
        print_stats: If True (default False), then stats, especially timing, will be printed.

    Returns:
        Sparse matrix in [??] format, where each value (i,j) shows if user j has commented on video i.

    Raises:
        ValueError: If the given filename does not end with '.npz' or leads to an existing file.
    """

    if not filename.endswith(".npz"):
        raise ValueError("The given filename does not end with '.npz'.")

    if os.path.isfile(filename):
        raise ValueError("The given filename leads to a file that exists. Delete or rename"
                         "the file in order to regenerate it.")
    

    # Create mapping for user ids (take all users in the user list (i.e., the cluster) and give 
    # them indices which correspond to their column in the matrix)
    user_id_int_mapping = get_int_mapping(users_to_consider)
    
    # Now for the actual calculations:

    # initiate empty row and column index lists
    rows = []
    cols = []

    time_duration_avg = 0

    # Go through the comments csv file in chunks of the defined chunk size
    for i, df_comments_news_pol_chunk in enumerate(pd.read_csv(comment_data_filepath, 
                                                               chunksize=chunksize)):
        
        if print_stats:
            print(f"Chunk {i} of {(391295476 / chunksize):.3f}:")
        
        time_start = time.time()
        
        # Replace the video_id in comment data by int indices using our mapping from above
        df_comments_news_pol_chunk.video_id = df_comments_news_pol_chunk.video_id.map(video_id_int_mapping)
        # and the same for the author
        df_comments_news_pol_chunk.author = df_comments_news_pol_chunk.author.map(user_id_int_mapping)
                
        # go through chunk i, get row and column indices for the current chunk
        (newrows, newcols) = get_video_user_entry_data_for_chunk(df_comments_news_pol_chunk, 
                                                                 user_id_int_mapping)

        # append the new row and column data from this chunk to the lists
        rows.extend(newrows)
        cols.extend(newcols)

        # every 10 chunks, we have to remove duplicate entries from the lists (every comment 
        # corresponds to one entry,however if a user has commented several times under one video
        # we only want one entry.) 
        # By creating a sparse array, we can remove these duplicates. In theory, it would be enough
        # to remove all duplicates at the end, however we need to do it more often, because otherwise
        # our lists of entries become too large and need too much memory.
        if i % 10 == 0 and len(rows) > 10000000:
            data = np.ones_like(rows, dtype=bool)  # create data for the entries

            if print_stats:
                old_row_len = len(rows)
                old_col_len = len(cols)
                print(f"Old row length: {old_row_len}, Old col length: {old_col_len}")

            # Create the array
            intermediate_coo_array = scipy.sparse.coo_array((data, (rows, cols)),
                                                            shape=(len(video_id_int_mapping),
                                                                    len(user_id_int_mapping)),
                                                            dtype=bool)
            intermediate_coo_array.sum_duplicates()  # remove the duplicates
            (rows, cols) = intermediate_coo_array.coords  # get the new rows and cols from the array
            
            # delete data not needed anymore and garbage collect
            del intermediate_coo_array
            del data
            if print_stats:
                print(f"{gc.collect()} garbages collected")
            else:
                gc.collect()

            # convert the data to lists again
            rows = rows.tolist()
            cols = cols.tolist()
            
            if print_stats:
                new_row_len = len(rows)
                new_col_len = len(cols)
                print(f"New row length: {new_row_len}, New col length: {new_col_len}")

        time_end = time.time()

        if print_stats:
            # Just some code to print time
            time_duration = time_end-time_start
            time_duration_avg = (time_duration_avg * i + time_duration) / (i+1)  # update the average time per chunk
            
            time_left_all_chunks_avg = ((391295476 / chunksize) - i) * (time_duration_avg)

            print(f"{time_duration:.3f} s ({time_duration / 60:.3f} min) total for chunk {i}.   |   "
                + f"{time_left_all_chunks_avg:.3f} s ({time_left_all_chunks_avg / 60:.3f} min) left total.\n")
    
    # Now create an array full of "True", the same length as the list of row indices
    data = np.ones_like(rows, dtype=bool)

    # Create the matrix and convert to csc sparse matrix format because this is much 
    # faster for getting columns than coo format
    video_author_matrix = scipy.sparse.coo_matrix((data, (rows, cols)), 
                                                 shape=(len(video_id_int_mapping),
                                                        len(user_id_int_mapping)),
                                                 dtype=bool).tocsc()
    # delete data not needed anymore and garbage collect
    del rows
    del cols
    del data
    del newrows
    del newcols
    del user_id_int_mapping
    if print_stats:
        print(f"{gc.collect()} garbages collected")
    else:
        gc.collect()
    
    # Save the array
    scipy.sparse.save_npz(filename, 
                          video_author_matrix)
    print(f"Saved file {filename}")

    return video_author_matrix

#first functions to calculate jaccard index : using jaccard_score function from sklearn.metrics library
#for 2 different clusters
def get_jacc_between_two_clusters_and_get_mean_sklearn_function(cluster1, cluster2, original_userid1, original_userid2):
    jaccard_list = []
    for i in range(cluster1.shape[1]):
        original_user1 = original_userid1[i]
        user1_array = cluster1.getcol(i).toarray().flatten()
        for j in range(cluster2.shape[1]):
            original_user2 = original_userid2[j]
            if original_user1!=original_user2 : 
                user2_array = cluster2.getcol(j).toarray().flatten()
                jaccard_list.append(jaccard_score(user1_array,user2_array))
    
    overall_jaccard = sum(jaccard_list)/len(jaccard_list)

    return overall_jaccard, jaccard_list
#for the same cluster
def get_jacc_between_same_cluster_and_get_mean_sklearn_function(cluster1):
    jaccard_list = []
    for i in range(cluster1.shape[1]):
        user1_array = cluster1.getcol(i).toarray().flatten()
        for j in range(i+1, cluster1.shape[1]): 
            user2_array = cluster1.getcol(j).toarray().flatten()
            jaccard_list.append(jaccard_score(user1_array,user2_array))
    
    overall_jaccard = sum(jaccard_list)/len(jaccard_list)
    
    return overall_jaccard, jaccard_list

#other way of coding the function to calculate jaccard index : do the calculations by hand of the jaccard index = 1-(TF+FT/TT+TF+FT) (see definition of jaccard index)
#for 2 different clusters
def get_jacc_between_two_clusters_and_get_mean_by_hand(cluster1, cluster2, original_userid1, original_userid2):
    jaccard_list = []
    for i in range(cluster1.shape[1]):
        original_user1 = original_userid1[i]
        for j in range(cluster2.shape[1]):
            original_user2 = original_userid2[j]
            if original_user1!=original_user2 : 
                TT = cluster1.getcol(i).multiply(cluster2.getcol(j))
                TF = (cluster1.getcol(i)-cluster2.getcol(j)).multiply(cluster1.getcol(i))
                FT = (cluster2.getcol(j)-cluster1.getcol(i)).multiply(cluster2.getcol(j))
                jaccard_list.append(1-((TF.sum()+FT.sum())/(TT.sum()+TF.sum()+FT.sum())))

    overall_jaccard = sum(jaccard_list)/len(jaccard_list)

    return overall_jaccard, jaccard_list
#for the same cluster
def get_jacc_between_same_cluster_and_get_mean_by_hand(cluster1):
    jaccard_list = []
    for i in range(cluster1.shape[1]):
        for j in range(i+1, cluster1.shape[1]): 
            TT = cluster1.getcol(i).multiply(cluster1.getcol(j))
            TF = (cluster1.getcol(i)-cluster1.getcol(j)).multiply(cluster1.getcol(i))
            FT = (cluster1.getcol(j)-cluster1.getcol(i)).multiply(cluster1.getcol(j))
            jaccard_list.append(1-((TF.sum()+FT.sum())/(TT.sum()+TF.sum()+FT.sum())))
    
    overall_jaccard = sum(jaccard_list)/len(jaccard_list)
    
    return overall_jaccard, jaccard_list

def add_zero_cols_to_sparse_matrix(matrix: scipy.sparse.csc_matrix, col_indices) -> scipy.sparse.csc_matrix:
    """
    Adds columns with all zeros (a.k.a. empty values) at all the speified column indices.
    The given row indices refer to the matrix in its finished state.

    For example, if a matrix with 3 columns is given (initial indices 0, 1, 2), together with
    col_indices=[2, 4, 5], zero columns will be put after the second original column, after
    the fourth column which was the third initial column, and another one after the fifth 
    column, which is the empty column just created.

    Example: 
                                                    1   2   3
    col_indices=[2, 4, 5] with initial matrix:  A=  4   5   6
                                                    7   8   9
                        1   2   0   3   0   0
    Resulting matrix:   4   5   0   5   0   0
                        7   8   0   9   0   0
    
    Args:
        matrix: sparse matrix to add the empty columns tp
        col_indices: places where the columns are to be put. For example, if a matrix with 
            three columns are given, and 
    """

    list_of_sparse_blocks = []

    empty_col = scipy.sparse.csc_matrix((matrix.shape[0], 1), dtype=matrix.dtype)
    old_index = 0
    for i, index in enumerate(col_indices):
        list_of_sparse_blocks.append(matrix[:,old_index:index - i].copy())
        old_index = index - i
        list_of_sparse_blocks.append(empty_col)
    
    list_of_sparse_blocks.append(matrix[:,old_index:])

    return scipy.sparse.hstack(list_of_sparse_blocks)


def get_video_user_matrices_with_equal_columns(video_user_matrix_1: scipy.sparse.csc_matrix,
                                               video_user_matrix_2: scipy.sparse.csc_matrix,
                                               user_ids_1: pd.Series, 
                                               user_ids_2: pd.Series) -> Tuple[scipy.sparse.csc_matrix,
                                                                               scipy.sparse.csc_matrix]:
    """
    Takes two video user matrices with the columns possibly corresponding to different users.
    Expands these two matrices by adding empty columns, so that both matrices share the same
    columns. This makes a comparison afterwards more easy, because we know that the user in
    column i in one of the matrices is the same as user i in the other matrix.

    Args:
        video_user_matrix_1: sparse video user matrix for one cluster
        video_user_matrix_2: sparse video user matrix for another cluster
        user_ids_1: mapping containing the original user id's as values, for matrix 1
        user_ids_2: the same for matrix 2
    """
    # raise Exception("This doesn't work yet!")

    # First verify that the two user id mappings are ordered in the same way, namely that
    # the original user id's are sorted in ascending order.
    # Otherwise the transformation we are going to do will not work
    print("Verifying that given user id Series are sorted corectly....")
    if (user_ids_1.sort_values().values != user_ids_1.values).any():
        raise ValueError("The given user id mapping 1 does not have the index sorted correctly.")
    elif (user_ids_2.sort_values() != user_ids_2).any():
        raise ValueError("The given user_id_mapping_2 does not have the index sorted correctly.")
    print("Done.")

    
    # get Series of all users that are in one of the matrices (= union of the users in both
    # given mappings)
    # These users will be the columns in the resulting matrices
    print("Concatting the users from both clusters and dropping duplicate values...")
    all_user_ids = pd.concat([user_ids_1, user_ids_2])
    # drop duplicate original user id's
    all_user_ids.drop_duplicates(inplace=True)
    print("Done.")
    # print(all_user_ids)
    print("Sorting the user ids in ascending order....")
    all_user_ids.sort_values(inplace=True)
    print("Done.")
    # print(all_user_ids)
    
    # find which of these users are "missing" in the users of the two given matrices
    print("Getting indices where empty columns are to be added for both matrices....")
    columns_to_add_to_matrix_1 = np.flatnonzero((~all_user_ids.isin(user_ids_1)).values)
    columns_to_add_to_matrix_2 = np.flatnonzero((~all_user_ids.isin(user_ids_2)).values)
    print("Done.")

    # add empty columns in order to get both matrices to the same columns
    print("Adding the empty columns....")
    matrix_1_with_added_cols = add_zero_cols_to_sparse_matrix(video_user_matrix_1, columns_to_add_to_matrix_1)
    print("Matrix 1 done.")
    matrix_2_with_added_cols = add_zero_cols_to_sparse_matrix(video_user_matrix_2, columns_to_add_to_matrix_2)
    print("Both matrices done.")

    return (matrix_1_with_added_cols, matrix_2_with_added_cols)


def remove_entries_for_duplicate_user_pairs(matrix: scipy.sparse.csc_matrix | np.ndarray, 
                                            users_1: pd.Series, 
                                            users_2: pd.Series,
                                            nan_instead: Optional[bool] = False) -> scipy.sparse.csc_matrix | np.ndarray:
    """
    Takes the given matrix and the given information about which users its rows and columns
    correspond to, and sets any entry corresponding to one user matched to themselves to zero (or np.nan,
    if nan_instead=True)

    This is to be used for a jaccard index matrix, as we are not interested in the comparison of
    one user to themselves, as the jaccard index will always be 1.

    Args:
        matrix: with some values (e.g. jaccard index) corresponding to pairs of users
        users_1: Series of users corresponding to the rows of the given matrix
        users_2: Series of users corresponding to the columns of the given matrix
        nan_instead: if True (default False), will not set to zero but instead set to np.nan

    Returns:
        matrix: The matrix, but with entries where row and column correspond to the same user set to 0
        removed_entries: The number of entries removed (set to 0, or np.nan if nan_instead=True)
    """
    removed_entries = 0
    for i, user_1 in enumerate(users_1.values):
        for j, user_2 in enumerate(users_2.values):
            if user_1 == user_2:
                removed_entries += 1
                if nan_instead:
                    matrix[i, j] = np.nan
                else:
                    matrix[i, j] = 0
    print(f"{removed_entries} overlapping users found and set to 0 or np.nan.")

    if isinstance(matrix, scipy.sparse.csc_matrix):
        matrix.eliminate_zeros()

    return matrix, removed_entries


def get_c_true_true(video_user_matrix: scipy.sparse.csc_matrix, 
                    video_user_matrix_2: Optional[scipy.sparse.csc_matrix] = None) -> scipy.sparse.csc_matrix:
    """
    Gets a user-user "overlap" matrix, i.e. a matrix showing the amount of videos two users
    have both commented on, based on a given video-user matrix, or optionally two.
    If one matrix is given, calculates the "overlap" for all pairs of users from the matrix.
    If two matrices are given, calculates the overlap for all pairs of users from the two
    different matrices.

    No filtering or anything is done, i.e., the resulting matrix might include the overlap
    of users with themselves, etc.

    Args:
        video_user_matrix: video user matrix containing information about which videos 
            users have commented on
        video_user_matrix_2: optional second video user matrix. If this is given, the users 
            in the first matrix


    Returns:
        c_true_true matrix, meaning that the entries are true when both users have commented
    
    Raises:
        ValueError: If two matrices are given and don't have the same number of rows
    """
    # we can use int16 (can display values up to +32767), because we will never have values
    # which are higher than the number of videos any user has commented on, and no user has 
    # commented on more than 17262 videos, in all of our clusters.
    # By using int16 instead of int, we save memory
    video_user_matrix = video_user_matrix.astype(np.int16)

    if video_user_matrix_2 is None:
        return (video_user_matrix.T @ video_user_matrix).astype(np.int16)
    elif video_user_matrix.shape[0] != video_user_matrix_2.shape[0]:
        raise ValueError("Given matrices don't share the same row length.")
    else:
        video_user_matrix_2 = video_user_matrix_2.astype(np.int16)
        return video_user_matrix.T @ video_user_matrix_2
    

def get_c_false_true_matrix(video_user_matrix: scipy.sparse.csc_matrix, 
                            video_user_matrix_2: Optional[scipy.sparse.csc_matrix] = None,
                            where: Optional[np.ndarray] = None) -> np.ndarray | scipy.sparse.csc_matrix:
    """
    Gets matrix showing for which entries user i has not commented but user j has.
    If one matrix is given, does it for all pairs of users from this matrix,
    otherwise does it for the first to the second matrix.

    No filtering w.r.t duplicates etc. is done here.

    Args:
        video_user_matrix: video user matrix containing information about which videos 
            users have commented on
        video_user_matrix_2: optional second video user matrix. If this is given, the users 
            in the first matrix
        where: optional bool array of same size as the result. If given, then the C_ft matrix
            will have value 0 in all entries where this matric is false. It will also be returned
            as a sparse csc matrix instead of a np array, to make use of this sparsity

    Returns:
        false true matrix
    """
    # we can use int16 (can display values up to +32767), because we will never have values
    # which are higher than the number of videos any user has commented on, and no user has 
    # commented on more than 17262 videos, in all of our clusters.
    # By using int16 instead of int, we save memory
    video_user_matrix = video_user_matrix.astype(np.int16)

    # We use a trick here: The thing we want to calculate is X^T * (1 - Y).
    # But if we do that directly, we need to store a matrix which is as big as Y (i.e. very big)
    # which is full of ones, hence not sparse at all. This would take up too much memory.
    # So instead, we calculate X^T * 1 - X^T * Y
    # (Note that the 1 here is not the scalar 1, it is a matrix full of ones.)
    # And, the trick is that X^T times a matrix full of ones is just the sum of each column of
    # X^T, so we can calculate that directly instead (no need to actually generate the matrix
    # full of ones)

    matrix_T_times_ones = video_user_matrix.T.sum(axis=1, dtype=np.int16)  # Corresponds to X^T * 1
    print(f"Dtype of matrix_T_times_ones is {matrix_T_times_ones.dtype}")
    print(f"It looks like this:\n{matrix_T_times_ones}")

    if video_user_matrix_2 is None:
        
        result = matrix_T_times_ones - (video_user_matrix.transpose() @ video_user_matrix).astype(np.int16)
        if where is None:
            return result
        else:
            return scipy.sparse.csc_matrix(np.multiply(result, where))
        
    elif video_user_matrix.shape[0] != video_user_matrix_2.shape[0]:
        raise ValueError("Given matrices don't share the same row length.")
    
    else:
        video_user_matrix_2 = video_user_matrix_2.astype(np.int16)
        result = matrix_T_times_ones - (video_user_matrix.T @ video_user_matrix_2).astype(np.int16)
        if where is None:
            return result
        else:
            return scipy.sparse.csc_matrix(np.multiply(result, where))

    

def get_c_true_false_matrix(video_user_matrix: scipy.sparse.csc_matrix, 
                            video_user_matrix_2: Optional[scipy.sparse.csc_matrix] = None,
                            where: Optional[np.ndarray] = None) -> np.ndarray | scipy.sparse.csc_matrix:
    """
    Gets matrix showing for which entries user i has commented on but user j has not.
    If one matrix is given, does it for all pairs of users from this matrix,
    otherwise does it for the first to the second matrix.

    No filtering w.r.t duplicates etc. is done here.

    Args:
        video_user_matrix: video user matrix containing information about which videos 
            users have commented on
        video_user_matrix_2: optional second video user matrix. If this is given, the users 
            in the first matrix
        where: optional bool array of same size as the result. If given, then the C_ft matrix
            will have value 0 in all entries where this matric is false. It will also be returned
            as a sparse csc matrix instead of a np array, to make use of this sparsity


    Returns:
        true false matrix
    """

    # we can use int16 (can display values up to +32767), because we will never have values
    # which are higher than the number of videos any user has commented on, and no user has 
    # commented on more than 17262 videos, in all of our clusters.
    # By using int16 instead of int, we save memory
    video_user_matrix = video_user_matrix.astype(np.int16)

    if video_user_matrix_2 is None:
        
        # Again we use the trick of calculating 1 * X, where 1 is a matrix full of ones,
        # without actually creating the matrix full of ones, because 1 * X is the sum of each
        # column of X. See also the function get_c_false_true_matrix above

        ones_times_matrix = video_user_matrix.sum(axis=0, dtype=np.int16)

        result = ones_times_matrix - (video_user_matrix.T @ video_user_matrix).astype(np.int16)
        
        if where is None:
            return result
        else:
            return scipy.sparse.csc_matrix(np.multiply(result, where))
        
    elif video_user_matrix.shape[0] != video_user_matrix_2.shape[0]:
        raise ValueError("Given matrices don't share the same row length.")
    
    else:
        # convert to np.int16 again
        video_user_matrix_2 = video_user_matrix_2.astype(np.int16)
        ones_times_matrix = video_user_matrix_2.sum(axis=0, dtype=np.int16)

        result = ones_times_matrix - (video_user_matrix.T @ video_user_matrix_2).astype(np.int16)

        if where is None:
            return result
        else:
            return scipy.sparse.csc_matrix(np.multiply(result, where))
    

def get_jaccard_index_matrix(video_user_matrix: scipy.sparse.csc_matrix,
                             video_user_matrix_2: Optional[scipy.sparse.csc_matrix] = None,
                             precision: int = 32,
                             sparse: bool = False) -> np.array:
    """
    Gets the jaccard distance, defined as (c_tt) / (c_tt + c_tf + c_ft)
    If one matrix is given, does it for all pairs of users in this matrix,
    if two are given, does it for all pairs from these two matrices.
    
    Args:
        video_user_matrix: video user matrix
        video_user_matrix: optional second video user matrix (eg., from another cluster)
        precision: optional specification of float precision to use for the final division.
            Default is 32
        sparse: if True (default is False), will calculate c_tt, c_tf etc a sparse matrices.
            Note that this only makes sense when we expect there to be many users that don't
            have any video in common (for example when comparing two different clusters)
        
    Returns:
        Jaccard index matrix. Note that no filtering or removal of duplicate users is done here.
    
    """

    if precision not in [16, 32]:
        raise ValueError("Given precision must be 16 or 32")

    # get C_tt matrix
    if sparse:
        c_tt = get_c_true_true(video_user_matrix, video_user_matrix_2)
        print("C_tt is a sparse matrix like this:")
        print(c_tt)
    else:
        c_tt = get_c_true_true(video_user_matrix, video_user_matrix_2).toarray()
    print("c_tt calculated")

    if sparse:
        c_tt_non_zero = c_tt.astype(bool).toarray()

    # get C_tf matrix

    # c_tf = get_c_true_false_matrix(video_user_matrix, video_user_matrix_2)
    # print(c_tf)
    # del c_tf
    
    # remove all entries which are 0 in the C_tt matrix from the C_tf matrix, then make sparse
    # (idea: when C_tt is 0, then the division result of C_tt / (C_tt + C_tf + C_ft) (for 
    #  for jaccard index) will be 0 anyway, so no need to consider these entries in C_tf and 
    #  C_ft. In this way, we save space and can make the matrices more sparse.)
    # (Because C_tt is much more sparse than C_tt, because there will be user pairs which have
    #  no videos in common (meaning C_tt = 0), but there will hardly be any user pairs which
    #  have no videos which one has commented on but the other hasn't (which would be needed
    #  for C_tf = 0))

    # c_tf = scipy.sparse.csc_matrix(np.multiply(c_tf ,c_tt.astype(bool).astype(np.int32).toarray()))
    # c_tf = get_c_true_false_matrix(video_user_matrix, video_user_matrix_2, where=c_tt_non_zero)
    if sparse:
        denominator = c_tt + get_c_true_false_matrix(video_user_matrix, video_user_matrix_2, where=c_tt_non_zero)
    else:
        denominator = c_tt + get_c_true_false_matrix(video_user_matrix, video_user_matrix_2)#, where=c_tt_non_zero)
    print("c_tf calculated")

    # get C_ft matrix

    # c_ft = get_c_false_true_matrix(video_user_matrix, video_user_matrix_2)
    # print(c_ft)
    # del c_ft

    # remove all entries which are 0 in the C_tt matrix from the C_ft matrix, then make sparse
    # c_ft = scipy.sparse.csc_matrix(np.multiply(c_ft,c_tt.astype(bool).astype(np.int32).toarray()))
    # c_ft = get_c_false_true_matrix(video_user_matrix, video_user_matrix_2, where=c_tt_non_zero)
    if sparse:
        denominator += get_c_false_true_matrix(video_user_matrix, video_user_matrix_2, where=c_tt_non_zero)    
    else:
        denominator += get_c_false_true_matrix(video_user_matrix, video_user_matrix_2)#, where=c_tt_non_zero)

    # calculate the denominator
    # print(c_tt)
    # print(c_tf)
    # print(c_ft)
    # denominator = c_tt + c_tf + c_ft
    print("Calculated c_ft, and summing nominator done. Looks like this:")
    print(denominator)
    print("And has this shape")
    print(denominator.shape)
    print("The numeraor will have this shape:")
    print(c_tt.shape)
    # print("This is where the true values of the numerator are:")
    # print(c_tt.astype(bool))
    # denominator = np.multiply(denominator,c_tt.astype(bool).astype(np.int32).toarray())#np.multiply(denominator, c_tt.astype(bool))
    # print("Removing elements in denominator which are 0 in numerator done. Still an array, looks like this:")
    # print(denominator)
    # denominator = scipy.sparse.csc_matrix(denominator, dtype=np.float32)
    # print("Converting denominator to sparse done, looks like this:")
    # print(denominator)
    # numerator = c_tt#.astype(np.float32)
    print("This is the numerator:")
    print(c_tt)
    # del c_tt
    # del c_tf
    # del c_ft
    print("Got all the matrices. Starting division....")
    if precision == 32:
        result = np.zeros(c_tt.shape, dtype=np.float32)
    elif precision == 16:
        result = np.zeros(c_tt.shape, dtype=np.float16)

    if sparse:
        np.divide(c_tt.toarray(),
              denominator.toarray(),
              where=c_tt_non_zero,
              out=result)
    else:
        np.divide(c_tt,#.toarray(),
                denominator,#.toarray(),
                #where=c_tt_non_zero,
                out=result)
        
    print("Division done. Result is:")
    print(result)
    return result


def get_mean_without_duplicates(matrix: np.ndarray) -> float:
    """
    Takes the mean of all values of a matrix, but ignores the values on the diagonal.
    The assumtion is that on the diagonal, we are comparing one user to themselves,
    which is not a relevant metric.
    
    Args:
        matrix: a sparse square matrix with distance values, e.g. jaccard distance, for all pairs
            of users.
    Returns:
        mean of all values excluding the original
    
    Raises:
        ValueError: if given matrix is not square
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Given matrix is not square")
    
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)

    np.fill_diagonal(matrix, np.nan)

    return np.nanmean(matrix)
    # sum_of_entries = matrix.sum(axis=None)
    # print(f"Sum of all entries without diag is {sum_of_entries}")

    # num_of_entries_without_diag = matrix.shape[0] * matrix.shape[1] - matrix.shape[0]
    # print(f"Numbre of entries without the diagonal is {num_of_entries_without_diag}")
    # return sum_of_entries / num_of_entries_without_diag


def get_median_without_duplicates(matrix: np.ndarray) -> float:
    """
    Takes the median of all values of a matrix, but ignores the values on the diagonal.
    The assumtion is that on the diagonal, we are comparing one user to themselves,
    which is not a relevant metric.
    
    Args:
        matrix: a sparse square matrix with distance values, e.g. jaccard distance, for all pairs
            of users.
    Returns:
        mean of all values excluding the original
    
    Raises:
        ValueError: if given matrix is not square
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Given matrix is not square")
    
    if matrix.dtype != np.float32:
        matrix = matrix.astype(np.float32)

    np.fill_diagonal(matrix, np.nan)
    
    return np.nanmedian(matrix)


def get_jacc_between_same_cluster_and_get_mean(video_user_matrix: scipy.sparse.csc_matrix,
                                               filename: str) -> float:
    """
    Gets jaccard index matrix between all pairs of users from one cluster, and calculates the mean
    of the jaccard indices excluding the indices between one user with themselves.

    Calculates and saves the jaccard index matrix if it doesnt exist yet

    Args:
        video_user_matrix: sparse matrix with video user data for the cluster
        filename: filename to look for the jaccard index matrix, or to save it
    
    Returns:
        mean jaccard index of the given cluster

    Side effects:
        Calculates jaccard index matrix for the given cluster if it doesn't exist yet
    """

    try:
        jaccard = np.load(filename)
        print("    Loaded jaccard index matrix from file")
    except FileNotFoundError:
        print("    Jaccard index matrix does not exist yet, generating....")
        jaccard = get_jaccard_index_matrix(video_user_matrix)
        print("    Done.")
        np.save(filename, jaccard)

    print("    Calculating mean of jaccard index matrix, excluding the diagonal....")
    mean_jaccard = get_mean_without_duplicates(jaccard)
    print("    Done.")
    del jaccard
    gc.collect()

    return mean_jaccard


def get_jacc_between_same_cluster_and_get_median(video_user_matrix: scipy.sparse.csc_matrix,
                                                 filename: str) -> float:
    """
    Gets jaccard index matrix between all pairs of users from one cluster, and calculates the median
    of the jaccard indices excluding the indices between one user with themselves.

    Calculates and saves the jaccard index matrix if it doesnt exist yet

    Args:
        video_user_matrix: sparse matrix with video user data for the cluster
        filename: filename to look for the jaccard index matrix, or to save it
    
    Returns:
        median jaccard index of the given cluster

    Side effects:
        Calculates jaccard index matrix for the given cluster if it doesn't exist yet
    """

    try:
        jaccard = np.load(filename)
        print("    Loaded jaccard index matrix from file")
    except FileNotFoundError:
        print("    Jaccard index matrix does not exist yet, generating....")
        jaccard = get_jaccard_index_matrix(video_user_matrix)
        print("    Done.")
        np.save(filename, jaccard)

    print("    Calculating median of jaccard index matrix, excluding the diagonal....")
    mean_jaccard = get_median_without_duplicates(jaccard)
    print("    Done.")
    del jaccard
    gc.collect()

    return mean_jaccard

def get_mean_jacc(video_user_matrix: scipy.sparse.csc_matrix) -> float:
    """
    Gets jaccard index matrix between all pairs of users from one cluster, and calculates the mean
    of the jaccard indices excluding the indices between one user with themselves.

    Calculates and saves the jaccard index matrix if it doesnt exist yet

    Args:
        video_user_matrix: sparse matrix with video user data for the cluster
        filename: filename to look for the jaccard index matrix, or to save it
    
    Returns:
        mean jaccard index of the given cluster

    Side effects:
        Calculates jaccard index matrix for the given cluster if it doesn't exist yet
    """
    
    jaccard = get_jaccard_index_matrix(video_user_matrix)
    
    mean_jaccard = get_mean_without_duplicates(jaccard)
    del jaccard
    gc.collect()

    return mean_jaccard

def get_jacc_between_two_clusters_and_get_mean(matrix_1: scipy.sparse.csc_matrix,
                                               matrix_2: scipy.sparse.csc_matrix, filename: str,
                                               users_to_consider_1: pd.Series,
                                               users_to_consider_2: pd.Series) -> float:
    """
    Get jaccard matrix between two different clusters, save it, and calculate the mean of the
    entries excluding the diagonal.

    Note: this function will take the two given matrices and put empty columns in them so that 
    they share the same column. 
    So if one matrix contains user 1 and user 3, and the other matrix conatins user 2 and user 3, 
    then the matrices are enlarged so that both contain user 1, 2 and 3, however the first matrix
    will have zeros for user 2, and the second matrix will have zeros for user 1.

    The reason for this is that then, all values for one user with themselves are on the diagonal,
    so that we can exclude them (because that will always have jaccard index 1).
    
    Args:
        matrix_1: the video user matrix for the first cluster
        matrix_2: the video user matrix for the second cluster
        users_to_consider_1: the authors contained in the first cluster (with original user ids)
        users_to_consider_2: the authors contained in the second cluster (with original user ids)

    Returns:
        Mean of the jaccard index for all pairs of users from the two clusters, excluding pairs of
        the same user

    Side effects:
        Calculates the jaccard matrix for these clusters and saves it, if not already existing.
    """

    try:  # try to load the jaccard matrix
        jaccard_with_duplicates = np.load(filename)
        print("    Loaded jaccard index matrix from file")
    except FileNotFoundError:  # otherwise, generate it

        # # add empty columns to the matrices, so that the same column in both matrices also corresponds
        # # to the same user
        # print("Getting the matrices with added empty columns....")
        # (matrix_1_w_empty_cols, 
        #  matrix_2_with_empty_cols) = dp.get_video_user_matrices_with_equal_columns(matrix_1, matrix_2, 
        #                                                                            users_to_consider_1, 
        #                                                                            users_to_consider_2)
        # print("Done.")
        # generate jaccard index matrix for these matrices
        # (Note that sparse is True, because many columns are 0, because of the step above)
        print("    Jaccard index matrix does not exist yet, generating....")
        # jaccard = dp.get_jaccard_index_matrix(matrix_1_w_empty_cols, matrix_2_with_empty_cols,
        #                                       sparse=True)
        jaccard_with_duplicates = get_jaccard_index_matrix(matrix_1, matrix_2)
        print("    Done.")

        # save the file
        np.save(filename, jaccard_with_duplicates)
    
    # Remove all entries of the jaccard matrix where the row and column correspond to the same user
    print("    Removing entries corresponding to pairs of the same user....")
    jaccard_without_duplicates, removed_entries = remove_entries_for_duplicate_user_pairs(
        jaccard_with_duplicates, 
        users_to_consider_1,
        users_to_consider_2)
    del jaccard_with_duplicates
    print("    Done.")
    # calculate the mean of the jaccard indices, excluding the diagonal
    print("    Calculating the mean of the jaccard index matrix....")
    mean_jaccard = jaccard_without_duplicates.sum(axis=None) / (jaccard_without_duplicates.size 
                                                                - removed_entries)
    print("    Done.")
    del jaccard_without_duplicates
    gc.collect()
    return mean_jaccard


def get_jacc_between_two_clusters_and_get_median(matrix_1: scipy.sparse.csc_matrix,
                                                 matrix_2: scipy.sparse.csc_matrix, filename: str,
                                                 users_to_consider_1: pd.Series,
                                                 users_to_consider_2: pd.Series) -> float:
    """
    Get jaccard matrix between two different clusters, save it, and calculate the median of the
    entries excluding the diagonal.

    Note: this function will take the two given matrices and put empty columns in them so that 
    they share the same column. 
    So if one matrix contains user 1 and user 3, and the other matrix conatins user 2 and user 3, 
    then the matrices are enlarged so that both contain user 1, 2 and 3, however the first matrix
    will have zeros for user 2, and the second matrix will have zeros for user 1.

    The reason for this is that then, all values for one user with themselves are on the diagonal,
    so that we can exclude them (because that will always have jaccard index 1).
    
    Args:
        matrix_1: the video user matrix for the first cluster
        matrix_2: the video user matrix for the second cluster
        users_to_consider_1: the authors contained in the first cluster (with original user ids)
        users_to_consider_2: the authors contained in the second cluster (with original user ids)

    Returns:
        Mean of the jaccard index for all pairs of users from the two clusters, excluding pairs of
        the same user

    Side effects:
        Calculates the jaccard matrix for these clusters and saves it, if not already existing.
    """

    try:  # try to load the jaccard matrix
        jaccard_with_duplicates = np.load(filename)
        print("    Loaded jaccard index matrix from file")
    except FileNotFoundError:  # otherwise, generate it

        # # add empty columns to the matrices, so that the same column in both matrices also corresponds
        # # to the same user
        # print("Getting the matrices with added empty columns....")
        # (matrix_1_w_empty_cols, 
        #  matrix_2_with_empty_cols) = dp.get_video_user_matrices_with_equal_columns(matrix_1, matrix_2, 
        #                                                                            users_to_consider_1, 
        #                                                                            users_to_consider_2)
        # print("Done.")
        # generate jaccard index matrix for these matrices
        # (Note that sparse is True, because many columns are 0, because of the step above)
        print("    Jaccard index matrix does not exist yet, generating....")
        # jaccard = dp.get_jaccard_index_matrix(matrix_1_w_empty_cols, matrix_2_with_empty_cols,
        #                                       sparse=True)
        jaccard_with_duplicates = get_jaccard_index_matrix(matrix_1, matrix_2)
        print("    Done.")

        # save the file
        np.save(filename, jaccard_with_duplicates)
    
    # Remove all entries of the jaccard matrix where the row and column correspond to the same user
    print("    Removing entries corresponding to pairs of the same user....")
    jaccard_without_duplicates, removed_entries = remove_entries_for_duplicate_user_pairs(
        jaccard_with_duplicates, 
        users_to_consider_1,
        users_to_consider_2,
        nan_instead=True)
    
    del jaccard_with_duplicates
    print("    Done.")
    # calculate the median of the jaccard indices, excluding the diagonal
    print("    Calculating the median of the jaccard index matrix....")
    median_jaccard = np.nanmedian(jaccard_without_duplicates)
    print("    Done.")
    del jaccard_without_duplicates
    gc.collect()
    return median_jaccard


def get_mean_jaccard_value_table(video_author_matrices: Dict[str, scipy.sparse.csc_matrix],
                                 users_in_clusters: Dict[str, pd.Series],
                                 jaccard_filenames: Dict[str, str],
                                 mean_jaccard_value_table_filename: str,
                                 mode: str='mean') -> pd.DataFrame:
    """
    Assembles a table for given clusters, where the value is the mean or median jaccard index between users
    in the two clusters.

    Will calculate jaccard index matrices for the clusters if it doesn't exist yet.

    Note on the dict keys: video_author_matrices and users_in_clusters must share the same keys. 
    Furthermore, all keys of jaccard_filenames must be a combination of the keys of the other dicts, 
    in the format "name1_name2", such as "cnn_fox" or "cnn_cnn".

    Args:
        video_author_matrices: Dict with the video_author matrices of all clusters. Will be used to
            calculate the jaccard index matrices (Note that if the jaccard index matrices already exists,
            the video author matrices aren't actually used, but they still have to be given here.)
        users_in_clusters: Dict with Series containing the users contained in each cluster (i.e., this is
            data about which user id each column in the video author matrices correspond to)
        jaccard_filenames: Dict with filenames for the jaccard index matrices. If such a file is found, 
            it will be used directly, if not, it will be calculated using the given video author matrices 
            and saved under this filename. Note that this dict must contain filenames both for all 
            jaccard matrices for one cluster with itself AND for jaccard matrices for each cluster with 
            all other clusters.
        mean_jaccard_value_table_filename: The mean jaccard value table will be saved here. If this file
            already exists, raises FileExistsError
        mode: can be "mean" (default) or "median"

    Returns:
        DataFrame with cluster names as rows and columns, and the mean or median jaccard index between users in #
        these clusters as values
    
    Side effects:
        Calculates needed jaccard matrices if these don't exist yet. Saves the mean jaccard table to csv
    
    Raises:
        ValueError: If the keys in the given dicts do not have the right format (see above)
        FileExistsError: If a file exists under the given filename for the mean jaccard table
    """

    if users_in_clusters.keys() != video_author_matrices.keys():
        raise ValueError("Keys do not match in given dicts 'video_author_matrices' and'users_in_clusters'")
    else:
        for key in jaccard_filenames.keys():
            if not np.array([key_i in users_in_clusters.keys() for key_i in key.split('_')]).all():
                raise ValueError("Keys of given dict 'jaccard_filenames' do not match the given keys "
                                 "in the other dicts.")

    # initiate empty array
    mean_jaccard_array = np.zeros((len(users_in_clusters), len(users_in_clusters)), dtype=np.float32)

    # go through all combinations of clusters 
    # (but skip (cluster2, cluster1) of (cluster1, cluster2) has already been calculated)
    for i, name1 in enumerate(list(users_in_clusters.keys())):
        for j, name2 in enumerate(list(users_in_clusters.keys())[i:]):

            print(f"Getting Jaccard matrix for cluster {name1} with cluster {name2}...")

            # Case 1: We are comparing a cluster to itself
            if name1 == name2:
                # call the corresponding function to get the mean jaccard index and save to the array
                if mode == 'mean':
                    mean_jaccard = get_jacc_between_same_cluster_and_get_mean(
                        video_author_matrices[name1],
                        jaccard_filenames[name1 + "_" + name1])
                elif mode == 'median':
                    mean_jaccard = get_jacc_between_same_cluster_and_get_median(
                        video_author_matrices[name1],
                        jaccard_filenames[name1 + "_" + name1])
                else:
                    raise ValueError("'mode' parameter must be either 'mean' or 'median'.")
                
                mean_jaccard_array[i, i] = mean_jaccard

            # Case 2: We compare two different clusters to each other
            else:
                # call the corresponding function to get the mean jaccard index and save to the array
                if mode == 'mean':
                    mean_jacc = get_jacc_between_two_clusters_and_get_mean(
                        video_author_matrices[name1],
                        video_author_matrices[name2],
                        filename=jaccard_filenames[name1 + "_" + name2],
                        users_to_consider_1=users_in_clusters[name1],
                        users_to_consider_2=users_in_clusters[name2])
                elif mode == 'median':
                    mean_jacc = get_jacc_between_two_clusters_and_get_median(
                        video_author_matrices[name1],
                        video_author_matrices[name2],
                        filename=jaccard_filenames[name1 + "_" + name2],
                        users_to_consider_1=users_in_clusters[name1],
                        users_to_consider_2=users_in_clusters[name2])
                else:
                    raise ValueError("'mode' parameter must be either 'mean' or 'median'.")
                
                mean_jaccard_array[i, j+i] = mean_jacc
            print("Done.")

    # convert the array to a dataframe and label the rows and columns
    df_mean_jaccard_values = pd.DataFrame(mean_jaccard_array, 
                                          index=users_in_clusters.keys(),
                                          columns=users_in_clusters.keys())

    gc.collect()

    # save the mean jaccard table as a csv file
    df_mean_jaccard_values.to_csv(mean_jaccard_value_table_filename, mode='x')
    if mode == 'mean':
        print(f"Generated and saved mean jaccard index value table under {mean_jaccard_value_table_filename}.")
    elif mode == 'median':
        print(f"Generated and saved median jaccard index value table under {mean_jaccard_value_table_filename}.")

    return df_mean_jaccard_values


def plot_histograms_of_jaccard_indices_from_matrix(jaccard_index_matrix: np.ndarray,
                                                   fig_linlog_filepath: str,
                                                   fig_loglog_filepath: str,
                                                   color: str,
                                                   cluster_name_1: str,
                                                   cluster_name_2: Optional[str] = None,
                                                   users_in_rows: Optional[pd.Series] = None,
                                                   users_in_cols: Optional[pd.Series] = None,
                                                   show: bool = True):  # add filepath thing here, then try the function on something, then stop for today
    """
    Plots histograms displaying the distribution of jaccard indices for all pairs in a jaccard index matrix.

    Once a lin-log plot, and one a log-log plot. Saving to file is mandatory, showing the plot is optional.

    Args:
        jaccard_index_matrix: A matrix with all jaccard indices
        fig_linlog_filepath: filepath where the lin log plot is saved
        fig_loglog_filepath: filepath where the log log plot is saved
        color: color of the graphs
        cluster_name_1: name of the first, or only cluster which the given jaccard matrix describes
        cluster_name_2: optional name of the second cluster which the given jaccard matrix describes
        users_in_rows: optional list of user ids corresponding to the rows in the given jaccard matrix.
            Only required when the jaccard matrix describes two different clusters
        users in cols: the same but for the user ids corresponding to the columns in the given jaccard matrix
        show: if True (default), will show the plot. Otherwise it is only saved
    
    """

    print("Jaccard index matrix loaded.")
    print(f"Jaccard dtype is {jaccard_index_matrix.dtype}")
    jaccard_index_matrix = jaccard_index_matrix.astype(np.float16)
    print(f"Jaccard dtype is {jaccard_index_matrix.dtype}")
    if cluster_name_2 is None or cluster_name_2 == cluster_name_1:
        
        for i in range(jaccard_index_matrix.shape[0]):
            jaccard_index_matrix[i, i:] = np.nan  # set everything in upper tri including diagonal to nan
            # this excludes for example the pair (user 1, user 1) and the pair (user 2, user 1) when 
            # the pair (user 1, user 2) has already been considered.
        
        values_in_non_duplicate_entries_of_jaccard_matrix = jaccard_index_matrix.flatten()
        
    
    else:
        jaccard_index_matrix = jaccard_index_matrix.flatten()
        
        for i, user1 in enumerate(users_in_rows.to_list()):
            for j, user2 in enumerate(users_in_cols.to_list()):
                if user1 == user2:
                    # set entries corresponding to the same user to nan
                    jaccard_index_matrix[i*len(users_in_cols) + j] = np.nan
                    
        values_in_non_duplicate_entries_of_jaccard_matrix = jaccard_index_matrix
        

    fig_linlog = plt.figure(figsize=(10, 6))
    values_array = plt.hist(values_in_non_duplicate_entries_of_jaccard_matrix, alpha=0.7,
                            color=color, log=True, bins=100)
    if cluster_name_2 is None or cluster_name_2 == cluster_name_1:
        plt.title(f'Distribution of Jaccard indices of pairs of users from cluster {cluster_name_1}')
    else:
        plt.title(f'Distribution of Jaccard indices of pairs of users from cluster {cluster_name_1} and {cluster_name_2}')
    plt.xlabel('Jaccard index')
    # plt.ylim(0, ylim)
    plt.ylabel('Number of Pairs')
    plt.grid(True)
    plt.tight_layout()
    

    plt.savefig(fig_linlog_filepath)

    if show:
        plt.show()
    
    del fig_linlog
    print(gc.collect())

    fig_loglog = plt.figure(figsize=(10,6))
    
    
    plt.loglog(values_array[1][1:], values_array[0], color=color)

    # sns.histplot(values_in_non_duplicate_entries_of_jaccard_matrix, alpha=0.7, color=color, log=(True, True), bins=100)
    if cluster_name_2 is None or cluster_name_2 == cluster_name_1:
        plt.title(f'Distribution of Jaccard indices of pairs of users from cluster {cluster_name_1}')
    else:
        plt.title(f'Distribution of Jaccard indices of pairs of users from cluster {cluster_name_1} and {cluster_name_2}')
    plt.xlabel('Jaccard index')
    # plt.ylim(0, ylim)
    plt.ylabel('Number of Pairs')
    plt.grid(True)
    plt.tight_layout()
    
    plt.savefig(fig_loglog_filepath)
    if show:
        plt.show()

    
    del fig_loglog
    print(gc.collect())


def create_jaccard_index_histograms_for_all_cluster_combinations(
        video_author_matrices: Dict[str, scipy.sparse.csc_matrix],
        users_in_clusters: Dict[str, pd.Series],
        jaccard_filenames: Dict[str, str],
        base_filename: str,
        show: bool | List[bool] = False) -> pd.DataFrame:
    """
    Creates histograms for the jaccard indics for all possible combinations of clusters given.

    Will calculate jaccard index matrices for the clusters if it doesn't exist yet.

    Note on the dict keys: video_author_matrices and users_in_clusters must share the same keys. 
    Furthermore, all keys of jaccard_filenames must be a combination of the keys of the other dicts, 
    in the format "name1_name2", such as "cnn_fox" or "cnn_cnn".

    Args:
        video_author_matrices: Dict with the video_author matrices of all clusters. Will be used to
            calculate the jaccard index matrices (Note that if the jaccard index matrices already exists,
            the video author matrices aren't actually used, but they still have to be given here.)
        users_in_clusters: Dict with Series containing the users contained in each cluster (i.e., this is
            data about which user id each column in the video author matrices correspond to)
        jaccard_filenames: Dict with filenames for the jaccard index matrices. If such a file is found, 
            it will be used directly, if not, it will be calculated using the given video author matrices 
            and saved under this filename. Note that this dict must contain filenames both for all 
            jaccard matrices for one cluster with itself AND for jaccard matrices for each cluster with 
            all other clusters.
        base_filename: path and common part of the filenames of the plots. The plots will be saved under
            base_filename + name1 + name2 + .png            
        show: either a single boolean value or a list of bools. If ith value is True, displays the ith plot. 
            If True for all, will result in very many plots shown. Default is False

    Returns:
        DataFrame with cluster names as rows and columns, and the mean jaccard index between users in #
        these clusters as values
    
    Side effects:
        Calculates needed jaccard matrices if these don't exist yet. Saves the mean jaccard table to csv
    
    Raises:
        ValueError: If the keys in the given dicts do not have the right format (see above)
        FileExistsError: If a file exists under the given filename for the mean jaccard table
    """
    if users_in_clusters.keys() != video_author_matrices.keys():
        raise ValueError("Keys do not match in given dicts 'video_author_matrices' and'users_in_clusters'")
    else:
        for key in jaccard_filenames.keys():
            if not np.array([key_i in users_in_clusters.keys() for key_i in key.split('_')]).all():
                raise ValueError("Keys of given dict 'jaccard_filenames' do not match the given keys "
                                 "in the other dicts.")

    # load colormap
    cmap = colormaps['Set3']

    # go through all combinations of clusters 
    # (but skip (cluster2, cluster1) of (cluster1, cluster2) has already been calculated)
    for i, name1 in enumerate(list(users_in_clusters.keys())):
        for j, name2 in enumerate(list(users_in_clusters.keys())[i:]):

            print(f"Getting Jaccard index histograms for cluster {name1} with cluster {name2}...")
            
            if isinstance(show, List):
                show_this_plot = show[i*len(users_in_clusters)+j]
            else:
                show_this_plot = show

            if (os.path.isfile(base_filename + name1 + '_' + name2 + '_linlog.png') 
                and os.path.isfile(base_filename  + name1 + '_' + name2 + '_loglog.png')):
                print("Plots are already existing, skipping...")
                continue

            plot_histograms_of_jaccard_indices_from_matrix(jaccard_index_matrix=np.load(jaccard_filenames[name1 + "_" + name2]),
                                                           fig_linlog_filepath=base_filename + name1 + '_' + name2 + '_linlog.png',
                                                           fig_loglog_filepath=base_filename  + name1 + '_' + name2 + '_loglog.png',
                                                           color=cmap(i*len(users_in_clusters)+j),
                                                           cluster_name_1=name1,
                                                           cluster_name_2=name2,
                                                           users_in_rows=users_in_clusters[name1],
                                                           users_in_cols=users_in_clusters[name2],
                                                           show=show_this_plot)
            
            print("Done.")

    print(f"Saved plots under file starting with {base_filename}.")
    gc.collect()




def get_jacc_mean_between_two_clusters(matrix_1: scipy.sparse.csc_matrix, 
                                               matrix_2: scipy.sparse.csc_matrix,
                                               users_to_consider_1: pd.Series,
                                               users_to_consider_2: pd.Series) -> float:
    """
    Get jaccard matrix between two different clusters, save it, and calculate the mean of the
    entries excluding the diagonal.

    Note: this function will take the two given matrices and put empty columns in them so that 
    they share the same column. 
    So if one matrix contains user 1 and user 3, and the other matrix conatins user 2 and user 3, 
    then the matrices are enlarged so that both contain user 1, 2 and 3, however the first matrix
    will have zeros for user 2, and the second matrix will have zeros for user 1.

    The reason for this is that then, all values for one user with themselves are on the diagonal,
    so that we can exclude them (because that will always have jaccard index 1).
    
    Args:
        matrix_1: the video user matrix for the first cluster
        matrix_2: the video user matrix for the second cluster
        users_to_consider_1: the authors contained in the first cluster (with original user ids)
        users_to_consider_2: the authors contained in the second cluster (with original user ids)

    Returns:
        Mean of the jaccard index for all pairs of users from the two clusters, excluding pairs of
        the same user

    Side effects:
        Calculates the jaccard matrix for these clusters and saves it, if not already existing.
    """


    # # add empty columns to the matrices, so that the same column in both matrices also corresponds
    # # to the same user
    # print("Getting the matrices with added empty columns....")
    # (matrix_1_w_empty_cols, 
    #  matrix_2_with_empty_cols) = dp.get_video_user_matrices_with_equal_columns(matrix_1, matrix_2, 
    #                                                                            users_to_consider_1, 
    #                                                                            users_to_consider_2)
    # print("Done.")
    # generate jaccard index matrix for these matrices
    # (Note that sparse is True, because many columns are 0, because of the step above)
        
    # jaccard = dp.get_jaccard_index_matrix(matrix_1_w_empty_cols, matrix_2_with_empty_cols,
    #                                       sparse=True)
    jaccard_with_duplicates = get_jaccard_index_matrix(matrix_1, matrix_2)
    print("Done.")

    
    # Remove all entries of the jaccard matrix where the row and column correspond to the same user
    print("Removing entries corresponding to pairs of the same user....")
    jaccard_without_duplicates, removed_entries = remove_entries_for_duplicate_user_pairs(
        jaccard_with_duplicates, 
        users_to_consider_1,
        users_to_consider_2)
    del jaccard_with_duplicates
    print("Done.")
    # calculate the mean of the jaccard indices, excluding the diagonal
    print("Calculating the mean of the jaccard index matrix....")
    mean_jaccard = jaccard_without_duplicates.sum(axis=None) / (jaccard_without_duplicates.size 
                                                                - removed_entries)
    print("Done.")
    del jaccard_without_duplicates
    gc.collect()
    return mean_jaccard


def percentage_users(sparse_matrix):
    """
    Get the percentage of users in a certain bubble that have commented under a specific video. 
    I.e. for each video, compute the perentage of users that have commented under it, divided by the total number of users withiin the bubble.


    
    Args:
        matrix: the video user matrix for the bubble you want to compute the percentage

    Returns:
        An array of percentage for each video 

    
    """
    row_percentage = sparse_matrix.sum(axis=1).A1 / sparse_matrix.shape[1]  
    return row_percentage



def word_interest(sparse_matrix, percentage, videos_new_pol:pd, word_interest:str, map ):
     """
    Find a word of interest in the title of the videos that a certain pourcentage of users has commented at (above acertain thresold).
    If a video has more than a certain thresold of its users that has commented under this video, then it is retrieving the title and looking for a specific word.
    The goal is to see if most of the videos as a specific subject (word) in common. 
    
    Args:
        sparse_matrix: the video user matrix for the bubble you want 
        percentage: percentage array of the sparse matrix, compute with percentage_users()
        videos_new_pol : dataframe of all the videos in news and politics and categories, need to be load previously from the csv file
        word of interest: word you want to look at within the videos
        map: the mapping you want to apply 

    Returns:
    The number of videos with the word of interest
    The dataframe with all the videos above the percentage with the word of interest, for more analysis purpose
    """
    #retrieve videos where at list one users in the cluser as commented under
     mask = percentage > 0
     sub_news_pol = videos_new_pol[mask]

    #filtered video that contain word of interest, with at least one user on the cluster that has commented under it
     filtered_df = sub_news_pol[sub_news_pol['title'].str.contains(word_interest, case=False, na=False)]

    #mapping users back to sparse matrix 
     video_ids_news_pol_int_mapping_new = map.loc[filtered_df.display_id]

    #looking for sparse matrix row we're interested in
     specific_rows = sparse_matrix[video_ids_news_pol_int_mapping_new.values, :]
     non_zero_columns = specific_rows.getnnz(axis=0) > 0  

     columns_with_non_zero_values = np.where(non_zero_columns)[0]

     number_colum = len(columns_with_non_zero_values)
     total_number_user = sparse_matrix.shape[1]
     percentage_total = number_colum/total_number_user

     return filtered_df, number_colum, total_number_user, percentage_total
