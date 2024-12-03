"""Functions for processing (big) datasets. Will be called from the notebook"""
import time
import pandas as pd
from typing import Tuple, Optional
import gc
import os
import numpy as np
import scipy.sparse

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
