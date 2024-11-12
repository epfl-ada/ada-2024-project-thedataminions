"""Functions for processing (big) datasets. Will be called from the notebook"""
import time
import pandas as pd
from typing import Tuple, Optional
import gc


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
                if not not save_every:
                    if (i+1) % save_every == 0:
                        result.to_csv(save +  f"__{i + 1 - save_every}_{i}.csv" + compress)  # save the df concatenated so far
                        result = pd.DataFrame()  # clear the df to avoid memory becoming too big
                        print(f"saved data under {save}__{i + 1 -save_every}_{i}.csv" + compress)
                        collector = gc.collect()
                        print(f"Collected {collector} garbages.")
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
                if not not save_every:
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

            if not not save and not save_every:
                result.to_csv(save + ".csv" + compress)
                print(f"saved data under {save}.csv" + compress)
            elif not not save_every:
                result.to_csv(save + f"__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)  # save the remaining data (if the number of chunks is not a multiple of save_every)
                print(f"saved data under {save}__{i + 1 - (i+1) % save_every}_{i}.csv" + compress)

            return result


def run_simple_function_on_chunks_save_csv(reader, fct, filename: str, every: int = 1,
                                           print_time: bool | Tuple = False):
    """
    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The function returns nothing, but saves the results into a single csv.

    Args:
        reader: iterator object that returns the chunks, you can get it for example by calling pd.read_csv(...., chunksize=something).
        filename: file path and name, including the ending .csv, and optionally an ending for compression (such as .gz)
        every: will save every <every> chunks. Default is 1 (saves after every chunk)
        print_time: If False (default), does not print time data. If True, prints the average time per chunk.
            If a tuple with two entries is given, where the fist is the chunk size used in the reader, 
            and the second is the total number of entries in the dataset,
            then additional data about estimated time left is printed.
        
    """
    
    with reader:
        
        # result = pd.DataFrame()
        if not print_time:
            header = True  # when writing to the csv the first time, include header
            result = pd.DataFrame()
            for i, chunk in enumerate(reader):
                result = pd.concat([result, fct(chunk)])
                if (i + 1) % every == 0:
                    result.to_csv(filename, header=header, mode='a')
                    header = False  # when appending new rows to the csv, don't include header
                    print("Appended new rows to the csv file")
                    del result
                    gc.collect()
                    result = pd.DataFrame()
            
            if not result.empty:  # if there is something left to apppend to the csv, do that now
                result.to_csv(filename, header=header, mode='a')
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
                    result.to_csv(filename, header=header, mode='a')
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
                result.to_csv(filename, header=header, mode='a')
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
                    result.to_csv(filename, header=header, mode='a')
                    time_save_end = time.time()
                    print("Appended new rows to the csv file")                   
                    print(f"Time spent saving: {time_save_end-time_save_start} secs\n")
                    header = False  # when appending new rows to the csv, don't include header
                    del result
                    gc.collect()
                    result = pd.DataFrame()
                time_end = time.time()
                processed_entries = (i+1)*print_time[0]
                entries_left = print_time[1] - processed_entries
                avg_time_per_chunk = (time_end-time_start_global)/(i+1)
                print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk /(print_time[0]* 60):.3f} minutes left.\n")

            if not result.empty:  # if there is something left to apppend to the csv, do that now
                time_save_start = time.time()
                result.to_csv(filename, header=header, mode='a')
                time_save_end = time.time()
                print("Appended last rows to the csv file")                   
                print(f"Time spent saving: {time_save_end-time_save_start} secs")
            return 

def run_simple_function_on_chunks_sum(reader, fct, sum_by_column: str, print_time: bool | Tuple = False) -> pd.DataFrame:
    """
    DON'T USE YET, DOESN'T WORK!!

    Runs a given function that works on a (single) dataframe, but runs it on the given reader. 
    The function returns a single dataframe with the results from all chunks summed, 
    i.e., if in two chunks similar entries are found, a single row is created with the values summed.

    Args:
        reader: iterator object that returns the chunks, you can get it for example by calling pd.read_csv(...., chunksize=something).
        sum_by_column: specify the column where it is to be looked for similar entries.
        print_time: If False (default), does not print time data. If True, prints the average time per chunk.
            If a tuple with two entries is given, where the fist is the chunk size used in the reader, 
            and the second is the total number of entries in the dataset,
            then additional data about estimated time left is printed.
    """
    
    with reader:
        
        result = pd.DataFrame()
        if not print_time:
            for chunk in reader:
                result = pd.concat([result, fct(chunk)])
            return result
        elif print_time is True:
            time_start_global = time.time()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                time_end = time.time()
                print(f"{(time_end-time_start_global)/(i+1):.3f} secs per chunk on average.")
            return result
        else:
            time_start_global = time.time()
            for i, chunk in enumerate(reader):
                print(f"Going through chunk {i}...")
                result = pd.concat([result, fct(chunk)])
                time_end = time.time()
                processed_entries = (i+1)*print_time[0]
                entries_left = print_time[1] - processed_entries
                avg_time_per_chunk = (time_end-time_start_global)/(i+1)
                print(f"The first {processed_entries} entries have been processed. {entries_left} left.")
                print(f"{avg_time_per_chunk:.3f} secs per chunk on average. Meaning  {entries_left * avg_time_per_chunk /(print_time[0]* 60):.3f} minutes left.")


            return result


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
        

