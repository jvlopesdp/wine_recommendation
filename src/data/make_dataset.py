#%%
import pandas as pd
import numpy as np
import os

class CreateDataframe:
    
    def __init__(self,input_directory, output_directory) -> None:
        self.input_directory = input_directory
        self.output_directory = output_directory
        pass   
    
    #Melhorar função -> df temp e concat
    def create_dataframes(self):
        df_list = []
        csv_files = [file for file in os.listdir(self.input_directory) if file.endswith('.csv')]
        for csv_file in csv_files:
            csv_file_directory = os.path.join(self.input_directory, csv_file)
            df_name = 'df_'+csv_file.replace('.csv','')
            df_name = pd.read_csv(csv_file_directory,index_col=None)
            df_name.drop(df_name.columns[0], axis=1, inplace=True)
            df_list.append(df_name)
        concatenated_df = pd.concat(df_list,ignore_index = True, axis = 0)
        concatenated_df.to_parquet(os.path.join(self.output_directory, "concatenated_dfs.parquet"), index=False)
