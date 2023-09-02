#%%
import os
import zipfile
# %%

directory = '..\\..\\data\\raw'
# %%
class GetData:
    def __init__(self, directory) -> None:
        self.directory = directory
        pass
    
    def find_zip_files(self):
        self.zip_files = [file for file in os.listdir(self.directory) if file.endswith('.zip')]
        
    def extract_zip_files(self):
        for file in self.zip_files:
            zip_filepath = os.path.join(self.directory,file)
            with zipfile.ZipFile(zip_filepath ,'r') as zip:
                zip_files = zip.namelist()
                for zip_file in zip_files:
                    destination_path = os.path.join(self.directory, zip_file)
                    if not os.path.exists(destination_path):
                        zip.extract(zip_file,path = self.directory)
                        print(f"Extraído: {zip_file}")
                    else:
                        print(f"Arquivo já existe: {zip_file}")
                
    def process_zip_files(self):
        self.find_zip_files()
        self.extract_zip_files()

