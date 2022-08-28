import os

def create_data_collection_folders(DATA_PATH,actions,no_sequences):
    for action in actions:
        for sequence in range(no_sequences):
            try:
                os.makedirs(os.path.join(DATA_PATH,action,str(sequence)))
            except:
                pass
    
    
    
