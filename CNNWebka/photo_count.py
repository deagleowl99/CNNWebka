import os

def photo_count(path):
    filenum = 0  
    for lists in os.listdir(path):
        sub_path = os.path.join(path, lists)
        if os.path.isfile(sub_path):
            filenum = filenum+1
    return filenum