import os


def check_folder(folder_name):
    path="Output/%s//" % folder_name
    if not os.path.exists(path):
        os.makedirs(path)
    return path