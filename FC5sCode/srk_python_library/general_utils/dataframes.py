from .notification import *

def rename_columns (Data,Variables):
    for var in Variables: # Recorre la lista variables para renombrar o crear columnas según el nombre del primer elemento de la lista
        original_name = var[1]
        new_name = var[0]
        if original_name in Data.columns:
            Data.rename(columns={original_name: new_name}, inplace=True)
        else:# new_name not in Data.columns:
            warning('WARNING: The column "{vrb1}" assigned for "{vrb0}" was not found'.format(vrb1=var[1],vrb0=var[0]))
            Data[new_name] = None