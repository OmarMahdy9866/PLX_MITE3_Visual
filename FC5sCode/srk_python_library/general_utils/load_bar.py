from ipywidgets import IntProgress
from IPython.display import display, clear_output

import time

def set_load_bar(max_count):
    '''
    This function creates a progress bar widget. It does not update the progress.

    Parameters:
        max_count (int): Total number of cases.

    Returns:
        f (ipywidget object): The progress bar.
    '''

    f = IntProgress(min=0, max=max_count) # instantiate the bar
    display(f) # display the bar
    return f


def update_load_bar(f,value):
    '''
    This function updates the progress bar.

    Parameters:
        f (ipywidget object): The progress bar to update.
        value (int): The value to update the progress bar.

    Returns:
        f (ipywidget object): The updated progress bar.
    '''

    f.value = value # valor que adopta la barra
    display(f) # display the bar
    

    return f
 