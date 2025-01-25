import numpy as np
import sys
from scipy.interpolate import interp1d

def Find_value(x:np.ndarray,y:np.ndarray,seed:float, find='y'):
    '''
    This function finds a value given its pair, as is:
        - Find 'x' given a 'y' value
        - Find 'y' given a 'x' value
        - Find the minimum 'y' value from the nearests
        - Find the average 'y' value from the nearests
    
    Parameters:
        - x (np.array): x set of values
        - y (np.array): y set of values
        - seed (float): seed value from which the indicated in "find" will be searched
        - find (str): Default: 'y', but can be 'x', 'y'
    Returns:
        - value (float): value found
    '''
    def check_range(value,data):
        if value > np.max(data) or value < np.min(data):
            raise ValueError('The seed value is outside the range of the data')
    if find=='x': # Finds 'x' given a 'y'
        check_range(seed,y)
        interp = interp1d(y, x, kind='linear', fill_value='extrapolate')
        return interp(seed)

    elif find=='y':
        check_range(seed,x)
        return np.interp(seed, x, y)
    else:
        raise ValueError('Insert \'x\' or \'y\' as \'find\' entry')