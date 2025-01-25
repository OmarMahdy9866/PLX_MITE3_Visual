def Significant_fig(x,n):
    
    '''
    This function returns the "n" significant figure of a "x" number
    '''
    st='%.'+str(n)+'g'
    return float('%s' % float(st % x))