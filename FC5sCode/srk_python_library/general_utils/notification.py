class bcolors:
    HEADER = '\033[95m'
    BLACK = '\033[90m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[31m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def warning(text):

    '''
    This function simply prints text in red.

    Parameters:
        text (str): The text you want to print in red.

    Returns:
        None
    '''
    print(bcolors.WARNING,text,bcolors.ENDC)
    

def header(text):

    print(bcolors.BLACK,bcolors.BOLD,text,bcolors.BLACK,bcolors.BOLD)
def notification_text(text):

    '''
    This function simply prints text in red.

    Parameters:
        text (str): The text you want to print in red.

    Returns:
        None
    '''
    print(bcolors.OKGREEN,text,bcolors.OKGREEN)
    

