import pygsheets

def read_sheet(creds,file):
    gc = pygsheets.authorize(service_file=creds)
    sh = gc.open_by_key(file)
    wks = sh[0]
    
    return wks


def take_case(wks,persona):
    df = wks.get_as_df(start='A22')

    sheetidx = df.loc[df['ESTADO']==''].index[0]
    sheetrow = int(sheetidx + 23)

    wks.update_value('C'+str(sheetrow),persona)
    wks.update_value('B'+str(sheetrow),'Running')
    return df,sheetidx,sheetrow
        



def done_case(wks,sheetidx):
    df = wks.get_as_df(start='A22')
    wks.update_value('B'+str(int(sheetidx+23)),'Done')
        

def check_numeric(x):
    try:
        float(x)
        return True
    except:
        return False

def read_material(creds,file):
    gc = pygsheets.authorize(service_file=creds)
    sh = gc.open_by_key(file)
    wks = sh[0]
    df = wks.get_as_df()
    return df

def read_seismic(creds,file):
    gc = pygsheets.authorize(service_file=creds)
    sh = gc.open_by_key(file)
    wks = sh[0]
    df = wks.get_as_df(start='A1')
    return df

def get_id(filename,idfolder,drive):
    file_list = drive.ListFile({'q': "'{}' in parents and trashed=false".format(idfolder)}).GetList()
    #print(file_list)
    for file in file_list:
        if file['title'] == filename:
            id = file['id']
            break
    return id

def check_bool(id):
    if id == 'TRUE':
        return True
    elif id == 'FALSE':
        return False