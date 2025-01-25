import pandas as pd
import functools
import numpy as np
from plxscripting.easy import *
from ..utils.varinfo import load_varinfo
import ipywidgets as widgets
import re
from ..utils.project import NodesPerElement
from tqdm.notebook import tnrange
from ..utils.connect import *
import os
from ...general_utils.notification import *
from ..utils.version import *
np.seterr(divide='ignore', invalid='ignore')

def extract_ContourMaps(PLX_ports, outputfolder = 'Data',varpath='info\Var_ver2023.1.xlsx',default_value=False):
    '''  
    This function allows the extraction of selected variables from selected phases of a specific PLAXIS model.
    It creates an interactive tab widget for the user to select which variables to download and which phases to download them from.  

    Parameters:  
        PLX_ports (dict): dictionary containing the PLAXIS connection (see function plaxis_connect).  
        varpath (str): path to the file containing information about the variables to be extracted from the PLAXIS Output (default: info\Var.csv).  

    Returns:  
        tab_nest (widgets.Tab()): a widget with tabs to download the information.  
    '''
    # Redefino los puertos de input y output
    g_i = PLX_ports['g_i']
    g_o = PLX_ports['g_o']

    # Detecto las phases del modelo
    phaseinfo = detect_phases(g_i,default_value=default_value)

    # Detecto las variables que se pueden descargar desde el PLAXIS Output
    varinfo = load_varinfo(g_o,varpath,default_value=default_value)

    # Creo el widget de pestañas
    tab_nest = widgets.Tab()

    # Indico los nombres de las pestañas (tienen que ser cortitos)
    tab_names = ['STEP 1 - PHASES','STEP 2 - VARS','STEP 3 - GO']

    # Lleno las pestañas
    tab_nest.children = [widget_list(phaseinfo['widgets'].to_list(),'Select phases to export:'),
                         widget_list(varinfo['widgets'].to_list(),'Select variables to export:'),
                         widget_go(PLX_ports, phaseinfo, varinfo, outputfolder)]
    
    # Lleno los nombres de pestañas
    [tab_nest.set_title(i, title) for i, title in enumerate(tab_names)]   

    return tab_nest

def detect_phases(g_i,default_value=False):
    '''  
    This function detects the phases that have a model using the PLAXIS input.  

    Parameters:  
        g_i (PlxProxyGlobalObject): Object that represents the link between Python and the PLAXIS input.  
        default_value (boolean): default value of the checkbox.  

    Returns:  
        phaseinfo (dict): A dictionary with all the phases found in the PLAXIS input.  
    '''

    # Creo el diccionario
    phaseinfo = {}

    # Creo los keys que van a contener las variables
    phaseinfo['name'] = []
    phaseinfo['id'] = []
    phaseinfo['widgetid'] = []
    phaseinfo['widgets'] = []

    # Para cada phase
    for i,phase in enumerate(g_i.Phases):

        # Nombre de la phase
        phasenumber = str(phase.Name)
        phaseinfo['name'].append(phasenumber)

        # Identificacion de la phase
        phaseinfo['id'].append(str(phase.Identification))
        # id del widget
        if i == 0:
            widgetid = 0
        else:
            widgetid = int(phasenumber.split('_')[-1])
        phaseinfo['widgetid'].append(widgetid)
        
        # widget de la phase que se guarda en el diccionario paseinfo
        phaseinfo['widgets'].append(widgets.Checkbox(value=default_value,
                                             description=str(phase.Identification)+' - ['+str(phase.Name)+']',
                                             disabled=False,
                                             indent=False))
    phaseinfo = pd.DataFrame(phaseinfo)
    phaseinfo = phaseinfo.sort_values(by=['widgetid'])
    return phaseinfo

def load_varinfo(g_o,varpath,default_value = False): 
    '''  
    This function loads all the variables that can be downloaded from PLAXIS and that are found in the file located at varpath.
    This function returns a dictionary with the name of each variable, the dictionary, logo, unit, and point type that should be used to extract information from the PLAXIS Output.  

    Parameters:  
        g_o (PlxProxyGlobalObject): Object representing the link between Python and the PLAXIS Output.  
        varpath (str): path to the file containing information about the variables to be extracted from the PLAXIS Output (default: info\Var.csv).  
        default_value (Boolean): value that the Checkboxes representing the variables have by default (default: False).  

    Returns:  
        Varinfo (dict): A dictionary with all the information about each variable that can be extracted from the PLAXIS Output.  
    '''

    # Levanto la data del CSV
    # Vardata = np.loadtxt(varpath, delimiter=';', encoding='utf-8', dtype=str)[1:]  
    Vardata = pd.read_excel(varpath)

    # Creo un diccionario 
    Varinfo = {}

    # Creo los keys que van a contener las variables
    Varinfo['logo'] = []
    Varinfo['name'] = []
    Varinfo['unit'] = []
    Varinfo['command'] = []
    Varinfo['point'] = []
    Varinfo['widgets'] = []
    
    # Defino variables obligatorias
    obligatory_vars = ['X stresspoint coordinate','Y stresspoint coordinate','X node coordinate',
                               'Y node coordinate','Material ID','Steady-state pore pressure',
                            #    'Incremental displacement',
                               'Horizontal displacement','Vertical displacement',]
    # Para cada variable
    for i in range(len(Vardata)):

        # Incluyo el logo en el subdiccionario
        Varinfo['logo'].append(Vardata['simbolo'].values[i])

        # Incluyo el nombre
        Varinfo['name'].append(Vardata['Nombre'].values[i])
        
        # Incluyo la unidad en el subdiccionario
        if Vardata['unidad'].values[i] == '-':
            Varinfo['unit'].append(Vardata['unidad'].values[i])
        else:
            Varinfo['unit'].append(eval(Vardata['unidad'].values[i]).replace('"',''))

        # Incluyo el comando en el subdiccionario
        Varinfo['command'].append(Vardata['comando'].values[i])

        # Incluyo el tipo de punto: StressPoint o Nodo
        Varinfo['point'].append(Vardata['tipo'].values[i])

        # Agrego al diccionario Varinfo el checkbox
        Varinfo['widgets'].append(widgets.Checkbox(value=default_value,
                                           description=Varinfo['logo'][-1]+' - '+Varinfo['name'][-1]+' ['+Varinfo['unit'][-1]+']',
                                           disabled = False,
                                           indent = False))
        if Varinfo['name'][-1] in obligatory_vars:
            Varinfo['widgets'][-1].value = True
            Varinfo['widgets'][-1].disabled = True
        
    Varinfo = pd.DataFrame(Varinfo)
    return Varinfo

def widget_go(PLX_ports, phaseinfo, varinfo, outputfolder, texto='Write a file name (without file extension):'):
    '''  
    This function is a collection of widgets. On one hand, it allows indicating the name that the file containing
    the downloaded information from PLAXIS Output will have. On the other hand, it has a button that calls the function that starts downloading the data.  

    Parameters:  
        texto (str): text that is indicated below the name of the tab.  

    Returns:  
        VBOX (widgets.VBox): a widget that contains a place to write the name of the file and a button to download the information from PLAXIS.  
    '''
    # widget del texto
    text = widgets.HTML(value="<b>"+texto+"</b>")

    # widget para poder anotar el nombre del archivo
    filename = widgets.Text(value='Write here...')

    # widget que representa el boton de extraer
    button = widgets.Button(description='Extract baby!',button_style='danger')

    # cuando se clickea, se activa la funcion principal de descargar
    button.on_click(functools.partial(extract_DataContour, PLX_ports, phaseinfo, varinfo, filename, outputfolder))
    
    # widget VBox que contiene todo lo definido previamente
    VBOX = widgets.VBox([text,filename,button])

    return VBOX

def widget_list(var,texto):
    '''  
    This function lists the Checkbox variables in 3 columns grouped in an array.  

    Parameters:  
        var (array): array with the previously defined checkboxes.  
        texto (str): text that is indicated above the columns.  

    Returns:  
        VBOX (widgets.VBox): a widget that contains the 3 columns with the checkboxes.  
    '''

    # Defino k como un tercio de la longitud del aray
    k = len(var)/3

    # la cajita de la izquierda tiene el primer tercio de checkboxes
    left_box = widgets.VBox(var[0:int(k)])

    # la del medio tiene el segundo tercio
    mid_box = widgets.VBox(var[int(k):int(k*2)])

    # la de la derecha tiene el ultimo tercio
    right_box = widgets.VBox(var[int(k*2):])
    
    # agrupo cada cajita en un widget HBox
    variables = widgets.HBox([left_box, mid_box,right_box])

    # creo un texto para poner arriba
    text = widgets.HTML(value="<b>"+texto+"</b>")

    # agrupo el texto y el widget HBox que contiene todos los checkboxes
    VBOX = widgets.VBox([text,variables])

    return VBOX

def extract_DataContour(PLX_ports, phaseinfo, varinfo, filename, outputfolder, button):
    '''  
    This function extracts the PLAXIS variables in the selected phases and saves them to a file.  

    Parameters:  
        PLX_ports (dict): dictionary with the information of the PLAXIS ports (see function plaxis_connect()).  
        phaseinfo (dict): dictionary with the information of the PLAXIS phases (see function detect_phases()).  
        varinfo (dict): dictionary with the information of the PLAXIS variables (see function load_varinfo()).  
        filename (str): name of the file that will have the database.  
        button (widget): button widget (useless).  

    Returns:  
        None  
    '''
    
    # Redefino los puertos de PLAXIS
    g_i = PLX_ports['g_i']
    g_o = PLX_ports['g_o']
    s_i = PLX_ports['s_i']

    # inicializo el dataframe
    data = pd.DataFrame()
    
    # Extraigo la data de los materiales
    materials = export_materials(PLX_ports)

    
    # Extraigo la cantidad de nodos por elementos
    typeelement = pd.DataFrame({'typeelement':[NodesPerElement(g_i)]})
    
    # Uno todos los df en uno solo
    Extra_data = pd.concat([materials.drop(['id','DrainageType'],axis=1).rename(columns={'name':'mat_name'}),
                            varinfo[['logo','name','unit','point']].rename(columns={'name':'var_name'}),
                            phaseinfo[['name','id']].rename(columns={'name':'phase_name','id':'phase_id'}),
                            typeelement],
                           axis=0, sort=False,ignore_index=True)

    non_porous_id = materials.index[materials['DrainageType'] == 'NonPorous']  
    for i_non_porous in range(len(non_porous_id)):
        print('The material "'+Extra_data['mat_name'].iloc[i_non_porous+1]+'" is Non-Porous')
    
    # inicializo otras variables
    varnamepack = []

    # Para cada phase
    for phaseid in tnrange(len(phaseinfo)):

        # Si seleccionaste la phase
        if phaseinfo['widgets'][phaseid].value == True:
            # selecciono la phase del output
            phase = get_equivalent(g_i.Phases[phaseid], g_o)

            # nombre de la phase
            phasename = phaseinfo['name'][phaseid]
            
            for check_non_porous in range(1): # Toda esta parte es para chequear si hay un material non-porous
                stressarraysize = 0 # Size that stresspoint arrays must have (needed to fill NonPorous materials arrays)
                varname = 'Material ID'
                varcommand = varinfo.loc[varinfo['name']==varname]['command'].iloc[0]
                point = 'stresspoint'
                varcommand_separated = re.split('{|}',varcommand)
                command = varcommand_separated[1]
                command = 'np.fromstring(g_o.getresults(phase,'+command+', point).echo()[1:-1], dtype=float, sep=\',\')'
                
                materialid_df = pd.DataFrame({varname:eval(command)})

                if old_plaxis_version(s_i):
                    materialid_df[varname] = materialid_df[varname]-1 # Esto es porque versiones anteriores a 2023.2 arrancan desde el 1 y no desde el 0
                
                non_porous_index = []
                for indx in non_porous_id:
                    non_porous_index+=materialid_df.index[materialid_df[varname]==indx].to_list()
                non_porous_index.sort()
                value_to_insert = np.nan
                
                new_non_porous_index = []
                cont=0
                for indx in non_porous_index:
                    new_non_porous_index.append(indx-cont)
                    cont+=1
            
            # Para cada variable
            for variableid in range(len(varinfo['widgets'])):

                # Si seleccionaste la variable
                if varinfo['widgets'][variableid].value == True:
                    # informacion de la variable
                    varcommand = varinfo['command'][variableid]
                    point = varinfo['point'][variableid]
                    varname = varinfo['name'][variableid]
                    varnamepack.append(varname)

                    # Si el comando es complejo, primero lo separo
                    varcommand_separated = re.split('{|}',varcommand)

                    # Lo rearmo, para así poder evaluarlo y extraer la data de una
                    varcommand_prepared = ''
                    for command in varcommand_separated:
                        if command == '':
                            continue
                        elif command.startswith('g_o'):
                            varcommand_prepared+= 'np.fromstring(g_o.getresults(phase,'+command+', point).echo()[1:-1], dtype=float, sep=\',\')'
                        else:
                            varcommand_prepared+= command

                    # nombre de la columna del dataframe
                    columnname = str(phasename) + '_' + str(varname)

                    # guardo la data en un df
                    data_ind = pd.DataFrame()
                    data_ind[columnname] = eval(varcommand_prepared)
                    
                    if np.isinf(data_ind[columnname]).any():
                        data_ind[columnname].replace([np.inf, -np.inf], np.nan, inplace=True)
                    
                    if 'X stresspoint coordinate' in columnname:
                        stressarraysize = data_ind.shape[0]
                        
                    if data_ind.shape[0] < stressarraysize and point=='stresspoint':
                        
                        df_update = pd.DataFrame({columnname: [value_to_insert]}, index=new_non_porous_index)
                        data_ind = pd.concat([df_update, data_ind], axis=0)
                        data_ind.sort_index(inplace=True,ignore_index=True)
                    
                    if varname == 'Material ID':
                        if old_plaxis_version(s_i):
                            data_ind[columnname] = data_ind[columnname]-1 # Esto es porque versiones anteriores a 2023.2 arrancan desde el 1 y no desde el 0

                    # lo anexo al dataframe principal
                    data = pd.concat([data,data_ind],axis=1)
    
    # Concateno el df de información general con el de data
    data = pd.concat([data,Extra_data],axis=0, sort=False,ignore_index=True)
    
    if not os.path.exists(outputfolder):  
        # Si no existe, creamos la carpeta  
        os.makedirs(outputfolder)
    # Guardo la data en formato parquet
    data.to_parquet(outputfolder+'\\'+str(filename.value)+'.parquet',engine='fastparquet')

    return print('Data exported! -  filename: '+filename.value+'.parquet')

def export_materials(PLX_ports):
    '''  
    This function extracts some attributes of the materials from the Plaxis output: the name, the ID, and the color.  

    Parameters:  
        PLX_ports (dict): dictionary with the information of the PLAXIS ports (see function plaxis_connect()).  

    Returns:  
        materials (dict): dictionary with the information of the materials.  
    '''  
    # Redefino los puertos de PLAXIS
    g_o = PLX_ports['g_o']
    
    # Inicializo el diccionario
    materials = {}
    
    # Creo los keys que van a contener las variables
    materials['name'] = []
    materials['color'] = []
    materials['id'] = []
    materials['DrainageType'] = []

    for i in range(len(g_o.Materials[:])):
        materials['name'].append(g_o.Materials[i].Name.echo().split('"')[1])
        materials['color'].append(int(str(g_o.Materials[i].Colour)))
        materials['id'].append(i)
        if int(str(g_o.Materials[i].DrainageType)) == 4:
            materials['DrainageType'].append('NonPorous')
        else:
            materials['DrainageType'].append('Porous')
    materials = pd.DataFrame(materials)
    return materials