from ipywidgets import VBox, HBox, Checkbox, widgets, Layout, interact
import numpy as np
from ..python_plaxis.utils.varinfo import load_varinfo

def select_phases(g_i,default_value = False):
    phasename = []
    phaseid = []
    phasewidgets = []

    for phase in g_i.Phases:
        phasename.append(str(phase.Name))
        phaseid.append(str(phase.Identification))
        phasewidgets.append(widgets.Checkbox(value=default_value,
                                             description=str(phase.Identification)+' - ['+str(phase.Name)+']',
                                             disabled=False,
                                             indent=False))

    k = len(phasewidgets)/3

    left_box = VBox(phasewidgets[0:int(k)])
    mid_box = VBox(phasewidgets[int(k):int(k*2)])
    right_box = VBox(phasewidgets[int(k*2):])
    return HBox([left_box,mid_box,right_box])

def select_variables(g_o,varpath = 'info\Var.csv',default_value = False):
    
    Varinfo = load_varinfo(g_o,varpath)
    varwidgets = []

    for name in Varinfo.keys():
        logo = Varinfo[name]['logo']
        unit = Varinfo[name]['unit']
        varwidgets.append(widgets.Checkbox(value=default_value,
                                           description=logo+' - '+name+' ['+unit+']',
                                           disabled = False,
                                           indent = False))
                                             
    
    id_selected = [0,1,2,3,4, # Fill
                   8,9,10,
                   16,
                   36,
                   40,
                   42,
                   46,
                   49,
                   61,
                   63]
    
    for idwidget in id_selected:
        varwidgets[idwidget].value = True

    k = len(Varinfo.keys())/3

    left_var = VBox(varwidgets[0:int(k+1)])
    mid_var = VBox(varwidgets[int(k+1):int(k*2+1)])
    right_var = VBox(varwidgets[int(k*2+1):])
    return HBox([left_var, mid_var,right_var])

def unpack(widgetpack):
    widgetlist = []
    for column in widgetpack:
        for widget in column:
            widgetlist.append(widget)
    return widget