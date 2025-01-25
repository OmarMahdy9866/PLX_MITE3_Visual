###################                                                     TO DO                                                        ###################
########################################################################################################################################################
################### - Consider using jslink instead of observe                                                                       ###################
########################################################################################################################################################
################### - Add a dotted line that follows the original geometry when plotting the deformed mesh                           ###################
########################################################################################################################################################
################### - Keep scales from a particular plot to the rest of them                                                         ###################
########################################################################################################################################################
################### - If the same phase is plotted twice (or more), use the same polygons                                            ###################
########################################################################################################################################################

from ...general_utils.notification import *
from ...general_utils.significant_fig import *
import numpy as np
import os
import copy as cp
from shapely.geometry import Polygon
from shapely.geometry.collection import GeometryCollection as GeoColl
from shapely.ops import unary_union
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt
from ipywidgets import VBox, HBox, widgets, interact, Text
from tqdm.notebook import tnrange
from shapely.geometry import MultiPoint
from shapely import concave_hull
import matplotlib.tri as tri
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.ticker import (MultipleLocator)
import matplotlib
from ..utils.plx_colors import *
import geopandas as gpd
import matplotlib.colors as mcolors

class Contour:
    def __init__(self,data_folder = 'Data'):
        '''
        ContourPlot class. This class allows to plot contour data extracted from Plaxis
        '''
        self.Model_folder(data_folder)
        pass

    def Search_df(self,df,where,to_search,out):
        '''
        This function searches for a value "to_search" in the "where" column of a dataframe "df" and returns the corresponding value in the "out" column.
        
        Parameters:
        
            df: Dataframe where to look
            where: Column name where to look
            to_search: Value to search
            out: Column to excract
            
        Returns:
            
            Obj or list of obj
        
        '''
        if isinstance(to_search, list): # If is a list of values
            result = []  
            for name in to_search:  
                value = df.loc[df[where].str.contains(name)][out].values[0]  
                result.append(value)  
            return result
        else: # If is a single value
            return df.loc[df[where].str.contains(to_search)][out].values[0]
        
    def Apply_mask(self,triang, alpha,x,y):
        # Mask triangles with sidelength bigger some alpha
        triangles = triang.triangles
        # Mask off unwanted triangles.
        xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
        ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)
            # apply masking
        triang.set_mask(maxi > alpha)

    def Plotclusters(self,Dat,NumNodes,din):
        '''
        This function creates a cluster per soil material and a main cluster
        
        Parameters:
            Dat (DataFrame): NodeData DataFrame
            NumNodes (int): Number of nodes of each element of the model
            din (bool): Display or not clusters borders
            
        Return:
            total (shapely.Polygon or shapely.MultiPolygon): Polygon or MultiPolygon of the entire geometry
            cluster_summary (DataFrame): DataFrame with {'cluster': Polygon or MultiPolygon of each cluster, 'id': Number of id to identify the soil material}
        '''
        kpack = np.unique(Dat['MaterialID'].values)
        kpack = kpack[~np.isnan(kpack)]
        P_tot = []
        for k in kpack:
            P = []
            xp = Dat.loc[Dat['MaterialID']==k]['X'].values
            yp = Dat.loc[Dat['MaterialID']==k]['Y'].values
            for i in range(0,len(xp)-2,NumNodes):
                P.append(Polygon(zip(xp[i:i+NumNodes],yp[i:i+NumNodes])).convex_hull) 
            cluster_k = unary_union(P)
            P_tot.append(cluster_k)
            if cluster_k.geom_type == 'Polygon':
                xe,ye = cluster_k.exterior.coords.xy
                if din == True:
                    plt.plot(xe,ye,color='k', linewidth=1,zorder=9999998)
            elif cluster_k.geom_type == 'MultiPolygon':
                for cluster in cluster_k.geoms:
                    xe,ye = cluster.exterior.coords.xy
                    if din == True:
                        plt.plot(xe,ye,color='k', linewidth=1,zorder=9999998)
        cluster_summary = pd.DataFrame({'cluster':P_tot,'id':kpack.astype(int)})
        for i_cl in range(len(cluster_summary)):
            if isinstance(cluster_summary.iloc[i_cl]['cluster'],GeoColl):
                cluster_summary = cluster_summary.drop(cluster_summary.index[i_cl],axis=0)
        total = unary_union(P_tot)
        return total,cluster_summary

    def Plotmesh(self,Data,NumNodes):
        '''
        This function plots the mesh of the model
        
        Parameters:
            Data (DataFrame): NodeData DataFrame
            NumNodes (int): Number of nodes of each element of the model
            
        Return:
            None. Just plt.plot(model_mesh)
        '''
        
        xp = Data['X'].values
        yp = Data['Y'].values
        for i in tnrange(0,len(xp)-2,NumNodes):
            #print(i)
            plt.plot(np.append(xp[i:i+3],xp[i]),np.append(yp[i:i+3],yp[i]),linewidth=0.5,color='gray',alpha =0.5, zorder=9999997)

    def Clusters_to_plot(self,clus_sum,varsoil):
        '''
        This function allows to turn off materials that not wanted to be shown
        
        Parameters:
            clus_sum (DataFrame): DataFrame with clusters polygons and id (from self.Plotclusters())
            varsoil (list): List of widgets. A widget per soil material. The .value indicates if the soil material has to be shown or not
        
        Return:
            total (shapely.Polygon or shapely.MultiPolygon): Polygon or MultiPolygon of the shownable geometry
        '''
        i = 0
        P = []
        for id in clus_sum['id'].values: # id of an existing material in current phase
            if varsoil[id].value: # Searches the id in the varsoil list
                pol = clus_sum.loc[clus_sum['id']==id]['cluster'].values[0]
                P.append(pol)
            i+=1
        total = unary_union(P)
        return total

    def IsInTailings(self,Data,polygon, il_float):
        xoff = []
        yoff = []
        xpon = []
        ypon = []
        xverif = []
        xverifcmap = [] 
        pipa = MultiPoint(list(zip(Data['X'].values,Data['Y'].values)))
        Z = pipa.intersection(polygon)
        Data['xverif'] = 'g'
        Data['xverifcmap'] = np.ones(len(Data['xverif']))
        for point in Z.geoms:
            xpon.append(point.x)
            ypon.append(point.y)
        other = pd.DataFrame({'X':xpon,'Y':ypon})
        xoff = pd.merge(Data.reset_index(),other,how='inner').set_index('index')[self.varlogo_p]*-1
        yoff = pd.merge(Data.reset_index(),other,how='inner').set_index('index')[self.varlogo_q]
        xponf = pd.merge(Data.reset_index(),other,how='inner').set_index('index')['X']
        yponf = pd.merge(Data.reset_index(),other,how='inner').set_index('index')['Y']

        Data.loc[Data[self.varlogo_q]>Data[self.varlogo_p]*il_float*-1,'xverif']  = 'r'
        Data.loc[Data[self.varlogo_q]>Data[self.varlogo_p]*il_float*-1,'xverifcmap']  = -1

        xverif = pd.merge(Data.reset_index(),other,how='inner').set_index('index')['xverif']
        xverifcmap = pd.merge(Data.reset_index(),other,how='inner').set_index('index')['xverifcmap']  

        return xoff.values, yoff.values, xponf.values, yponf.values, xverif.values, xverifcmap.values

    def Model_folder(self,data_folder = 'Data'):
        '''
        This function allows to see the database available to plot from .parquet files
        
        Parameters:
            data_folder (str): Name of the relative folder that contains the .parquet files
        '''
        
        def link_folders(b): # This function updates the widget
            foutput.value=data_folder+'/%s'% model_name.value
            self.input_folder = foutput.value
        it=0
        fold_names=[]
        self.input_file_ext = '.parquet'
        for root, dirs, files in os.walk(data_folder): # Searches every data
            it+=1
            if it==1:
                for i in range(len(files)):
                    if files[i].endswith(self.input_file_ext):
                        fold_names.append(files[i].split(self.input_file_ext)[0])
                for i in range(len(dirs)):
                    fold_names.append(dirs[i])
        # Widgets
        model_name=widgets.RadioButtons(#value=False,
                            options=fold_names,
                            disabled=False, indent=False)
        foutput=widgets.Text(
            value=data_folder+'/%s'% model_name.value,
            placeholder=data_folder,
            description=data_folder,
            disabled=False
            )
        model_name.observe(link_folders)
        self.input_folder = foutput.value
        display(model_name)
        
    def Process_contour_df(self):
        '''
        ContourPlot function
        This function processes the input data
        Also adds the Inestability Line figure option
        '''
        df_input = pd.read_parquet(self.input_folder+self.input_file_ext) # Reads the file
        self.info_mat = df_input[['mat_name','color']].dropna().reset_index(drop=True) # Extract general information about materials
        self.info_var = df_input[['logo','var_name','unit','point']].dropna().reset_index(drop=True) # Extract general information about variables
        self.info_ph = df_input[['phase_name','phase_id']].dropna().reset_index(drop=True) # Extract general information about phases
        self.info_typeelement = int(df_input[['typeelement']].dropna().reset_index(drop=True).to_numpy()[0][0]) # Extract the number of nodes each element has
        index0 = df_input[['mat_name']].dropna().index[0]
        
        self.data = df_input.filter(regex='^Phase')[:index0] # Filters the phases/variables output data
        self.phases = sorted(list(np.unique(np.array(['_'.join(col.split("_")[0:2]) for col in self.data.columns.to_list()]))), key=lambda x: int(x.split("_")[1])) # Creates a list of Phases available to use
        vars = [col.split("_")[2:] for col in self.data.columns] # Extract variables available
        indx = np.unique(vars,return_index=True)[1] # Extract the index (to sort the list the original way. This is because np.unique() function sorts the elements)
        self.variable_names = [vars[ind][0] for ind in sorted(indx)] # Creates the originally sorted list of variable names
        
        # Adds the Inestability Line option
        self.varname_IL = 'Instability Line'
        varlogo_IL = 'IL' # Is an attribute just because it will be used for the warnings
        varunit_IL = '-'
        
        # Adds the Material ID option
        self.varname_matid = 'Material ID'
        
        df = pd.DataFrame({'logo':[varlogo_IL],'var_name':[self.varname_IL],'unit':[varunit_IL]})
        self.info_var = pd.concat([self.info_var,df],ignore_index = True) # Adds the IL info into the df
        self.variable_names.append(self.varname_IL) # Adds the IL info into the list
        
        # Creates two widgets that will be used to
        self.il_float = widgets.FloatText(value=0.60,description='η_'+varlogo_IL)
        self.phicv = widgets.FloatText(value=30,description='φ [°]')
        
    def Materials_to_plot(self):
        '''
        This function allows to select which soil materials will be shown
        '''
        
        self.mat_to_plot = []
        for mat in self.info_mat['mat_name']:
            self.mat_to_plot.append(widgets.Checkbox(value=True,
                                    description=mat,
                                    disabled=False, indent=False))
        
        k = len(self.mat_to_plot)/3
        left_var = VBox(self.mat_to_plot[0:int(k+1)])
        mid_var = VBox(self.mat_to_plot[int(k+1):int(k*2+1)])
        right_var = VBox(self.mat_to_plot[int(k*2+1):])
        B = HBox([left_var, mid_var,right_var])
        print('Please, uncheck the materials that must not be shown in the figures')
        display(B)
        
    def Select_variables_to_plot(self,clever_boxes = True):
        '''
        ContourPlot function
        Creates widgets for each phase and variable, so is possible to select which ones will be shown in figures
        
        Parameters:
            clever_boxes (bool=True): Autodisable boxes when N were selected (set False if kernel breakes)
        '''
        
        self.Process_contour_df() # Executes the function, that is needed to continue
        
        self.nwid=widgets.IntSlider(
                                value=2,
                                min=1,
                                max=6,
                                step=1,
                                description='Num of plots:',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='d')
        display(self.nwid)
        
        # Creates a layout to place a warning that is automatically updated (widget)
        layout = widgets.Layout(width='auto', height='20px',style= {'description_width': 'initial'}) #set width and height
        # Creates the widget
        ILwarningwid= widgets.Button(description=' ',disabled=False,display='flex',flex_flow='column',align_items='stretch',layout = layout,style = {'button_color': 'transparent'})
        
        # Prepare the text that may appear in the automatic warning
        ILwarningN='INESTABILITY LINE - PLOT INFO: To plot the '+self.varname_IL+' figure you have to select Num of plots = 1.'
        ILwarningQP='INESTABILITY LINE - PLOT INFO: You need to export the MeanEffStress and DeviatoricStress data from Plaxis to plot the '+self.varname_IL+' figure. \nPlease, check both were exported'
        
        # This part communicates the warning widget with the widgets that trigger the warning itself
        if clever_boxes: # If "Clever boxes" option is True
            def check_N_selected(b):
                n=0
                avoidIL=False
                if IL_fail:
                    avoidIL=True
                if self.nwid.value>1:
                    avoidIL=True
                    ILwarningwid.description=ILwarningN
                else:
                    ILwarningwid.description=' '
                    
                for phase in self.phases:
                    for wid in self.varwidgets[phase]:
                        if wid.value:
                            n+=1
                if n>=self.nwid.value:
                    for phase in self.phases:
                        for wid in self.varwidgets[phase]:
                            if not wid.value:
                                wid.disabled=True
                if n<self.nwid.value:
                    for phase in self.phases:
                        for wid in self.varwidgets[phase]:
                            if not wid.value:
                                if self.varname_IL in wid.description and avoidIL:
                                    wid.disabled=True
                                else:
                                    wid.disabled=False
        # This part checks if the needed information to create IL figure is available or not
        if 'Mean effective stress' in self.variable_names and 'Deviatoric stress' in self.variable_names:
            IL_fail=False
        elif 'K0' in self.variable_names:
            IL_fail=False
        else: # If the info is not available
            IL_fail=True # Trigger the warning
            ILwarningwid.description=ILwarningQP # Update the warning widget
        display(ILwarningwid)
        
        if clever_boxes: # If "Clever boxes" option is True
            self.nwid.observe(check_N_selected) # Communicate the slider "number of plots" widget with the warning
            ILwarningwid.observe(check_N_selected)
        
        # A little adjustment to consider singular or plural
        if self.nwid.value==1:
            vbls=' VARIABLE:'
        else:
            vbls=' VARIABLES:'
        # Creates a sign widget, that tells how many variables the user has to select
        text= widgets.Button(description='SELECT JUST '+str(self.nwid.value)+vbls,
                            disabled=False,display='flex',flex_flow='column',align_items='stretch',layout = layout,style = {'button_color': 'transparent'})
        def link_text_slider(b): # Function to communicate this sign with the slider widget
            if self.nwid.value==1:
                vbls=' VARIABLE:'
            else:
                vbls=' VARIABLES:'
            text.description = 'SELECT JUST '+str(self.nwid.value)+vbls
        text.observe(link_text_slider) # Communicate the sign with the slider "number of plots" widget
        self.nwid.observe(link_text_slider) # Communicate the slider "number of plots" widget with the sign
        display(text)
        
        self.varwidgets = {} # Creates a dict to save each variable box
        
        for phase in self.phases: # Each phase

            print("\033[1m" + phase + ' - ' + self.Search_df(self.info_ph,'phase_name',phase,out='phase_id') + "\033[0m") # Prints the phase name
            self.varwidgets[phase] = [] # Creates an specific column for "phase". Each column will contain every variable widget (that corresponds with that specific phase)
            for var in self.variable_names: # Each variable
                self.varwidgets[phase].append(widgets.Checkbox(value=False, # Creates a widget
                                    description=var+' || '+self.Search_df(self.info_var,'var_name',var,out='logo')+' ['+self.Search_df(self.info_var,'var_name',var,out='unit')+']',
                                    disabled=False, indent=False))
                if clever_boxes: # Communicate the box widget of each variable with the number of figures selected
                    self.varwidgets[phase][-1].observe(check_N_selected)
            
            # Divides the variables in three columns
            k = len(self.variable_names)/3 
            left_var = VBox(self.varwidgets[phase][0:int(k+1)])
            mid_var = VBox(self.varwidgets[phase][int(k+1):int(k*2+1)])
            right_var = VBox(self.varwidgets[phase][int(k*2+1):])
            A = HBox([left_var, mid_var,right_var])
            display(A)
        
        # Disables or not the IL box widget
        for phase in self.phases:
            for wid in self.varwidgets[phase]:
                if self.varname_IL in wid.description:
                    if IL_fail:
                        wid.disabled=True
                    if self.nwid.value>1:
                        wid.disabled=True
                        
        self.Materials_to_plot() # Executes the function, that is needed to continue

    def Plot_config(self):
        '''
        This function creates widgets to set the needed data to plot every figure, such as x and y limits, contourmap limits, ticks, and others
        '''
        self.clusterok = []
        self.meshok = []
        self.externalok = []
        self.nfok = []
        self.extremeok = []
        self.deformok = []
        self.xmin = []
        self.xmax = []
        self.eqok = []
        self.ymin = []
        self.ymax = []
        self.cmin = []
        self.cmax = []
        self.cdiv = []
        self.Tmax = []
        self.xticksmin = []
        self.xticksmax = []
        self.yticksmin = []
        self.yticksmax = []
        self.scale = []
        self.dimx = []
        self.dimy = []
        self.dimc = []
        self.ticksx = []
        self.ticksy = []
        self.dimx = []
        self.dimy = []
        self.dimc = []
        self.zminT = []
        self.zmaxT = []
        self.ph_to_plot = []
        self.var_to_plot = []
        self.r_cmap = []

        self.multiplier = []

        self.ord_wid = []

        for p in self.phases: # For each phase and variable will verify if is checked to plot or not
            i = 0
            for v in self.varwidgets[p]: # Each variable
                if v.value == True: # Check if was selected
                    self.ph_to_plot.append(p)
                    self.var_to_plot.append(self.varwidgets[p][i].description.split(' || ')[0])
                i+=1
        
        if len(self.ph_to_plot)<self.nwid.value: # If the number of boxes selected is lower than the number of plots that was selected to plot (slider)
            text1 = 'Please select {N} variable'.format(N=str(self.nwid.value))
            text3 = 'to plot in "Select variable to plot" section. Then, run again this cell'
            if self.nwid.value == 1: # Just a simple check to consider plural in the warning
                text2 = " "
            else:
                text2 = "s "
            return warning(text1+text2+text3) # Warns about this issue
        
        separamelosejes = 0.025
        
        for plot in range(self.nwid.value):

            self.multiplier.append(widgets.FloatText(value=1,description='α$_{Shansep}$'))
            self.zminT.append(0)
            self.zmaxT.append(0)
            
            if self.var_to_plot[plot]==self.varname_IL:
                self.zminT[plot] = -1
                self.zmaxT[plot] = 1
            else:
                colZ = self.ph_to_plot[plot]+'_'+self.var_to_plot[plot]
                self.zminT[plot] = self.data[colZ].min()
                self.zmaxT[plot] = self.data[colZ].max()
            colX,colY = False,False
            for col in self.data.columns:
                if self.ph_to_plot[plot] in col and 'X node coordinate' in col:
                    colX = col
                elif self.ph_to_plot[plot] in col and 'Y node coordinate' in col:
                    colY = col
            if colX==False or colY==False:
                warning('Check if "node coordinates" columns exist in "data" dataframe')
                return
            nodes_x = self.data[colX]
            nodes_y = self.data[colY]
            
            xminT = min(nodes_x)
            xmaxT = max(nodes_x)
            yminT = min(nodes_y)
            ymaxT = max(nodes_y)
            
            deltayT = ymaxT-yminT
            deltaxT = xmaxT-xminT
            
            str_horizdispl = self.ph_to_plot[plot]+'_Horizontal displacement'
            str_vertdispl = self.ph_to_plot[plot]+'_Vertical displacement'

            if str_horizdispl in self.data.columns and str_vertdispl in self.data.columns:
                Uxmax = self.data[str_horizdispl].max()
                Uymax = self.data[str_vertdispl].max()
                Utotmax = np.sqrt(Uxmax**2+Uymax**2)
                TotSize = np.sqrt(deltayT**2+deltaxT**2)
                defscale = Significant_fig(0.01 * TotSize/Utotmax,3) # La escala de deformaciones es tal que el desplazamiento en la malla deformada es del 1% del tamaño medio de la malla
            else:
                defscale = 1

            self.xmin.append(widgets.FloatText(value=round(xminT-deltaxT*separamelosejes,2),description='Min x:'))
            self.xmax.append(widgets.FloatText(value=round(xmaxT+deltaxT*separamelosejes,2),description='Max x:'))
            self.eqok.append(widgets.Checkbox(value=True,
                                    description='Equal aspect ratio',
                                    disabled=False, indent=False))

            self.ymin.append(widgets.FloatText(value=round(yminT-deltayT*separamelosejes*10,2),description='Min y:'))
            self.ymax.append(widgets.FloatText(value=round(ymaxT+deltayT*separamelosejes*20,2),description='Max y:'))
            self.cmin.append(widgets.FloatText(value=Significant_fig(self.zminT[plot],6),description='Min scale:'))
            self.cmax.append(widgets.FloatText(value=Significant_fig(self.zmaxT[plot],6),description='Max scale:'))
            # cmin.append(widgets.FloatText(value=Significant_fig(zminT[plot]*alpha_shansep[plot].value,6),description='Min scale:'))
            # cmax.append(widgets.FloatText(value=Significant_fig(zmaxT[plot]*alpha_shansep[plot].value,6),description='Max scale:'))
            self.cdiv.append(widgets.FloatText(value=11,description='Divs cmap:'))
            self.r_cmap.append(widgets.Checkbox(value=False,
                                    description='Revert cmap',
                                    disabled=False, indent=False))
            var_name = self.var_to_plot[plot]
            if var_name=='Excess pore pressure' or var_name=='Active pore pressure' or var_name=='Pore water pressure'\
                    or var_name=='Steady-state pore pressure' or var_name=='SigmaEffective1Max' or var_name=='Sigma_eff_1 - Sigma_eff_v difference'\
                    or var_name=='Vertical displacement'\
                    or var_name=='Mean effective stress':
                self.r_cmap[-1].value = True
                
            tickxmax=20
            if self.nwid.value>=4:
                tickxmax=50
            
            # exec("""def link_alpha_with_scale{a}(b):
            #             self.cmin[{a}].value=Significant_fig(self.zminT[{a}]*self.multiplier[{a}].value,6)
            #             self.cmax[{a}].value=Significant_fig(self.zmaxT[{a}]*self.multiplier[{a}].value,6)
            #             """.format(a=plot))
            # self.cmin[plot].observe(eval("link_alpha_with_scale%s"%plot))
            # self.cmax[plot].observe(eval("link_alpha_with_scale%s"%plot))
            # self.multiplier[plot].observe(eval("link_alpha_with_scale%s"%plot))
            
            self.xticksmin.append(widgets.FloatText(value=2,description='Min x tick:'))
            self.xticksmax.append(widgets.FloatText(value=tickxmax,description='Max x tick:'))
            self.yticksmin.append(widgets.FloatText(value=2,description='Min y tick:'))
            self.yticksmax.append(widgets.FloatText(value=20,description='Max y tick:'))
            self.scale.append(widgets.FloatText(value=defscale, description='Deform. scale'))

            self.ticksx.append(HBox([self.xticksmin[plot], self.xticksmax[plot]]))
            self.ticksy.append(HBox([self.yticksmin[plot], self.yticksmax[plot]]))
            self.dimx.append(HBox([self.xmin[plot],self.xmax[plot]]))
            self.dimy.append(HBox([self.ymin[plot],self.ymax[plot], self.eqok[plot]]))
            self.dimc.append(HBox([self.cmin[plot], self.cmax[plot], self.cdiv[plot]]))
    #         dimt.append(HBox([Tmin[plot],Tmax[plot],Tdiv[plot]]))
            
            self.clusterok.append(widgets.Checkbox(value=True,
                                    description='Show material borders',
                                    disabled=False, indent=False))
            self.meshok.append(widgets.Checkbox(value=False,
                                    description='Display mesh elements (takes a while)',
                                    disabled=False, indent=False))
            self.externalok.append(widgets.Checkbox(value=False,
                                    description='Show external boundary of mesh',
                                    disabled=False, indent=False))
            self.nfok.append(widgets.Checkbox(value=True,
                                    description='Show water table',
                                    disabled=False, indent=False))
            self.extremeok.append(widgets.Checkbox(value=False,
                                    description='Show extreme values',
                                    disabled=False, indent=False))
            self.deformok.append(widgets.Checkbox(value=False,
                                    description='Show deformed mesh',
                                    disabled=False, indent=False))
            self.ord_wid.append(widgets.IntSlider(
                                    value=int(plot+1),
                                    min=1,
                                    max=self.nwid.value,
                                    step=1,
                                    description='Order in plot',
                                    orientation='horizontal',))

        def show_boxes(x):
            box1 = HBox([self.meshok[x],self.clusterok[x],self.ord_wid[x]])
            box2 = HBox([self.externalok[x],self.nfok[x]])
            box3 = HBox([self.extremeok[x], self.deformok[x]])
            if self.var_to_plot[x] == self.varname_IL:
                return_boxes = VBox([box1,box2,HBox([self.il_float,self.phicv]),self.dimx[x],self.dimy[x],self.ticksx[x],self.ticksy[x]])#,dimt[x]])#, layout=Layout(border = '1px solid black'))
            elif self.var_to_plot[x] == self.varname_matid:
                return_boxes = VBox([box1,box2,box3,self.dimx[x],self.dimy[x],self.scale[x],self.ticksx[x],self.ticksy[x]])
            else:
                return_boxes = VBox([box1,box2,box3,self.dimx[x],self.dimy[x],self.dimc[x],HBox([self.scale[x],self.r_cmap[x]]),self.ticksx[x],self.ticksy[x]])#,dimt[x]])#, layout=Layout(border = '1px solid black'))
            return return_boxes

        graphs=[]
        for i in range(self.nwid.value):
            graphs.append((self.var_to_plot[i]+" - "+self.ph_to_plot[i],i))
            
        interact(show_boxes,x=graphs);

    def Make_contour_plot(self,figure_name='',cmap = 'jet',show_NF_in_non_shown_clusters=True,figsize=[49.6/2.54,10/2.54]):
        '''
        This function creates the ContourPlots
        
        Parameters:
            figure_name (str=''): If == '', no figure will be save into the computer
            cmap (str='jet'): Matplotlib cmap. https://matplotlib.org/stable/gallery/color/colormap_reference.html
            shown_NF_in_non_shown_clusters (bool=True): Allows to select if Phreatic Level will be shown above non shown clusters or not
            
        Returns:
            fig,axes (matplotlib fig,axes)
        '''
        if len(self.ph_to_plot)<self.nwid.value: # If the number of boxes selected is lower than the number of plots that was selected to plot (slider)
            text1 = 'Please select {N} variable'.format(N=str(self.nwid.value))
            text3 = 'to plot in "Select variable to plot" section. Then, run again from "Plot_config" cell'
            if self.nwid.value == 1: # Just a simple check to consider plural in the warning
                text2 = " "
            else:
                text2 = "s "
            return warning(text1+text2+text3),0 # Warns about this issue
        
        figaxes = []
        ord_list = []
        self.clusters = []
        for i in range(self.nwid.value):
            ord_list.append(self.ord_wid[i].value-1)
        repeated = set([x for x in ord_list if ord_list.count(x) > 1])
        if len(repeated)!=0:
            return warning("ERROR. Repeated plot order values: Check the plot order defined in the previous cell"),0

        if self.nwid.value==1:
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [[0.05,0.15,0.95,0.85]] 
            axesdesc = [[0.05,0.05,0.803,0.10]]
        if self.nwid.value==2:
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [[0.05,0.15,0.45,0.85],[0.55,0.15,0.45,0.85]] 
            axesdesc = [[0.05,0.05,0.3805,0.10],[0.55,0.05,0.3805,0.10]]
        if self.nwid.value==3:
            # axes = [[0.05,0.60,0.95,0.35],
            #         [0.05,0.15,0.45,0.35],[0.55,0.15,0.45,0.35]] 
            # axesdesc = [[0.05,0.55,0.803,0.05],
            #             [0.05,0.10,0.3805,0.05],[0.55,0.10,0.3805,0.05]]
            
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [[0.05,0.60,1,0.35],
                    [0.05,0.15,0.45,0.35],[0.5145,0.15,0.45,0.35]] 
            axesdesc = [[0.05,0.55,0.845,0.05],
                        [0.05,0.10,0.3805,0.05],[0.5145,0.10,0.3805,0.05]]
        if self.nwid.value==4:
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [[0.05,0.60,0.45,0.35],[0.55,0.60,0.45,0.35],
                    [0.05,0.15,0.45,0.35],[0.55,0.15,0.45,0.35]] 
            axesdesc = [[0.05,0.55,0.3805,0.05],[0.55,0.55,0.3805,0.05],
                        [0.05,0.10,0.3805,0.05],[0.55,0.10,0.3805,0.05]]
        if self.nwid.value==5:
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [                     [0.55,0.7,0.45,0.23],
                    [0.05,0.4,0.45,0.23],[0.55,0.4,0.45,0.23],
                    [0.05,0.1,0.45,0.23],[0.55,0.1,0.45,0.23],]
            axesdesc = [                      [0.55,0.67,0.3805,0.03],
                        [0.05,0.37,0.3805,0.03],[0.55,0.37,0.3805,0.03],
                        [0.05,0.07,0.3805,0.03],[0.55,0.07,0.3805,0.03],]
        if self.nwid.value==6:
            fig = plt.figure(figsize=(figsize[0],figsize[1]))
            axes = [[0.05,0.60,0.3,0.30],[0.40,0.60,0.3,0.30],[0.75,0.60,0.3,0.30],
                    [0.05,0.15,0.3,0.30],[0.40,0.15,0.3,0.30],[0.75,0.15,0.3,0.30],]
            axesdesc = [[0.05,0.55,0.254,0.05],[0.40,0.55,0.254,0.05],[0.75,0.55,0.254,0.05],
                        [0.05,0.1,0.254,0.05],[0.40,0.1,0.254,0.05],[0.75,0.1,0.254,0.05],]
        if self.nwid.value>6:
            warning('Just can plot grids of 1, 2, 3, 4, 5 and 6 plots. Please, use Power Point')
            return
        texts1,texts2=[],[]
        ax = {}
        i=-1
        for plotid in range(self.nwid.value):
            i+=1
            if self.var_to_plot[plotid]==self.varname_IL:
                ax[plotid] = fig.add_axes([0.05, 0.15,0.5,0.85])
            else:
                ax[plotid] = fig.add_axes(axes[ord_list[plotid]])
            
            str_x_node = self.ph_to_plot[plotid]+'_X node coordinate'
            str_y_node = self.ph_to_plot[plotid]+'_Y node coordinate'
            str_matid = self.ph_to_plot[plotid]+'_'+self.varname_matid
            str_horizdispl = self.ph_to_plot[plotid]+'_Horizontal displacement'
            str_vertdispl = self.ph_to_plot[plotid]+'_Vertical displacement'
            NodeData = pd.DataFrame({'X':self.data[str_x_node].to_list(),
                                     'Y':self.data[str_y_node].to_list(),
                                     'MaterialID':self.data[str_matid].to_list()}).dropna()
            
            if str_horizdispl in self.data.columns and str_vertdispl in self.data.columns:
                NodeData = pd.concat([NodeData,pd.DataFrame({'Ux':self.data[str_horizdispl].to_list(),
                                                            'Uy':self.data[str_vertdispl].to_list()})]
                                    ,axis=1)
            
            str_x_stress = self.ph_to_plot[plotid]+'_X stresspoint coordinate'
            str_y_stress = self.ph_to_plot[plotid]+'_Y stresspoint coordinate'
            str_psteady = self.ph_to_plot[plotid]+'_Steady-state pore pressure'
            StressData = pd.DataFrame({'X':self.data[str_x_stress].to_list(),
                                       'Y':self.data[str_y_stress].to_list(),
                                       'PSteady':self.data[str_psteady].to_list(),
                                       'MaterialID':self.data[str_matid]})
            
            var_name = self.var_to_plot[plotid]
            var_logo = self.info_var.loc[self.info_var['var_name']==var_name]['logo'].values[0]
            var_unit = self.info_var.loc[self.info_var['var_name']==var_name]['unit'].values[0]
            var_point = self.info_var.loc[self.info_var['var_name']==var_name]['point'].values[0]
            var_info = [var_name,var_logo,var_unit]
            
            if self.var_to_plot[plotid]==self.varname_IL:
                self.varlogo_qp = 'DeviatoricStress/MeanEffStress ratio'
                self.varlogo_q = 'Deviatoric stress'
                self.varlogo_p = 'Mean effective stress'
                Data = StressData
                if self.ph_to_plot[plotid]+'_'+self.varlogo_q in self.data.columns and self.ph_to_plot[plotid]+'_'+self.varlogo_p in self.data.columns:
                    self.q = self.data[self.ph_to_plot[plotid]+'_'+self.varlogo_q].values
                    self.p = self.data[self.ph_to_plot[plotid]+'_'+self.varlogo_p].values
                    self.qp = -1* self.data[self.ph_to_plot[plotid]+'_'+self.varlogo_q].values/self.data[self.ph_to_plot[plotid]+'_'+self.varlogo_p].values
                    Data[self.varlogo_qp] = self.qp
                    Data[self.varlogo_q] = self.q
                    Data[self.varlogo_p] = self.p
                
                else:
                    return warning('Is not possible to plot the Instability Line graph. q, p\' or both are missing'),1

            else:
                str_data = self.ph_to_plot[plotid]+'_'+self.var_to_plot[plotid]
                Data = self.data[str_data].dropna()

            if self.deformok[plotid].value == True:
                if 'Ux' in NodeData.columns and 'Uy' in NodeData.columns:
                    NodeData['X'] = NodeData['X']+self.scale[plotid].value*NodeData['Ux']
                    NodeData['Y'] = NodeData['Y']+self.scale[plotid].value*NodeData['Uy']
                else:
                    warning('Is not possible to plot the deformation of the mesh because Ux and Uy were not exported')

            zmin = self.zminT[plotid]*self.multiplier[plotid].value
            zmax = self.zmaxT[plotid]*self.multiplier[plotid].value
            
            if zmin != zmax:

                total,clusters = self.Plotclusters(NodeData, self.info_typeelement,self.clusterok[plotid].value)
                self.clusters.append(clusters)
                # Esto es para verificar si los materiales que están seleccionados para plotear existen en el plot actual
                mat_exist = False
                for i_mat_exist,mat in enumerate(self.mat_to_plot):
                    if mat.value:
                        if i_mat_exist in clusters['id'].values:
                            mat_exist = True
                if not mat_exist:
                    warning('Please check the materials that were selected. No materials available in plot number {plot}'.format(plot=plotid+1))
                    warning('The materials available are, in order, the following:')
                    return display(clusters['id'].values),0
                
                xn, yn = total.exterior.coords.xy
                if self.externalok[plotid].value == True:
                    plt.plot(xn,yn,color='k',linewidth=2,zorder=99999999)
                plt.fill(xn,yn,color='gainsboro',zorder=1)
                if self.meshok[plotid].value == True:
                    self.Plotmesh(NodeData, self.info_typeelement)
                    
                clusterplot = self.Clusters_to_plot(clusters,self.mat_to_plot)
                
                vr=Data*self.multiplier[plotid].value
                
                if self.var_to_plot[plotid]==self.varname_IL: # Cambiar para que reconozca el cluster de tailings con material ID
                    xdata,ydata,xpon,ypon,verifcolor, verifcmapcolor = self.IsInTailings(Data.dropna(),clusterplot, self.il_float.value)
                    verif = verifcmapcolor
                    zmin= verif.min()
                    zmax= verif.max()
                    triangmask = tri.Triangulation(np.array(xpon),np.array(ypon))
                else:
                    if var_point == 'stresspoint':
                        triangmask = tri.Triangulation(np.array(StressData['X'].dropna()),np.array(StressData['Y'].dropna()))
                        if len(vr) != len(triangmask.x): # Esto es porque con materiales no porosos len(PSteady)!=len(X_stresspoints)
                            triangmask = tri.Triangulation(np.array(StressData.dropna()['X']),np.array(StressData.dropna()['Y']))
                            
                    elif var_point == 'node':
                        triangmask = tri.Triangulation(np.array(NodeData['X'].dropna()),np.array(NodeData['Y'].dropna()))
                    else:
                        return warning('The data is neither stresspoint nor node. Please check')
                    
                cdiv_value=int(self.cdiv[plotid].value)
                lvls=np.linspace(self.cmin[plotid].value,self.cmax[plotid].value,int(self.cdiv[plotid].value))
                if self.r_cmap[plotid].value:
                    cm = cmap+'_r'
                else:
                    cm = cmap
                
                if self.var_to_plot[plotid]==self.varname_IL:
                    cm='RdYlGn'
                    cdiv_value=20
                    lvls=np.linspace(zmin-(zmax-zmin)/20,zmax+(zmax-zmin)/20,31)
                    vr=verif
                
                multipol_flag = False
                xtai,ytai = [],[]
                if clusterplot.geom_type == 'Polygon':
                    xtai_i,ytai_i = clusterplot.exterior.coords.xy
                    xtai.append(xtai_i)
                    ytai.append(ytai_i)
                    
                elif clusterplot.geom_type == 'MultiPolygon':
                    multipol_flag = True
                    for cluster in clusterplot.geoms:
                        xtai_i,ytai_i = cluster.exterior.coords.xy
                        xtai.append(xtai_i)
                        ytai.append(ytai_i)

                if self.var_to_plot[plotid]==self.varname_matid:

                    # To change the axes size
                    newbox = axes[ord_list[plotid]] # Toma como base el original
                    newbox[2] = axesdesc[ord_list[plotid]][2] # Achica el tamaño horizontal para hacerlo igual al del box de descripción
                    fig.axes[-1].set_position(newbox)
                    
                    # Plots clusters with the correspondient colour
                    # rgb_list = [PLX_Colors(self.info_mat['color'].iloc[i_color]) for i_color in range(len(self.info_mat))]
                    rgb_list = [PLX_Colors(self.info_mat['color'].iloc[i_color]) for i_color in self.clusters[plotid]['id']]

                    clusters_plot = gpd.GeoDataFrame(geometry=[self.clusters[plotid]['cluster'][i_clus] for i_clus in range(len(self.clusters[plotid]))])

                    ################################################################################
                    # Hay algo que plotea los poligonos en gris y tapa el color del material. Le
                    # agregué un zorder para evitar eso
                    clusters_plot.plot(ax=ax[plotid], color=rgb_list,zorder=9999995)
                    ################################################################################

                    # cmap = mcolors.LinearSegmentedColormap.from_list("MaterialID_cmap", rgb_list)
                else:
                    cont = ax[plotid].tricontourf(triangmask, vr, cdiv_value,cmap=cm, levels=lvls, zorder=2)
                    
                if multipol_flag:
                    xtai_aux = []
                    ytai_aux = []
                    codes = []
                    for xtai_aux_i in xtai:
                        codes += [Path.MOVETO] + [Path.LINETO]*(len(xtai_aux_i)-2) + [Path.CLOSEPOLY]
                        xtai_aux+=xtai_aux_i
                    for ytai_aux_i in ytai:
                        ytai_aux+=ytai_aux_i
                    xtai = xtai_aux
                    ytai = ytai_aux

                    clippath = Path(list(zip(xtai, ytai)),codes)
                else:
                    clippath = Path(np.c_[xtai[0],ytai[0]])
                patch = PathPatch(clippath, facecolor='none')
                ax[plotid].add_patch(patch)
                if self.var_to_plot[plotid]!=self.varname_matid:
                    for c in cont.collections:
                        c.set_clip_path(patch)

                if self.nfok[plotid].value == True:
                    SD = cp.copy(StressData)
                    SD = SD.dropna()
                    triangmask2 = tri.Triangulation(np.array(SD['X']),np.array(SD['Y']))

                    ################################################################################
                    # Le bajé el levels de -0.05 a -1. Esto porque generaba superficies de NF muy
                    # feas. El -1 correspondería a 10cm por debajo de la realidad, no lo veo tan mal
                    # para las magnitudes que manejamos
                    cont2 = ax[plotid].tricontour(triangmask2 , SD['PSteady'], 1,colors=['magenta'],levels=[-1],linestyles='solid', zorder=9999999)
                    ################################################################################

                    if show_NF_in_non_shown_clusters:
                        pass
                    else:
                        for c in cont2.collections:
                            c.set_clip_path(patch)

                fmt = matplotlib.ticker.ScalarFormatter(useMathText=False)
                if self.var_to_plot[plotid]!=self.varname_IL and self.var_to_plot[plotid]!=self.varname_matid:
                    cbar = fig.colorbar(cont, pad=0.005, aspect=10, ticks=np.linspace(self.cmin[plotid].value,self.cmax[plotid].value,int(self.cdiv[plotid].value)),format=fmt)
                    cbar.set_label(str(var_info[0])+' ['+str(var_info[2])+']', rotation=270, labelpad=10)

                if self.eqok[plotid].value == True:
                    plt.axis('equal')
                plt.ylim(self.ymin[plotid].value,self.ymax[plotid].value)
                plt.xlim(self.xmin[plotid].value,self.xmax[plotid].value)

                ax[plotid].tick_params(which='major', length=12)
                ax[plotid].tick_params(which='minor', length=6)
                ax[plotid].xaxis.set_minor_locator(MultipleLocator(self.xticksmin[plotid].value))
                ax[plotid].xaxis.set_major_locator(MultipleLocator(self.xticksmax[plotid].value))
                ax[plotid].yaxis.set_minor_locator(MultipleLocator(self.yticksmin[plotid].value))
                ax[plotid].yaxis.set_major_locator(MultipleLocator(self.yticksmax[plotid].value))
                ax[plotid].xaxis.tick_top()
                
                if self.var_to_plot[plotid]==self.varname_IL:
                    ax[plotid] = fig.add_axes([0.05, 0.05,0.5,0.10])
                else:
                    ################################################################################
                    # Le agregué el 'desc' que sobreescribre al anterior (y no se puede llamar fuera de la funcion)
                    ax[str(plotid)+'desc'] = fig.add_axes(axesdesc[plotid])
                    ################################################################################
                fig.axes[-1].set_label('Text bar')
                
                plt.gca().axes.get_yaxis().set_visible(False)
                plt.gca().axes.get_xaxis().set_visible(False)
                if self.var_to_plot[plotid]==self.varname_matid:
                    texts1.append('Model Geometry')
                else:
                    texts1.append(str(var_info[0])+' - '+str(var_info[1]))
                
                if self.var_to_plot[plotid]==self.varname_IL:
                    ax = fig.add_axes([0.65, 0.15,0.3,0.85])
                    plt.grid(color='whitesmoke', zorder=0)
                    plt.scatter(xdata,ydata, facecolor=verifcolor, alpha=0.2, s=30, zorder=10)
                    ax.tick_params(which='major', length=6)
                    ax.tick_params(which='minor', length=3)
                    ax.xaxis.set_minor_locator(MultipleLocator(20))
                    ax.xaxis.set_major_locator(MultipleLocator(100))
                    ax.yaxis.set_minor_locator(MultipleLocator(20))
                    ax.yaxis.set_major_locator(MultipleLocator(100))
                    plt.ylabel('Deviatoric Stress, q [kPa]')
                    plt.xlabel('Mean Effective Stress. p\' [kPa]')
                    xverif = pd.DataFrame({'color':verifcolor})
                    my_dict = {}
                    my_dict['g'] = len(xverif.loc[xverif['color']=='g'])
                    my_dict['r'] = len(xverif.loc[xverif['color']=='r'])

                    CSL=(6*np.sin(np.radians(self.phicv.value)))/(3-np.sin(np.radians(self.phicv.value)))

                    plt.xlim(0,400)
                    plt.ylim(0,400)
                    plt.plot([0,400],[0,400*self.il_float.value], label = 'IL', linestyle='dashed', color='tab:blue',zorder = 9999999)
                    plt.plot([0,400],[0,400*CSL], label = 'CSL', color='r',linestyle='dashed',zorder = 9999999)
    #                 plt.scatter([], [], facecolor='g', alpha=0.2, s=30, label="Verify: "+str(my_dict['g'])+' points')
    #                 plt.scatter([], [], facecolor='r', alpha=0.2, s=30, label="Do not verify: "+str(my_dict['r'])+' points')
                    plt.legend(loc='best')
                    warning('Please, note that IL is not always a straight line')
                else:
                    i_max = Data.idxmax()
                    i_min = Data.idxmin()
                    maxvalue = str("{:.2e}".format(zmax))
                    minvalue = str("{:.2e}".format(zmin))
                    unit = var_info[2]
                    xmax = str(Significant_fig(NodeData.iloc[i_max]['X'],2))
                    ymax = str(Significant_fig(NodeData.iloc[i_max]['Y'],2))
                    xmin = str(Significant_fig(NodeData.iloc[i_min]['X'],2))
                    ymin = str(Significant_fig(NodeData.iloc[i_min]['Y'],2))
                    texts2.append('Max value = '+maxvalue+' '+unit+' at ('+xmax+' '+unit+', '+ymax+' '+unit+')  -  Min value = '+minvalue+' '+unit+' at ('+xmin+' '+unit+', '+ymin+' '+unit+')')
                    
            else:
                warning(var_name)
                warning('All values are the same')
                warning('max: ', zmax)
                warning('min: ', zmin)
                pass
        
        texts1=[x for _, x in sorted(zip(ord_list, texts1))]
        texts2=[x for _, x in sorted(zip(ord_list, texts2))]
        extremeok2=[x for _, x in sorted(zip(ord_list, self.extremeok))]
        
        if self.var_to_plot[0]==self.varname_IL:
            # return warning('IL WIP'),6
            fig.axes[1].text(0.01,0.35,texts1[0], weight='bold', fontsize=12)
        else:
            it=-1
            for i in range(len(fig.axes)):
                if fig.axes[i].get_label() == 'Text bar':
                    it+=1
                    fig.axes[i].text(0.01,0.35,texts1[it], weight='bold', fontsize=12)
                    if extremeok2[it].value == True:
                        fig.axes[i].text(0.99,0.35,texts2[it], fontsize=12, horizontalalignment='right')
        if figure_name != '':
            plt.savefig(figure_name, bbox_inches='tight')
        ###########################################################################################
        # Desactive el plt.show() para agregarle mas cosas fuera de la funcion
        # plt.show()
        ###########################################################################################
        ph_names = [str(self.info_ph.loc[self.info_ph['phase_name']==ph]['phase_id'].values[0]) +' ['+ph+']' for ph in self.ph_to_plot]
        print([x for _, x in sorted(zip(ord_list, ph_names))])
        return fig,ax