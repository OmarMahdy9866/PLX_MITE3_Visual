import numpy as np
import matplotlib.pyplot as plt

def Frequency_plot (data,conf,
                        title="Histogram",
                        x_label = "Settlement [mm]",
                        y1_label = "Frequency",
                        y2_label = "Likelihood of occurence",
                        fontsize_title = 14,
                        fontsize_axeslabels = 12,
                        fontsize_axesticks = 10,):
    '''
    Frequency plot
    
    Parameters:
        data (np.ndarray): Data array
        conf (float): Confidence level [%]
        title (str): Figure title
        x_label (str): X-axis label
        y1_label (str): Left Y-axis label
        y2_label (str): Right Y-axis label (CDF)
        fontsize_title (str): Fontsize of titles
        fontsize_axeslabels (str): Fontsize of axes labels
        fontsize_axesticks (str): Fontsize of axes ticks
    Return:
        fig,axes: (matplotlib) Frequency plot
    '''

    n = len(data)
    if len(np.unique(data))>1:
        fig, ax1 = plt.subplots(figsize=(10,6))  
        ax1.set_title(title , fontsize = fontsize_title)  
        ax1.set_xlabel(x_label , fontsize = fontsize_axeslabels)  
        ax1.set_ylabel(y1_label , fontsize = fontsize_axeslabels)  
        ax1.grid(color='gray',alpha=0.5)
        ax1.tick_params(axis='both', which='major', labelsize=fontsize_axesticks)
        
        data2 = np.sort(data)  
        pos = int(n * conf / 100)  
        set_rounded = round(data2[pos],2)
        count, bins = np.histogram(data2, int(n/40))  

        # Encontrar el índice del bin correspondiente a DeltaH2[Pos]  
        bin_index = np.digitize(data2[pos], bins) - 1  

        # Crear las barras del histograma con colores personalizados  
        for i in range(len(bins) - 1):  
            if i < bin_index:  
                ax1.bar(bins[i], count[i], width=bins[i + 1] - bins[i], color='tab:blue')  
            else:  
                ax1.bar(bins[i], count[i], width=bins[i + 1] - bins[i], color='tab:gray')  

        # Crear el eje secundario a la derecha  
        ax2 = ax1.twinx()  
        ax2.set_ylabel(y2_label, fontsize=fontsize_axeslabels)  

        # Calcular la distribución acumulativa y normalizarla (de 0 a 1)  
        cdf = np.cumsum(count)  
        cdf_normalized = cdf / cdf[-1]  

        # Trazar la curva acumulada en el eje secundario  
        ax2.plot(bins[:-1], cdf_normalized, color='darkorange', linewidth=2)  
        ax2.set_ylim(0,1)
        ax2.tick_params(axis='both', which='major', labelsize=fontsize_axesticks)

        ax2.axhline(conf / 100, linestyle='dashed', color='gray', linewidth=1)
        intersection_index = np.where(cdf_normalized >= conf / 100)[0][0]  
        intersection_point = (bins[intersection_index], conf / 100)  
        ax2.plot(intersection_point[0], intersection_point[1], 'ro')
        ax2.axvline(intersection_point[0], linestyle='dashed', color='gray', linewidth=1)
        ax2.text(intersection_point[0], conf*0.8/100, str(set_rounded), ha='right', va='top', fontsize=fontsize_axeslabels, transform=ax2.get_xaxis_transform(),rotation=90)
        try:
            print("Characteristic value: ",set_rounded," (confidence level:",conf,"%) ")
        except:
            pass
        plt.show()
    else:
        set_rounded = round(data[0],2)
        try:
            print("Characteristic value: ",set_rounded)
        except:
            pass
        
    return fig,[ax1,ax2]