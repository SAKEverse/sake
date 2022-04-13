### ---------------------- IMPORTS ---------------------- ###
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.widgets import SpanSelector
from PyQt5 import QtCore
from processing.stft import f_fill
from processing.outlier_detection import get_outliers
### ----------------------------------------------------- ###

def find_nearest(array, values):
    """
    Find index of values in array.

    Parameters
    ----------
    array : np.array to index
    values : list/array values used to search array

    Returns
    -------
    idx : array, index of values from array

    """
    array = np.asarray(array)
    idx = np.zeros(len(values), dtype=int)
    for i, value in enumerate(values):
        idx[i] = int((np.abs(array - value)).argmin())
    return idx

def replace_nans_with_row_median(pmat):
    """
    Replace NaNs with row median.

    Parameters
    ----------
    pmat : np.ndarray

    Returns
    -------
    pmat : np.ndarray

    """
    
    # find row median value
    row_med = np.nanmedian(pmat, axis=1)
    
    # find indices that you need to replace
    inds = np.where(np.isnan(pmat))
    
    # place row medians in the indices.
    pmat[inds] = np.take(row_med, inds[0])
    
    return pmat
    

def remove_outliers(pmat:np.ndarray, outlier_window, outlier_threshold):
    """
    Remove outliers based on MAD.

    Parameters
    ----------
    pmat : np.ndarray

    Returns
    -------
    pmat : np.ndarray
    outliers : np.ndarray

    """
    # get outliers
    outliers = get_outliers(np.mean(pmat, axis=0), outlier_window, outlier_threshold)

    # replace outliers with nans
    pmat[:, outliers] = np.nan
    
    # interpolate missing data
    pmat = f_fill(pmat, axis=1)
    
    # replace nans row median value
    pmat = replace_nans_with_row_median(pmat)
    
    return pmat, outliers
        
def remove_regions(pmat, idx):
    """
    Remove regions as indicated in index.

    Parameters
    ----------
    pmat : np.ndarray
    idx : 2d array (rows = regions, columns = (start,stop))

    Returns
    -------
    pmat : np.ndarray

    """
    if np.sum(idx) == 0:
        return pmat
    
    for i in range(idx.shape[0]):
        
        # replace outliers with nans
        pmat[:, idx[i,0]:idx[i,1]] = np.nan
        
    # replace nans row median value
    pmat = replace_nans_with_row_median(pmat)
    
    return pmat
    

class matplotGui:
    """
        Matplotlib GUI for user seizure verification.
    """
    
    ind = 0 # set internal counter
       
    def __init__(self, settings, index_df, power_df):

        # get values from dictionary
        for key, value in settings.items():
               setattr(self, key, value)

        # wait time after plot
        self.wait_time = 0.1 # in seconds
        self.bcg_color = {-1:'w', 0:'salmon', 1:'palegreen'}

        # pass object attributes to class
        self.power_df = power_df                      
        self.index_df = index_df
        
        # if first time verifying initialize values
        if 'time_rejected' not in self.index_df.columns:
            self.index_df.insert(0, 'time_rejected', '')       
        if 'accepted' not in self.index_df.columns:
            self.index_df.insert(0, 'accepted', -1)
            
        # replace nans with empty string
        self.index_df['time_rejected'].fillna('', inplace=True)
        
        # create array to store remove index
        self.remove_idx = np.zeros(len(self.index_df), dtype=object)
        
        # create figure and axis
        self.fig, self.axs = plt.subplots(2, 1, sharex=False, figsize=(8,8))

        # remove top and right axis
        self.axs[0].spines["top"].set_visible(False)
        self.axs[0].spines["right"].set_visible(False)
        self.axs[1].spines["top"].set_visible(False)
        self.axs[1].spines["right"].set_visible(False)
               
        # create first plot
        self.plot_data()
        
        # set useblit True on gtkagg for enhanced performance
        _ = SpanSelector(self.axs[0], self.on_select, 'horizontal', useblit=True,
            rectprops=dict(alpha=0.5, facecolor='tab:blue'))
        
        # connect callbacks and add key legend 
        plt.subplots_adjust(bottom=0.15)
        self.fig.suptitle('Select PSDs', fontsize=12)   
        self.fig.text(0.5, 0.04, '** Accept/Reject = a/r,         Previous/Next = \u2190/\u2192,         \
Increase/Decrease threshold = \u2191/\u2193 **\
\n**     Clear Highlighted (rejected) Regions = c,         Enter = Save, Esc = Close(no Save)      **' ,
                      ha="center", bbox=dict(boxstyle="square", ec=(1., 1., 1.), fc=(0.9, 0.9, 0.9),))
        self.fig.canvas.callbacks.connect('key_press_event', self.keypress)
        self.fig.canvas.callbacks.connect('close_event', self.close_event)
        
        # disable x button
        win = plt.gcf().canvas.manager.window
        win.setWindowFlags(win.windowFlags() | QtCore.Qt.CustomizeWindowHint)
        win.setWindowFlags(win.windowFlags() & ~QtCore.Qt.WindowCloseButtonHint)

        plt.show()

        
    def get_index(self):
        """
        Get dataframe index, reset when limits are exceeded
        
        Returns
        -------
        None.
        """

        # reset counter when limits are exceeded
        if self.ind >= len(self.index_df):
            self.ind = 0 
            
        elif self.ind < 0:
            self.ind = len(self.index_df)-1
       
        # set counter to internal counter
        self.i = self.ind
        
    def get_index_from_str(self, t):
        """
        Get index from string (whitespace sep).

        Parameters
        ----------
        t : array, time

        Returns
        -------
        idx : 2d array (rows = regions, columns = (start,stop))

        """
        
        selected_span = np.array([int(s) for s in str.split(self.index_df['time_rejected'][self.i]) if s.isdigit()])
        idx = find_nearest(t, selected_span)
        idx = np.reshape(idx, (int(len(idx)/2),2))
        return idx
        
    def set_background_color(self, axis =  [0,1]):
        """
        Set background color

        Returns
        -------
        None.

        """
        clr = self.bcg_color[self.index_df['accepted'][self.i]]
        for ax in axis:
            self.axs[ax].set_facecolor(clr)

    def plot_data(self, **kwargs):
        """
        Plot power area time plot and PSD for outlier and experiment verification
        """
        
        # get index
        self.get_index()

        # clear graph
        self.axs[0].clear()
        self.axs[1].clear() 
        
        # get pmat and freq
        pmat = self.power_df['pmat'][self.i].copy()
        freq = self.power_df['freq'][self.i].copy()
        
        # get time plot
        time_plot = np.mean(pmat, axis = 0)
        t = np.arange(0, time_plot.shape[0], 1) * self.fft_overlap * self.fft_win

        # plot time plot (before outlier removal)
        self.axs[0].plot(t, time_plot, color='black', label = self.index_df.index[self.i])
        
        # pass index to store array
        idx = self.get_index_from_str(t)
        self.remove_idx[self.i] = idx
        
        # plot highlighted region (bad)
        for i in range(idx.shape[0]):
            y_bad_section = time_plot[idx[i, 0]:idx[i, 1]+1]
            t_bad_section = t[idx[i, 0]:idx[i, 1]+1]
            self.axs[0].plot(t_bad_section, y_bad_section, color='darkmagenta')
        
        # plot PSD (before outlier removal)
        psd = np.mean(pmat, axis = 1)
        sem = np.std(pmat, axis = 1) / np.sqrt(pmat.shape[1])
        self.axs[1].plot(freq, psd, color='black', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs[1].fill_between(freq, psd+sem, psd-sem, color = 'black', alpha=0.2)
        
        # remove bad regions and outliers
        pmat = remove_regions(pmat, idx)
        pmat, outliers = remove_outliers(pmat, self.outlier_window, self.outlier_threshold)
        
        # plot time plot (after outlier removal)
        self.axs[0].plot(t[outliers], time_plot[outliers], color='orange', linestyle='', marker='x')
        
        # plot PSD (after outlier removal)
        psd = np.mean(pmat, axis = 1)
        sem = np.std(pmat, axis = 1) / np.sqrt(pmat.shape[1])
        self.axs[1].plot(freq, psd, color='orange', linewidth=1.5, alpha=0.9, label = self.index_df.index[self.i])
        self.axs[1].fill_between(freq, psd+sem, psd-sem, color = 'orange', alpha=0.2)
        self.axs[1].set_ylim(np.min(psd), np.max(psd)*1.3)
        
        # format graphs
        self.set_background_color()
        self.axs[0].legend(loc = 'upper right')
        self.axs[0].set_xlabel('Time (Seconds)')
        self.axs[0].set_ylabel('Mean Power')
        self.axs[1].legend(['Outlier_threshold = {:.1f}'.format(self.outlier_threshold)], loc = 'upper right')
        self.axs[1].set_xlabel('Frequency (Hz)')
        self.axs[1].set_ylabel('Power (V^2/Hz)')
        self.fig.canvas.draw()
       

            
    def save_idx(self):
        """
        Saves accepted PSD index and mat files.

        Returns
        -------
        None.
        """

        # check if all PSDs were verified
        if np.any(self.index_df['accepted'] == -1):
            print('\n****** Some PSDs were not verified ******\n')
            
        # store index csv
        self.index_df.to_csv(self.index_path, index=False)
            
        # get accepted PSDs
        accepted_idx = self.index_df['accepted'] == 1
        self.index_df = self.index_df[accepted_idx]
        self.power_df = self.power_df[accepted_idx]
        self.remove_idx = self.remove_idx[accepted_idx]
        
        # drop extra columns
        self.index_df = self.index_df.drop(columns = ['accepted'])
        
        # reset index
        self.index_df.reset_index(drop=True, inplace=True)
        self.power_df.reset_index(drop=True, inplace=True)
        
        # remove outliers and bad regions
        for i in tqdm(range(len(self.power_df))):
            pmat = self.power_df['pmat'][i]
            pmat = remove_regions(pmat, self.remove_idx[i])
            self.power_df.at[i, 'pmat'], _ = remove_outliers(pmat, self.outlier_window, self.outlier_threshold)
        
        # save verified index and power_df file
        self.index_df.to_csv(self.index_verified_path, index = False)
        self.power_df.to_pickle(self.power_mat_verified_path)
        
        print(f"Verified PSDs were saved in '{self.search_path}'.\n")  
    
    
    ### ------ Span Select ------ ###
    def on_select(self, xmin, xmax):
        """
        User selection of x span.
        
        Parameters
        ----------
        xmin : float/int
            Xmin-user selection.
        xmax : float/int
            Xmax-user selection.

        """
        
        # do not allow values smaller than zero
        if xmin < 0:
            xmin = 0
        
        # pass to index
        self.index_df.at[self.i, 'time_rejected'] += ' ' + str(int(xmin)) + ' '  + str(int(xmax))
        
        # highlight user selected region
        self.plot_data()
    
    ### --- Clear highlighted region --- ###
    def clear_traces(self):
        
        self.index_df.at[self.i, 'time_rejected'] = ''
        
        # highlight user selected region
        self.plot_data()
    
    ## ------  Cross press ------ ## 
    def close_event(self, event):
        plt.close()

    ## ------  Keyboard press ------ ##     
    def keypress(self, event):
        # print(event.key)
        if event.key == 'right':
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'left':
            self.ind -= 1 # subtract one to class index
            self.plot_data() # plot
        
        if event.key == 'up':
            self.outlier_threshold += 0.5
            self.plot_data() # plot
            
        if event.key == 'down':
            self.outlier_threshold -= 0.5
            self.plot_data() # plot

        if event.key == 'a':
           # set values to arrays
           self.index_df.at[self.i, 'accepted'] = 1
                    
           # draw and pause for user visualization
           self.set_background_color()
           plt.draw()
           plt.pause(self.wait_time)
           
           # plot next event
           self.ind += 1 # add one to class index
           self.plot_data() # plot
          
        if event.key == 'r':
            # set values to arrays
            self.index_df.at[self.i, 'accepted'] = 0
            
            # draw and pause for user visualization
            self.set_background_color()
            plt.draw()
            plt.pause(self.wait_time)
            
            # plot next event
            self.ind += 1 # add one to class index
            self.plot_data() # plot
            
        if event.key == 'c':
            self.clear_traces()
            
        if event.key == 'ctrl+a':
           # set values to arrays
           self.index_df.at[:, 'accepted'] = 1
           
           # draw and pause for user visualization
           self.set_background_color()
           plt.draw()
           plt.pause(self.wait_time)

        if event.key == 'ctrl+r':
            # set values to arrays
            self.index_df.at[:, 'accepted'] = 0
            
            # draw and pause for user visualization
            self.set_background_color()
            plt.draw()
            plt.pause(self.wait_time)
            
        if event.key == 'escape':
              plt.close()
        
        if event.key == 'enter':
            self.save_idx()
            plt.close() # trigger close callback


            
# if __name__ == '__main__':
#     import yaml,os
#     import pandas as pd
#     from load_index import load_index

#     parent_folder = r'X:\Alyssa\EtOH_paper\acute_raw_data\etoh'

#     # define frequencies of interest
#     with open('settings.yaml', 'r') as file:
#         settings = yaml.load(file, Loader=yaml.FullLoader)
        
#     settings.update({'outlier_threshold':5, 'outlier_window':11})
    
#     # get path to files
#     settings.update({'index_path': os.path.join(parent_folder, settings['file_index'])})
#     settings.update({'power_mat_path': os.path.join(parent_folder, settings['file_power_mat'])})
#     settings.update({'index_verified_path': os.path.join(parent_folder, settings['file_index_verified'])})
#     settings.update({'power_mat_verified_path': os.path.join(parent_folder, settings['file_power_mat_verified'])})
    
#     #### ---------------------------------------------------------------- ####
    
#     # load index and power dataframe
#     index_df = load_index(os.path.join(parent_folder, settings['file_index']))
#     power_df = pd.read_pickle(os.path.join(parent_folder, settings['file_power_mat']))
       
#     # pmat, outliers = remove_outliers(power_df['pmat'][0], 11, 5)
#     callback = matplotGui(settings, index_df, power_df)






























