####### -------------------------- IMPORTS ------------------------ ########
import os
import sys
import yaml
import click
import pandas as pd
settings_yaml = 'settings.yaml'
load_path_yaml = 'path.yaml'

# add plot path
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path = os.path.join(parent_path, 'plots')
if path not in sys.path:
    sys.path.append(path)
    
######## ---------------------------------------------------------- ########

def load_yaml(settings_path):
    with open(settings_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def save_yaml(settings, settings_path):
    with open(settings_path, 'w') as file:
        yaml.dump(settings, file)
        
# define template settings path and load template settings
temp_settings = load_yaml(settings_yaml)

# if load path yaml does not exist create
if not os.path.isfile(load_path_yaml):
    save_yaml({'search_path':''}, load_path_yaml)
######## ---------------------------------------------------------- ########
        
def main_check(ctx):
    """
    Check if files and settings are present.

    Parameters
    ----------
    ctx : context class

    Returns
    -------
    dict, with settings

    """
    # get path and settings
    path = load_yaml(load_path_yaml)
    settings_path = os.path.join(path['search_path'], settings_yaml)
    if not os.path.isfile(settings_path):
        settings_path = settings_yaml 
    settings = load_yaml(settings_path)
    
    # check if keys match otherwise load original settings
    if settings.keys() != temp_settings.keys():
        settings = temp_settings
        
    # set variables and pass to context
    ctx.obj = settings.copy()
        
    # save original settings (keep separate from context)
    ctx.obj.update({'settings': settings,
                    'search_path': path['search_path'],
                     'settings_path':settings_path}) 

    # get path to files
    paths = {'index_path': 'file_index', 'power_mat_path': 'file_power_mat',
             'index_verified_path': 'file_index_verified',
             'power_mat_verified_path': 'file_power_mat_verified'}
    for file_path, file_name in paths.items():
        ctx.obj.update({file_path: os.path.join(ctx.obj['search_path'], ctx.obj[file_name])})
    return ctx.obj
    

@click.group()
@click.pass_context
def main(ctx):
    """
    
 \b   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄    ▄  ▄▄▄▄▄▄▄▄▄▄▄                                                         
 \b   ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌▐░░░░░░░░░░░▌                             
 \b   ▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░▌ ▐░▌ ▐░█▀▀▀▀▀▀▀▀▀                              
 \b   ▐░▌          ▐░▌       ▐░▌▐░▌▐░▌  ▐░▌                                       
 \b   ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌░▌   ▐░█▄▄▄▄▄▄▄▄▄                           
 \b    ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌    ▐░░░░░░░░░░░▌                             
 \b     ▀▀▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌░▌   ▐░█▀▀▀▀▀▀▀▀▀                              
 \b             ▐░▌▐░▌       ▐░▌▐░▌▐░▌  ▐░▌                                       
 \b    ▄▄▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░▌ ▐░▌ ▐░█▄▄▄▄▄▄▄▄▄                              
 \b   ▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░▌  ▐░▌▐░░░░░░░░░░░▌                             
 \b    ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀    ▀  ▀▀▀▀▀▀▀▀▀▀▀                              
                                                                   
    """
    ctx.obj = main_check(ctx)

  
@main.command()
# @click.pass_context ctx
def checkpath():
    """Reserved for gui use (runs main with checks)""" 
    return
           
### ------------------------------ SET PATH ------------------------------ ### 
@main.command()
@click.argument('path', type = str)
@click.pass_context
def setpath(ctx, path):
    """Set path to index file parent directory"""    
    # save new path and settings
    save_yaml({'search_path':path}, load_path_yaml)
    ctx.obj.update({'index_path': os.path.join(path, ctx.obj['file_index'])})
    
    # check for index file
    if not os.path.isfile(ctx.obj['index_path']):
        click.secho(f"\n --> File '{ctx.obj['file_index']}' "  +\
                        f"was not found in '{ctx.obj['search_path']}'.\n",
                        fg = 'yellow', bold = True)
        return
            
    save_yaml(ctx.obj['settings'], os.path.join(path, settings_yaml))
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg = 'green', bold = True)


### ---------------------- SET NORMALIZE SETTINGS ------------------------- ###  
@main.command()
@click.option('--enable', type = bool, help = 'Enable normalization')
@click.option('--column', type = str, help = 'Category to normalize')
@click.option('--group', type = str, help = 'Group to use for normalization')
@click.pass_context
def normalize(ctx, enable, column, group):
    """Update settings file with norm data""" 

    if enable:
        ctx.obj['settings']['normalize'] = 1
        ctx.obj['settings']['norm_groups'] = [column, group]
    else:
        ctx.obj['settings']['normalize'] = 0 

    # write to file  
    save_yaml(ctx.obj['settings'], os.path.join(ctx.obj['search_path'], settings_yaml))
    return


### ------------------------------ STFT ---------------------------------- ###     
@main.command()
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.option('--njobs', type = str, help = 'Enter number of threads to use')
@click.pass_context
def stft(ctx, freq, njobs):
    """Runs the Short-time Fourier Transform"""

    # check for index file
    if not os.path.isfile(ctx.obj['index_path']):
        click.secho(f"\n\n --> File '{ctx.obj['file_index']}' "  +\
                        f"was not found in '{ctx.obj['search_path']}'.\n",
                        fg = 'yellow', bold = True)
        return

    # get frequency
    if freq is not None:
        freq_range = [int(i) for i in freq.split('-')]    
        if len(freq_range) !=2:
            
            click.secho(f"\n -> '{freq}' could not be parsed."  +\
                            "Please use the following format: 1-30.\n",
                            fg = 'yellow', bold = True)
            return
        ctx.obj['fft_freq_range'] = freq_range
        
        
    from processing.batch_stft import BatchStft
    import pandas as pd

    # load index 
    index_df = pd.read_csv(ctx.obj['index_path'])
                          
    # get power
    if njobs:
        njobs = int(njobs)
   
    obj = BatchStft(ctx.obj, index_df, njobs)
    power_df = obj.get_pmat_batch()
    
    # save index and power
    power_df.to_pickle(ctx.obj['power_mat_path'])
    if len(power_df)!=0:
        obj.index_df.to_csv(ctx.obj['index_path'], index = False)
    
    # get freq_range for display
    freq_range = '-'.join(list(map(str, ctx.obj['fft_freq_range']))) + ' Hz'
    
    click.secho(f"\n -> Analysis completed: {freq_range} and" +\
                f" file saved in:'{ctx.obj['search_path']}'.\n",
                fg = 'green', bold = True)

### ------------------------------ VERIFY PSDs ---------------------------------- ###      
@main.command()
@click.option('--outlier_threshold', type = str, help = 'Enter outlier threshold, e.g. 4.5')
@click.pass_context
def verify(ctx, outlier_threshold):
    """
    Manual verification of PSDs
    """
    
    # update outlier if present
    if outlier_threshold is not None:
        ctx.obj.update({'outlier_threshold': float(outlier_threshold)})

    # check if index file was not found
    if not os.path.isfile(ctx.obj['power_mat_path']):
            click.secho(f"\n -> File '{ctx.obj['file_power_mat']}' "  +\
                            f"was not found in '{ctx.obj['search_path']}'.\n",
                            fg = 'yellow', bold = True)
            return
    
    from processing.verify_psd import matplotGui
        
    # load files
    index_df = pd.read_csv(ctx.obj['index_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_path'])

    # init gui object
    matplotGui(ctx.obj, index_df, power_df)

 
### ------------------------------ PLOT ---------------------------------- ###     
@main.command()
@click.option('--plot_type', type = str, help = 'Enter plot type, e.g. psd')
@click.option('--freq', type = str, help = 'Enter frequency range, e.g. 1-30')
@click.option('--kind', type = str, help = 'Enter graph kind, e.g. violin')
@click.pass_context
def plot(ctx, freq, plot_type, kind):
    """Enter plot menu"""
    
    # update kind if present
    if kind is not None:
        ctx.obj.update({'summary_plot_type':kind})
    
    # check if power mat exists 
    if not os.path.isfile(ctx.obj['power_mat_verified_path']):
        click.secho(f"\n -> File '{ctx.obj['power_mat_verified_path']}' was not found in '{ctx.obj['search_path']}'" + 
                    "Need to run 'stft' and 'verify' before plotting.\n", fg = 'yellow', bold = True)
        return
    
    # check if plot type is present in the correct format
    plot_type_options = ['power_area', 'power_ratio', 'psd', 'dist']
    
    if plot_type is None:
        click.secho("\n -> 'Missing argument 'plot_type'. Please use the following format: --plot_type power_area.\n"
                    , fg = 'yellow', bold = True)
        return
    
    if plot_type not in plot_type_options:
        click.secho(f"\n -> Got'{plot_type}' instead of {plot_type_options}\n",
                    fg = 'yellow', bold = True)
        return
        
    # load index and power mat
    index_df = pd.read_csv(ctx.obj['index_verified_path'])
    power_df = pd.read_pickle(ctx.obj['power_mat_verified_path'])  
    
    # normalize psds based on condition
    if ctx.obj['settings']['normalize']:
        from plots.psd_analysis import norm_power, norm_power_unpaired, norm_mean_power
        norm_func = norm_power
        
        if not ctx.obj['settings']['paired']:
            norm_func = norm_power_unpaired
        
        if 'transform' == ctx.obj['settings']['norm_groups'][0]:
            power_df = norm_mean_power(power_df)
        else:
            index_df, power_df = norm_func(index_df, power_df, ctx.obj['settings']['norm_groups'])
    
    # get categories
    categories = list(index_df.columns[index_df.columns.get_loc('stop_time')+1:])
    
    # remove unwanted categories from yaml file
    for category in ctx.obj['exclude_categories']:
        if category in categories:
            categories.remove(category)
    
    from facet_plot_gui import GridGraph
    if plot_type == 'power_area':
        from plots.psd_analysis import melted_power_area
        # get power area
        data = melted_power_area(index_df, power_df, ctx.obj['freq_ranges'], categories)
        
        # graph interactive summary plot
        GridGraph(ctx.obj['search_path'], 
                  ctx.obj['power_mat_verified_path'],
                  data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    if plot_type == 'power_ratio':
        from plots.psd_analysis import melted_power_ratio
        # get power ratio
        data = melted_power_ratio(index_df, power_df, ctx.obj['freq_ratios'], categories)
        
        # graph interactive summary plot
        GridGraph(ctx.obj['search_path'], 
                  ctx.obj['power_mat_verified_path'], 
                  data).draw_graph(ctx.obj['summary_plot_type'])
        return
    
    # get frequency
    if freq is not None:
        freq_range = [int(i) for i in freq.split('-')]
        if len(freq_range) !=2:
              click.secho(f"\n -> '{freq}' could not be parsed." +\
                          " Please use the following format: 1-30.\n", 
                          fg = 'yellow', bold = True)
              return
    else:
        click.secho("\n -> 'Missing argument 'freq'. " +\
                    "Please use the following format: --freq 1-30.\n",
                    fg = 'yellow', bold = True)
        return
    
    if plot_type == 'psd':
        from plots.psd_analysis import melted_psds
        # get psd data
        psd_data = melted_psds(index_df, power_df, freq_range, categories)
        
        # Graph interactive PSD
        GridGraph(ctx.obj['search_path'],  ctx.obj['psd_mat'], psd_data).draw_psd()
        
    if plot_type == 'dist':
        from plots.psd_analysis import melted_power_dist
        # get psd data
        pdf_data = melted_power_dist(index_df, power_df, freq_range, categories)

        # Graph interactive PSD
        GridGraph(ctx.obj['search_path'],  ctx.obj['psd_mat'],
                  pdf_data, x='power').draw_dist()
        
        
# Execute if module runs as main program
if __name__ == '__main__':
    
    # start
    main(obj={})

    
    
    
    
    
    
    
    
    