# -*- coding: utf-8 -*-
##### ----------------------------- IMPORTS ----------------------------- #####
import os
import sys
import yaml
import click

# add plot path
parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path = os.path.join(parent_path, 'plots')
if path not in sys.path:
    sys.path.append(path)

# create settings file if it does not exists
temp_path_yaml = 'temp_settings.yaml'
load_path_yaml = 'settings.yaml'
if not os.path.isfile(load_path_yaml):
    import shutil
    shutil.copy(temp_path_yaml, load_path_yaml)
##### ------------------------------------------------------------------- #####

def load_yaml(settings_path):
    with open(settings_path, 'r') as file:
        return yaml.load(file, Loader=yaml.FullLoader)

def save_yaml(settings, settings_path):
    with open(settings_path, 'w') as file:
        yaml.dump(settings, file)


@click.group()
@click.pass_context
def main(ctx):
    """
    
    \b---------------------------------------------------                       
    \b---------------- SAKE CONNECTIVITY ----------------                       
    \b---------------------------------------------------                     
    
    """
    
    # load yaml 
    settings = load_yaml(load_path_yaml)
    ctx.obj = settings.copy()
        
### ------------------------------ SET PATH ------------------------------ ### 
@main.command()
@click.argument('path', type=str)
@click.pass_context
def setpath(ctx, path):
    """Set path to index file parent directory"""
    
    # check if index file exists
    index_path = os.path.join(path, ctx.obj['sake_index'])
    if not os.path.isfile(index_path):
        click.secho(f"\n --> File '{ctx.obj['sake_index']}' "  +\
                        f"was not found in '{path}'.\n",
                        fg='yellow', bold=True)
        path=''
            
    # save path
    ctx.obj.update({'search_path': path})
    save_yaml(ctx.obj, load_path_yaml)
    click.secho(f"\n -> Path was set to:'{path}'.\n", fg='green', bold=True)
  
def check_path(ctx):
    if not ctx.obj['search_path']:
        print_str = "\n --> Path was not found, please use -setpath- command."
        click.secho(print_str, fg='yellow', bold=True)
        raise(Exception(print_str))
      
@main.command()
@click.option('--ws', type=str, help='Enter window size (s), e.g. 30')
@click.option('--method', type=str, help='Enter method type: E.g. tort')
@click.pass_context
def coupling(ctx, ws, method='tort'):
    """
    Calculate phase amplitude coupling
    """
    
    # check path
    check_path(ctx)

    if not ws:
        click.secho("\n -> Please enter window size' e.g. --ws 30.\n", fg='yellow', bold=True)
        return
    
    # downsample
    from preprocess import batch_downsample
    downsampled_df = batch_downsample(ctx.obj['search_path'], ctx.obj['sake_index'], new_fs=ctx.obj['new_fs'])
    click.secho(f"\n -> Data successfully downsampled to {ctx.obj['new_fs']} Hz'.\n", fg='green', bold=True)
    
    # get coupling index
    from phase_amp import phaseamp_batch
    data = phaseamp_batch(downsampled_df, ctx.obj['iter_freqs'], ctx.obj['new_fs'], int(ws))

    # store data
    data.to_pickle(os.path.join(ctx.obj['search_path'], 'phase_amp.pickle'))
    click.secho(f"\n -> Coupling completed and data were stored to {ctx.obj['search_path']}'.\n",
                fg='green', bold=True)

    
@main.command()
@click.option('--ws', type=str, help='Enter window size (s), e.g. 5')
@click.option('--method', type=str, help='Analysis type (s), e.g. coh plv')
@click.pass_context
def coherence(ctx, ws, method='coh'):
    """
    Calculate coherence
    """
    # check path
    check_path(ctx)

    if not ws:
        click.secho("\n -> Please enter window size' e.g. --ws 5.\n", fg='yellow', bold=True)
        return
    
    methods = ['coh', 'plv', 'pli']
    method = method.split(' ')
    if not set(method) <= set(methods):
        click.secho(f"\n -> Got '{method}' instead of  '{methods}", fg='yellow', bold=True)
        return
    
    # downsample
    from preprocess import batch_downsample
    downsampled_df = batch_downsample(ctx.obj['search_path'], ctx.obj['sake_index'], new_fs=ctx.obj['new_fs'])
    click.secho(f"\n -> Data successfully downsampled to {ctx.obj['new_fs']} Hz'.\n", fg='green', bold=True)
    
    # calculate coherence
    from coherence import coherence_batch
    data = coherence_batch(downsampled_df, ctx.obj['iter_freqs'], ctx.obj['new_fs'], int(ws), method=method)
    data.to_pickle(os.path.join(ctx.obj['search_path'], 'coherence.pickle'))
    click.secho(f"\n -> Coherence completed and data were stored to {ctx.obj['search_path']}'.\n",
                fg='green', bold=True)

@main.command()
@click.option('--method', type=str, help='Analysis type (s), e.g. coherence')
@click.option('--plottype', type=str, help='Analysis type (s), e.g. box')
@click.pass_context
def plot(ctx, method, plottype):
    """
    Interactive summary plot.

    """
    # check path
    check_path(ctx)
    
    # import modules
    import pandas as pd
    from facet_plot_gui import GridGraph
    
    if method == 'pac':
        file = 'phase_amp.pickle'
    elif method == 'coherence':
        file = 'coherence.pickle'
     
    # get data
    data = pd.read_pickle(os.path.join(ctx.obj['search_path'], file))
    
    # convert data to appropriate plotting format
    if plottype == 'time':
        plotdf = data.drop(columns='method', axis=1).set_index('animal')
        graph = GridGraph(ctx.obj['search_path'], method+'.csv', plotdf, x=plottype)
        graph.draw_psd()
    else:
        group_cols = list(data.columns[data.columns.get_loc('time') +1 :-1]) +['animal']
        plotdf = data.groupby(group_cols).mean().reset_index().drop(columns='time', axis=1)
        graph = GridGraph(ctx.obj['search_path'], method+'.csv', plotdf.set_index('animal'), x='band')
        graph.draw_graph(plottype)

    
# Execute if module runs as main program
if __name__ == '__main__':
    
    # start
    main(obj={})




# def normalize(df, y, base_condition, match_cols):
#     """
#     Normalize column in dataframe.

#     Parameters
#     ----------
#     df: pd.DataFrame
#     y : str, var name, e.g. power
#     base_condition : dict, {col:column matching condition, e.g. treatment
#                        val: str,  baseline  value, e.g. baseline}
#     match_cols: list, column to match for normalization

#     Returns
#     -------
#     None.

#     """
    
#     # create copy and add normalize
#     data = df.copy()
#     norm_y = 'norm_' + y
#     data[norm_y] = 0

#     # get unique conditions
#     combi_list=[]
#     for col in match_cols:
#         combi_list.append(data[col].unique())

#     # normalize to baseline
#     combinations = itertools.product(*combi_list)
#     pbar = tqdm(desc='Normalizing', total=len(list(combinations)))
#     for comb in itertools.product(*combi_list):       
        
#         # find unique treatments for each set of conditions
#         idx = np.zeros((len(data), len(match_cols)))
#         for i, col in enumerate(match_cols):
#             idx[:,i] = data[col] == comb[i]
#         temp = data[np.all(idx, axis=1)]
        
#         # normalize by baseline
#         baseline_idx = temp.index[temp[base_condition['col']] == base_condition['val']][0]
#         data.at[temp.index, norm_y] = data[y][temp.index] / data[y][baseline_idx]
        
#         pbar.update(1)   
#     pbar.close()
        
#     return data, norm_y
