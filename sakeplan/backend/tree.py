### --------- IMPORTS --------- ###
import pandas as pd
import plotly.graph_objects as go
import plotly
import numpy as np
import itertools
### --------------------------- ###

def drawSankey(data):
    
    group_tree={col:{unique:len(data[col][data[col]==unique]) for unique in data[col].unique()} for col in data}
    uniques=[[unique for unique in data[col].unique()] for col in data]
    
    
    all_combo=[combo for col in range(1,len(uniques)+1) for combo in list(itertools.product(*uniques[:col]))]
    value=[(data.iloc[:,:len(combo)]==combo).all(axis=1).sum() for combo in all_combo]  
    
    
    labels=['Total']
    source=[]
    target=[]
    multiplier=1 #number of repeats due to previous groups
    k=0#cumulative multiplier (how many groups cames before this)
    
    for i,col in enumerate(group_tree):
        list(itertools.product(*uniques[:1]))
        labels+=list(group_tree[col].keys())*multiplier
        k+=multiplier
        target+=list(range(k,k+len(group_tree[col])*multiplier))
        source+= [[t]*len(group_tree[col]) for t in range(k-multiplier,k)]
        multiplier*=len(group_tree[col])
        
    source=[item for sublist in source for item in sublist]
    custom=np.array(value)
    colors=['rgb(250,250,250)' if value>=1 else 'lightgrey' for value in custom]
    

    fig = go.Figure(data=[go.Sankey(
        textfont = plotly.graph_objects.sankey.Textfont(size=18, color='black',family='Droid Serif'),
        arrangement='perpendicular',
        node = dict(
          pad = 50,
          thickness = 10,
          line = dict(color = 'black', width = 3),
          label = labels,
          hovertemplate='Mice: %{value}<extra></extra>',
          color = 'rgb(190,45,45)',
        ),
        link = dict(
          line = dict(color = 'rgb(130,130,130)', width = 3),
          source = source, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = target,
          value = value,
          customdata=custom,
          hovertemplate=' %{source.label}->%{target.label} Mice: %{customdata}<extra></extra>',
          color=colors
      ))])

    return fig

if __name__ == '__main__':
    example_path=r'C:\Users\gweiss01\Documents\GitHub\SAKE\example_data\sankey_data.csv'
    data=pd.read_csv(example_path,index_col=0)
    fig = drawSankey(data)
    plotly.offline.plot(fig)