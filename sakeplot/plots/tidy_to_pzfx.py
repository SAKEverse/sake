# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import pandas as pd
##### ------------------------------------------------------------------- #####

head_string = '''<?xml version="1.0" encoding="UTF-8"?>
<GraphPadPrismFile xmlns="http://graphpad.com/prism/Prism.htm" PrismXMLVersion="5.00">
<Created>
<OriginalVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="6.0.1.298" RegisteredTo="      " Login=" " DateTime=" "/>
<MostRecentVersion CreatedByProgram="GraphPad Prism" CreatedByVersion="6.0.1.298" RegisteredTo="      " Login=" " DateTime=" "/>
</Created>
<InfoSequence>
<Ref ID="Info0" Selected="1"/>
</InfoSequence>
<Info ID="Info0">
<Title>Project info 1</Title>
<Notes>
</Notes>
<Constant><Name>Experiment Date</Name><Value>2021-11-11</Value></Constant>
<Constant><Name>Experiment ID</Name><Value/></Constant>
<Constant><Name>Notebook ID</Name><Value/></Constant>
<Constant><Name>Project</Name><Value/></Constant>
<Constant><Name>Experimenter</Name><Value/></Constant>
<Constant><Name>Protocol</Name><Value/></Constant>
</Info>

'''
foot_string='''<!--Analyses, graphs and layouts as compressed binary. Don't edit this part of the file.-->

<Template xmlns:dt="urn:schemas-microsoft-com:datatypes" dt:dt="bin.base64">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
kvEvAAAA//8DALVrVS0=</Template></GraphPadPrismFile>'''

def tidy_to_grouped(data,x,y,group):
    
    largest_n=0
    #iterate through the groups
    all_groups_string=''
    for cond in data[group].unique():
        # make new table with the filter index
        filtered=data[data[group]==cond]
        
        #check for larger n
        n=len(filtered[filtered[x]==filtered[x].unique()[0]]) 
        if n > largest_n:largest_n=n
        #setup group title
        group_string='''<YColumn Width="{}" Decimals="6" Subcolumns="{}">
<Title>{}</Title>\n'''.format((n*4+1)*n,n,cond)
        #iterate through data points
        for i in range(n):
            #iterate through rows (x)
            subcol_string='<Subcolumn>\n'
            for row in filtered[x].unique():
                row_string='<d>{:e}</d>\n'.format(filtered[filtered[x]==row][y].iloc[i])
                subcol_string+=row_string
            subcol_string+='</Subcolumn>\n'
            group_string+=subcol_string
        group_string+='</YColumn>\n'
        all_groups_string+=group_string
        
    #set up rows
    row_names="<d>"+"</d>\n<d>".join(list(data[x].unique()))+'</d>'
    row_string = '''<TableSequence Selected="1">

<Ref ID="Table0" Selected="1"/>
</TableSequence>
<Table ID="Table0" XFormat="none" YFormat="replicates" Replicates="{}" TableType="TwoWay" EVFormat="AsteriskAfterNumber">
<Title>Data 1</Title>
<RowTitlesColumn Width="{}">
<Subcolumn>
{}
</Subcolumn>
</RowTitlesColumn>'''.format(largest_n,largest_n*4+1,row_names)

    table_string = row_string+all_groups_string+'</Table>\n\n'
    
    out_string= head_string + table_string + foot_string
    return out_string
    
  
if __name__ == '__main__':
    path= r"C:\Users\Grant\Downloads"
    filename=r"melt_index.csv"
    data=pd.read_csv(os.path.join(path,filename),index_col=0)
    out=tidy_to_grouped(data,'freq','power_area','treatment')
    text_file = open(os.path.join(path,"sample.pzfx"), "w")
    n = text_file.write(out)
    text_file.close()
