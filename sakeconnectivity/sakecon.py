# -*- coding: utf-8 -*-

##### ----------------------------- IMPORTS ----------------------------- #####
import os
import sys
import subprocess
import webbrowser
from PyQt5 import QtCore, QtWidgets, QtTest 
from gui.sake_con_ui import Ui_mainWindow
from cli import load_yaml
import os
from PyQt5.QtGui import QMovie
import pandas as pd
##### ------------------------------------------------------------------- #####


# init gui app
app = QtWidgets.QApplication(sys.argv)
Dialog = QtWidgets.QMainWindow()
ui = Ui_mainWindow()
ui.setupUi(Dialog)
Dialog.show()
_translate = QtCore.QCoreApplication.translate
script_dir=os.path.dirname(os.path.realpath(__file__))

coher_funcs = {'Spectral':'coh',"Phase Locking Value":'plv',"Phase Lag Index":"pli"}
pac_funcs = {'Tort':'tort'}
plot_types = {'Time Series':'time','Bar':'bar','Scatter':'strip','Violin':'violin'}
ui.coherFuncBox.addItems(coher_funcs.keys())
ui.pacFuncBox.addItems(pac_funcs.keys())
ui.pacPlotBox.addItems(plot_types.keys())
ui.coherPlotBox.addItems(plot_types.keys())

logo_path = os.path.join(script_dir,'logo',r'sake connectivity logo.gif')
movie=QMovie(logo_path)
ui.logoLabel.setMovie(movie)
movie.start()

def update_norms():
    try:
        index=pd.read_csv(os.path.join(ui.pathEdit.text(),'index.csv'))

        ui.normCol.clear()
        ui.normGroup.clear()
        ui.normCol.addItems(list(index.columns)[list(index.columns).index('stop_time')+1:-1])
        norm_col_changed()
    except:pass

def setpath():
    """Set path to index file parent directory"""    
    # add path to original ctx.obj
    widget=QtWidgets.QFileDialog()
    path=widget.getExistingDirectory(None,_translate("mainWindow", 'Set path for index.csv'),r"C:")
    ui.pathEdit.setText(_translate("mainWindow",path))
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "setpath", path],capture_output=True)
    ui.errorBrowser.setText(_translate("mainWindow",str(msg.stdout.decode())))
    update_norms()
    

    
ui.pathButton.clicked.connect(lambda:setpath())

def pac():

    ui.errorBrowser.setText(_translate("mainWindow","Running PAC... Check Terminal for Progess Bar"))
    
    QtTest.QTest.qWait(100)
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "coupling",
                        "--ws", int(ui.binEdit.text()),
                        "--method", pac_funcs[ui.pacFuncBox.currentText()],
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not perform PAC... \nCheck terminal for errors..."))
        return

    
    ui.errorBrowser.setText(_translate("mainWindow",'PAC Complete!'))
    
    
ui.pacCalcButton.clicked.connect(lambda:pac())

def coherence():
    ui.errorBrowser.setText(_translate("mainWindow","Running Coherence... Check Terminal for Progess Bar"))
    
    QtTest.QTest.qWait(100)
    print( os.path.join(script_dir,r"cli.py"))
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "coherence",
                        "--ws", ui.binEdit.text(),
                        "--method", coher_funcs[ui.coherFuncBox.currentText()],
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not perform Coherence Calc... \nCheck terminal for errors..."))
        return

    
    ui.errorBrowser.setText(_translate("mainWindow",'Coherence Calculation Complete!'))
    
ui.coherCalcButton.clicked.connect(lambda:coherence())

def openSettings():
    webbrowser.open(os.path.join(script_dir, os.path.join(script_dir,r"settings.yaml")))
    
ui.actionSettings.triggered.connect(lambda:openSettings())

def get_norm_string():
    #check for normalize
    norm_string="false"
    if ui.normCheckBox.isChecked():
        norm_string="{}^{}".format(ui.normCol.currentText(),ui.normGroup.currentText())
    print(norm_string)
    return norm_string

def plot_pac():
    
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "plot",
                        "--method", "pac",
                        "--plottype", plot_types[ui.coherPlotBox.currentText()],
                        "--norm", get_norm_string(),
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not plot... \nCheck terminal for errors..."))
        return
    
ui.pacPlotButton.clicked.connect(lambda:plot_pac())

def plot_coher():
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "plot",
                        "--method", "coherence",
                        "--plottype", plot_types[ui.pacPlotBox.currentText()],
                        "--norm", get_norm_string(),
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not plot... \nCheck terminal for errors..."))
        return

ui.coherPlotButton.clicked.connect(lambda:plot_coher())


def norm_col_changed():
    """
    Manual verification of PSDs
    """
    try:
        index=pd.read_csv(os.path.join(ui.pathEdit.text(),'index.csv'))
        ui.normGroup.clear()
        if ui.normCol.currentText() == 'transform':
            ui.normGroup.addItems(['total_power'])
        else:
            ui.normGroup.addItems(index[ui.normCol.currentText()].unique())
    except:pass
        
    
ui.normCol.activated.connect(lambda:norm_col_changed())

# Execute if module runs as main program
if __name__ == '__main__': 
    settings = load_yaml(os.path.join(script_dir,r"settings.yaml"))
    
    ui.pathEdit.setText(_translate("SAKEDSP", settings['search_path']))
    update_norms()
    app.exec_()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    