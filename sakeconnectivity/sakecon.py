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
plot_types = {'Time Series':'time','Bar':'bar','Scatter':'scatter','Violin':'violin'}
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
    #check for normalize
    norm_string="false"
    if ui.normCheckBox.isChecked():
        norm_string="{}^{}".format(ui.normCol.currentText(),ui.normGroup.currentText())
    ui.errorBrowser.setText(_translate("mainWindow","Running PAC... Check Terminal for Progess Bar"))
    
    QtTest.QTest.qWait(100)
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "coupling",
                        "--ws", int(ui.pacBinEdit.text()),
                        "--method", pac_funcs[ui.pacFuncBox.currentText()],
                        "--norm", norm_string])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not perform PAC... \nCheck terminal for errors..."))
        return

    
    ui.errorBrowser.setText(_translate("mainWindow",'PAC Complete!'))
    
    
ui.pacCalcButton.clicked.connect(lambda:pac())

def coherence():
    #check for normalize
    norm_string="false"
    if ui.normCheckBox.isChecked():
        norm_string="{}^{}".format(ui.normCol.currentText(),ui.normGroup.currentText())
        
    ui.errorBrowser.setText(_translate("mainWindow","Running Coherence... Check Terminal for Progess Bar"))
    
    QtTest.QTest.qWait(100)
    print( os.path.join(script_dir,r"cli.py"))
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "coherence",
                        "--ws", ui.pacBinEdit.text(),
                        "--method", coher_funcs[ui.coherFuncBox.currentText()],
                        "--norm", norm_string])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not perform Coherence Calc... \nCheck terminal for errors..."))
        return

    
    ui.errorBrowser.setText(_translate("mainWindow",'Coherence Calculation Complete!'))
    
ui.coherCalcButton.clicked.connect(lambda:coherence())

def openSettings():
    webbrowser.open(os.path.join(script_dir, os.path.join(script_dir,r"settings.yaml")))
    
ui.actionSettings.triggered.connect(lambda:openSettings())

def plot_pac():

    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "plot",
                        "--method", "pac",
                        "--type", plot_types[ui.coherPlotBox.currentText()],
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not plot... \nCheck terminal for errors..."))
        return
    
ui.pacPlotButton.clicked.connect(lambda:plot_pac())

def plot_coher():
    msg=subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "plot",
                        "--method", "coherence",
                        "--type", plot_types[ui.pacPlotBox.currentText()],
                        ])
    if msg.returncode != 0:
        ui.errorBrowser.setText(_translate("mainWindow","ERROR: Could not plot... \nCheck terminal for errors..."))
        return

ui.coherPlotButton.clicked.connect(lambda:plot_coher())

def norm_changed():
    """
    Manual verification of PSDs
    """
    
    # if ui.normCheckBox.isChecked():
    #     subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "normalize", "--enable", "true", "--column", ui.normCol.currentText(), "--group", ui.normGroup.currentText()])
    # else:
    #     subprocess.run(["python", os.path.join(script_dir,r"cli.py"), "normalize", "--enable", "false", "--column", ui.normCol.currentText(), "--group", ui.normGroup.currentText()])
        
    
ui.normCheckBox.stateChanged.connect(lambda:norm_changed())

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
        norm_changed()
    except:pass
        
    
ui.normCol.activated.connect(lambda:norm_col_changed())
ui.normGroup.activated.connect(lambda:norm_changed())

# Execute if module runs as main program
if __name__ == '__main__': 
    settings = load_yaml(os.path.join(script_dir,r"settings.yaml"))
    
    ui.pathEdit.setText(_translate("SAKEDSP", settings['search_path']))
    update_norms()
    app.exec_()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    