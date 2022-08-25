# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\gui\sakeplot.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_SAKEDSP(object):
    def setupUi(self, SAKEDSP):
        SAKEDSP.setObjectName("SAKEDSP")
        SAKEDSP.resize(829, 461)
        self.centralwidget = QtWidgets.QWidget(SAKEDSP)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.normCol = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.normCol.setFont(font)
        self.normCol.setObjectName("normCol")
        self.gridLayout.addWidget(self.normCol, 5, 2, 1, 2)
        self.normGroup = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.normGroup.setFont(font)
        self.normGroup.setObjectName("normGroup")
        self.gridLayout.addWidget(self.normGroup, 5, 4, 1, 1)
        self.checkBoxNorm = QtWidgets.QCheckBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.checkBoxNorm.setFont(font)
        self.checkBoxNorm.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.checkBoxNorm.setTristate(False)
        self.checkBoxNorm.setObjectName("checkBoxNorm")
        self.gridLayout.addWidget(self.checkBoxNorm, 5, 1, 1, 1)
        self.labelPSDRange = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelPSDRange.setFont(font)
        self.labelPSDRange.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelPSDRange.setObjectName("labelPSDRange")
        self.gridLayout.addWidget(self.labelPSDRange, 6, 1, 1, 1)
        self.distEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.distEdit.setFont(font)
        self.distEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.distEdit.setObjectName("distEdit")
        self.gridLayout.addWidget(self.distEdit, 9, 2, 1, 2)
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 10, 1, 1, 1)
        self.labelDist = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.labelDist.setFont(font)
        self.labelDist.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelDist.setObjectName("labelDist")
        self.gridLayout.addWidget(self.labelDist, 9, 1, 1, 1)
        self.distButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.distButton.setFont(font)
        self.distButton.setObjectName("distButton")
        self.gridLayout.addWidget(self.distButton, 9, 4, 1, 1)
        self.picLabel = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picLabel.sizePolicy().hasHeightForWidth())
        self.picLabel.setSizePolicy(sizePolicy)
        self.picLabel.setMinimumSize(QtCore.QSize(300, 400))
        self.picLabel.setText("")
        self.picLabel.setObjectName("picLabel")
        self.gridLayout.addWidget(self.picLabel, 0, 0, 12, 1)
        self.pathButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pathButton.setFont(font)
        self.pathButton.setObjectName("pathButton")
        self.gridLayout.addWidget(self.pathButton, 0, 4, 1, 1)
        self.labelCores = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.labelCores.setFont(font)
        self.labelCores.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.labelCores.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelCores.setObjectName("labelCores")
        self.gridLayout.addWidget(self.labelCores, 3, 1, 1, 1)
        self.plotType = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plotType.setFont(font)
        self.plotType.setObjectName("plotType")
        self.gridLayout.addWidget(self.plotType, 8, 3, 1, 1)
        self.coresEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.coresEdit.setFont(font)
        self.coresEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.coresEdit.setObjectName("coresEdit")
        self.gridLayout.addWidget(self.coresEdit, 3, 2, 1, 2)
        self.STFTButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.STFTButton.setFont(font)
        self.STFTButton.setObjectName("STFTButton")
        self.gridLayout.addWidget(self.STFTButton, 3, 4, 1, 1)
        self.verifyButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.verifyButton.setFont(font)
        self.verifyButton.setObjectName("verifyButton")
        self.gridLayout.addWidget(self.verifyButton, 4, 4, 1, 1)
        self.errorBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.errorBrowser.setFont(font)
        self.errorBrowser.setObjectName("errorBrowser")
        self.gridLayout.addWidget(self.errorBrowser, 11, 1, 1, 4)
        self.PSDButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PSDButton.setFont(font)
        self.PSDButton.setObjectName("PSDButton")
        self.gridLayout.addWidget(self.PSDButton, 6, 4, 1, 1)
        self.pathEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.pathEdit.setFont(font)
        self.pathEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.pathEdit.setReadOnly(True)
        self.pathEdit.setObjectName("pathEdit")
        self.gridLayout.addWidget(self.pathEdit, 0, 1, 1, 3)
        self.plotValue = QtWidgets.QComboBox(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.plotValue.setFont(font)
        self.plotValue.setObjectName("plotValue")
        self.gridLayout.addWidget(self.plotValue, 8, 2, 1, 1)
        self.PowerAreaButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PowerAreaButton.setFont(font)
        self.PowerAreaButton.setObjectName("PowerAreaButton")
        self.gridLayout.addWidget(self.PowerAreaButton, 8, 4, 1, 1)
        self.labelThresh = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelThresh.setFont(font)
        self.labelThresh.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelThresh.setObjectName("labelThresh")
        self.gridLayout.addWidget(self.labelThresh, 4, 1, 1, 1)
        self.PSDEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.PSDEdit.setFont(font)
        self.PSDEdit.setFrame(True)
        self.PSDEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.PSDEdit.setObjectName("PSDEdit")
        self.gridLayout.addWidget(self.PSDEdit, 6, 2, 1, 2)
        self.threshEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.threshEdit.setFont(font)
        self.threshEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.threshEdit.setObjectName("threshEdit")
        self.gridLayout.addWidget(self.threshEdit, 4, 2, 1, 2)
        self.labelPlotType = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.labelPlotType.setFont(font)
        self.labelPlotType.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.labelPlotType.setObjectName("labelPlotType")
        self.gridLayout.addWidget(self.labelPlotType, 8, 1, 1, 1)
        self.timeSeriesButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.timeSeriesButton.setFont(font)
        self.timeSeriesButton.setObjectName("timeSeriesButton")
        self.gridLayout.addWidget(self.timeSeriesButton, 7, 4, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 7, 1, 1, 1)
        self.timeWindowEdit = QtWidgets.QLineEdit(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.timeWindowEdit.setFont(font)
        self.timeWindowEdit.setAlignment(QtCore.Qt.AlignCenter)
        self.timeWindowEdit.setObjectName("timeWindowEdit")
        self.gridLayout.addWidget(self.timeWindowEdit, 7, 2, 1, 2)
        SAKEDSP.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(SAKEDSP)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 829, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        SAKEDSP.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(SAKEDSP)
        self.statusbar.setObjectName("statusbar")
        SAKEDSP.setStatusBar(self.statusbar)
        self.actionSettings = QtWidgets.QAction(SAKEDSP)
        self.actionSettings.setObjectName("actionSettings")
        self.menuFile.addAction(self.actionSettings)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(SAKEDSP)
        QtCore.QMetaObject.connectSlotsByName(SAKEDSP)

    def retranslateUi(self, SAKEDSP):
        _translate = QtCore.QCoreApplication.translate
        SAKEDSP.setWindowTitle(_translate("SAKEDSP", "SAKE Plot"))
        self.checkBoxNorm.setText(_translate("SAKEDSP", "Normalize"))
        self.labelPSDRange.setText(_translate("SAKEDSP", "PSD Range (hz):"))
        self.distEdit.setText(_translate("SAKEDSP", "40-70"))
        self.label.setText(_translate("SAKEDSP", "Notifications:"))
        self.labelDist.setText(_translate("SAKEDSP", "Distribution Range:"))
        self.distButton.setText(_translate("SAKEDSP", "Plot Dist"))
        self.pathButton.setText(_translate("SAKEDSP", "Set Path..."))
        self.labelCores.setText(_translate("SAKEDSP", "Number of Cores:"))
        self.coresEdit.setText(_translate("SAKEDSP", "4"))
        self.STFTButton.setText(_translate("SAKEDSP", "Fourier Transform (STFT)"))
        self.verifyButton.setText(_translate("SAKEDSP", "Verify"))
        self.PSDButton.setText(_translate("SAKEDSP", "Plot PSD"))
        self.PowerAreaButton.setText(_translate("SAKEDSP", "Plot Power"))
        self.labelThresh.setText(_translate("SAKEDSP", "Outlier Threshold:"))
        self.PSDEdit.setText(_translate("SAKEDSP", "1-30"))
        self.threshEdit.setText(_translate("SAKEDSP", "4"))
        self.labelPlotType.setText(_translate("SAKEDSP", "Power Area Type:"))
        self.timeSeriesButton.setText(_translate("SAKEDSP", "Plot Time Series"))
        self.label_2.setText(_translate("SAKEDSP", "Window (sec)"))
        self.timeWindowEdit.setText(_translate("SAKEDSP", "60"))
        self.menuFile.setTitle(_translate("SAKEDSP", "File"))
        self.actionSettings.setText(_translate("SAKEDSP", "Settings"))

