# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 11:22:10 2021

@author: DELL
"""

import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QLineEdit
from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QMessageBox
from PyQt5.QtCore import QSize
from hdr import main    

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.setMinimumSize(QSize(320, 140))    
        self.setWindowTitle("Offline Handwritten Text Recognition") 

        self.nameLabel = QLabel(self)
        self.nameLabel.setText('FileName:')
        self.line = QLineEdit(self)

        self.line.move(80, 20)
        self.line.resize(200, 32)
        self.nameLabel.move(20, 20)
        
        
        pybutton = QPushButton('Convert', self)
        pybutton.clicked.connect(self.clickMethod)        
        pybutton.resize(200,32)
        pybutton.move(80, 60)
        
        
    def clickMethod(self):
        f=main(self.line.text())
        data = f
        msgbox = QMessageBox(QMessageBox.Information, "Title", "Predicted output: %s" % data, QMessageBox.Ok)
        msgbox.exec_()
    
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit( app.exec_() )