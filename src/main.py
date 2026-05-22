import sys
import os
from PIL import Image
import numpy as np
import cv2
from spectral import *
import spectral.io.envi as envi
import HSIHelper


from PyQt6 import QtWidgets, uic
from PyQt6.QtGui import QIcon,QPixmap,QImage, QActionGroup
from PyQt6.QtWidgets import QFileDialog,QGraphicsView,QGraphicsScene, QMessageBox

from MainWindow import Ui_MainWindow

def numpy_to_qpixmap(image):
    if image.dtype == np.uint8:
        height, width, channels = image.shape
        bytes_per_line = channels * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimage)
    else:
        raise ValueError("Unsupported image dtype")

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, *args, obj=None, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)

        self.viewer.mainui = self # used for viewer to access mainwindow widgets
        # self.viewer_context_menu = QtWidgets.QMenu()
        # self.viewer_context_menu.addAction("Spectrum Plot", self.spectrumPlot)
        # self.viewer_context_menu.addAction("Clear Selection", self.clearSelection)

        # self.viewer_index_menu = QtWidgets.QMenu(self.viewer, title="Index Mean")
        # self.viewer_index_menu.addAction("NDVI", lambda:self.showMeanIndex("NDVI"))
        # self.viewer_index_menu.addAction("EVI", lambda:self.showMeanIndex("EVI"))
        # self.viewer.setMenu(self.viewer_index_menu)

        view_widget = uic.loadUi("qt/Visualization.ui")
        self.verticalLayoutBottomRight.addWidget(view_widget)

        self.visualizationButton.clicked.connect(lambda:self.selectFunctionality("Visualization"))
        self.superResolutionButton.clicked.connect(lambda:self.selectFunctionality("Super-resolution"))
        self.calibrationButton.clicked.connect(lambda:self.selectFunctionality("Calibration"))
        self.classificationButton.clicked.connect(lambda:self.selectFunctionality("Classification"))

        self.actionLoadImage.triggered.connect(self.loadImage)
        self.actionSaveImage.triggered.connect(self.saveImage) 
        
    def selectFunctionality(self, functionality): 

        # hard-coded: 3rd widget is the widget below the file path and horizontal line, it is to be replaced every time a new functionality is selected
        third_widget = self.verticalLayoutBottomRight.itemAt(2).widget()
        self.verticalLayoutBottomRight.removeWidget(third_widget)
        third_widget.deleteLater()

        if functionality == "Visualization":
            view_widget = uic.loadUi("qt/Visualization.ui")
            self.verticalLayoutBottomRight.addWidget(view_widget)
        elif functionality == "Super-resolution":
            view_widget = uic.loadUi("qt/Super-esolution.ui")
            self.verticalLayoutBottomRight.addWidget(view_widget)
        elif functionality == "Calibration":
            view_widget = uic.loadUi("qt/Calibration.ui")
            self.verticalLayoutBottomRight.addWidget(view_widget)
        elif functionality == "Classification":
            view_widget = uic.loadUi("qt/Classification.ui")
            self.verticalLayoutBottomRight.addWidget(view_widget)        

    def spectrumPlot(self):
        pass

    def clearSelection(self):
        pass

    def showMeanIndex(self, index):
        pass

    def loadImage(self):        
        imagePath, selectedFilter = QFileDialog.getOpenFileName(self, 'Open file', None,("Hyperspc Images(*.bil *.bip *.bsq)"))
        if imagePath == "":
            return
        
        self.image_path = imagePath                                                
       
        directory = os.path.dirname(self.image_path)
        base_name = os.path.basename(self.image_path)
        filename, extension = os.path.splitext(base_name)
        headerPath = directory + "/"+ filename + ".hdr"
        if (not os.path.exists(headerPath)): #if no header file, show error and return
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle("Error")
            msg_box.setText("Header file not found!")
            msg_box.exec()
            return
        
        # check if it's PSI image format
        with open(headerPath, "r") as file:
            first_line = file.readline().strip()

        if first_line.startswith("BYTEORDER"): # PSI format
            dictMeta = HSIHelper.read_PSI_header(headerPath)
            headerPath = directory + "/"+ filename + "_envi.hdr" # new header file
            HSIHelper.create_envi_header(headerPath, dictMeta)            

        self.hsi = envi.open(headerPath, self.image_path)
        tuple_rgb_bands = HSIHelper.find_RGB_bands([float(i) for i in self.hsi.metadata['wavelength']]) # metadata['wavelength'] is read as string; for CSIRO image, can use self.hsi.bands.centers
        rgb_image = get_rgb(self.hsi, tuple_rgb_bands) #(100, 54, 31)
        rgb_image = (rgb_image*255).astype(np.uint8)
        rgb_image = rgb_image.copy() # Spy don't load it to memory automatically, so must be copied
        # rgb_image = np.array(rgb_image)

        self.viewer.rgb = rgb_image
        self.viewer.mask_array = np.zeros((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.uint8) # initialize mask image, to make sure some functionalities (e.g, export mask/statistics dlg) work even if mask is empty
        self.viewer.setPhoto(numpy_to_qpixmap(rgb_image))

    def saveImage(self): 
        pass

app = QtWidgets.QApplication(sys.argv)
# app.setWindowIcon(QIcon('logo.ico'))

window = MainWindow()
window.show()
app.exec()