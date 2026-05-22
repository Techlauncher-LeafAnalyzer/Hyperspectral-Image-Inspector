######### HSI Viewer                  ######################
######### Author: Tao Hu  2024.08.29  ######################
######### Organization: APPN          ######################
############################################################

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem, QDialog, QMessageBox
from PyQt6.QtGui import QIcon,QImage,QPixmap,QKeyEvent,QColor,QPen, QPainterPath
from PyQt6.QtCore import Qt, QRectF, QPointF
import numpy as np
import torch

class HSIViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPointF)

    def __init__(self, parent):
        super(HSIViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self.avatar = QtWidgets.QGraphicsPixmapItem()
        self.avatarArray = None
        

        
        self.input_points = np.empty((0, 2), dtype=np.uint32)         
        self.rgb = None # numpy image
        self.mask_pixmapitem = QtWidgets.QGraphicsPixmapItem()
        self.newInputBoxList = [] # newly added boxes at one time
        self.allInputBoxList = [] # all boxes added        

        self.history = [] 
        self.redo_stack = []           

         
        self._scene.addItem(self._photo)
        self._scene.addItem(self.avatar)
        self._scene.addItem(self.mask_pixmapitem)       
        self.setScene(self._scene)        
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)

        self.text_item = QGraphicsTextItem("APPN-Tech")
        font = self.text_item.font()
        font.setPointSize(45)
        font.setBold(True)
        font.setItalic(True)
        self.text_item.setFont(font)
        self.text_item.setDefaultTextColor(QColor("#eaecee"))  # Set the text color
        self.text_item.setTextWidth(400)  # Set the text width for word wrapping

        # Set the text item position to the center of the scene
        self.text_item.setPos(-self.text_item.boundingRect().width() / 2, -self.text_item.boundingRect().height() / 2)

        # Add the text item to the scene
        self._scene.addItem(self.text_item)

        self.text_item.setZValue(0)
        self._photo.setZValue(1)
        self.avatar.setZValue(2)
        self.mask_pixmapitem.setZValue(3)        
    

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def removeListItems(self, list):
        for list_item in list:
                self._scene.removeItem(list_item)
        list = []

    def clear(self):
        
        self.text_item.setVisible(False) # if removed multiple times, it will give error               
        self.mask_pixmapitem.setPixmap(QPixmap()) # empty previous mask

        self.input_points = np.empty((0, 2), dtype=np.uint32) # everytime a new image loaded, points must be cleared        
      
         
        self.history = [] 
        self.redo_stack = []

        # self.mainui.actionUndo.setEnabled(False)
        # self.mainui.actionRedo.setEnabled(False)
        # self.mainui.actionClear.setEnabled(False)    

        
    def setPhoto(self, pixmap=None):
        self.clear() # remove all promopts

        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    def setAvatar(self, pixmap):
        self.avatar.setPixmap(pixmap)
            

    def doSegmentation(self):   
        
        converted_mask = np.zeros((self.mask_array.shape[0], self.mask_array.shape[1], 4), dtype=np.uint8)    
        # Set blue color where mask is 1 and transparency to 0.5
        converted_mask[:,:,2] = self.mask_array * 255  # Blue channel
        converted_mask[:,:,3] = self.mask_array * 128  # Transparency channel

        self.mask_pixmapitem.setPixmap(QPixmap.fromImage(QImage(converted_mask.data, converted_mask.shape[1], converted_mask.shape[0], QImage.Format.Format_RGBA8888)))
        # QImage.Format
      

    def undo(self):
        if len(self.history):
            prompt_type, prompt_data = self.history.pop()
            self.redo_stack.append((prompt_type, prompt_data))

            self.mainui.actionRedo.setEnabled(True)
            if not len(self.history): # if no more history, disable Redo/Clear button
                self.mainui.actionUndo.setEnabled(False)
                self.mainui.actionClear.setEnabled(False)
    
            if prompt_type == 'point':
                self.input_points = self.input_points[:-1] # remove last row
                self.input_labels_list.pop(-1)                
                # self._scene.removeItem(self.circle_item_list.pop())
            elif prompt_type == 'box':
                # self._scene.removeItem(self.rect_item)
                self.start_point = None 
                self.rect_item = None
                self.input_box = None

            items = self._scene.items()    
            self._scene.removeItem(items[0]) # remove last item from scene, note last item index is 0 instead of -1

            self.doSegmentation()     

    def redo(self):
        if len(self.redo_stack):
            prompt_type, prompt_data = self.redo_stack.pop()

            self.history.append((prompt_type, prompt_data))

            if not len(self.redo_stack): # no more to redo, disable Redo button
                self.mainui.actionRedo.setEnabled(False)

            if prompt_type == 'point':
                self.input_points = np.vstack((self.input_points, prompt_data[1]))
                self.input_labels_list.append(prompt_data[0])
                self.drawCircle(prompt_data[1], prompt_data[0])
            elif prompt_type == 'rectangle':
                self.input_box = prompt_data 
                self.rect_item = self._scene.addRect(self.input_box.tolist(), QPen(Qt.GlobalColor.blue))               
                action = ('box', self.input_box)
                self.history.append(action)

            self.mainui.actionUndo.setEnabled(True)
            self.mainui.actionClear.setEnabled(True)    

            self.doSegmentation()    
    

    def wheelEvent(self, event):
        if self.hasPhoto():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

    def mousePressEvent(self, event):                    

        if self._photo.isUnderMouse():
            self.photoClicked.emit(self.mapToScene(event.position().toPoint()))

        super(HSIViewer, self).mousePressEvent(event)

        # modifiers = QtWidgets.QApplication.keyboardModifiers()
        # if modifiers & Qt.KeyboardModifier.ControlModifier:
        #     self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag) # if not set, cursor will not change
        #     self.setCursor(Qt.CursorShape.CrossCursor)
       

        print("clicked ", self.mapToScene(event.position().toPoint()))

    def mouseMoveEvent(self, event):
        super(HSIViewer, self).mouseMoveEvent(event)
          

    def mouseReleaseEvent(self, event):
        print("relased...")
        super(HSIViewer, self).mouseReleaseEvent(event)

        clicked_position = self.mapToScene(event.position().toPoint()) 
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if self.prompt == 0: # points
                input_point = np.array([clicked_position.x(), clicked_position.y()]).astype(np.uint32)    
                self.input_points = np.vstack((self.input_points, input_point))
                if self.isSplit:
                    self.new_input_points = np.vstack((self.new_input_points, input_point)) # only used to check which grid is to update
                
                if event.button() == Qt.MouseButton.LeftButton:
                    self.input_labels_list.append(1) # Ctrl+Left to add fg
                    prompt = ('point', (1, input_point))
                    self.drawCircle(input_point, 1)
                elif event.button() == Qt.MouseButton.RightButton:
                    self.input_labels_list.append(0) # Ctrl+Right to add bg
                    prompt = ('point', (0, input_point))
                    self.drawCircle(input_point, 0)

                                
                self.history.append(prompt)
                # self.doSegmentation()
                    
            elif self.prompt == 1: # boxes
                if self.start_point is not None:
                    self.draw_rectangle(self.start_point, clicked_position)
                    print(" point is ", self.start_point.x())
                    rect = [self.start_point.x(), self.start_point.y(), clicked_position.x(), clicked_position.y()]
                    print("rect is:", rect)
                    print("tensor is :", torch.tensor(rect))
                    self.input_box = np.array(rect)
                    self.newInputBoxList.append(np.array(rect))
                    self.allInputBoxList.append(np.array(rect))
                    action = ('box', self.input_box)
                    self.history.append(action)
                    # self.boxes_tensor = torch.cat((self.boxes_tensor, torch.tensor(rect)), dim=0)
                    self.start_point = None       
            
            if len(self.history):
                self.mainui.actionUndo.setEnabled(True)
                self.mainui.actionClear.setEnabled(True)
                    

            #when Ctrl pressed and mouse released, but Shift not pressed
            if not event.modifiers() & Qt.KeyboardModifier.ShiftModifier: # if Shift not pressed, finished adding prompts and start segmenting
                print("into seg...")
                self.doSegmentation() 
                if self.isSplit:
                    self.new_input_points = np.empty((0, 2), dtype=np.uint32) # after every segmentation, reinitialate         
                

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Control:
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)
            self.setCursor(Qt.CursorShape.CrossCursor)
        if isinstance(event, QKeyEvent):
            key_text = event.text()
            # self.key_label.setText(f"Last Key Pressed: {key_text}")

    def keyReleaseEvent(self, event):
        print(event.key())
        if event.key() == Qt.Key.Key_Control:
            # self.setCursor(Qt.CursorShape.DragMoveCursor)
            self.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        if event.key() == Qt.Key.Key_Shift:
            self.newInputBoxList = []  # reset input box list
            self.doSegmentation()    
            
        # if isinstance(event, QKeyEvent):
        #     key_text = event.text()
            # self.key_label.setText(f"Key Released: {key_text}")

    def contextMenuEvent(self, event):

        # # self.viewer_context_menu = QtWidgets.QMenu()
        # # self.viewer_context_menu.addAction("Spectrum Plot", self.spectrumPlot)
        # # self.viewer_context_menu.addAction("Clear Selection", self.clearSelection)

        # # self.viewer_index_menu = QtWidgets.QMenu(self.viewer, title="Index Mean")
        # # self.viewer_index_menu.addAction("NDVI", lambda:self.showMeanIndex("NDVI"))
        # # self.viewer_index_menu.addAction("EVI", lambda:self.showMeanIndex("EVI"))
        # # self.viewer.setMenu(self.viewer_index_menu)

        viewer_context_menu =  QtWidgets.QMenu(self)
        viewer_context_menu.addAction("Spectrum Plot", self.spectrumPlot)
        viewer_context_menu.addAction("Clear Selection", self.clearSelection)
        viewer_index_menu = QtWidgets.QMenu(self, title="Index Mean")
        viewer_index_menu.addAction("NDVI", lambda:self.showMeanIndex("NDVI"))
        viewer_index_menu.addAction("EVI", lambda:self.showMeanIndex("EVI"))
        viewer_context_menu.addMenu(viewer_index_menu)
        
        viewer_context_menu.exec(event.globalPos())

        # context_menu = QtWidgets.QMenu(self)

        # # # Add actions to the context menu
        # # action1 = QAction("Action 1", self)
        # # action2 = QAction("Action 2", self)
        # # action3 = QAction("Action 3", self)

        # context_menu.addAction("Spectrum Plot", self.spectrumPlot)
        # context_menu.addAction("Spectrum Plot", self.spectrumPlot)
        # context_menu.addAction("Spectrum Plot", self.spectrumPlot)

        # # # Connect actions to functions (optional)
        # # action1.triggered.connect(lambda: print("Action 1 triggered"))
        # # action2.triggered.connect(lambda: print("Action 2 triggered"))
        # # action3.triggered.connect(lambda: print("Action 3 triggered"))

        # # Show the context menu at the mouse position
        # context_menu.exec(event.globalPos())

    def spectrumPlot(self):
        pass

    def clearSelection(self):
        pass

    def showMeanIndex(self, index):
        pass    

class Window(QtWidgets.QWidget):
    def __init__(self):
        super(Window, self).__init__()
        self.viewer = HSIViewer(self)
        
        # # Button to change from drag/pan to getting pixel info
        # self.btnPixInfo = QtWidgets.QToolButton(self)
        # self.btnPixInfo.setText('Enter pixel info mode')
        # self.btnPixInfo.clicked.connect(self.pixInfo)
        # self.editPixInfo = QtWidgets.QLineEdit(self)
        # self.editPixInfo.setReadOnly(True)
        self.viewer.photoClicked.connect(self.photoClicked)
        # Arrange layout
        VBlayout = QtWidgets.QVBoxLayout(self)
        VBlayout.addWidget(self.viewer)
        HBlayout = QtWidgets.QHBoxLayout()
        HBlayout.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        HBlayout.addWidget(self.btnLoad)
        HBlayout.addWidget(self.btnPixInfo)
        HBlayout.addWidget(self.editPixInfo)
        VBlayout.addLayout(HBlayout)

    # def loadImage(self):
    #     self.viewer.setPhoto(QtGui.QPixmap('image.jpg'))

    # def pixInfo(self):
    #     self.viewer.toggleDragMode()

    # def photoClicked(self, pos):
    #     if self.viewer.dragMode() == QtWidgets.QGraphicsView.DragMode.NoDrag:
    #         self.editPixInfo.setText('%d, %d' % (pos.x(), pos.y()))