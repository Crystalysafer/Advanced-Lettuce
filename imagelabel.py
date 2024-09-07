from PyQt5.QtWidgets import QFileDialog, QMenu, QLabel
from PyQt5.QtGui import QPainter, QPen, QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QPoint, QRect
import math
import numpy as np
from PIL import Image

class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        self.drawing = False
        self.cropping = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.scale = 1.0
        self.set_img(self.img)

    def showContextMenu(self, pos):
        contextMenu = QMenu(self)

        saveAction = contextMenu.addAction("Save Image")
        saveAction.triggered.connect(self.saveImage)

        loadAction = contextMenu.addAction("Load Image")
        loadAction.triggered.connect(self.loadImage)

        contextMenu.exec_(self.mapToGlobal(pos))

    def saveImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getSaveFileName(self, "Save Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if fileName:
            if not (fileName.endswith('.png') or fileName.endswith('.jpg') or fileName.endswith('.bmp')):
                fileName += '.png'
            img = Image.fromarray(self.img)
            img.save(fileName)

    def loadImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if fileName:
            # 使用 PIL.Image 打开图像
            pil_image = Image.open(fileName)
            pil_image = pil_image.convert("RGB")
            self.img = np.array(pil_image)  # RGB顺序

            # 获取图像的原始尺寸
            img_height, img_width, _ = self.img.shape
            long_size = max(img_height, img_width)

            # 创建一个黑色背景的图像
            result_img = np.zeros((long_size, long_size, 3), dtype=np.uint8)

            x_offset = (long_size - img_width) // 2
            y_offset = (long_size - img_height) // 2

            # 将缩放后的图像复制到黑色背景图像中
            result_img[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = self.img
            self.img = result_img
            self.set_img(result_img)

    def mousePressEvent(self, event):
        if self.drawing or self.cropping:
            self.start_point = event.pos()
            self.end_point = self.start_point
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing or self.cropping:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing or self.cropping:
            self.end_point = event.pos()
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if (self.drawing or self.cropping) and not self.start_point.isNull() and not self.end_point.isNull():
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            if self.drawing:
                painter.drawLine(self.start_point, self.end_point)
            elif self.cropping:
                rect = QRect(self.start_point, self.end_point)
                painter.drawRect(rect)

    def set_drawing(self, drawing):
        self.drawing = drawing
        if not drawing:
            self.start_point = QPoint()
            self.end_point = QPoint()
        self.update()

    def set_cropping(self, cropping):
        self.cropping = cropping
        if not cropping:
            self.start_point = QPoint()
            self.end_point = QPoint()
        self.update()

    def cropImage(self):
        if self.start_point.isNull() or self.end_point.isNull():
            return
        x1 = int(min(self.start_point.x(), self.end_point.x()) * self.scale)
        y1 = int(min(self.start_point.y(), self.end_point.y()) * self.scale)
        x2 = int(max(self.start_point.x(), self.end_point.x()) * self.scale)
        y2 = int(max(self.start_point.y(), self.end_point.y()) * self.scale)

        self.img = self.img[y1:y2, x1:x2].copy()
        self.set_img(self.img)

    def get_line_length(self):
        if not self.start_point.isNull() and not self.end_point.isNull():
            length = math.sqrt((self.end_point.x() - self.start_point.x()) ** 2 +
                               (self.end_point.y() - self.start_point.y()) ** 2)
            length = length * self.scale
            return length

    def get_img(self):
        return self.img

    def set_img(self, img):
        self.img = img
        img_height, img_width, _ = img.shape
        self.scale = img_width / self.width()
        qImg = QImage(img.data, img_width, img_height, 3 * img_width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        self.setPixmap(pixmap)
