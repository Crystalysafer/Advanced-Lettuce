import cv2

from mainWin_V1 import Ui_LettuceMain
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QCoreApplication, Qt
from Lettuce import Lettuce
import numpy as np
from PIL import Image
from qt_material import apply_stylesheet


class LettuceWin(QMainWindow, Ui_LettuceMain):
    def __init__(self):
        super(LettuceWin, self).__init__()
        self.setupUi(self)
        self.initUi()

    def initUi(self):
        self.num_r2.display("0.00")
        self.num_yield.display("0000")

        self.btn_measure.clicked.connect(self.on_measure_click)
        self.btn_redo.clicked.connect(self.redo)
        self.btn_rescale.clicked.connect(self.rescale)
        self.btn_crop.clicked.connect(self.on_crop_click)
        self.btn_measure.setCheckable(True)
        self.btn_crop.setCheckable(True)
        self.num_pixels.valueChanged.connect(self.pixel_num_changed)

        self.btn_seg.clicked.connect(lambda: self.start_task(self.segmentation, "Segmentation"))
        self.btn_generate.clicked.connect(lambda: self.start_task(self.generate, "Generation"))
        self.btn_classify.clicked.connect(lambda: self.start_task(self.classify, "Classification"))
        self.btn_reset.clicked.connect(lambda: self.start_task(self.reset, "Reset"))
        self.btn_yield.clicked.connect(lambda: self.start_task(self.on_PredYield_click, "Yield Prediction"))
        self.btn_loadImg.clicked.connect(self.openImage)
        self.comboBox_classify.currentTextChanged.connect(self.type_changed)
        self.num_fakeNum.valueChanged.connect(self.fake_num_changed)
        self.num_DAS.valueChanged.connect(self.days_num_changed)
        self.btn_hint.clicked.connect(self.on_hint_click)

        self.length10cm = 120
        self.length10cm_std = 120
        self.pred_type = self.comboBox_classify.currentText()
        self.days = [19, 24, 29, 34, 39, 44]
        self.blankImg = np.ones((100, 100, 3), dtype=np.uint8) * 0
        self.rawImg = np.ones((100, 100, 3), dtype=np.uint8) * 0
        self.label_raw.set_img(self.rawImg)

        self.days_num_changed()
        self.reset()

        # self.group_seg.setEnabled(False)
        # self.group_generate.setEnabled(False)

    def openImage(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                  "Image Files (*.png *.jpg *.bmp);;All Files (*)", options=options)
        if fileName:
            # 使用 cv2 读取图像
            pil_image = Image.open(fileName)
            pil_image = pil_image.convert("RGB")
            img = np.array(pil_image)  # RGB顺序

            # 获取图像的原始尺寸
            img_height, img_width, _ = img.shape
            long_size = max(img_height, img_width)

            # 创建一个黑色背景的图像
            result_img = np.zeros((long_size, long_size, 3), dtype=np.uint8)

            x_offset = (long_size - img_width) // 2
            y_offset = (long_size - img_height) // 2

            # 将缩放后的图像复制到黑色背景图像中
            result_img[y_offset:y_offset + img_height, x_offset:x_offset + img_width] = img

            # 设置 QLabel 的图像
            self.label_raw.set_img(result_img)
            self.statusBar().showMessage('Image opened')
            self.rawImg = result_img  # 保存原始图像的引用，如果需要的话

    def start_task(self, func, func_name):
        sender_button = self.sender()
        self.worker = Worker(func, func_name)
        self.worker.finished.connect(lambda: self.on_task_finished(sender_button, func_name))
        self.worker.error.connect(self.handle_error)
        sender_button.setEnabled(False)
        self.worker.start()
        self.statusbar.showMessage(f"{func_name} Running...")

    def on_task_finished(self, sender_button, func_name):
        sender_button.setEnabled(True)
        self.generable_check()
        self.statusbar.showMessage(f"{func_name} Finished!")

    def handle_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)

    def pixel_num_changed(self):
        self.length10cm = self.num_pixels.value()

    def days_num_changed(self):
        self.DAS = self.num_DAS.value()
        self.fake_days = self.num_fakeNum.value() * 5 + self.DAS

    def on_crop_click(self):
        if self.btn_crop.isChecked():
            self.statusbar.showMessage("Crop a lettuce")
            self.btn_measure.setEnabled(False)
            self.btn_redo.setEnabled(False)
            self.label_raw.set_cropping(True)
            self.btn_crop.setText('Done')
        else:
            self.label_raw.cropImage()
            self.label_raw.set_cropping(False)
            self.btn_crop.setText("Crop")
            self.statusbar.showMessage("Stop Cropping")
            self.btn_measure.setEnabled(True)
            self.btn_redo.setEnabled(True)

    def on_measure_click(self):
        if self.btn_measure.isChecked():
            self.statusbar.showMessage("Draw a 10 cm scale")
            self.btn_crop.setEnabled(False)
            self.btn_redo.setEnabled(False)
            self.label_raw.set_drawing(True)
            self.btn_measure.setText('Done')
        else:
            self.num_pixels.setValue(int(self.label_raw.get_line_length()))
            self.label_raw.set_drawing(False)
            self.btn_measure.setText('Measure')
            self.statusbar.showMessage("Stop Drawing")
            self.btn_crop.setEnabled(True)
            self.btn_redo.setEnabled(True)

    def on_hint_click(self):
        QMessageBox.information(self, "Attention", "The results come from paper:\n haven't published \n\nFOR REFERENCE ONLY!")

    def rescale(self):

        raw_img = self.label_raw.get_img()
        img_height, img_width, _ = raw_img.shape

        # 计算缩放比例
        scale_factor = self.length10cm_std / self.length10cm

        # 计算新的尺寸
        new_width = int(img_width * scale_factor)
        new_height = int(img_height * scale_factor)

        scaled_img = cv2.resize(raw_img, (new_width, new_height))

        half_size = 500
        x1, y1 = max(new_width // 2 - half_size, 0), max(new_height // 2 - half_size, 0)
        x2, y2 = min(new_width // 2 + half_size, scaled_img.shape[1]), min(new_height // 2 + half_size,
                                                                           scaled_img.shape[0])

        # 裁剪并创建黑色背景的图像
        cropped = np.zeros((half_size * 2, half_size * 2, scaled_img.shape[2]), dtype=scaled_img.dtype)
        cropped_y1 = half_size - (new_height // 2 - y1)
        cropped_y2 = cropped_y1 + (y2 - y1)
        cropped_x1 = half_size - (new_width // 2 - x1)
        cropped_x2 = cropped_x1 + (x2 - x1)

        cropped[cropped_y1:cropped_y2, cropped_x1:cropped_x2] = scaled_img[y1:y2, x1:x2]
        # 更新 QLabel 中的图像
        self.label_raw.set_img(cropped)

    def redo(self):
        self.label_raw.set_img(self.rawImg)
        self.num_pixels.setValue(int(self.length10cm_std))

    def type_changed(self):
        self.pred_type = self.comboBox_classify.currentText()

    def reset(self):
        self.label_seg.set_img(self.blankImg)
        self.fakeImgs = {0: self.blankImg}
        # self.fakeYields = {0: "0000"}
        # self.fakeRs = {0: "0.00"}
        self.num_fakeNum.setMinimum(0)
        self.num_fakeNum.setValue(0)
        self.num_fakeNum.setMaximum(0)
        self.btn_generate.setEnabled(True)
        self.num_r2.display("0.00")
        self.num_yield.display("0000")

    def fake_num_changed(self):
        self.fake_days = self.num_fakeNum.value() * 5 + self.DAS
        if self.num_fakeNum.value() != self.num_fakeNum.maximum():
            self.btn_generate.setEnabled(False)
        else:
            self.btn_generate.setEnabled(True)
            self.generable_check()
        self.label_seg.set_img(self.fakeImgs[self.num_fakeNum.value()])
        # self.num_yield.display(self.fakeYields.get(self.num_fakeNum.value(), "0000"))
        # self.num_r2.display(self.fakeRs.get(self.num_fakeNum.value(), "0.00"))

    def generable_check(self):
        if self.fake_days + 5 > self.days[-1]:
            self.btn_generate.setEnabled(False)

    def classify(self):
        # QMessageBox.information(self, "Classify", "Classify is not implemented yet")
        self.comboBox_classify.setEnabled(False)
        self.pred_type = Lettuce.classification(self.label_seg.get_img())
        self.comboBox_classify.setCurrentText(self.pred_type)
        self.comboBox_classify.setEnabled(True)

    def segmentation(self):
        seg = Lettuce.segmentation(self.label_raw.get_img())
        self.reset()
        self.label_seg.set_img(seg)
        self.fakeImgs = {0: seg}


    def generate(self):
        self.num_fakeNum.setMaximum(self.num_fakeNum.maximum() + 1)
        fake = Lettuce.generation(self.label_seg.get_img(), self.pred_type)
        self.fakeImgs[self.num_fakeNum.value() + 1] = fake
        self.num_fakeNum.setValue(self.num_fakeNum.value() + 1)

    def on_PredYield_click(self):
        maxR2 = 0.0
        maxNum = 0
        for i in range(self.num_fakeNum.maximum()+1):
            fakeNum = i
            fake_days = self.DAS + 5 * fakeNum
            _, R2 = self.yieldPrediction(self.blankImg, fake_days, fakeNum)
            if R2 > maxR2:
                maxR2 = R2
                maxNum = fakeNum
        pred, _ = self.yieldPrediction(self.fakeImgs[maxNum], self.DAS + 5 * maxNum, maxNum)
        # self.fakeYields[self.num_fakeNum.value()] = f"{int(pred)}"
        # self.fakeRs[self.num_fakeNum.value()] = f"{R2:.2f}"
        self.num_yield.display(f"{int(pred)}")
        self.num_r2.display(f"{maxR2:.2f}")

    def yieldPrediction(self, img, days, fakeNum):
        if fakeNum:
            r2df = Lettuce.fake_R2.copy()
            r2df["start"] = [int(x[:x.index("to")]) for x in r2df.index]
            r2df.index = [int(x[x.index("to") + 2:]) for x in r2df.index]
            r2df = r2df[(r2df.index - r2df["start"]) == fakeNum * 5].drop(["start"], axis=1)
        else:
            r2df = Lettuce.real_R2.copy()
            r2df.index = [int(x) for x in r2df.index]
        if days not in self.days:
            index1 = 0
            index2 = 0
            for i in range(len(self.days) - 1):
                if self.days[i] < days < self.days[i + 1]:
                    index1 = i
                    index2 = i + 1
                    break
            pred1 = Lettuce.yieldPrediction(img, self.pred_type, self.days[index1])
            pred2 = Lettuce.yieldPrediction(img, self.pred_type, self.days[index2])
            R21 = r2df.loc[self.days[index1], self.pred_type]
            R22 = r2df.loc[self.days[index2], self.pred_type]
            pred = pred1 + (pred2 - pred1) * (days - self.days[index1]) / (self.days[index2] - self.days[index1])
            R2 = R21 + (R22 - R21) * (days - self.days[index1]) / (self.days[index2] - self.days[index1])
        else:
            pred = Lettuce.yieldPrediction(img, self.pred_type, days)
            R2 = r2df.loc[days, self.pred_type]
        return pred, R2


class Worker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, func, func_name):
        super().__init__()
        self.func = func
        self.func_name = func_name

    def run(self):
        try:
            self.func()  # 执行传递的函数
            self.finished.emit(self.func_name)
        except Exception as e:
            self.error.emit(f"Error in {self.func_name}: {str(e)}")
            self.finished.emit(self.func_name)


if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication([])
    win = LettuceWin()
    apply_stylesheet(app, theme='dark_lightgreen.xml')
    win.show()
    app.exec_()
