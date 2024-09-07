import cv2
import sys
import numpy as np
import pandas as pd
from PIL import Image
from PyQt5.QtGui import QPixmap, QImage
import models
import Gnetworks
sys.modules['models.networks'] = Gnetworks

import torch
import torch.nn as nn
from torchvision import transforms
from mmpretrain import get_model
from mmpretrain import models as premodel

from paddle.inference import create_predictor
from paddle.inference import Config as PredictConfig
from paddleseg.deploy.infer import DeployConfig


class Segmenter:
    def __init__(self, args):
        """
        Prepare for prediction.
        The usage and docs of paddle inference, please refer to
        https://paddleinference.paddlepaddle.org.cn/product_introduction/summary.html
        """
        self.cfg = DeployConfig(args)

        self._init_base_config()
        self._init_cpu_config()
        self.predictor = create_predictor(self.pred_cfg)

    def _init_base_config(self):
        self.pred_cfg = PredictConfig(self.cfg.model, self.cfg.params)
        self.pred_cfg.disable_glog_info()
        self.pred_cfg.enable_memory_optim()
        self.pred_cfg.switch_ir_optim(True)

    def _init_cpu_config(self):
        """
        Init the config for x86 cpu.
        """
        self.pred_cfg.disable_gpu()
        self.pred_cfg.set_cpu_math_library_num_threads(10)

    def run(self, imgs_path):
        if not isinstance(imgs_path, (list, tuple)):
            imgs_path = [imgs_path]

        input_names = self.predictor.get_input_names()
        input_handle = self.predictor.get_input_handle(input_names[0])
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = []

        for i in range(0, len(imgs_path), 1):
            # inference

            data = np.array([
                self._preprocess(p) for p in imgs_path[i:i + 1]
            ])
            input_handle.reshape(data.shape)
            input_handle.copy_from_cpu(data)

            self.predictor.run()

            results = output_handle.copy_to_cpu()
        return results

    def _preprocess(self, img):
        data = {}
        data['img'] = img
        return self.cfg.transforms(data)['img']


class Generator:
    def __init__(self, pred_type):
        self.pred_type = pred_type
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        self.model = torch.load("./TrainedModels/netG.pth").module
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        ckptPth = f"./TrainedModels/{self.pred_type}_G.pth"
        ckpt = torch.load(ckptPth)
        self.model.load_state_dict(ckpt)
        self.model.to(self.device)
        self.model.eval()

    def generate(self, rawImg):
        img = self._CropAndResize(rawImg)
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            fakeImg = self.model(img)
        fakeImg = (fakeImg.squeeze(0).permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255.0
        fakeImg = fakeImg.astype(np.uint8)
        fakeImg = self._Resize2raw(fakeImg, rawImg)
        return fakeImg

    @staticmethod
    def _CropAndResize(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # 查找图像的轮廓
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        max_contour = max(contours, key=cv2.contourArea)

        # 获取最大轮廓的最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        # center = (int(x), int(y))
        radius = int(radius) + 100

        # 计算裁剪范围
        top = max(0, int(y) - radius)
        bottom = min(img.shape[0], int(y) + radius)
        left = max(0, int(x) - radius)
        right = min(img.shape[1], int(x) + radius)

        # 裁剪图像
        cropped_img = img[top:bottom, left:right]
        resized_img = cv2.resize(cropped_img, (256, 256))
        return resized_img

    @staticmethod
    def _Resize2raw(fakeImg, rawImg):
        ## remove_outlier_small_objects ##
        fake_gray = cv2.cvtColor(fakeImg, cv2.COLOR_BGR2GRAY)
        fake_contours, _ = cv2.findContours(fake_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        fake_max_contour = max(fake_contours, key=cv2.contourArea)
        mask = np.zeros_like(fake_gray)
        cv2.drawContours(mask, [fake_max_contour], -1, (255), thickness=cv2.FILLED)
        fakeImg = cv2.bitwise_and(fakeImg, fakeImg, mask=mask)

        dst = np.zeros_like(rawImg)
        gray = cv2.cvtColor(rawImg, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        radius = int(radius) + 100

        top = max(0, int(y) - radius)
        bottom = min(rawImg.shape[0], int(y) + radius)
        left = max(0, int(x) - radius)
        right = min(rawImg.shape[1], int(x) + radius)

        cropped_img = rawImg[top:bottom, left:right]
        resized_fakeImg = cv2.resize(fakeImg, (cropped_img.shape[1], cropped_img.shape[0]))
        dst[top:bottom, left:right] = resized_fakeImg
        return dst


class Classifier:
    def __init__(self):
        self.typeDict = {
            0: 'Romaine',
            1: 'Looseleaf',
            2: 'Butterhead',
            3: 'Stem',
            4: 'Crisphead',
            5: 'Wildlettuce'
        }
        self.device = "cpu"
        ckpt_path = "./TrainedModels/classify.pth"
        self.model = get_model("convnext-tiny_in21k-pre_3rdparty_in1k", pretrained=ckpt_path, head=dict(num_classes=6))
        norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ])
        self.model.to(self.device)
        self.model.eval()

    def classify(self, img):
        img = self.crop(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img)
        index = pred.argmax(dim=1).item()
        return self.typeDict[index]

    @staticmethod
    def crop(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)

        # 获取最大轮廓的最小外接圆
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        # center = (int(x), int(y))
        radius = int(radius)

        # 计算裁剪范围
        top = max(0, int(y) - radius)
        bottom = min(img.shape[0], int(y) + radius)
        left = max(0, int(x) - radius)
        right = min(img.shape[1], int(x) + radius)

        cropped_img = img[top:bottom, left:right]
        pil_cropped_img = Image.fromarray(cropped_img)

        return pil_cropped_img


class Predictor:

    def __init__(self, pred_type, days):
        self.device = "cpu"
        self.model = self.LettuceYield()
        ckptName = f"./TrainedModels/yield_{pred_type}_{days}.pth"
        ckpt = torch.load(ckptName)
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        norm_cfg = dict(mean=[0.491, 0.482, 0.447], std=[0.202, 0.199, 0.201])
        self.transform = transforms.Compose([
            transforms.Resize((1000, 1000)),
            transforms.ToTensor(),
            transforms.Normalize(**norm_cfg)
        ])

    def predict(self, img):
        img = Image.fromarray(img)
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(img)
        pred = pred.cpu().numpy()[0][0]
        return pred

    class LettuceYield(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = premodel.backbones.ResNet(depth=34)
            self.neck = premodel.GlobalAveragePooling()
            self.head = nn.Sequential(
                nn.Linear(512, 1),
                nn.ReLU(inplace=True)
            )

        def forward(self, img):
            x = self.backbone(img)
            x = self.neck(x)
            x = self.head(x[0])
            return x

class Lettuce:
    # 定义数据
    data = {
        'Butterhead': [0.408537414, 0.471759082, 0.5681894, 0.647477412, 0.727199442, 0.789179365],
        'Crisphead': [0.496406134, 0.628468239, 0.682652992, 0.716267808, 0.828956388, 0.858332006],
        'Looseleaf': [0.492057149, 0.490566093, 0.542700894, 0.69565176, 0.725573723, 0.757538718],
        'Romaine': [0.318157709, 0.409099602, 0.577251852, 0.714177165, 0.758802615, 0.8122403],
        'Stem': [0.644779889, 0.632120494, 0.791520514, 0.791093327, 0.88012124, 0.832022092],
        'Wildlettuce': [0.200412473, 0.753047535, 0.728010227, 0.849012123, 0.851020589, 0.848694109]
    }
    real_R2 = pd.DataFrame(data, index=[19, 24, 29, 34, 39, 44])

    data = {
        'Butterhead': [0.451378121, 0.444066582, 0.45579756, 0.313140913, 0.197959171, 0.550871369, 0.598082373,
                       0.56850586, 0.52646343, 0.634805528, 0.704575815, 0.708899791, 0.72824932, 0.770461176,
                       0.768226691],
        'Crisphead': [0.615744122, 0.655686958, 0.608786284, 0.484378384, 0.496681524, 0.686454615, 0.710001837,
                      0.642393947, 0.65672533, 0.719948733, 0.779304119, 0.799299269, 0.825639832, 0.860393941,
                      0.854188598],
        'Looseleaf': [0.455542501, 0.481361074, 0.542932264, 0.463306688, 0.495483775, 0.519749006, 0.607334057,
                      0.581094316, 0.607080094, 0.652295076, 0.677237771, 0.671867342, 0.717743089, 0.718880717,
                      0.731589246],
        'Romaine': [0.346903445, 0.460646604, 0.51301439, 0.247544001, 0, 0.517243122, 0.613911601, 0.50792053,
                    0.317008458, 0.669378548, 0.705767657, 0.630978124, 0.748513733, 0.73806467, 0.75864336],
        'Stem': [0.614545526, 0.49905954, 0.721446311, 0.373091517, 0.359887572, 0.754585908, 0.774774561, 0.494746567,
                 0.48783452, 0.787493075, 0.52578361, 0.453053654, 0.590123161, 0.562790191, 0.828640419],
        'Wildlettuce': [0.742613996, 0.612518952, 0.597992081, 0.496690236, 0.615435062, 0.712581454, 0.6566005,
                        0.475792871, 0.68547124, 0.670364714, 0.481157991, 0.722630324, 0.499647935, 0.7183187,
                        0.804625305]
    }
    fake_R2 = pd.DataFrame(data, index=['19to24', '19to29', '19to34', '19to39', '19to44', '24to29', '24to34', '24to39',
                                        '24to44', '29to34', '29to39', '29to44', '34to39', '34to44', '39to44'])

    @staticmethod
    def segmentation(img):
        def _remove_black_border(img):
            # 转换为灰度图像
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 找到所有非黑色的像素点
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # 找到图像的轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 获取最大轮廓的边界框
                x, y, w, h = cv2.boundingRect(contours[0])
                # 裁剪图像
                cropped_img = img[y:y + h, x:x + w]
                return cropped_img
            else:
                # 如果没有找到轮廓，返回原图
                return img

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = _remove_black_border(img)
        segmenter = Segmenter("TrainedModels/seg.yaml")
        result = segmenter.run(img.astype(np.float32))[0]
        # 找出最大连通域
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result.astype(np.uint8), connectivity=8)

        if num_labels > 1:  # 只有背景，没有检测到连通域
            # 获取最大连通域（第一个连通域是背景）
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            largest_mask = (labels == largest_label).astype(np.uint8)
        else:
            largest_label = 0
            largest_mask = np.zeros_like(img)


        # 创建掩码，仅保留最大连通域
        seg = np.zeros_like(img)
        seg[largest_mask == 1] = img[largest_mask == 1]

        # 计算质心
        centroid = centroids[largest_label]
        cx, cy = int(centroid[0]), int(centroid[1])

        # 以质心为中心，裁剪1000x1000像素的图像
        half_size = 500
        x1, y1 = max(cx - half_size, 0), max(cy - half_size, 0)
        x2, y2 = min(cx + half_size, img.shape[1]), min(cy + half_size, img.shape[0])

        # 裁剪并创建黑色背景的图像
        cropped = np.zeros((half_size*2, half_size*2, img.shape[2]), dtype=img.dtype)
        cropped_y1 = half_size - (cy - y1)
        cropped_y2 = cropped_y1 + (y2 - y1)
        cropped_x1 = half_size - (cx - x1)
        cropped_x2 = cropped_x1 + (x2 - x1)

        cropped[cropped_y1:cropped_y2, cropped_x1:cropped_x2] = seg[y1:y2, x1:x2]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return cropped

    @staticmethod
    def classification(img):
        classifier = Classifier()
        pred_type = classifier.classify(img)
        return pred_type

    @staticmethod
    def generation(img, pred_type):

        if np.all(img == img[0, 0]):
            return img
        generator = Generator(pred_type)
        fakeimg = generator.generate(img)

        return fakeimg

    @staticmethod
    def yieldPrediction(img, pred_type, days):
        pred = 0
        if np.all(img == img[0, 0]):
            return pred
        predictor = Predictor(pred_type, days)
        pred = predictor.predict(img)

        return pred
