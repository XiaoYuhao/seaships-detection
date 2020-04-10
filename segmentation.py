import numpy as np 
import torch 
from PIL import Image

from xml.dom.minidom import parse
import xml.dom.minidom
import os
import utils
import transforms

class Segmentation(object):
    def __init__(self, root):
        self.root = root
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.annotation = list(sorted(os.listdir(os.path.join(root,"Annotations"))))
        #self.mask = list(sorted(os.listdir(os.path.join(root,"MaskImages"))))

    def _readxml(self, path):
        DOMTree = xml.dom.minidom.parse(path)
        annotaion = DOMTree.getElementsByTagName('annotation')
        objlist = annotaion[0].getElementsByTagName('object')
        boxes = []
        for obj in objlist:
            box = obj.getElementsByTagName('bndbox')[0]
            x1 = int(box.getElementsByTagName('xmin')[0].firstChild.data)
            y1 = int(box.getElementsByTagName('ymin')[0].firstChild.data)
            x2 = int(box.getElementsByTagName('xmax')[0].firstChild.data)
            y2 = int(box.getElementsByTagName('ymax')[0].firstChild.data)
            #print(x1,y1,x2,y2)
            boxes.append([x1,y1,x2,y2])
        return boxes

    def _gauss(self, box, image):
        arr = np.array(image)
        sub_arr = arr[box[1]:box[3],box[0]:box[2]]
        #img = Image.fromarray(sub_arr)
        arr_r = arr[:,:,0].flatten()
        arr_g = arr[:,:,1].flatten()
        arr_b = arr[:,:,2].flatten()
        X = np.vstack([arr_r, arr_g, arr_b])
        S = np.cov(X)
        mean = np.array([arr_r.mean(), arr_g.mean(), arr_b.mean()])
        #var = np.array([arr_r.var(), arr_g.var(), arr_b.var()])
        return S,mean                                   #返回协方差矩阵和均值

    def _dist(self, S, mean, X):
        ST = np.linalg.inv(S)
        arr = X - mean
        d = np.sqrt(np.dot(np.dot(arr.T,ST),arr))       #计算马氏距离
        return d

    def _segmentation(self, idx):
        path = self.imgs[idx]
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        mask = np.zeros((arr.shape[0],arr.shape[1]),dtype=np.float32)
        anno_path = os.path.join(self.root, "Annotations", self.annotation[idx])
        boxes = self._readxml(anno_path)
        for box in boxes:
            sub_arr = arr[box[1]:box[3],box[0]:box[2]]
            height = box[3] - box[1]
            width = box[2] - box[0]
            low = [box[0],box[3],box[0]+width,box[3]+height]
            up = [box[0],box[1]-height,box[0]+width,box[1]]
            low_S,low_mean = self._gauss(low,img)
            up_S,up_mean = self._gauss(up,img)
            area = height*width
            d_min = 5.0
            for iter in range(6):
                d_min = d_min - iter*0.5
                for i in range(height):
                    for j in range(width):
                        pix = sub_arr[i,j]
                        d_low = self._dist(low_S,low_mean,pix)
                        d_up = self._dist(up_S,up_mean,pix)
                        if d_up > d_min and d_low > d_min :
                            mask[i + box[1], j + box[0]] = 1
                mask = utils._morph_opening(mask,box)
                mask = utils._morph_closing(mask,box)
                count = utils.countmask(mask,box)
                if count > 0.2*area:
                    break

        mask = mask[np.newaxis,:,:]
        mask = torch.from_numpy(mask)
        #img = img.convert('RGBA')
        #utils.drawRectangle(img,boxes)
        mask = transforms.TensorToPIL(mask)
        mask_path = os.path.join(self.root, "MaskImages", self.imgs[idx].split('.')[0]+".png")
        print(mask_path)
        mask.save(mask_path)
        #img = utils.drawMasks(img, mask)
        #img.show()

    def _segmentation_faster(self, idx):
        path = self.imgs[idx]
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        mask = np.zeros((arr.shape[0],arr.shape[1]),dtype=np.float32)
        anno_path = os.path.join(self.root, "Annotations", self.annotation[idx])
        boxes = self._readxml(anno_path)
        for box in boxes:
            sub_arr = arr[box[1]:box[3],box[0]:box[2]]
            height = box[3] - box[1]
            width = box[2] - box[0]
            low = [box[0],box[3],box[0]+width,box[3]+height]
            up = [box[0],box[1]-height,box[0]+width,box[1]]
            low_S,low_mean = self._gauss(low,img)
            up_S,up_mean = self._gauss(up,img)
            d_min = 2.5

            for i in range(height):
                for j in range(width):
                    pix = sub_arr[i,j]
                    d_low = self._dist(low_S,low_mean,pix)
                    d_up = self._dist(up_S,up_mean,pix)
                    if d_up > d_min and d_low > d_min :
                        mask[i + box[1], j + box[0]] = 1
                
            mask = utils._morph_opening(mask,box)
            mask = utils._morph_closing(mask,box)

        mask = mask[np.newaxis,:,:]
        #mask = Image.fromarray(mask)
        mask = torch.from_numpy(mask)
        mask = transforms.TensorToPIL(mask)
        mask_path = os.path.join(self.root, "MaskImages", self.imgs[idx].split('.')[0]+".png")
        print(mask_path)
        mask.save(mask_path)       

    def work(self):
        for idx in range(1093,len(self.imgs)):
            self._segmentation_faster(idx)

if __name__ == '__main__':
    #Mahalanobis_distance()
    S = Segmentation('../')
    S.work()
    #print(S.imgs[0])