import os 
import numpy as np 
import torch
from PIL import Image

from xml.dom.minidom import parse
import xml.dom.minidom

class SeashipDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPEGImages"))))
        self.annotation = list(sorted(os.listdir(os.path.join(root,"Annotations"))))
        self.labels ={
            'background' : 0,
            'ore carrier' : 1,
            'passenger ship' : 2,
            'container ship' : 3,
            'bulk cargo carrier' : 4,
            'general cargo ship' : 5,
            'fishing boat' : 6
        } 
        self.classes =['background','ore carrier','passenger ship','container ship','bulk cargo carrier','general cargo ship','fishing boat']
    
    def readxml(self, path):
        DOMTree = xml.dom.minidom.parse(path)
        annotaion = DOMTree.getElementsByTagName('annotation')
        objlist = annotaion[0].getElementsByTagName('object')
        boxes = []
        labels = []
        for obj in objlist:
            name = obj.getElementsByTagName('name')[0].firstChild.data
            #print(name)
            labels.append(self.labels[name])
            box = obj.getElementsByTagName('bndbox')[0]
            x1 = int(box.getElementsByTagName('xmin')[0].firstChild.data)
            y1 = int(box.getElementsByTagName('ymin')[0].firstChild.data)
            x2 = int(box.getElementsByTagName('xmax')[0].firstChild.data)
            y2 = int(box.getElementsByTagName('ymax')[0].firstChild.data)
            #print(x1,y1,x2,y2)
            boxes.append([x1,y1,x2,y2])
        return labels,boxes
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPEGImages", self.imgs[idx])
        anno_path = os.path.join(self.root, "Annotations", self.annotation[idx])
        img = Image.open(img_path).convert("RGB")
        labels, boxes = self.readxml(anno_path)
        num_objs = len(labels)
        #print(labels)
        #print(boxes)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:,3] - boxes[:,1]) * (boxes[:,2] - boxes[:,0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target= {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)

'''
def total():
    name_dict = dict()
    annotation_path = list(sorted(os.listdir(os.path.join("SeaShips","Annotations"))))
    for idx in range(len(annotation_path)):
        path = os.path.join("SeaShips", "Annotations", annotation_path[idx])
        name_list = ReadXml(path)
        for name in name_list:
            if name not in name_dict.keys():
                name_dict[name] = 0
            name_dict[name] += 1
        print("%s is ok" %path)
    
    print(name_dict)
{   'ore carrier': 2199, 
    'passenger ship': 474, 
    'container ship': 901, 
    'bulk cargo carrier': 1952, 
    'general cargo ship': 1505, 
    'fishing boat': 2190    }
'''

if __name__ == '__main__':
    classes = ('__background__','ore carrier','passenger ship',
        'container ship','bulk cargo carrier','general cargo ship','fishing boat')
    print(classes)
    print(classes[0])


'''
一个值得注意的点：第一次统计数据的时候很慢，而过一会在统计第二次
则飞快。应该是刚开始从磁盘读取大量数据需要花费很大时间，而之后可
从缓存中读取数据，则会很快。
'''