import json
import os
import torch
import open_clip
import re
import cv2
import torch.utils.data as data
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np

Image.MAX_IMAGE_PIXELS = None

def get_class_names(dataset_name):
    if dataset_name == "mvtec":
        class_names = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
                    'transistor', 'metal_nut', 'screw', 'toothbrush', 'zipper', 'tile', 'wood']
    elif dataset_name == "visa":
        class_names = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2',
                    'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']
    elif dataset_name == "mpdd":
        class_names = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']
    elif dataset_name == "mad-real":
        class_names = ["Bear", "Bird", "Elephant", "Parrot", "Pig", "Puppy", "Scorpion", "Turtle", "Unicorn", "Whale"]
    elif dataset_name ==  "mad-sim":
        base_path = os.path.join(os.path.dirname(__file__), 'data/MAD-Sim')
        class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        class_names = [re.sub(r'^\d+', '', name)for name in class_names]
    elif dataset_name == "mvtec_loco":
        class_names = ["breakfast_box", "juice_bottle", "pushpins", "screw_bag", "splicing_connectors"]
    elif dataset_name == "real-iad":
        base_path = os.path.join(os.path.dirname(__file__), 'data/Real-IAD')
        class_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        class_names = [name for name in class_names]
    return class_names

class ASICDatasetv2(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, image_size, mode, k_shot, save_dir):
        super().__init__()
        
        self.all_data = []
        self.dataset_root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.class_name = ["asic_v3"]
        self.k_shot = k_shot
        
        with open(f'{self.dataset_root}/annotations/instances_train.json', 'r') as f:
            self.train_annotation = json.load(f)

        with open(f'{self.dataset_root}/annotations/instances_val.json', 'r') as f:
            self.val_annotation = json.load(f)

        self.train_data_annotation = self.train_annotation['annotations']
        self.train_data = self.train_annotation['images']
        self.val_data_annotation = self.val_annotation['annotations']
        self.val_data = self.val_annotation['images']

        defect_free_images = []
        fpr_testset_path = os.path.join(self.dataset_root, "fpr_testset")   
        try:
            self.defect_free_img_file = os.listdir(fpr_testset_path)
            for img_f in self.defect_free_img_file:
                img_path = os.path.join(fpr_testset_path, img_f)
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size

                    defect_free_images.append({
                        'width': width,
                        'height': height,
                        'id': None,
                        'file_name': os.path.join('fpr_testset', img_f)
                    })
                except Exception as e:
                    print(f"Error processing image {img_path} : {e}")
        except Exception as e:
            print(f"Error accessing directory {fpr_testset_path}: {e}")

        if mode == "few_shot":
            self.all_data = []
            save_dir = os.path.join(save_dir, 'k_shot.txt')
            for cls_name in self.class_name:
                indices = torch.randint(0, len(defect_free_images), (int(self.k_shot), ))
                for i in range(len(indices)):
                    temp_data = defect_free_images[indices[i]]
                    self.all_data.append(temp_data)
                    with open(save_dir, "a") as f:
                        f.write(temp_data['file_name'] + '\n')

        else:
            train_image_path = os.path.join(self.dataset_root, 'images', 'train')
            try:
                self.defect_train_images = os.listdir(train_image_path)
                self.annotated_train_data = {ann['image_id'] for ann in self.train_data_annotation}
                self.filtered_train_data = [{**data, "file_name": os.path.join("train", data["file_name"])}
                                            for data in self.train_data
                                    if data['file_name'] in self.defect_train_images and data['id'] in self.annotated_train_data]
            
                self.all_data.extend(self.filtered_train_data)
            except Exception as e:
                print(f"Error processing train data {e}")

            val_image_path = os.path.join(self.dataset_root, 'images', 'val')
            try:
                self.defect_val_images = os.listdir(val_image_path)
                self.annotated_val_data = {ann['image_id'] for ann in self.val_data_annotation}
                self.filtered_val_data = [{**data, "file_name": os.path.join("val", data["file_name"])}
                                           for data in self.val_data
                                    if data['file_name'] in self.defect_val_images and data['id'] in self.annotated_val_data]
            
                self.all_data.extend(self.filtered_val_data)

            except Exception as e:
                print(f"Error processing val data {e}")

            self.all_data.extend(defect_free_images)

        self.length = len(self.all_data)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.all_data[index]
        if data['id'] is None:
            file_path = os.path.join(self.dataset_root, data['file_name'])
            img = Image.open(file_path)
            img_width, img_height = img.size
            annotation = [{'category_id': 1, 'bbox' :[]}]
            mask = Image.new('L', (img_width, img_height), 0)
            anomaly = 0
        else:
            file_path = os.path.join(self.dataset_root, 'images', data['file_name'])
            img = Image.open(file_path)
            img_width, img_height = img.size
            if data['file_name'] in [d['file_name'] for d in self.filtered_train_data]:
                annotation = [annotation for annotation in self.train_data_annotation if annotation['image_id'] == data['id'] ]
            elif data['file_name'] in [d['file_name'] for d in self.filtered_val_data]:
                annotation = [annotation for annotation in self.val_data_annotation if annotation['image_id'] == data['id'] ]
            anomaly = 1
            mask = Image.new('L', (img_width, img_height), 0)
            draw = ImageDraw.Draw(mask)

            for ann in annotation:
                bbox = ann['bbox']
                x_min, y_min, width, height = bbox
                draw.rectangle([x_min, y_min, x_min + width, y_min + height], fill=255)
        file_name = file_path.split('/')[-1]
        mask = np.array(mask)
        cv2.imwrite(f'./test_asic_v3/{file_name}', mask)

        img = self.transform(img) if self.transform is not None else img
        mask = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        mask = self.target_transform(mask).squeeze() if self.target_transform is not None and mask is not None else mask
        return {'img_path': file_path, 'img': img, 'mask': mask, 'anomaly':anomaly, 'class_name': 'asic_v3'}
        

class ASICDataset(data.Dataset):
    def __init__(self, root, transform, target_transform, dataset_name, image_size, mode, k_shot, save_dir):
        super().__init__()

        self.dataset_root = root
        self.transform = transform
        self.target_transform = target_transform
        self.dataset_name = dataset_name
        self.image_size = image_size
        self.mode = mode
        self.k_shot = k_shot
        
        self.data = json.load(open(f'{self.dataset_root}/annotations.json', 'r'))
        self.annotations = self.data['annotations']
        self.data = self.data['images']
        self.class_name = ["asic"]

        normal_images = []
        self.normal_image_file = os.listdir(os.path.join(self.dataset_root, 'images/normal'))
        for img_f in self.normal_image_file:
            img_path = os.path.join(self.dataset_root, 'images/normal', img_f)

            with Image.open(img_path) as img:
                width, height = img.size
            
            normal_images.append({
                'width': width,
                'height': height,
                'id': None,
                'file_name': f"images/normal/{img_f}"
            })

        if self.mode == "few_shot":
            self.all_data = []
            save_dir = os.path.join(save_dir, 'k_shot.txt')
            for cls_name in self.class_name:
                indices = torch.randint(0, len(normal_images), (int(self.k_shot), ))
                for i in range(len(indices)):
                    temp_data = normal_images[indices[i]]
                    self.all_data.append(temp_data)
                    with open(save_dir, "a") as f:
                        f.write(temp_data['file_name'] + '\n')
        else:

            self.image_file = os.listdir(os.path.join(self.dataset_root, 'images'))
            filtered_images = [data for data in self.data if os.path.basename(data['file_name']) in self.image_file]
            annotated_ids = {annotation['image_id'] for annotation in self.annotations}
            self.all_data = [data for data in filtered_images if data['id'] in annotated_ids]
            self.all_data.extend(normal_images)

        self.length = len(self.all_data)
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.all_data[index]
        file_path = os.path.join(self.dataset_root, data['file_name'])
        file_name = file_path.split('/')[-1]
        img = Image.open(file_path)
        img_width, img_height = img.size
        if data['id'] is None:
            annotation = [{'category_id': 1, 'bbox' :[]}]
            mask = Image.new('L', (img_width, img_height), 0)
            anomaly = 0
        else:
            annotation = [annotation for annotation in self.annotations if annotation['image_id'] == data['id'] ]
            anomaly = 1

            mask = Image.new('L', (img_width, img_height), 0)
            draw = ImageDraw.Draw(mask)

            for ann in annotation:
                bbox = ann['bbox']
                x_min, y_min, width, height = bbox
                draw.rectangle([x_min, y_min, x_min + width, y_min + height], fill=255)

        mask = np.array(mask)
        cv2.imwrite(f'./test/{file_name}', mask)

        img = self.transform(img) if self.transform is not None else img
        mask = Image.fromarray(mask.astype(np.uint8), mode='L')
        
        mask = self.target_transform(mask).squeeze() if self.target_transform is not None and mask is not None else mask
        return {'img_path': file_path, 'img': img, 'mask': mask, 'anomaly':anomaly, 'class_name': 'asic'}

class Dataset(data.Dataset):
    def __init__(self, root, transform, target_transform, mode, k_shot, dataset_name, save_dir = None, obj_name = None):
        super(Dataset, self).__init__()

        self.dataset_root = root
        self.transform = transform
        self.target_transform = target_transform
        self.all_data = []
        self.dataset_name = dataset_name
        self.k_shot = k_shot

        meta_info = json.load(open(f'{self.dataset_root}/meta.json', 'r'))
        meta_info = meta_info[mode]
        
        if mode == "train":
            self.class_names = [obj_name]
            save_dir = os.path.join(save_dir, 'k_shot.txt')
        else:
            self.class_names = list(meta_info.keys())

        for cls_name in self.class_names:
            if mode == "train":
                data_tmp = meta_info[cls_name]
                indices =  torch.randint(0, len(data_tmp), (int(self.k_shot),))
                for i in range(len(indices)):
                    self.all_data.append(data_tmp[indices[i]])
                    with open(save_dir, "a") as f:
                        f.write(data_tmp[indices[i]]['img_path'] + '\n')
            else:
                self.all_data.extend(meta_info[cls_name])

        self.length = len(self.all_data)
        self.class_name = get_class_names(self.dataset_name)

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        data = self.all_data[index]

        img_path, mask_path, cls_name, anomaly, defect_type = data['img_path'], data['mask_path'], data['cls_name'], data['anomaly'], data['specie_name']
        
        #img_path = os.path.join(cls_name, img_path)    
        img = Image.open(os.path.join(self.dataset_root, img_path))
        
        if anomaly == 0:
            mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
        else:
            #mask_path = os.path.join(cls_name, mask_path)
            mask = np.array(Image.open(os.path.join(self.dataset_root, mask_path)).convert('L')) > 0
            mask = Image.fromarray(mask.astype(np.uint8) * 255, mode='L')
            #mask.save(os.path.join(self.dataset_root, mask_path))
        img = self.transform(img) if self.transform is not None else img
        mask = self.target_transform(mask) if self.target_transform is not None and mask is not None else mask


        return {"img_path": os.path.join(self.dataset_root, img_path), "img": img, "mask": mask, "class_name": cls_name, "anomaly": anomaly, "defect_type": defect_type}
    
if __name__ == "__main__":
    project_root = os.path.dirname(__file__)
    dataset_root = os.path.join(project_root, 'data/mvtec')
    model, _ , pre_process = open_clip.create_model_and_transforms('ViT-L-14-336-quickgelu', 'openai')

    target_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop(336),
        transforms.ToTensor()
    ])
    dataset = Dataset(dataset_root, transform=pre_process, target_transform= target_transform, mode="train", k_shot=0)
    train_loader = data.DataLoader(dataset, batch_size=2, shuffle=True)
    
    print(next(iter(train_loader))['mask'].shape)