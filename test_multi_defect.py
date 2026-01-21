import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from sklearn.preprocessing import label_binarize

import open_clip
from few_shot import memory
from adapter import LinearAdapter
from dataset import Dataset, ASICDataset, ASICDatasetv2
from learnable_prompt import LearnablePrompt
from visual_prompt_tuning import VisualPromptTuning
from train import compute_similarity, compute_similarity_map
product_type2defect_type_mvtec = {
    'bottle': ['good', 'broken', 'contamination'],
    'cable': ['good', 'bent', 'misplaced', 'combined', 'cut', 'missing', 'poke'],
    'capsule': ['good', 'crack', 'faulty imprint','poke', 'scratch', 'squeeze'],
    'carpet': ['good', 'color', 'cut', 'hole', 'contamination', 'thread'],
    'grid': ['good', 'bent', 'broken', 'glue', 'contamination', 'thread'],
    'hazelnut': ['good', 'crack', 'cut', 'hole', 'faulty imprint'],
    'leather': ['good', 'color', 'cut', 'misplaced', 'glue', 'poke'],
    'metal_nut': ['good', 'bent', 'color', 'misplaced', 'scratch'],
    'pill': ['good', 'color', 'combined', 'contamination', 'crack', 'faulty imprint', 'damaged', 'scratch'],
    'screw': ['good', 'fabric', 'scratch', 'thread'],
    'tile': ['good', 'crack', 'glue', 'damaged', 'liquid', 'rough'],
    'toothbrush': ['good', 'damaged'],
    'transistor': ['good', 'bent', 'cut', 'damaged', 'misplaced'],
    'wood': ['good', 'color', 'combined', 'hole', 'liquid', 'scratch'],
    'zipper': ['good', 'broken', 'combined', 'fabric', 'rough', 'misplaced', 'squeeze']
}


product_type2defect_type_visa = {
    'candle': ['normal', 'damage', 'weird wick', 'partical', 'melded', 'spot', 'extra', 'missing'],
    'capsules': ['normal', 'bubble'],
    'cashew': ['normal', 'scratch', 'breakage','burnt', 'stuck', 'hole', 'spot'],
    'chewinggum': ['normal', 'scratch', 'spot', 'missing'],
    'fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot'],
    'macaroni1': ['normal', 'scratch', 'crack', 'spot', 'chip'],
    'macaroni2': ['normal', 'scratch', 'breakage', 'crack', 'spot', 'chip'],
    'pcb1': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb2': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb3': ['normal', 'scratch', 'melt', 'bent', 'missing'],
    'pcb4': ['normal', 'scratch', 'damage', 'extra', 'burnt', 'missing', 'wrong place'],
    'pipe_fryum': ['normal', 'scratch', 'breakage', 'burnt', 'stuck', 'spot']
}


product_type2defect_type_mpdd = {
    'bracket_black': ['good', 'hole', 'scratch'],
    'bracket_brown': ['good', 'mismatch', 'bent'],
    'bracket_white': ['good', 'defective painting', 'scratch'],
    'connector': ['good', 'mismatch'],
    'metal_plate': ['good', 'rust', 'scratch'],
    'tubes': ['good', 'flattening']
}

product_type2defect_type_mad_real = {
    'Bear': ['good', 'Stains'],
    'Bird': ['good', 'Missing'],
    'Elephant': ['good', 'Missing'],
    'Parrot': ['good', 'Missing'],
    'Pig': ['good', 'Missing'],
    'Puppy': ['good', 'Stains'],
    'Scorpion': ['good', 'Missing'],
    'Turtle': ['good', 'Stains'],
    'Unicorn': ['good', 'Missing'],
    'Whale': ['good', 'Stains']
}

product_type2defect_type_mad_sim = {
    'Gorilla': ['good', 'Stains', 'Burrs', 'Missing'],
    'Unicorn': ['good', 'Stains', 'Burrs', 'Missing'],
    'Mallard': ['good', 'Stains', 'Burrs', 'Missing'],
    'Turtle': ['good', 'Stains', 'Burrs', 'Missing'],
    'Whale': ['good', 'Stains', 'Burrs', 'Missing'],
    'Bird': ['good', 'Stains', 'Burrs', 'Missing'],
    'Owl': ['good', 'Stains', 'Burrs', 'Missing'],
    'Sabertooth': ['good', 'Stains', 'Burrs', 'Missing'],
    'Swan': ['good', 'Stains', 'Burrs', 'Missing'],
    'Sheep': ['good', 'Stains', 'Burrs', 'Missing'],
    'Pig': ['good', 'Stains', 'Burrs', 'Missing'],
    'Zalika': ['good', 'Stains', 'Burrs', 'Missing'],
    'Pheonix': ['good', 'Stains', 'Burrs', 'Missing'],
    'Elephant': ['good', 'Stains', 'Burrs', 'Missing'],
    'Parrot': ['good', 'Stains', 'Burrs', 'Missing'],
    'Cat': ['good', 'Stains', 'Burrs', 'Missing'],
    'Scorpion': ['good', 'Stains', 'Burrs', 'Missing'],
    'Obesobeso': ['good', 'Stains', 'Burrs', 'Missing'],
    'Bear': ['good', 'Stains', 'Burrs', 'Missing'],
    'Puppy': ['good', 'Stains', 'Burrs', 'Missing'],
}
product_type2defect_type_real_iad = {
    'switch': ['good', 'missing', 'contamination', 'scratch'], 
    'eraser': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'woodstick': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'zipper': ['good', 'contamination', 'deformation', 'missing', 'damage'], 
    'fire_hood': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'pcb': ['good', 'contamination', 'scratch', 'missing', 'foreign'], 
    'toothbrush': ['good', 'abrasion', 'contamination', 'missing'], 
    'plastic_nut': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'wooden_beads': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'transistor1': ['good', 'missing', 'contamination', 'deformation'], 
    'bottle_cap': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'u_block': ['good', 'abrasion', 'contamination', 'scratch', 'missing'], 
    'sim_card_set': ['good', 'abrasion', 'contamination', 'scratch'], 
    'end_cap': ['good', 'contamination', 'scratch', 'missing', 'damage'], 
    'usb': ['good', 'contamination', 'deformation', 'scratch', 'missing'], 
    'regulator': ['good', 'missing', 'scratch'], 
    'plastic_plug': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'audiojack': ['good', 'contamination', 'deformation', 'scratch', 'missing'], 
    'mint': ['good', 'missing', 'contamination', 'foreign'], 
    'toy_brick': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'toy': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'rolled_strip_base': ['good', 'pit', 'missing', 'contamination'], 
    'terminalblock': ['good', 'pit', 'missing', 'contamination', ], 
    'mounts': ['good', 'missing', 'contamination', 'pit'], 
    'button_battery': ['good', 'abrasion', 'contamination', 'scratch', 'pit'], 
    'porcelain_doll': ['good', 'abrasion', 'contamination', 'scratch'], 
    'phone_battery': ['good', 'contamination', 'scratch', 'damage', 'pit'], 
    'usb_adaptor': ['good', 'abrasion', 'contamination', 'scratch', 'pit'], 
    'vcpill': ['good', 'contamination', 'scratch', 'missing', 'pit'], 
    'tape': ['good', 'missing', 'contamination', 'damage']
}

import re
from tqdm import tqdm

import pdb


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def test(args):
    img_size = args.image_size
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device =  "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained, force_image_size=img_size)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device("cpu"))

    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    # datasets
    if args.dataset == 'mvtec':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, aug_rate=-1, mode='test')
        gt_defect = {"good":0, "bent":1, "bent_lead":1, "bent_wire":1, "broken":2, "broken_large":2, "broken_small":2, "broken_teeth":2, "color":3, "combined":4, "contamination":5, "metal_contamination":5, "crack":6, "cut":7, "cut_inner_insulation":7, "cut_lead":7, "cut_outer_insulation":7, "fabric":8, "manipulated_front":8, "fabric_border":8, "fabric_interior":8, "faulty_imprint":9, "print":9, "glue":10, "glue_strip":10, "hole":11, "missing":12, "missing_wire":12, "missing_cable":12, "poke":13, "poke_insulation":13, "rough":14, "scratch":15, "scratch_head":15, "scratch_neck":15, "squeeze":16, "squeezed_teeth":16, "thread":17, "thread_side":17, "thread_top":17, "liquid":18, "oil":18, "misplaced":19, "cable_swap":19, "flip":19, "fold":19, "split_teeth":19, "damaged_case":20, "defective":20, "gray_stroke":20, "pill_type":20}  
        defects = ['good', 'bent', 'broken', 'color', 'combined', 'contamination', 'crack', 'cut', 'fabric', 'faulty imprint', 'glue', 'hole', 'missing', 'poke', 'rough', 'scratch', 'squeeze', 'thread', 'liquid', 'misplaced', 'damaged']
        p_cls2d_cls = product_type2defect_type_mvtec
        
    elif args.dataset == 'visa':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, dataset_name=dataset_name, mode="test", k_shot=0)
        gt_defect = {'normal': 0, 'damage': 1, 'scratch':2, 'breakage': 3, 'burnt': 4, 'weird wick': 5, 'stuck': 6, 'crack': 7, 'wrong place': 8, 'partical': 9, 'bubble': 10, 'melded': 11, 'hole': 12, 'melt': 13, 'bent':14, 'spot': 15, 'extra': 16, 'chip': 17, 'missing': 18}
        defects = ['normal', 'damage', 'scratch', 'breakage',
                    'burnt', 'weird wick', 'stuck', 'crack', 'wrong place', 'partical', 'bubble', 'melded', 'hole',
                      'melt', 'bent', 'spot', 'extra', 'chip', 'missing']
        p_cls2d_cls = product_type2defect_type_visa
    elif args.dataset == 'mpdd':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, dataset_name=dataset_name, mode="test", k_shot=0)
        gt_defect =  {"good":0, 'hole':1, 'scratches':2, 'bend_and_parts_mismatch':3, 'parts_mismatch':4, 'defective_painting':5, 'major_rust':6, 'total_rust':6, 'flattening':7}
        defects = ['good', 'hole', 'scratch', 'bent', 'mismatch', 'defective painting', 'rust', 'flattening']
        p_cls2d_cls = product_type2defect_type_mpdd
    elif args.dataset == 'mad_sim':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, dataset_name=dataset_name, mode="test", k_shot=0)
        gt_defect =  {"good":0, 'Stains':1, 'Missing':2, 'Burrs':3}
        defects = ['good', 'Stains', 'Missing', 'Burrs']
        p_cls2d_cls = product_type2defect_type_mad_sim
    elif args.dataset == 'mad-real':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, dataset_name=dataset_name, mode="test", k_shot=0)
        gt_defect =  {"good":0, 'Stains':1, 'Missing':2}
        defects = ['good', 'Stains', 'Missing']
        p_cls2d_cls = product_type2defect_type_mad_real

    elif args.dataset == 'real_iad':
        test_data = Dataset(root=dataset_dir, transform=preprocess, target_transform=transform, dataset_name=dataset_name, mode="test", k_shot=0)
        gt_defect =  {"good":0, 'pit':1, 'deformation':2, 'abrasion':3, 'scratch':4, 'damage':5, 'missing':6, 'foreign':7, 'contamination':8}
        defects = ['good', 'pit', 'deformation', 'abrasion', 'scratch', 'damage', 'missing', 'foreign', 'contamination']
        p_cls2d_cls = product_type2defect_type_real_iad
    
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.class_name

    visual_prompt = VisualPromptTuning(model=model, total_d_layer=24, num_tokens=4, device=device)
    prompt_learner = LearnablePrompt(model, prompt_state=defects[1:], normal_token_count=5, prompt_count=10, abnormal_token_count=5, tokenizer=tokenizer, device=device)
    trainable_adapter = LinearAdapter(dim_in=1024, dim_out=768, k=4)

    visual_prompt.load_state_dict(checkpoint["visual_prompt"])
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    trainable_adapter.load_state_dict(checkpoint["linear_adapter"])
    visual_prompt.to(device)

    prompt_learner.to(device)
    trainable_adapter.to(device)

    visual_prompt.eval()
    prompt_learner.eval()
    trainable_adapter.eval()

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    class_ids = []
    
    for items in tqdm(test_dataloader):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        paths = items['img_path']
        results['cls_names'].append(cls_name[0])

        img_masks = items['mask']
        if args.dataset == 'mvtec' or args.dataset == 'mpdd' :
            defect_cls_id = []               
            for i in paths:
                match = re.search(r'\/([^\/]+)\/[^\/]*$', i) # './data/mvtec/transistor/test/good/004.png', './data/mvtec/carpet/test/hole/002.png', './data/mvtec/metal_nut/test/scratch/004.png',
                defect_cls_id.append(int(gt_defect[str(match.group(1))]))
        elif args.dataset == 'visa' or args.dataset == 'mad_sim' or args.dataset == 'mad-real' or args.dataset =='real_iad':
            defect_cls = items['specie_name']
            defect_cls_id = [gt_defect[name] for name in defect_cls]

        gt_mask = items['mask']
        
        for i in range(gt_mask.size(0)):
            gt_mask[i][gt_mask[i] > 0.5], gt_mask[i][gt_mask[i] <= 0.5] = p_cls2d_cls[cls_name[i]].index(defects[defect_cls_id[i]]), 0 
        

        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())

        with torch.no_grad():
            visual_tokens = visual_prompt()
            image_features, patch_features, attn_weights = model.encode_image([image, visual_tokens])
            image_features /= image_features.norm(dim=-1, keepdim=True)

            prompt, learnable_tokens, template_tokens = prompt_learner(image_features)
            text_features, learned_tokens = model.encode_learn_prompts(prompt, learnable_tokens, template_tokens)
            text_features = text_features/text_features.norm(dim=-1, keepdim=True)
            text_prompts = []
            for cls in cls_name:
                defects_indices = [defects.index(d) for d in p_cls2d_cls[cls]]
                text_prompts.append(text_features[defects_indices])
            
            text_prompts = torch.stack(text_prompts, dim=0)

            # sample
            text_probs = ((image_features @ text_prompts[0].T) / 0.07).softmax(dim=-1) # B, H, W
            # pdb.set_trace()
            results['pr_sp'].append(sum(text_probs[0][1:]).cpu().item())

            # pixel
            patch_features = trainable_adapter(patch_features)
            anomaly_maps = []

            for _, patch in enumerate(patch_features):
                patch /= patch.norm(dim=-1, keepdim=True)
                similarity = compute_similarity(patch, text_prompts[0]).softmax(dim=-1)
                similarity_map = compute_similarity_map(similarity, img_size)
                anomaly_maps.append(similarity_map.cpu().numpy())

            anomaly_map = np.sum(anomaly_maps, axis=0)
            results['anomaly_maps'].append(anomaly_map)

    # metrics
    table_ls = []
    auroc_px_ls = []
    f1_px_ls = []
    aupro_ls = []
    ap_px_ls = []
    auroc_px_per_defect_cls_all = {}
    ap_px_per_defect_cls_all = {}
    f1_px_per_defect_cls_all = {}

    for obj in obj_list:
        table = []
        gt_px = []
        pr_px = []
        table.append(obj)
        for idxes in range(len(results['cls_names'])):
            if results['cls_names'][idxes] == obj:
                gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
                pr_px.append(results['anomaly_maps'][idxes])

        gt_px = np.array(gt_px) # (N_p, B, H, W)
        pr_px = np.array(pr_px) # (N_p, B, C, H, W)
        C = pr_px.shape[4]
        gt_px = gt_px.ravel()  # (N,)
        gt_px_b = label_binarize(gt_px, classes=range(C)) # (N, C)
        if C == 2:
            gt_px_b = np.hstack((1-gt_px_b, gt_px_b))
        pr_px = pr_px.reshape(-1, C) # (N, C)
        
        try:
            auroc_px = roc_auc_score(gt_px_b, pr_px, multi_class='ovr') #, multi_class='ovo', labels = class_ids)
            roc_auc_per_class = roc_auc_score(gt_px_b, pr_px, multi_class='ovr', average=None)
            for i, auc in enumerate(roc_auc_per_class):
                if p_cls2d_cls[obj][i] in auroc_px_per_defect_cls_all:
                    auroc_px_per_defect_cls_all[p_cls2d_cls[obj][i]].append(auc)
                else:
                    auroc_px_per_defect_cls_all[p_cls2d_cls[obj][i]] = [auc]
        except:
            print(obj)
            pdb.set_trace()
            exit()
        ap_px = average_precision_score(gt_px_b, pr_px, average='macro')
        ap_px_per_class = average_precision_score(gt_px_b, pr_px, average=None)
        for i, ap in enumerate(ap_px_per_class):
            if p_cls2d_cls[obj][i] in ap_px_per_defect_cls_all:
                ap_px_per_defect_cls_all[p_cls2d_cls[obj][i]].append(ap)
            else:
                ap_px_per_defect_cls_all[p_cls2d_cls[obj][i]] = [ap]

        # f1_px
        f1_px = f1_score(gt_px, np.argmax(pr_px, axis=-1), average='macro')
        f1_px_per_class = f1_score(gt_px, np.argmax(pr_px, axis=-1), average=None)
        for i, f1 in enumerate(f1_px_per_class):
            if p_cls2d_cls[obj][i] in f1_px_per_defect_cls_all:
                f1_px_per_defect_cls_all[p_cls2d_cls[obj][i]].append(f1)
            else:
                f1_px_per_defect_cls_all[p_cls2d_cls[obj][i]] = [f1]

        table.append(str(np.round(auroc_px * 100, decimals=1)))
        table.append(str(np.round(f1_px * 100, decimals=1)))
        table.append(str(np.round(ap_px * 100, decimals=1)))

        table_ls.append(table)
        auroc_px_ls.append(auroc_px)
        f1_px_ls.append(f1_px)
        ap_px_ls.append(ap_px)
    # logger
    per_defect_cls_table = []
    for defect in auroc_px_per_defect_cls_all:
        auroc = np.round(np.mean(auroc_px_per_defect_cls_all[defect]) * 100, decimals=6)
        ap = np.round(np.mean(ap_px_per_defect_cls_all[defect]) * 100, decimals=6)
        f1 = np.round(np.mean(f1_px_per_defect_cls_all[defect]) * 100, decimals=6)
        per_defect_cls_table.append([defect, auroc, ap, f1])
    results = tabulate(per_defect_cls_table, headers=['defects', 'auroc_px', 'f1_px', 'ap_px'], tablefmt="pipe")
    logger.info("\n%s", results)
    
    table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
                     str(np.round(np.mean(aupro_ls) * 100, decimals=1))])
    results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px'], tablefmt="pipe")
    logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MultiADS", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa_v2", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/visa_multi_type_seg/zero_shot/', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./checkpoint/deepppt/best_checkpoint/epoch_1_depth12.pth', help='path to save results')
    
    # model
    parser.add_argument("--dataset", type=str, default='visa', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-L-14-336-quickgelu", help="model used")
    parser.add_argument("--pretrained", type=str, default="openai", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="features used")
    parser.add_argument("--image_size", type=int, default=518, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()
    setup_seed(args.seed)
    test(args)
