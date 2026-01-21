import torch
import torch.nn.functional as F
import numpy as np
import os
import json
import argparse
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from tabulate import tabulate
from sklearn.metrics import pairwise, ConfusionMatrixDisplay, RocCurveDisplay
from PIL import Image
import open_clip

from logger import get_logger
from learnable_prompt import LearnablePrompt
from dataset import Dataset, ASICDataset, ASICDatasetv2
from visualization import visualizer
from train import compute_similarity, compute_similarity_map
from metrics import image_level_metrics, pixel_level_metrics
#from few_shot import memory
from adapter import LinearAdapter
from visual_prompt_tuning import VisualPromptTuning


class Tester:
    def __init__(self, args):
        """
        Initialize the tester
        
        Args:
            args: Testing arguments containing model config, paths, etc.
        """
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize logger
        # self.logger = get_logger(args.save_path)
        
        # Dataset-specific configurations
        self.dataset_configs = {
            "mvtec": {
                "prompt_states": ["bent", "broken", "color", "combined", "contamination", 
                                "crack", "cut", "fabric", "faulty imprint", "glue", "hole", 
                                "missing", "poke", "rough", "scratch", "squeeze", "thread", 
                                "liquid", "misplaced", "damaged"],
                "dataset_class": "Dataset"
            },
            "asic": {
                "prompt_states": ["contamination"],
                "dataset_class": "ASICDataset"
            },
            "asic_v2": {
                "prompt_states": ["contamination"],
                "dataset_class": "ASICDatasetv2"
            },
            "visa": {
                "prompt_states": ['damage', 'scratch', 'breakage', 'burnt', 'irregular wick', 
                                'stuck', 'crack', 'wrong place', 'partical', 'bubble', 'melded', 
                                'hole', 'melt', 'bent', 'spot', 'extra', 'chip', 'missing'],
                "dataset_class": "Dataset"
            },
            "mpdd": {
                "prompt_states": ["hole", "scratch", "bent", "rust", "flattening", 
                                "mismatch", "defective painting"],
                "dataset_class": "Dataset"
            },
            "mvtec_loco": {
                "prompt_states": ["misplaced", "damaged"],
                "dataset_class": "Dataset"
            },
            "mad-real": {
                "prompt_states": ["stains", "missing"],
                "dataset_class": "Dataset"
            },
            "mad-sim": {
                "prompt_states": ["missing", "stains", "burrs"],
                "dataset_class": "Dataset"
            },
            "real-iad": {
                "prompt_states": ["hole", "deformed", "scratch", "damage", 
                                "missing parts", "contamination"],
                "dataset_class": "Dataset"
            }
        }
        if self.args.dataset in self.dataset_configs:
            pass
        else:
            new_config = {
                "prompt_states": self.args.defect_types,
                "dataset_class": "CustomDatasetClass",
            }
            self.dataset_configs[self.args.dataset] = new_config
        
        # Initialize components
        self.model_components = None
        self.test_loader = None
        self.class_names = None
        self.results = {}
        self.metrics = {}
        
    def setup_model(self):
        """Setup CLIP model and preprocessing"""
        print(f"Setting up model: {self.args.model_name}")
        
        model, _, pre_process = open_clip.create_model_and_transforms(
            self.args.model_name, 
            pretrained=self.args.pretrained, 
            force_image_size=self.args.image_size
        )
        
        # Modify preprocessing
        pre_process.transforms[0] = transforms.Resize(
            size=(self.args.image_size, self.args.image_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
            max_size=None, antialias=None
        )
        pre_process.transforms[1] = transforms.CenterCrop(size=(self.args.image_size, self.args.image_size))
        
        model.eval()
        model = model.to(self.device)
        
        tokenizer = open_clip.get_tokenizer(self.args.model_name)
        
        return {
            'model': model,
            'preprocessor': pre_process,
            'tokenizer': tokenizer
        }
    
    def setup_dataset(self):
        """Setup test dataset and data loader"""
        print(f"Setting up dataset: {self.args.dataset}")
        
        target_transform = transforms.Compose([
            transforms.Resize((self.args.image_size, self.args.image_size)),
            transforms.CenterCrop(self.args.image_size),
            transforms.ToTensor()
        ])
        
        dataset_config = self.dataset_configs.get(self.args.dataset)
        if not dataset_config:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
        
        # Create dataset based on type
        if self.args.dataset == "asic":
            test_data = ASICDataset(
                self.args.test_data_path, 
                transform=self.model_components['preprocessor'],
                target_transform=target_transform,
                dataset_name=self.args.dataset,
                image_size=self.args.image_size,
                mode="test",
                k_shot=0,
                save_dir=self.args.save_path
            )

        elif self.args.dataset == "asic_v2":
            test_data = ASICDatasetv2(
                self.args.test_data_path,
                transform=self.model_components['preprocessor'],
                target_transform=target_transform,
                dataset_name=self.args.dataset,
                image_size=self.args.image_size,
                mode="test",
                k_shot=0,
                save_dir=self.args.save_path
            )

        else:
            test_data = Dataset(
                self.args.test_data_path,
                transform=self.model_components['preprocessor'],
                target_transform=target_transform,
                dataset_name=self.args.dataset,
                mode="test",
                k_shot=0
            )
        
    
        self.class_names = test_data.class_name
        return DataLoader(test_data, shuffle=False, batch_size=1)
    
    
    def setup_learnable_components(self):
        """Setup learnable prompt, visual prompt, and adapter"""
        dataset_config = self.dataset_configs[self.args.dataset]
        prompt_state_all = dataset_config["prompt_states"]
        
        prompt_learner = LearnablePrompt(
        clip_model=self.model_components["model"],
        prompt_state=prompt_state_all,
        normal_token_count=self.args.normal_token_cnt,
        prompt_count=self.args.prompt_count,
        abnormal_token_count=self.args.abnormal_token_cnt,
        token_depth=self.args.text_depth,
        learnable_token_length=self.args.layer_token_cnt,
        tokenizer=self.model_components["tokenizer"],
        device=self.device
    )

        # Initialize visual prompt tuning
        visual_prompt = VisualPromptTuning(
            model=self.model_components["model"],
            total_d_layer=self.args.depth,
            num_tokens=self.args.prefix_token_cnt,
            device=self.device
        )
        with open(os.path.join(self.args.model_configs, f"{self.args.model_name}.json"), "r") as f:
            model_config = json.load(f)
        trainable_adapter = LinearAdapter(dim_in=model_config["vision_cfg"]["width"], dim_out=model_config["embed_dim"], k=len(self.args.feature_layers))
        
        return {
            'visual_prompt': visual_prompt,
            'prompt_learner': prompt_learner,
            'trainable_adapter': trainable_adapter
        }
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load learnable components
        if checkpoint["visual_prompt_state_dict"]:
            self.learnable_components['visual_prompt'].load_state_dict(checkpoint["visual_prompt_state_dict"])
        else:
            self.learnable_components['visual_prompt'].load_state_dict(checkpoint["visual_prompt"])
        self.learnable_components['visual_prompt'].to(self.device)
        self.learnable_components['visual_prompt'].eval()
        
        if checkpoint["prompt_learner_state_dict"]:
            self.learnable_components['prompt_learner'].load_state_dict(checkpoint["prompt_learner_state_dict"])
        else:
            self.learnable_components['prompt_learner'].load_state_dict(checkpoint["prompt_learner"])
        self.learnable_components['prompt_learner'].to(self.device)
        self.learnable_components['prompt_learner'].eval()
        
        if checkpoint["trainable_adapter_state_dict"]:
            self.learnable_components['trainable_adapter'].load_state_dict(checkpoint["trainable_adapter_state_dict"])
        else:
             self.learnable_components['trainable_adapter'].load_state_dict(checkpoint["linear_adapter"])
        self.learnable_components['trainable_adapter'].to(self.device)
        self.learnable_components['trainable_adapter'].eval()
        
        return checkpoint
    
    def initialize_results(self):
        """Initialize results and metrics dictionaries"""
        self.results = {}
        self.metrics = {}
        
        for cls in self.class_names:
            self.results[cls] = {
                'gt_sp': [],
                'pr_sp': [],
                'pr_label': [],
                'img_mask': [],
                'anomaly_map': []
            }
            
            self.metrics[cls] = {
                'pixel_auroc': 0,
                'pixel_aupro': 0,
                'image_auroc': 0,
                'image_ap': 0
            }
    
    def process_image(self, items):
        """Process a single image through the model"""
        class_name = items['class_name']
        image = items['img'].to(self.device)
        label = items['anomaly']
        gt_mask = items['mask']
        
        # Process ground truth mask
        for i in range(gt_mask.size(0)):
            gt_mask[i][gt_mask[i] > 0.5], gt_mask[i][gt_mask[i] <= 0.5] = 1, 0
        
        # Store ground truth
        self.results[class_name[0]]['img_mask'].append(gt_mask)
        self.results[class_name[0]]['gt_sp'].append(label.item())
        
        with torch.no_grad():
            # Forward pass through model
            local_anomaly_map, anomaly_score = self.forward_pass(image)
            pr_label = 1 if anomaly_score >= 0.5 else 0
            
            # Store predictions
            self.results[class_name[0]]['pr_label'].append(pr_label)
            self.results[class_name[0]]['pr_sp'].append(anomaly_score.detach().cpu().item())
            self.results[class_name[0]]['anomaly_map'].append(local_anomaly_map.detach().cpu())
            
            # Visualize results
            visualizer(items['img_path'], local_anomaly_map.detach().cpu().numpy(), 
                      self.args.image_size, self.args.save_path, class_name)
            
    def forward_pass(self, image):
        """Performs a forward pass for inference on a single image tensor."""
        visual_tokens = self.learnable_components['visual_prompt']()
        image_features, patch_features, attn_weights = self.model_components['model'].encode_image([image, visual_tokens], self.args.feature_layers)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        print(patch_features[0].shape)
        patch_features = self.learnable_components['trainable_adapter'](patch_features)
        
        # Text prompt processing
        prompt, learnable_tokens, template_tokens = self.learnable_components['prompt_learner'](image_features)
        text_features, learned_tokens = self.model_components['model'].encode_learn_prompts(prompt, learnable_tokens, template_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Separate normal and abnormal features
        normal_feature = text_features[0:1]
        abnormal_feature = text_features[1:]
        
        avg_abnormal_text_feature = abnormal_feature.mean(dim=0, keepdim=True)
        avg_abnormal_text_feature = avg_abnormal_text_feature / avg_abnormal_text_feature.norm(dim=-1, keepdim=True)
        normal_abnormal_feature = torch.cat([normal_feature, avg_abnormal_text_feature], dim=0)
        
        # Compute probabilities
        probs = (image_features @ normal_abnormal_feature.T) / 0.07
        
        # Process patches for local anomaly maps
        anomaly_map_list = []
        patch_similarity_list = []
        
        for _, patch in enumerate(patch_features):
            patch = patch / patch.norm(dim=-1, keepdim=True)
            similarity = compute_similarity(patch, text_features).softmax(dim=-1)
            binary_similarity = compute_similarity(patch, normal_abnormal_feature).softmax(dim=-1)
            similarity_map = compute_similarity_map(similarity, self.args.image_size)
            
            anomaly_map = torch.sum(similarity_map[:, :, :, 1:], dim=-1)
            anomaly_map_list.append(anomaly_map)
            patch_similarity_list.append(binary_similarity)
        
        local_anomaly_map = torch.stack(anomaly_map_list).sum(dim=0)
        probs = probs.softmax(dim=-1)
        anomaly_score = probs[:, 1]

        return local_anomaly_map, anomaly_score

    def run_single_image_inference(self):
        """Run inference on a single image"""
        print(f"--- Running Single Image Inference ---")
        print(f"Image Path: {self.args.image_path}")

        try:
            img = Image.open(self.args.image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image not found at {self.args.image_path}")
            return
        
        checkpoint_path = f"{self.args.checkpoint_path}epoch_{self.args.epoch}.pth"
        checkpoint = self.load_checkpoint(checkpoint_path)

        # Preprocess the image
        preprocessor = self.model_components['preprocessor']
        img_tensor = preprocessor(img).unsqueeze(0).to(self.device)
        print(img_tensor.shape, self.device)

        # Perform inference
        with torch.no_grad():
            local_anomaly_map, anomaly_score = self.forward_pass(img_tensor)

        # Display and save results
        anomaly_score_value = anomaly_score.detach().cpu().item()
        predicted_label = "Anomaly" if anomaly_score_value >= 0.5 else "Non-defective"

        print(f"\nAnomaly Score: {anomaly_score_value:.4f}")
        print(f"Predicted Label: {predicted_label}")
    
        image_class_name = ["ASIC"]
        visualizer([self.args.image_path], local_anomaly_map.detach().cpu().numpy(),
                   self.args.image_size, self.args.save_path, image_class_name)
        
        base_name = os.path.basename(self.args.image_path)
        file_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(self.args.save_path, image_class_name[0], f"{file_name_no_ext}.png")
        print(f"Anomaly map saved to: {output_path}")
        return predicted_label 

    def compute_metrics(self):
        """Compute evaluation metrics for all classes"""
        table_ls = []
        image_auroc_list = []
        image_ap_list = []
        image_f1_list = []
        pixel_auroc_list = []
        pixel_aupro_list = []
        
        for cls in self.class_names:
            table = [cls]
            
            # Concatenate results
            self.results[cls]['img_mask'] = torch.cat(self.results[cls]['img_mask'])
            self.results[cls]['anomaly_map'] = torch.cat(self.results[cls]['anomaly_map']).numpy()
            
            if self.args.metrics == 'image_level':
                image_auroc = image_level_metrics(self.results, cls, 'image_auroc')
                image_ap = image_level_metrics(self.results, cls, 'image_ap')
                
                img_precision, img_recall, thresholds = image_level_metrics(self.results, cls, 'precision_recall')
                f1_score = (2 * img_precision * img_recall) / (img_precision + img_recall)
                f1_sp = np.max(f1_score[np.isfinite(f1_score)])
                
                table.extend([
                    str(np.round(image_auroc * 100, decimals=1)),
                    str(np.round(image_ap * 100, decimals=1)),
                    str(np.round(f1_sp * 100, decimals=1))
                ])
                
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap)
                image_f1_list.append(f1_sp)
                
                # Generate confusion matrix and ROC curve
                self.generate_confusion_matrix(cls)
                self.generate_roc_curve(cls)
                
            elif self.args.metrics == 'pixel_level':
                pixel_auroc = pixel_level_metrics(self.results, cls, 'pixel_auroc')
                pixel_aupro = pixel_level_metrics(self.results, cls, 'pixel_aupro')
                
                table.extend([
                    str(np.round(pixel_auroc * 100, decimals=1)),
                    str(np.round(pixel_aupro * 100, decimals=1))
                ])
                
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
                
            elif self.args.metrics == "image_pixel_level":
                image_auroc = image_level_metrics(self.results, cls, "image_auroc")
                image_ap = image_level_metrics(self.results, cls, "image_ap")
                img_precision, img_recall, thresholds = image_level_metrics(self.results, cls, 'precision_recall')
                f1_score = (2 * img_precision * img_recall) / (img_precision + img_recall)
                f1_sp = np.max(f1_score[np.isfinite(f1_score)])
                pixel_auroc = pixel_level_metrics(self.results, cls, "pixel_auroc")
                pixel_aupro = pixel_level_metrics(self.results, cls, "pixel_aupro")
                
                table.extend([
                    str(np.round(pixel_auroc * 100, decimals=1)),
                    str(np.round(pixel_aupro * 100, decimals=1)),
                    str(np.round(image_auroc * 100, decimals=1)),
                    str(np.round(image_ap * 100, decimals=1)),
                    str(np.round(f1_sp * 100, decimals=1))
                ])
                
                image_auroc_list.append(image_auroc)
                image_ap_list.append(image_ap)
                image_f1_list.append(f1_sp)
                pixel_auroc_list.append(pixel_auroc)
                pixel_aupro_list.append(pixel_aupro)
            
            table_ls.append(table)
        
        # Add mean row and create table
        if self.args.metrics == 'image_level':
            table_ls.append(['mean',
                           str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                           str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                           str(np.round(np.mean(image_f1_list) * 100, decimals=1))])
            headers = ['objects', 'image_auroc', 'image_ap', 'image_f1_score']
            
        elif self.args.metrics == 'pixel_level':
            table_ls.append(['mean',
                           str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                           str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1))])
            headers = ['objects', 'pixel_auroc', 'pixel_aupro']
            
        elif self.args.metrics == 'image_pixel_level':
            table_ls.append(['mean',
                           str(np.round(np.mean(pixel_auroc_list) * 100, decimals=1)),
                           str(np.round(np.mean(pixel_aupro_list) * 100, decimals=1)),
                           str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                           str(np.round(np.mean(image_ap_list) * 100, decimals=1)),
                           str(np.round(np.mean(image_f1_list) * 100, decimals=1))])
            headers = ['objects', 'pixel_auroc', 'pixel_aupro', 'image_auroc', 'image_ap', 'image_f1_score']
        
        results_table = tabulate(table_ls, headers=headers, tablefmt="pipe")
        
        return results_table
    
    def generate_confusion_matrix(self, cls):
        """Generate and save confusion matrix"""
        confusion_matrix = image_level_metrics(self.results, cls, 'confusion_matrix')
        cm_disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, 
                                       display_labels=["Non-defective", "Anomaly"])
        cm_disp.plot()
        plt.title(f"Confusion Matrix {cls}")
        plt.savefig(os.path.join(self.args.save_path, f"confusion_matrix_{cls}.png"))
        plt.close()
    
    def generate_roc_curve(self, cls):
        """Generate and save ROC curve"""
        fpr, tpr, threshold = image_level_metrics(self.results, cls, 'fpr')
        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, 
                                    estimator_name="Anomaly Detection").plot()
        plt.title(f"ROC Curve {cls}")
        plt.savefig(os.path.join(self.args.save_path, f"roc_curve_{cls}.png"))
        plt.close()
        
        # Plot TPR/FPR vs thresholds
        plt.figure(figsize=(6, 6))
        plt.plot(threshold, tpr, color='g', label='TPR')
        plt.plot(threshold, fpr, color='r', label='FPR')
        plt.xlabel('Threshold')
        plt.ylabel('Rate')
        plt.title(f'TPR/FPR vs Thresholds - {cls}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.args.save_path, f"tpr_fpr_threshold_{cls}.png"))
        plt.close()
    
    def test_epoch(self, epoch):
        """Test a single epoch checkpoint"""
        print(f"Testing epoch {epoch}")
        
        checkpoint_path = f"{self.args.checkpoint_path}epoch_{epoch}.pth"
        checkpoint = self.load_checkpoint(checkpoint_path)
        
        # Initialize results for this epoch
        self.initialize_results()
        
        # Process all test images
        for idx, items in enumerate(tqdm(self.test_loader, desc=f"Testing Epoch {epoch}")):
            self.process_image(items)
        
        # Compute and return metrics
        results_table = self.compute_metrics()
        
        return results_table
    
    def run_test(self):
        """Main testing pipeline"""
        print(f"Starting testing pipeline for {self.args.dataset}")
        print(f"Device: {self.device}")
        
        # Setup all components
        self.model_components = self.setup_model()
        self.learnable_components = self.setup_learnable_components()
        
        # Create save directory
        os.makedirs(self.args.save_path, exist_ok=True)
        
        # Run testing
        if self.args.inference_mode:
            label = self.run_single_image_inference()
            return label
        else:
            self.test_loader = self.setup_dataset()
            results_table = self.test_epoch(self.args.epoch)
            print("\nFinal Results:")
            return results_table


def create_test_args_parser():
    """Create argument parser for testing"""
    project_root = os.path.dirname(__file__)
    dataset_root = os.path.join(project_root, 'data')
    parser = argparse.ArgumentParser("Prompt Learning Testing")
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default="ViT-L-14-336-quickgelu")
    parser.add_argument("--model_configs", type=str, default= os.path.join(project_root, "open_clip", "model_configs"))
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument('--image_size', type=int, default=518)
    
    # Dataset arguments
    parser.add_argument('--dataset', type=str, default="asic", 
                       choices=['mvtec', 'asic', 'asic_v2', 'visa', 'mpdd', 
                               'mvtec_loco', 'mad-real', 'mad-sim', 'real-iad'])
    parser.add_argument("--test_data_path", type=str, default=f"{dataset_root}/asic")
    parser.add_argument("--defect_types", type=str, nargs='+', default="contamination")

    #Image encoder
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--prefix_token_cnt", type=int, default=4)
    parser.add_argument("--feature_layers", type=list, default=[6,12,18,24])

    #Text Encoder
    parser.add_argument("--text_depth", type=int, default=12)
    parser.add_argument("--normal_token_cnt", type=int, default=5)
    parser.add_argument("--abnormal_token_cnt", type=int, default=5)
    parser.add_argument("--prompt_count", type=int, default=10)
    parser.add_argument("--layer_token_cnt", type=int, default=4)
    
    # Testing arguments
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint/")
    parser.add_argument("--save_path", type=str, default="./results/")
    parser.add_argument("--metrics", type=str, default="image_level",
                       choices=['image_level', 'pixel_level', 'image_pixel_level'])
    parser.add_argument("--mode", type=str, default="zero_shot")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for testing (usually 1 for anomaly detection)")
    
    # arguments for single image inference
    parser.add_argument('--inference_mode', action='store_true', 
                        help='Run inference on a single image instead of a dataset.')
    parser.add_argument('--image_path', type=str, default=f"{dataset_root}/asic/2_19FH0215_CS990DD_6cb9843f20284313bb8cb5e707617188_19FH0215_irr_Q3Q4_020x_filter1100nm_03.png", 
                        help='Path to the single image.')
    
    return parser


class TestConfig:
    """Configuration class for testing parameters"""
    
    def __init__(self, args):
        self.args = args
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration parameters"""
        if self.args.inference_mode:
            if self.args.image_path is None:
                raise ValueError("Argument --image_path must be provided in inference mode.")
            if not os.path.exists(self.args.image_path):
                raise FileNotFoundError(f"Image for inference not found: {self.args.image_path}")
            
        else:
            if not os.path.exists(self.args.test_data_path):
                raise FileNotFoundError(f"Test data path not found: {self.args.test_data_path}")
        
        if not os.path.exists(self.args.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint path not found: {self.args.checkpoint_path}")
    
    def print_config(self):
        """Print configuration summary"""
        print("\n" + "="*50)
        print("TESTING CONFIGURATION")
        print("="*50)
        print(f"Model: {self.args.model_name}")
        print(f"Checkpoint Path: {self.args.checkpoint_path}")
        print(f"Save Path: {self.args.save_path}")
        print(f"Image Size: {self.args.image_size}")
        print(f"Epochs to Test: {self.args.epoch}")
        if self.args.inference_mode:
            print(f"Mode: Single Image Inference")
            print(f"Image Path: {self.args.image_path}")
        else:
            print(f"Testing Mode: {self.args.mode}")
            print(f"Dataset: {self.args.dataset}")
            print(f"Test Data Path: {self.args.test_data_path}")
            print(f"Metrics: {self.args.metrics}")
        
        print("="*50 + "\n")


def main():
    """Main testing function"""
    # Parse arguments
    parser = create_test_args_parser()
    args = parser.parse_args()
    
    # Create and validate configuration
    config = TestConfig(args)
    config.print_config()
    
    # Setup logging
    logger = get_logger(args.save_path)
    logger.info("Starting anomaly detection testing")
    logger.info(f"Configuration: {vars(args)}")
    
    try:
        # Initialize tester
        tester = Tester(args)
        
        # Run testing
        results = tester.run_test()
        
        logger.info("Testing completed successfully")
        logger.info("\n%s", results)

    except Exception as e:
        logger.error(f"Testing failed with error: {str(e)}")
        raise
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()