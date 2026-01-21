import torch
import torch.nn.functional as F
import numpy as np
import random
import os
import json
import time
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import open_clip

from dataset import Dataset
from loss import FocalLoss, BinaryDiceLoss, DiceLoss
from learnable_prompt import LearnablePrompt
from visual_prompt_tuning import VisualPromptTuning
from tqdm import tqdm
from adapter import LinearAdapter

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_similarity(image_features, text_features):
    """Compute similarity between image and text features"""
    token_similarities = image_features @ text_features.T
    similarity = token_similarities / 0.07
    return similarity


def compute_similarity_map(sim, img_size):
    """Compute similarity map for visualization"""
    patch_dim = int(sim.shape[1] ** 0.5)
    sim = sim.reshape(sim.shape[0], patch_dim, patch_dim, -1).permute(0, 3, 1, 2)
    sim = F.interpolate(sim, img_size, mode="bilinear", align_corners=True)
    sim = sim.permute(0, 2, 3, 1)
    return sim


def compute_pairwise_similarity(embeddings):
    """Compute pairwise similarity matrix"""
    similarity_matrix = embeddings @ embeddings.T
    return similarity_matrix


def compute_loss(abnormal_prompt_embedding):
    """Compute constraint loss for prompt diversity"""
    similarity_matrix = compute_pairwise_similarity(abnormal_prompt_embedding)
    mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device)
    similarity_matrix = similarity_matrix.masked_fill(mask == 1, 0)
    loss = torch.sum(torch.triu(similarity_matrix, diagonal=1))
    return loss


class Trainer:
    def __init__(self, args, model_components, data_loaders, losses, checkpoint_dir, 
                 checkpoint_interval=1):
        """
        Initialize the trainer
        
        Args:
            args: Training arguments
            model_components: Dict containing model, prompt_learner, visual_prompt, trainable_adapter
            data_loaders: Dict containing train_loader and optional val_loader
            losses: Dict containing focal_loss and dice_loss
            checkpoint_dir: Directory to save checkpoints
            checkpoint_interval: Save checkpoint every N epochs
        """
        self.args = args
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Model components
        self.model = model_components['model']
        self.prompt_learner = model_components['prompt_learner']
        self.visual_prompt = model_components['visual_prompt']
        self.trainable_adapter = model_components['trainable_adapter']
        
        # Data loaders
        self.train_loader = data_loaders['train_loader']
        
        # Losses
        self.focal_loss = losses['focal_loss']
        self.dice_loss = losses['dice_loss']
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.prompt_learner.parameters()) + 
            list(self.trainable_adapter.parameters()) + 
            list(self.visual_prompt.parameters()),
            lr=args.learning_rate, 
            betas=(0.5, 0.999)
        )
        
        # Training state
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Training hyperparameters
        self.weight_factor = 4
        self.weight_factor_2 = 1
        
        
        self.gt_defect = {
            "good": 0, "bent": 1, "bent_lead": 1, "bent_wire": 1, "manipulated_front": 1,
            "broken": 2, "broken_large": 2, "broken_small": 2, "broken_teeth": 2,
            "color": 3, "combined": 4, "contamination": 5, "metal_contamination": 5,
            "crack": 6, "cut": 7, "cut_inner_insulation": 7, "cut_lead": 7,
            "cut_outer_insulation": 7, "fabric": 8, "fabric_border": 8,
            "fabric_interior": 8, "faulty_imprint": 9, "print": 9, "glue": 10,
            "glue_strip": 10, "hole": 11, "missing": 12, "missing_wire": 12,
            "missing_cable": 12, "poke": 13, "poke_insulation": 13, "rough": 14,
            "scratch": 15, "scratch_head": 15, "scratch_neck": 15, "squeeze": 16,
            "squeezed_teeth": 16, "thread": 17, "thread_side": 17, "thread_top": 17,
            "liquid": 18, "oil": 18, "misplaced": 19, "cable_swap": 19, "flip": 19,
            "fold": 19, "split_teeth": 19, "damaged_case": 20, "defective": 20,
            "gray_stroke": 20, "pill_type": 20
        }

    def save_checkpoint(self, additional_info=None):
        """Save model and training state checkpoint"""
        checkpoint_name = f'epoch_{self.epoch}.pth'            
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint_data = {
            'epoch': self.epoch,
            'prompt_learner': self.prompt_learner.state_dict(),
            'visual_prompt': self.visual_prompt.state_dict(),
            'linear_adapter': self.trainable_adapter.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
            
        torch.save(checkpoint_data, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

    def load_checkpoint(self, checkpoint_path=None):
        """Load model and training state from checkpoint"""
        if checkpoint_path is None:
            checkpoints = [f for f in os.listdir(self.checkpoint_dir) 
                          if f.startswith('checkpoint_') and f.endswith('.pt')]
            if not checkpoints:
                print("No checkpoints found, starting from scratch")
                return
            checkpoint_path = os.path.join(self.checkpoint_dir, sorted(checkpoints)[-1])
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint {checkpoint_path} not found, starting from scratch")
            return
            
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        self.prompt_learner.load_state_dict(checkpoint['prompt_learner'])
        self.visual_prompt.load_state_dict(checkpoint['visual_prompt'])
        self.trainable_adapter.load_state_dict(checkpoint['linear_adapter'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.epoch = checkpoint['epoch']
        
        print(f"Resumed training from epoch {self.epoch}, best loss: {self.best_loss:.4f}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.eval()  # Keep CLIP backbone frozen
        self.prompt_learner.train()
        self.trainable_adapter.train()
        self.visual_prompt.train()
        
        loss_list = []
        image_loss_list = []
        
        torch.autograd.set_detect_anomaly(True)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1}/{self.args.epoch}')
        
        for batch_idx, items in enumerate(pbar):
            # Move data to device
            image = items['img'].to(self.device)
            label = items['anomaly']
            defect_type = items['defect_type']
            gt = items['mask'].to(self.device).to(torch.long)
            
            # Process class IDs
            cls_id = []
            for defect in defect_type:
                cls_id.append(int(self.gt_defect[defect]))
            cls_id = torch.tensor(cls_id, dtype=torch.long).to(self.device)
            
            # Process ground truth masks
            gt_b = gt.clone().to(torch.long)
            
            for i in range(gt.size(0)):
                gt[i][gt[i] > 0.5] = cls_id[i]
                gt[i][gt[i] <= 0.5] = 0
                
                gt_b[i][gt_b[i] > 0.5] = 1
                gt_b[i][gt_b[i] <= 0.5] = 0
            
            # Forward pass
            with torch.amp.autocast(enabled=False, device_type=self.device):
                # Visual prompt and image encoding
                visual_tokens = self.visual_prompt()
                image_features, patch_features, _ = self.model.encode_image([image, visual_tokens], self.args.feature_layers)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                patch_features = self.trainable_adapter(patch_features)
                
                # Text prompt learning
                prompt, learnable_tokens, template_tokens = self.prompt_learner(image_features)
                
                # Text encoding
                text_features, learned_tokens = self.model.encode_learn_prompts(
                    prompt, learnable_tokens, template_tokens
                )
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Separate normal and abnormal features
                normal_text_feature = text_features[:1]
                abnormal_text_feature = text_features[1:]
                
                avg_abnormal_text_feature = abnormal_text_feature.mean(dim=0, keepdim=True)
                avg_abnormal_text_feature = avg_abnormal_text_feature / avg_abnormal_text_feature.norm(dim=-1, keepdim=True)
                
                normal_abnormal_feature = torch.cat([normal_text_feature, avg_abnormal_text_feature], dim=0)
                probabilities = ((image_features @ normal_abnormal_feature.T) / 0.07)
                
                # Process patch features for local predictions
                similarity_map_list = []
                patch_similarity_list = []
                
                for idx, patch_feature in enumerate(patch_features):
                    patch_feature = patch_feature / patch_feature.norm(dim=-1, keepdim=True)
                    similarity = compute_similarity(patch_feature, text_features).softmax(dim=-1)
                    binary_similarity = compute_similarity(patch_feature, normal_abnormal_feature)
                    similarity_map = compute_similarity_map(similarity, self.args.image_size).permute(0, 3, 1, 2)
                    similarity_map_list.append(similarity_map)
                    patch_similarity_list.append(binary_similarity)
            
            # Compute losses
            # Global image-level loss
            image_loss = F.cross_entropy(probabilities, label.long().to(self.device))
            image_loss = self.weight_factor_2 * image_loss
            image_loss_list.append(image_loss.cpu().item())
            
            # Local pixel-level loss
            local_loss = 0
            for i in range(len(similarity_map_list)):
                # Focal loss between multiple defect classes
                local_loss += self.focal_loss(similarity_map_list[i], gt)
                
                # Dice loss for normal score
                Mn = similarity_map_list[i][:, 0, :, :].unsqueeze(1)
                local_loss += self.dice_loss(Mn, 1 - gt_b)
                
                # Dice loss for anomaly score
                Ma = torch.sum(similarity_map_list[i][:, 1:, :, :], dim=1).unsqueeze(1)
                local_loss += self.dice_loss(Ma, gt_b)
            
            local_loss = self.weight_factor * local_loss
            total_loss = local_loss + image_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            loss_list.append(local_loss.cpu().item())
            
            # Update progress bar
            pbar.set_postfix({
                'Local Loss': f'{local_loss.item():.4f}',
                'Global Loss': f'{image_loss.item():.4f}',
                'Total Loss': f'{total_loss.item():.4f}'
            })
        
        avg_loss = np.mean(loss_list)
        avg_image_loss = np.mean(image_loss_list)
        
        print(f'Epoch [{self.epoch + 1}/{self.args.epoch}], '
              f'Local Loss: {avg_loss:.4f}, Global Loss: {avg_image_loss:.4f}')
        
        return avg_loss, avg_image_loss

    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.args.epoch} epochs")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.learning_rate}")
        
        for epoch in range(self.epoch, self.args.epoch):
            self.epoch = epoch
            
            # Train one epoch
            train_loss, train_image_loss = self.train_epoch()
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0:
                additional_info = {
                    'train_loss': train_loss,
                    'train_image_loss': train_image_loss
                }
                self.save_checkpoint(additional_info=additional_info)
        
        print("Training completed!")
        
        # Save final checkpoint
        self.save_checkpoint(is_best=False, additional_info={'final_epoch': True})


def create_model_components(args):
    """Create and initialize model components"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create CLIP model
    model, _, pre_process = open_clip.create_model_and_transforms(
        args.model_name, 
        pretrained=args.pretrained, 
        force_image_size=args.image_size
    )
    
    # Modify preprocessing
    pre_process.transforms[0] = transforms.Resize(
        size=(args.image_size, args.image_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        max_size=None, antialias=None
    )
    pre_process.transforms[1] = transforms.CenterCrop(size=(args.image_size, args.image_size))
    
    model.eval()
    model = model.to(device)
    
    tokenizer = open_clip.get_tokenizer(args.model_name)
    
    # Create prompt state for MVTec

    if args.dataset == "mvtec":
        prompt_state_all = [
            "bent", "broken", "discoloration", "compound", "contamination",
            "crack", "cut", "fabric", "faulty imprint", "gluing", "hole",
            "missing", "puncture", "rough", "scratch", "compressed",
            "threading", "liquid", "misaligned", "wear"
        ]
    
    # Initialize prompt learner
    prompt_learner = LearnablePrompt(
        clip_model=model,
        prompt_state=prompt_state_all,
        normal_token_count=args.normal_token_cnt,
        prompt_count=args.prompt_count,
        abnormal_token_count=args.abnormal_token_cnt,
        token_depth=args.text_depth,
        learnable_token_length=args.layer_token_cnt,
        tokenizer=tokenizer,
        device=device
    )

    # Initialize visual prompt tuning
    visual_prompt = VisualPromptTuning(
        model=model,
        total_d_layer=args.depth,
        num_tokens=args.prefix_token_cnt,
        device=device
    )

    # Initialize linear adapter
    with open(os.path.join(args.model_configs, f"{args.model_name}.json"), "r") as f:
        model_config = json.load(f)
    trainable_adapter = LinearAdapter(dim_in=model_config["vision_cfg"]["width"], dim_out=model_config["embed_dim"], k=len(args.feature_layers)).to(device)
    
    # Return components dictionary
    return {
        'model': model,
        'prompt_learner': prompt_learner,
        'visual_prompt': visual_prompt,
        'trainable_adapter': trainable_adapter,
        'preprocessor': pre_process,
        'tokenizer': tokenizer
    }


def create_data_loaders(args, preprocessor):
    """Create data loaders"""
    target_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    
    train_data = Dataset(
        root=args.train_data_path,
        transform=preprocessor,
        target_transform=target_transform,
        dataset_name='mvtec',
        mode="test",
        k_shot=0
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    
    return {
        'train_loader': train_loader
    }


def main():
    # Parse arguments
    project_root = os.path.dirname(__file__)
    dataset_root = os.path.join(project_root, 'data')
    parser = argparse.ArgumentParser("Defect Aware Prompt Learning")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336-quickgelu")
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--dataset", type=str, default="mvtec")
    parser.add_argument("--train_data_path", type=str, default=f"{dataset_root}/mvtec")
    parser.add_argument("--model_configs", type=str, default= os.path.join(project_root, "open_clip", "model_configs"))
    
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
    
    
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint/")
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create model components
    model_components = create_model_components(args)
    
    # Create data loaders
    data_loaders = create_data_loaders(args, model_components['preprocessor'])
    
    # Create loss functions
    losses = {
        'focal_loss': FocalLoss(),
        'dice_loss': BinaryDiceLoss()
    }
    
    # Initialize trainer
    trainer = Trainer(
        args=args,
        model_components=model_components,
        data_loaders=data_loaders,
        losses=losses,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()