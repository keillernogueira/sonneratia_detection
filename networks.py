import os
import rasterio
import numpy as np
import geopandas as gpd

import cv2
from pathlib import Path
from requests import patch
from tqdm import tqdm
from rasterio.windows import Window
from shapely.geometry import box
from coco_eval import CocoEvaluator

import torch
import torchvision
from torchvision.ops import box_iou
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
from utils import plot_bb


class ObjectDetectionTrainer:
    """
    Trainer for object detection on georeferenced patches
    """
    
    def __init__(self, num_classes, output_path, trained_model=None, 
                 learning_rate=0.005, weight_decay=0.0005, device='cuda'):
        self.num_classes = num_classes
        self.output_path = output_path
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load pre-trained model
        self.model = self.get_model()
        # self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        if trained_model is not None:
            self.model.load_state_dict(torch.load(trained_model)['model_state_dict'])
        self.model.to(self.device)
        
        # Optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(
            params, lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=25, gamma=0.1
        )
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(self.output_path, 'runs/geo_detection'))
        
        print(f"Model initialized on {self.device}")
    
    def get_model(self):
        """Get Faster R-CNN model with custom number of classes"""
        # Load pre-trained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes + 1)
        
        return model
    
    def calculate_map(self, predictions, targets, iou_threshold=0.5):
        """Calculate mean Average Precision"""
        
        # Collect all detections and ground truths
        all_pred_boxes = []
        all_pred_scores = []
        all_gt_boxes = []
        
        for pred, target in zip(predictions, targets):
            if len(pred['boxes']) > 0:
                all_pred_boxes.append(pred['boxes'])
                all_pred_scores.append(pred['scores'])
            else:
                all_pred_boxes.append(torch.zeros((0, 4)))
                all_pred_scores.append(torch.zeros(0))
            
            if len(target['boxes']) > 0:
                all_gt_boxes.append(target['boxes'])
            else:
                all_gt_boxes.append(torch.zeros((0, 4)))
        
        # Calculate precision, recall
        if len(all_pred_boxes) == 0 or len(all_gt_boxes) == 0:
            return {
                'mAP': 0.0,
                'precision': 0.0,
                'recall': 0.0
            }
        
        # Simple mAP calculation
        total_tp = 0
        total_fp = 0
        total_gt = sum(len(boxes) for boxes in all_gt_boxes)
        
        for pred_boxes, pred_scores, gt_boxes in zip(all_pred_boxes, all_pred_scores, all_gt_boxes):
            if len(pred_boxes) == 0:
                continue
            
            if len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            
            # Calculate IoU between predictions and ground truth
            ious = box_iou(pred_boxes, gt_boxes)
            
            # For each prediction, find best matching GT
            matched_gt = set()
            for i, (box, score) in enumerate(zip(pred_boxes, pred_scores)):
                if len(gt_boxes) == 0:
                    total_fp += 1
                    continue
                
                max_iou, max_idx = ious[i].max(), ious[i].argmax()
                
                if max_iou >= iou_threshold and max_idx.item() not in matched_gt:
                    total_tp += 1
                    matched_gt.add(max_idx.item())
                else:
                    total_fp += 1
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        
        metrics = {
            'mAP': precision,  # Simplified mAP
            'precision': precision,
            'recall': recall,
            'total_predictions': total_tp + total_fp,
            'total_gt': total_gt
        }
        
        return metrics
    
    def train_one_epoch(self, data_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        epoch_loss = 0
        progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
        
        for images, targets in progress_bar:
            # Move to device
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            # Update progress bar
            epoch_loss += losses.item()
            progress_bar.set_postfix({'loss': losses.item()})
        
        avg_loss = epoch_loss / len(data_loader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, data_loader, plot_samples=False):
        """Evaluate model"""
        self.model.eval()
        
        all_predictions = []
        all_targets = []

        for images, targets in tqdm(data_loader, desc='Evaluating'):
            images = list(image.to(self.device) for image in images)
            outputs = self.model(images, targets)
            print('preds', outputs)
            print('-----------------------')
            print('targets', targets)
            
            for pred, target in zip(outputs, targets):
                all_predictions.append({
                    'boxes': pred['boxes'].cpu(),
                    'scores': pred['scores'].cpu(),
                    'labels': pred['labels'].cpu()
                })
                all_targets.append({
                    'boxes': target['boxes'].cpu(),
                    'labels': target['labels'].cpu()
                })
                if plot_samples:
                    plot_bb(target['img_path'], pred['boxes'].cpu(), bbox_target=target['boxes'].cpu())
        
        # Calculate mAP
        metrics = self.calculate_map(all_predictions, all_targets, iou_threshold=0.5)
        
        return metrics
    
    def train(self, train_loader, val_loader, num_epochs=10):
        """Full training loop"""
        save_dir = os.path.join(self.output_path, 'models')
        os.makedirs(save_dir, exist_ok=True)
        best_map = 0.0
        
        for epoch in range(1, num_epochs + 1):           
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.evaluate(val_loader)
            
            # Update learning rate
            self.lr_scheduler.step()
            
            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('mAP', val_metrics['mAP'], epoch)
            self.writer.add_scalar('Precision', val_metrics['precision'], epoch)
            self.writer.add_scalar('Recall', val_metrics['recall'], epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val mAP: {val_metrics['mAP']:.4f}")
            print(f"Val Precision: {val_metrics['precision']:.4f}")
            print(f"Val Recall: {val_metrics['recall']:.4f}")
            
            # Save best model based on mAP
            if val_metrics['mAP'] > best_map:
                best_map = val_metrics['mAP']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mAP': val_metrics['mAP'],
                }, f'{save_dir}/best_model.pth')
                print(f"✓ Saved best model (mAP: {val_metrics['mAP']:.4f})")
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'mAP': val_metrics['mAP'],
                }, f'{save_dir}/checkpoint_epoch_{epoch}.pth')
        
        self.writer.close()
        print("\n✓ Training completed!")

 
class GeoRasterInference:
    """
    Run inference on full georeferenced raster
    """
    def __init__(self, model_path, raster_path, output_shapefile,
                 patch_size=512, overlap=0.3, confidence_threshold=0.5, device='cuda'):
        self.raster_path = raster_path
        self.output_shapefile = output_shapefile
        self.patch_size = patch_size
        self.stride = int(patch_size * (1 - overlap))
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        self.model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Open raster
        self.raster = rasterio.open(raster_path)
        
        print(f"Model loaded from {model_path}")
        print(f"Raster load from: {raster_path}")
    
    def predict_patch(self, patch):
        """Run prediction on a single patch"""
        # Normalize
        patch = patch.astype(np.float32) / 255.0
        patch = (patch - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        
        # To tensor
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float()
        # print(f"✓ Patch tensor shape: {patch_tensor.shape}")  # Debugging line
        patch_tensor = patch_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(patch_tensor)
        
        return predictions[0]
    
    def apply_nms_geo(self, gdf, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        from shapely.ops import unary_union
        
        # Sort by confidence
        gdf = gdf.sort_values('confidence', ascending=False).reset_index(drop=True)
        
        keep = []
        while len(gdf) > 0:
            # Keep highest confidence detection
            keep.append(gdf.iloc[0])
            
            if len(gdf) == 1:
                break
            
            # Calculate IoU with remaining boxes
            current_box = gdf.iloc[0].geometry
            remaining = gdf.iloc[1:]
            
            # Filter out overlapping boxes
            ious = remaining.geometry.apply(
                lambda x: current_box.intersection(x).area / current_box.union(x).area
            )
            
            gdf = remaining[ious < iou_threshold].reset_index(drop=True)
        
        return gpd.GeoDataFrame(keep, crs=gdf.crs)

    def run_inference(self):
        """Run inference on full raster"""
        detections = []
        
        raster_height, raster_width = self.raster.shape[0], self.raster.shape[1]
        
        for row_off in tqdm(range(0, raster_height - self.patch_size + 1, self.stride), 
                            desc="Processing raster"):
            for col_off in range(0, raster_width - self.patch_size + 1, self.stride):
                
                # Read patch
                window = Window(col_off, row_off, self.patch_size, self.patch_size)
                patch = self.raster.read(window=window)
                patch = np.transpose(patch, (1, 2, 0))
                patch = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)

                binc = np.bincount(patch.flatten())
                if binc[0] + binc[1] > binc[1:-1].sum() * 10:
                    continue
                
                # Predict
                pred = self.predict_patch(patch)
                # print(pred)
                
                # Filter by confidence
                keep = pred['scores'] > self.confidence_threshold
                boxes = pred['boxes'][keep].cpu().numpy()
                scores = pred['scores'][keep].cpu().numpy()
                
                # Convert to geographic coordinates
                for bbox, score in zip(boxes, scores):
                    x1_px, y1_px, x2_px, y2_px = bbox
                    
                    # Add patch offset
                    x1_px += col_off
                    y1_px += row_off
                    x2_px += col_off
                    y2_px += row_off
                    
                    # Convert to geo coordinates
                    x1_geo, y1_geo = rasterio.transform.xy(
                        self.raster.transform, y1_px, x1_px
                    )
                    x2_geo, y2_geo = rasterio.transform.xy(
                        self.raster.transform, y2_px, x2_px
                    )
                    
                    # Create bounding box
                    bbox_geo = box(
                        min(x1_geo, x2_geo), min(y1_geo, y2_geo),
                        max(x1_geo, x2_geo), max(y1_geo, y2_geo)
                    )
                    
                    detections.append({
                        'geometry': bbox_geo,
                        'confidence': float(score)
                    })
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(detections, crs=self.raster.crs)
        
        # Apply Non-Maximum Suppression in geographic space
        gdf = self.apply_nms_geo(gdf, iou_threshold=0.5)
        
        # Save
        gdf.to_file(self.output_shapefile)
        
        print(f"✓ Detected {len(gdf)} objects")
        print(f"✓ Saved to {self.output_shapefile}")
        
        return gdf
    
 