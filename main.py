import os
import argparse

from tqdm import tqdm
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

from utils import create_yolo_config
from networks import ObjectDetectionTrainer, GeoRasterInference
from preprocessing import create_bounding_boxes, GeoRasterPatchExtractor
from dataloader import GeoObjectDetectionDataset, get_transforms, collate_fn


def from_points_to_bboxes(input_shapefile, output_shapefile, box_width=0.00003, box_height=0.00003):
    return create_bounding_boxes(input_shapefile, output_shapefile, box_width, box_height)


# root@c32c4de8fca4:/mnt/DADOS_GRENOBLE_1/keiller/sonneratia_detection# python main.py --operation patch_generation --input_raster_path ../datasets/manguezal/OS_006129_AMBIPAR_ORTO_2.01cm_09.2025.tif --bbox_shapefile_path ../datasets/manguezal/100m_anotacao_2/bbs_shapefile/bounding_boxes.shp --output_patches_path ../datasets/manguezal/patches/
def main():
    parser = argparse.ArgumentParser(description='main')

    # general options
    parser.add_argument('--operation', type=str, required=True,
                        help='Operation', choices=['train', 'val', 'test', 'bbox_generation', 'patch_generation'])
    parser.add_argument('--output_path', type=str, required=False,
                        help='Path to save outcomes (like trained models)')

    # dataset options
    parser.add_argument('--point_shapefile_path', type=str, required=False, help='Path to shapefile with point annotations')
    parser.add_argument('--bbox_shapefile_path', type=str, required=False, help='Path to shapefile with bounding box annotations')
    parser.add_argument('--input_raster_path', type=str, required=False, help='Path to input raster file')
    parser.add_argument('--output_shapefile_path', type=str, required=False, help='Path to output shapefile for detections')
    parser.add_argument('--output_patches_path', type=str, required=False, help='Path to output patches for detections')

    # model options
    parser.add_argument('--model', type=str, required=False, help='Model', default='faster',
                        choices=['faster', 'yolo'])
    parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch_num', type=int, default=100, help='Number of epochs')
    parser.add_argument('--model_path', type=str, required=False, default=None,
                        help='Path to a trained model to be used during the inference')
    args = parser.parse_args()
    print(args)
    
    if args.operation == 'bbox_generation':
        bbox_gdf = from_points_to_bboxes(args.point_shapefile_path, args.output_shapefile_path)
    elif args.operation == 'patch_generation':
        extractor = GeoRasterPatchExtractor(
            raster_path=args.input_raster_path,
            bbox_shapefile=args.bbox_shapefile_path,
            output_dir=args.output_patches_path,
            patch_size=512,
            overlap=0.3,
            negative_samples=False
        )
        annotations, classes = extractor.extract_patches()
        train_ann, val_ann = extractor.split_train_val(annotations, classes, val_split=0.2)
    elif args.operation == 'train':
        # Create datasets and dataloaders
        train_dataset = GeoObjectDetectionDataset(
            annotations_file=os.path.join(args.output_patches_path, 'train_annotations.json'),
            images_dir=os.path.join(args.output_patches_path, 'images'),
            transforms=get_transforms(train=True)
        )
        val_dataset = GeoObjectDetectionDataset(
            annotations_file=os.path.join(args.output_patches_path, 'val_annotations.json'),
            images_dir=os.path.join(args.output_patches_path, 'images'),
            transforms=get_transforms(train=False)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
        )
        
        # Train model
        if args.model == 'faster':
            trainer = ObjectDetectionTrainer(num_classes=1, output_path=args.output_path,
                                            learning_rate=args.learning_rate, 
                                            weight_decay=args.weight_decay)
            trainer.train(train_loader, val_loader, num_epochs=args.epoch_num)
        else:
            create_yolo_config(args.output_patches_path)

            # Train YOLO
            model = YOLO('yolov8n.pt')  # Load pretrained model
            results = model.train(
                data='dataset/data.yaml',
                epochs=args.epoch_num,
                imgsz=512,
                batch=args.batch_size,
                name='geo_detection',
                patience=10
            )
    elif args.operation == 'val':
        val_dataset = GeoObjectDetectionDataset(
            annotations_file=os.path.join(args.output_patches_path, 'val_annotations.json'),
            images_dir=os.path.join(args.output_patches_path, 'images'),
            transforms=get_transforms(train=False)
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn
        )
        
        # Train model
        if args.model == 'faster':
            trainer = ObjectDetectionTrainer(num_classes=1, output_path=args.output_path,
                                            learning_rate=args.learning_rate, 
                                            weight_decay=args.weight_decay, trained_model=args.model_path)
            metrics = trainer.evaluate(val_loader, plot_samples=True)
            print(f"Validation metrics: {metrics}")
        else:
            create_yolo_config(args.output_patches_path)

            # Train YOLO
            model = YOLO('yolov8n.pt')  # Load pretrained model
            results = model.train(
                data='dataset/data.yaml',
                epochs=args.epoch_num,
                imgsz=512,
                batch=args.batch_size,
                name='geo_detection',
                patience=10
            )
    elif args.operation == 'test':
        # Run inference
        inference = GeoRasterInference(
            model_path=args.model_path,
            raster_path=args.input_raster_path,
            output_shapefile='output/inference_shp/detections.shp',
            patch_size=512,
            confidence_threshold=0.3
        )

        detections_gdf = inference.run_inference()
    else:
        raise NotImplementedError(f"Operation {args.operation} not implemented.")


if __name__ == "__main__":
    main()
