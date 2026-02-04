import cv2
import json
import rasterio

import numpy as np
from tqdm import tqdm
import geopandas as gpd

from pathlib import Path
from shapely.geometry import box, Point
from rasterio.windows import Window
from rasterio.mask import mask

from sklearn.model_selection import train_test_split


def create_bounding_boxes(input_shp, output_shp, box_width, box_height):
    """
    Create bounding boxes around each point in a shapefile
    
    Parameters:
    -----------
    input_shp : str
        Path to input point shapefile
    output_shp : str
        Path to output polygon shapefile
    box_width : float
        Width of bounding box (in same units as shapefile CRS)
    box_height : float
        Height of bounding box (in same units as shapefile CRS)
    """
    # Read input shapefile
    points_gdf = gpd.read_file(input_shp)
    
    print(f"Read {len(points_gdf)} points from {input_shp}")
    print(f"CRS: {points_gdf.crs}")
    
    # Create list to store bounding boxes
    bboxes = []
    
    # For each point, create a bounding box
    for idx, row in points_gdf.iterrows():
        point = row.geometry
        
        # Get point coordinates
        x, y = point.x, point.y
        
        # Calculate bounding box corners
        # Point is at center
        minx = x - box_width / 2
        maxx = x + box_width / 2
        miny = y - box_height / 2
        maxy = y + box_height / 2
        
        # Create bounding box polygon
        bbox = box(minx, miny, maxx, maxy)
        bboxes.append(bbox)
    
    # Create new GeoDataFrame with bounding boxes
    # Copy all attributes from original points
    bbox_gdf = points_gdf.copy()
    bbox_gdf['geometry'] = bboxes
    
    # Save to shapefile
    bbox_gdf.to_file(output_shp)
    
    print(f"✓ Created {len(bbox_gdf)} bounding boxes")
    print(f"✓ Saved to {output_shp}")
    
    return bbox_gdf



def convert_to_yolo_format(annotations, output_dir):
    """Convert annotations to YOLO format"""
    output_dir = Path(output_dir)
    labels_dir = output_dir / 'yolo_labels'
    labels_dir.mkdir(parents=True, exist_ok=True)

    for ann in annotations:
        label_file = labels_dir / f"{Path(ann['filename']).stem}.txt"

        with open(label_file, 'w') as f:
            for bbox_data in ann['bboxes']:
                bbox = bbox_data['bbox']  # [x1, y1, x2, y2]
                class_id = bbox_data['class']
                
                # Convert to YOLO format (class x_center y_center width height) normalized
                x_center = ((bbox[0] + bbox[2]) / 2) / ann['width']
                y_center = ((bbox[1] + bbox[3]) / 2) / ann['height']
                width = (bbox[2] - bbox[0]) / ann['width']
                height = (bbox[3] - bbox[1]) / ann['height']
                
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")


class GeoRasterPatchExtractor:
    """
    Extract patches from georeferenced raster using bounding box shapefile
    """
    
    def __init__(self, raster_path, bbox_shapefile, output_dir, 
                 patch_size=512, overlap=0.2, negative_samples=True):
        """
        Parameters:
        -----------
        raster_path : str
            Path to georeferenced raster image
        bbox_shapefile : str
            Path to bounding box shapefile
        output_dir : str
            Directory to save extracted patches
        patch_size : int
            Size of extracted patches (pixels)
        overlap : float
            Overlap between patches (0-1)
        negative_samples : bool
            Whether to extract negative samples (patches without objects)
        """
        self.raster_path = raster_path
        self.bbox_shapefile = bbox_shapefile
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.overlap = overlap
        self.negative_samples = negative_samples
        
        # Create output directories
        self.images_dir = self.output_dir / 'images'
        self.labels_dir = self.output_dir / 'labels'
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.bboxes_gdf = gpd.read_file(bbox_shapefile)
        self.raster = rasterio.open(raster_path)
        
        print(f"Raster shape: {self.raster.shape}")
        print(f"Raster CRS: {self.raster.crs}")
        print(f"Number of bounding boxes: {len(self.bboxes_gdf)}")
        
        # Ensure CRS match
        if self.bboxes_gdf.crs != self.raster.crs:
            print(f"Reprojecting bboxes from {self.bboxes_gdf.crs} to {self.raster.crs}")
            self.bboxes_gdf = self.bboxes_gdf.to_crs(self.raster.crs)
    
    def geo_to_pixel(self, x, y):
        """Convert geographic coordinates to pixel coordinates"""
        row, col = rasterio.transform.rowcol(self.raster.transform, x, y)
        return col, row
    
    def pixel_to_geo(self, col, row):
        """Convert pixel coordinates to geographic coordinates"""
        x, y = rasterio.transform.xy(self.raster.transform, row, col)
        return x, y
    
    def extract_patch_with_window(self, col_off, row_off, width, height):
        """Extract a patch from raster using window"""
        window = Window(col_off, row_off, width, height)
        
        # Read patch
        patch = self.raster.read(window=window)
        
        # Convert from (C, H, W) to (H, W, C)
        if patch.shape[0] < patch.shape[1]:  # Channels first
            patch = np.transpose(patch, (1, 2, 0))
        
        return patch
    
    def get_bboxes_in_patch(self, col_off, row_off, width, height):
        """Get bounding boxes that intersect with the patch"""
        # Get patch bounds in geo coordinates
        x_min, y_max = self.pixel_to_geo(col_off, row_off)
        x_max, y_min = self.pixel_to_geo(col_off + width, row_off + height)
        
        patch_bounds = box(x_min, y_min, x_max, y_max)
        
        # Find intersecting bboxes
        intersecting = self.bboxes_gdf[self.bboxes_gdf.intersects(patch_bounds)]
        
        # Convert to pixel coordinates relative to patch
        bboxes_pixel = []
        for idx, row in intersecting.iterrows():
            bbox_geo = row.geometry
            
            # Get bbox corners in geo coordinates
            bounds = bbox_geo.bounds  # (minx, miny, maxx, maxy)
            
            # Convert to pixel coordinates
            x1_px, y1_px = self.geo_to_pixel(bounds[0], bounds[3])  # top-left
            x2_px, y2_px = self.geo_to_pixel(bounds[2], bounds[1])  # bottom-right
            
            # Make relative to patch
            x1_rel = x1_px - col_off
            y1_rel = y1_px - row_off
            x2_rel = x2_px - col_off
            y2_rel = y2_px - row_off
            
            # Clip to patch bounds
            x1_rel = max(0, min(x1_rel, width))
            y1_rel = max(0, min(y1_rel, height))
            x2_rel = max(0, min(x2_rel, width))
            y2_rel = max(0, min(y2_rel, height))
            
            # Only keep if bbox has valid area
            if (x2_rel > x1_rel and y2_rel > y1_rel) and \
                (((x2_rel - x1_rel) * (y2_rel - y1_rel)) / (self.patch_size * self.patch_size) > 0.03):
                bboxes_pixel.append({
                    'bbox': [int(x1_rel), int(y1_rel), int(x2_rel), int(y2_rel)],
                    'class': 1  #,  # Single class for now
                    # 'attributes': row.to_dict()
                })
        
        return bboxes_pixel
    
    def extract_patches(self):
        """Extract all patches with bounding boxes"""
        patches_data = []
        classes = []
        patch_id = 0
        
        # Calculate grid
        raster_height, raster_width = self.raster.shape[0], self.raster.shape[1]
        stride = int(self.patch_size * (1 - self.overlap))
        
        print(f"Extracting patches with size={self.patch_size}, stride={stride}")
        print(f"Raster size: {raster_width} x {raster_height}")
        
        # Iterate over raster in sliding window fashion
        for row_off in tqdm(range(0, raster_height - self.patch_size + 1, stride), 
                           desc="Extracting patches"):
            for col_off in range(0, raster_width - self.patch_size + 1, stride):
                
                # Extract patch
                patch = self.extract_patch_with_window(
                    col_off, row_off, self.patch_size, self.patch_size
                )
                binc = np.bincount(patch.flatten())
                if binc[0] + binc[1] > binc[1:-1].sum() * 10:
                    continue
                
                # Get bboxes in this patch
                bboxes = self.get_bboxes_in_patch(
                    col_off, row_off, self.patch_size, self.patch_size
                )
                
                # Skip patches without objects if negative_samples is False
                if not self.negative_samples and len(bboxes) == 0:
                    continue
                
                # Save patch image
                patch_filename = f'patch_{patch_id:06d}.png'
                patch_path = self.images_dir / patch_filename
                
                # Normalize and save
                if patch.dtype == np.uint16:
                    patch = (patch / 256).astype(np.uint8)
                
                cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                
                # Save annotations in COCO format
                annotation = {
                    'image_id': patch_id,
                    'filename': patch_filename,
                    'width': self.patch_size,
                    'height': self.patch_size,
                    'bboxes': bboxes
                }
                
                classes.append(1 if len(bboxes) > 0 else 0)
                patches_data.append(annotation)
                patch_id += 1
        
        print(f"\n✓ Extracted {patch_id} patches")
        print(f"  Patches with objects: {sum(1 for p in patches_data if len(p['bboxes']) > 0)}")
        print(f"  Patches without objects: {sum(1 for p in patches_data if len(p['bboxes']) == 0)}")
        
        # Save annotations
        annotations_path = self.output_dir / 'annotations.json'
        with open(annotations_path, 'w') as f:
            json.dump(patches_data, f, indent=2)
        
        print(f"✓ Saved annotations to {annotations_path}")
        
        return patches_data, classes
    
    def split_train_val(self, annotations, classes, val_split=0.2, random_state=42):
        """Split patches into train and validation sets"""
        train_ann, val_ann = train_test_split(
            annotations, test_size=val_split, random_state=random_state, stratify=classes
        )
        
        # Save splits
        train_path = self.output_dir / 'train_annotations.json'
        val_path = self.output_dir / 'val_annotations.json'
        
        convert_to_yolo_format(train_ann, self.output_dir)
        convert_to_yolo_format(val_ann, self.output_dir)
        
        with open(train_path, 'w') as f:
            json.dump(train_ann, f, indent=2)
        
        with open(val_path, 'w') as f:
            json.dump(val_ann, f, indent=2)
        
        print(f"\nData split:")
        print(f"  Train: {len(train_ann)} patches")
        print(f"  Val: {len(val_ann)} patches")
        
        return train_ann, val_ann
