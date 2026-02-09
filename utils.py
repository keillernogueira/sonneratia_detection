import os
import yaml
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches


def create_yolo_config(dataset_path):
    # Create YOLO config
    config = {
        'path': 'dataset',
        'train': 'train/images',
        'val': 'val/images',
        'names': {0: 'object'}
    }
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(config, f)


def plot_bb(image_path, bbox, bbox_target=None):
    print('in plot_bb', image_path, bbox, bbox_target)
    img = Image.open(image_path)
    img_np = np.array(img)

    # Image size
    img_h, img_w, _ = img_np.shape

    for bb in bbox:
        x1, y1, x2, y2 = bb

        bbox_width = x2 - x1
        bbox_height = y2 - y1
        print(f"Bounding box size: {bbox_width:.2f} x {bbox_height:.2f} pixels")
        print(bbox_height * bbox_width, bbox_height * bbox_width / (img_h * img_w))

        # Plot image
        fig, ax = plt.subplots()
        ax.imshow(img_np)

        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    if bbox_target is not None:
        for bbt in bbox_target:
            x1_t, y1_t, x2_t, y2_t = bbt
            rect_target = patches.Rectangle(
                (x1_t, y1_t),
                x2_t - x1_t,
                y2_t - y1_t,
                linewidth=2,
                edgecolor="blue",
                facecolor="none"
            )
            ax.add_patch(rect_target)

    ax.axis("off")

    # plt.show()
    result_img_path = os.path.join("output/val_preds", os.path.basename(image_path).replace('.png', '.jpg'))
    fig.savefig(result_img_path)
    plt.close(fig)


if __name__ == "__main__":
    with open("C:\\Users\\keill\\Desktop\\annotations.json", 'r') as f:
        annotations = json.load(f)
        
    image_path = "C:\\Users\\keill\\Desktop\\patch_000648.png"
    id = int(os.path.basename(image_path).split('.')[0].split('_')[1])
    print(id)
    bb = None
    for ann in annotations:
        if ann['image_id'] == id:
            bb = ann['bboxes'][0]['bbox']
            break  # found

    print(bb)   
    plot_bb(image_path, bb)
