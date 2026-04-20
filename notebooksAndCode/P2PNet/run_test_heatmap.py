import argparse
import os
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as standard_transforms
from PIL import Image

from engine import *
from models import build_model

warnings.filterwarnings("ignore")


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation with heatmap', add_help=False)

    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--output_dir', default='outputs',
                        help='path where to save outputs')
    parser.add_argument('--weight_path', default='',
                        help='path where the trained weights are saved')
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='the gpu used for evaluation')
    parser.add_argument('--img_path', default='', type=str,
                        help='path to the input image')
    parser.add_argument('--threshold', default=0.5, type=float,
                        help='confidence threshold for predicted points')
    parser.add_argument('--sigma', default=8, type=float,
                        help='gaussian blur sigma for heatmap')
    return parser


def _get_resample_mode():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)

    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(args)
    model.to(device)

    if args.weight_path:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)

    model.eval()

    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])

    if not args.img_path:
        raise ValueError("Please provide --img_path")

    img_raw = Image.open(args.img_path).convert('RGB')

    width, height = img_raw.size
    new_width = max(128, (width // 128) * 128)
    new_height = max(128, (height // 128) * 128)

    img_raw = img_raw.resize((new_width, new_height), _get_resample_mode())
    img = transform(img_raw)

    samples = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(samples)
        outputs_scores = F.softmax(outputs['pred_logits'], dim=-1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

    keep = outputs_scores > args.threshold
    points = outputs_points[keep].detach().cpu().numpy()
    predict_cnt = len(points)

    img_bgr = cv2.cvtColor(np.array(img_raw), cv2.COLOR_RGB2BGR)

    # 1) Save point visualization
    dot_img = img_bgr.copy()
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < new_width and 0 <= y < new_height:
            cv2.circle(dot_img, (x, y), 2, (0, 0, 255), -1)

    dot_path = os.path.join(args.output_dir, f'pred_{predict_cnt}.jpg')
    cv2.imwrite(dot_path, dot_img)

    # 2) Build heatmap from predicted points
    heat = np.zeros((new_height, new_width), dtype=np.float32)
    for p in points:
        x, y = int(round(p[0])), int(round(p[1]))
        if 0 <= x < new_width and 0 <= y < new_height:
            heat[y, x] += 1.0

    heat = cv2.GaussianBlur(heat, (0, 0), args.sigma)

    if heat.max() > 0:
        heat_norm = heat / heat.max()
    else:
        heat_norm = heat

    heat_uint8 = (heat_norm * 255).astype(np.uint8)

    # 3) Save pure heatmap
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    heatmap_path = os.path.join(args.output_dir, f'heatmap_{predict_cnt}.jpg')
    cv2.imwrite(heatmap_path, heat_color)

    # 4) Save overlay on original image
    overlay = cv2.addWeighted(img_bgr, 0.6, heat_color, 0.4, 0)
    overlay_path = os.path.join(args.output_dir, f'overlay_{predict_cnt}.jpg')
    cv2.imwrite(overlay_path, overlay)

    # 5) Save raw numeric heatmap
    npy_path = os.path.join(args.output_dir, f'heatmap_{predict_cnt}.npy')
    np.save(npy_path, heat)

    print(f"Predicted count: {predict_cnt}")
    print(f"Saved point result to: {dot_path}")
    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved overlay to: {overlay_path}")
    print(f"Saved raw heatmap array to: {npy_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'P2PNet evaluation script with heatmap',
        parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
