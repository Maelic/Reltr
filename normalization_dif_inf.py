# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as T
import time, glob
import onnxruntime
import numpy as np
import torch
from scipy.special import softmax

transform = T.Compose([
    T.Resize([800,1200]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# for output bounding box post-processing
def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = np.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

# VG classes
CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
            'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
            'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
            'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
            'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
            'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
            'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
            'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
            'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
            'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
            'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
            'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
            'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
            'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
            'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
            'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
            'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
            'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
            'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--dataset', default='vg')

    # image path
    parser.add_argument('--img_path', type=str, default='demo/vg1.jpg',
                        help="Path of the test image")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_entities', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--num_triplets', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='ckpt/checkpoint0149.pth', help='resume from checkpoint')
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_iou_threshold', default=0.7, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")


    # distributed training parameters
    parser.add_argument('--return_interm_layers', action='store_true',
                        help="Return the fpn if there is the tag")
    return parser

def normalize(img_pil):
    # MEAN = 255 * np.array([0.485, 0.456, 0.406])
    # STD = 255 * np.array([0.229, 0.224, 0.225])
    # x = np.array(img_pil)
    # x = x.transpose(-1, 0, 1)
    # x = (x - MEAN[:, None, None]) / STD[:, None, None]
    np_image = np.array(img_pil)
    np_image = np_image.transpose(2, 0, 1) # CxHxW
    mean_vec = np.array([0.485, 0.456, 0.406])
    std_vec = np.array([0.229, 0.224, 0.225])
    norm_img_data = np.zeros(np_image.shape).astype('float32')
    for i in range(np_image.shape[0]):
        norm_img_data[i,:,:] = (np_image[i,:,:]/255 - mean_vec[i])/std_vec[i]
             
    np_image = np.expand_dims(norm_img_data, axis=0) # 1xCxHxW

    return np_image

def normalization_dif(img_torch, img_numpy):
    print('\nptImage.shape = ' + str(img_torch.shape))
    print('ptImage max = ' + str(torch.max(img_torch)))
    print('ptImage min = ' + str(torch.min(img_torch)))
    print('ptImage avg = ' + str(torch.mean(img_torch)))

    print('\nnumpyImage.shape = ' + str(img_numpy.shape))
    print('numpyImage max = ' + str(np.max(img_numpy)))
    print('numpyImage min = ' + str(np.min(img_numpy)))
    print('numpyImage avg = ' + str(np.mean(img_numpy)))

    print("average dif: " + str(np.mean(np.abs(img_torch.numpy() - img_numpy))))

def main(args):

    img_path = args.img_path
    im = Image.open(img_path)
    newsize = (640, 480)
    img2 = im.resize(newsize)
    img2 = normalize(img2)

    # mean-std normalize the input image (batch-size: 1)
    sess_options = onnxruntime.SessionOptions()

    sess_options.intra_op_num_threads = 8
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession("normal_resolution.onnx", sess_options)

    img = transform(im).unsqueeze_(0)
    #normalization_dif(img, img2)
    img = img.cpu().detach().numpy()
    # compute ONNX Runtime output prediction
    start = time.time()
    ort_inputs = {ort_session.get_inputs()[0].name: img2}
    ort_outs = ort_session.run(None, ort_inputs)
    print("Total time: ", time.time() - start)
    outputs = {}

    outputs['pred_logits'] = ort_outs[0]
    outputs['pred_boxes'] = ort_outs[1]
    outputs['sub_logits'] = ort_outs[2]
    outputs['sub_boxes'] = ort_outs[3]
    outputs['obj_logits'] = ort_outs[4]
    outputs['obj_boxes'] = ort_outs[5]
    outputs['rel_logits'] = ort_outs[6]

    # keep only predictions with 0.+ confidence
    probas = softmax(outputs['rel_logits'], axis=-1)[0, :, :-1]
    probas_sub = softmax(outputs['sub_logits'], axis=-1)[0, :, :-1]
    probas_obj = softmax(outputs['obj_logits'], axis=-1)[0, :, :-1]
    keep = np.logical_and(np.amax(probas, axis=-1) > 0.3, np.logical_and(np.amax(probas_sub, axis=-1) > 0.3,
                                                                            np.amax(probas_obj, axis=-1) > 0.3))

    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = 10
    keep_queries = np.nonzero(keep)[0]
    indices = np.argsort(-np.amax(probas[keep_queries], axis=-1) * np.amax(probas_sub[keep_queries], axis=-1) * np.amax(probas_obj[keep_queries], axis=-1))[:topk]
    keep_queries = keep_queries[indices]
    print("Detected relationships:")
    for idx in keep_queries:
        print("\t{} {} {}".format(CLASSES[np.argmax(probas_sub[idx])],
                                  REL_CLASSES[np.argmax(probas[idx])],
                                  CLASSES[np.argmax(probas_obj[idx])]))
    
    fig, axs = plt.subplots(ncols=len(indices), nrows=1, figsize=(22, 7))
    for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
            zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
        ax = ax_i
        ax.imshow(im)
        ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                    fill=False, color='blue', linewidth=2.5))
        ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                    fill=False, color='orange', linewidth=2.5))

        ax.axis('off')
        ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

    fig.tight_layout()
    plt.show()

def detect():

    # mean-std normalize the input image (batch-size: 1)
    sess_options = onnxruntime.SessionOptions()
    #sess_options.enable_profiling = True

    sess_options.intra_op_num_threads = 2
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    ort_session = onnxruntime.InferenceSession("normal_resolution-sim-quant.onnx", sess_options)
    total_time = 0
    nb_images = 0
    init = time.time()
    for img_path in glob.glob("demo/*.jpg"):
        im = Image.open(img_path)
        newsize = (640, 480)
        img2 = im.resize(newsize)
        img2 = normalize(img2)

        img = transform(im).unsqueeze_(0)
        #normalization_dif(img, img2)
        img = img.cpu().detach().numpy()
        # compute ONNX Runtime output prediction
        start = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: img2}
        ort_outs = ort_session.run(None, ort_inputs)
        total_time += time.time() - start
        nb_images += 1

    print("Number of images: ", nb_images)
    print("Total time: ", time.time() - init)
    print("Average time: ", total_time / nb_images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
#    main(args)
    detect()
