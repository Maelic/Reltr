# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
from PIL import Image

import time, sys
import numpy as np
from scipy.special import softmax
import qi 
import matplotlib.pyplot as plt
import cv2
import argparse
import torch
import torchvision.transforms as T
from models import build_model

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

class SceneGraphGenerator():
    def __init__(self, session, args):
        self.CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
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

        self.REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                    'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                    'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                    'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                    'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                    'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.session = session
        self.initCamerasNaoQi()

        self.model, _, _ = build_model(args)
        ckpt = torch.load(args.resume)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

    def initCamerasNaoQi(self):
        self.video_service = session.service("ALVideoDevice")
        fps = 2
        resolution = 2  	# 2 = Image of 640*480px ; 3 = Image of 1280*960px
        colorSpace = 11  	# RGB
        self.videosClient = self.video_service.subscribeCamera("cameras_pepper", 0, resolution, colorSpace, fps)
            
    def normalize(self, img_pil):
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

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def detect(self):
        

        while(True):
            naoImage = self.video_service.getImageRemote(self.videosClient)
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_bytes = bytes(bytearray(array))

            # Create a PIL Image from our pixel array.
            im = Image.frombytes("RGB", (imageWidth, imageHeight), image_bytes)

            # mean-std normalize the input image (batch-size: 1)
            img = self.transform(im).unsqueeze(0)

            # propagate through the model
            start=time.time()
            outputs = self.model(img)
            print("time inference :", time.time()-start)

             # keep only predictions with 0.+ confidence
            probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
            probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
            probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
            keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                                    probas_obj.max(-1).values > 0.3))

            # convert boxes from [0; 1] to image scales
            sub_bboxes_scaled = self.rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
            obj_bboxes_scaled = self.rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

            topk = 10
            keep_queries = torch.nonzero(keep, as_tuple=True)[0]
            indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
            keep_queries = keep_queries[indices]

            # use lists to store the outputs via up-values
            conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []

            hooks = [
                self.model.backbone[-2].register_forward_hook(
                    lambda self, input, output: conv_features.append(output)
                ),
                self.model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_sub.append(output[1])
                ),
                self.model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
                    lambda self, input, output: dec_attn_weights_obj.append(output[1])
                )
            ]
            with torch.no_grad():
                # propagate through the model
                outputs = self.model(img)

                for hook in hooks:
                    hook.remove()

                # don't need the list anymore
                conv_features = conv_features[0]
                dec_attn_weights_sub = dec_attn_weights_sub[0]
                dec_attn_weights_obj = dec_attn_weights_obj[0]

                # get the feature map shape
                h, w = conv_features['0'].tensors.shape[-2:]
                im_w, im_h = im.size

                print("Detected relationships:")

                fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(22, 7))
                for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                        zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
                    ax = ax_i[0]
                    ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
                    ax.axis('off')
                    ax.set_title(f'query id: {idx.item()}')
                    ax = ax_i[1]
                    ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
                    ax.axis('off')
                    ax = ax_i[2]
                    ax.imshow(im)
                    ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                            fill=False, color='blue', linewidth=2.5))
                    ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                            fill=False, color='orange', linewidth=2.5))

                    ax.axis('off')
                    ax.set_title(self.CLASSES[probas_sub[idx].argmax()]+' '+self.REL_CLASSES[probas[idx].argmax()]+' '+self.CLASSES[probas_obj[idx].argmax()], fontsize=10)
                    print(self.CLASSES[probas_sub[idx].argmax()]+' '+self.REL_CLASSES[probas[idx].argmax()]+' '+self.CLASSES[probas_obj[idx].argmax()])


            fig.canvas.draw()

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            # display image with opencv or any operation you like
            cv2.imshow("plot",img)

            k = cv2.waitKey(33) & 0xFF
            if k == 27:
                self.video_service.unsubscribe(self.videosClient)
                self.session.close()
                break

if __name__ == '__main__':
    session = qi.Session()
    try:
        session.connect("tcp://pepper2.local:9559")
    except RuntimeError:
        print(("[RoboBreizh - Vision]  Can't connect to Naoqi. Please check your script arguments. Run with -h option for help."))
        sys.exit(1)
    parser = argparse.ArgumentParser('RelTR inference', parents=[get_args_parser()])
    args = parser.parse_args()
    detector=SceneGraphGenerator(session, args)
    detector.detect()
