# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
from PIL import Image

import time, sys
import onnxruntime
import numpy as np
from scipy.special import softmax
import qi 
import matplotlib.pyplot as plt
import cv2
import networkx as nx
from distinctipy import distinctipy

class SceneGraphGenerator():
    def __init__(self, session=None):
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
        
        self.session = session
        self.initCamerasNaoQi()
        self.initONNX()

    def initONNX(self):
        start = time.time()
        sess_options = onnxruntime.SessionOptions()
        #sess_options.enable_profiling = True

        sess_options.intra_op_num_threads = 4
        sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session  = onnxruntime.InferenceSession("normal_resolution.onnx", sess_options)                      
        #self.ort_session = onnxruntime.InferenceSession("normal_resolution-sim.onnx")
        print("Loading time ONNX: ", time.time() - start)

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

    def IoU(self, box1, box2):
        x1, y1, x2, y2 = box1	
        x3, y3, x4, y4 = box2
        x_inter1 = max(x1, x3)
        y_inter1 = max(y1, y3)
        x_inter2 = min(x2, x4)
        y_inter2 = min(y2, y4)
        width_inter = abs(x_inter2 - x_inter1)
        height_inter = abs(y_inter2 - y_inter1)
        area_inter = width_inter * height_inter
        width_box1 = abs(x2 - x1)
        height_box1 = abs(y2 - y1)
        width_box2 = abs(x4 - x3)
        height_box2 = abs(y4 - y3)
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        area_union = area_box1 + area_box2 - area_inter
        iou = area_inter / area_union
        return iou
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, boxes):
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        boxes = np.stack((x1, y1, x2, y2), axis=-1)
        return boxes

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
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

            newsize = (640, 480)
            img = im.resize(newsize)
            img = self.normalize(img)

            # mean-std normalize the input image (batch-size: 1)

            # compute ONNX Runtime output prediction
            start = time.time()

            ort_inputs = {self.ort_session.get_inputs()[0].name: img}
            ort_outs = self.ort_session.run(None, ort_inputs)

            print("Total inference time: ", time.time() - start)
            outputs = {}

            outputs['pred_logits'] = ort_outs[0]
            outputs['pred_boxes'] = ort_outs[1]
            outputs['sub_logits'] = ort_outs[2]
            outputs['sub_boxes'] = ort_outs[3]
            outputs['obj_logits'] = ort_outs[4]
            outputs['obj_boxes'] = ort_outs[5]
            outputs['rel_logits'] = ort_outs[6]

            fig = self.draw_scene_graph_layout(ort_outs, im)

            # convert canvas to image
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

            # display image with opencv or any operation you like
            cv2.imshow("plot",img)
            time.sleep(1)

            k = cv2.waitKey(33) & 0xFF
            if k == 27:
                self.video_service.unsubscribe(self.videosClient)
                self.session.close()
                break

    def draw_splited_layout(self, outputs, im):
        # keep only predictions with 0.+ confidence
        probas = softmax(outputs['rel_logits'], axis=-1)[0, :, :-1]
        probas_sub = softmax(outputs['sub_logits'], axis=-1)[0, :, :-1]
        probas_obj = softmax(outputs['obj_logits'], axis=-1)[0, :, :-1]
        keep = np.logical_and(np.amax(probas, axis=-1) > 0.3, np.logical_and(np.amax(probas_sub, axis=-1) > 0.3,
                                                                                np.amax(probas_obj, axis=-1) > 0.3))

        # convert boxes from [0; 1] to image scales
        sub_bboxes_scaled = self.rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = self.rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = 10
        keep_queries = np.nonzero(keep)[0]
        indices = np.argsort(-np.amax(probas[keep_queries], axis=-1) * np.amax(probas_sub[keep_queries], axis=-1) * np.amax(probas_obj[keep_queries], axis=-1))[:topk]
        keep_queries = keep_queries[indices]
        print("Detected relationships:")
        for idx in keep_queries:
            print("\t{} {} {}".format(self.CLASSES[np.argmax(probas_sub[idx])],
                                    self.REL_CLASSES[np.argmax(probas[idx])],
                                    self.CLASSES[np.argmax(probas_obj[idx])]))

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
            ax.set_title(self.CLASSES[np.argmax(probas_sub[idx])]+' '+self.REL_CLASSES[np.argmax(probas[idx])]+' '+self.CLASSES[np.argmax(probas_obj[idx])], fontsize=10)

        fig.tight_layout()

        fig.canvas.draw()

        return fig
        

    def draw_scene_graph_layout(self, ort_outs, im):
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
        sub_bboxes_scaled = self.rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
        obj_bboxes_scaled = self.rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

        topk = 10
        keep_queries = np.nonzero(keep)[0]
        indices = np.argsort(-np.amax(probas[keep_queries], axis=-1) * np.amax(probas_sub[keep_queries], axis=-1) * np.amax(probas_obj[keep_queries], axis=-1))[:topk]
        keep_queries = keep_queries[indices]

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(20, 6))
        ax = axs.T[0]
        ax.imshow(im)

        G = nx.MultiDiGraph()
        labels = {}  
        color_map=[]
        colors = distinctipy.get_colors(len(keep_queries)*2)

        indexes=[i for i in range(len(sub_bboxes_scaled[indices]))]
        index_color={}
        for col, idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(indexes, keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            
            rel=[self.CLASSES[np.argmax(probas_sub[idx])], self.REL_CLASSES[np.argmax(probas[idx])], self.CLASSES[np.argmax(probas_obj[idx])]]

            if (np.argmax(probas_sub[idx]) not in index_color.keys()):
                index_color[np.argmax(probas_sub[idx])] = colors[col]
                ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                                    fill=False, color=colors[col], linewidth=2.5))
            
            index_color[rel[1]]="white"                              
            if np.argmax(probas_obj[idx]) not in index_color.keys():
                index_color[np.argmax(probas_obj[idx])] = colors[col+len(keep_queries)]
                ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                                    fill=False, color=colors[col+len(keep_queries)], linewidth=2.5))
        
            G.add_node(rel[0])
            G.add_node(rel[1])
            G.add_node(rel[2])
            G.add_edge(rel[0], rel[1])
            G.add_edge(rel[1], rel[2])  
            labels[rel[0]] = rel[0]
            labels[rel[1]] = rel[1]
            labels[rel[2]] = rel[2]
        
        for k, v in index_color.items():
            color_map.append(v)
        ax.axis('off')
        ax.set_title("Detected regions", fontsize=10)
        ax2 = axs.T[1]
        pos=nx.circular_layout(G)    
        options = {"node_size": 2000, "alpha": 0.9, "node_color": color_map}
        nx.draw_networkx_nodes(G, pos=pos, node_size=2000, alpha=1, node_color='w')
        # draw the nodes as desired
        nx.draw_networkx_nodes(G, pos=pos, node_size=2000, alpha=.3, node_color=color_map)
        #nx.draw_networkx_nodes(G, pos,  ax=ax2, **options)
        nx.draw_networkx_labels(G,pos,labels, font_size=16,font_color='r')
        nx.draw_networkx_edges(G, pos, ax=ax2, arrows=True, arrowsize=10, arrowstyle='-|>', node_size=2000)
        ax2.set_title("Relationships", fontsize=10)
        ax2.set_axis_off()
    
        fig.tight_layout()

        fig.canvas.draw()


        return fig

if __name__ == '__main__':
    session = qi.Session()
    try:
        session.connect("tcp://pepper2.local:9559")
    except RuntimeError:
        print(("[RoboBreizh - Vision]  Can't connect to Naoqi. Please check your script arguments. Run with -h option for help."))
        sys.exit(1)
    detector=SceneGraphGenerator(session)
    detector.detect()
