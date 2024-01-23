# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Institute of Information Processing, Leibniz University Hannover.
import argparse
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as T
import numpy as np
import onnxruntime
import time
from scipy.special import softmax
import cv2
import networkx as nx
from distinctipy import distinctipy
from IPython.display import Image as img2

transform = T.Compose([
    T.Resize([480,640]),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

model_fp32 = "normal_resolution.onnx"

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

def IoU(box1, box2):
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

def post_processing(ort_outs, im):
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

    # print("Detected relationships:")
    # for idx in keep_queries:
    #     print("\t{} {} {}".format(CLASSES[np.argmax(probas_sub[idx])],
    #                             REL_CLASSES[np.argmax(probas[idx])],
    #                             CLASSES[np.argmax(probas_obj[idx])]))
    
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
        ax.set_title(CLASSES[np.argmax(probas_sub[idx])]+' '+REL_CLASSES[np.argmax(probas[idx])]+' '+CLASSES[np.argmax(probas_obj[idx])], fontsize=10)

    fig.tight_layout()
    fig.canvas.draw()

    #plt.show()

    return(fig)

def webcam():

    ort_session = onnxruntime.InferenceSession(model_fp32, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])

    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Our operations on the frame come here
        # compute ONNX Runtime output prediction
        im_pil = Image.fromarray(frame)
        # Compile and run

        img=transform(im_pil).unsqueeze_(0).cpu().detach().numpy()

        start = time.time()
        ort_inputs = {ort_session.get_inputs()[0].name: img}
        ort_outs = ort_session.run(None, ort_inputs)
        print("inference time: ", time.time()-start)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Display the resulting frame
        #fig = post_processing(ort_outs, im_pil)
        fig = draw_scene_graph_layout(ort_outs, im_pil)
        
        # convert canvas to image
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        cv2.imshow("plot",img)

        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def draw_scene_graph_layout(ort_outs, im):
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

    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(22, 7))
    ax = axs.T[0]
    ax.imshow(im)

    G = nx.MultiDiGraph()
    labels = {}  
    color_map=[]
    edge_labels={}
    colors = distinctipy.get_colors(len(keep_queries)*2)

    indexes=[i for i in range(len(sub_bboxes_scaled[indices]))]
    index_color={}
    for col, idx, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
            zip(indexes, keep_queries, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
        
        rel=[CLASSES[np.argmax(probas_sub[idx])], REL_CLASSES[np.argmax(probas[idx])], CLASSES[np.argmax(probas_obj[idx])]]

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
    #print(reversed(G.edges()[0]))

    # curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
    # straight_edges = list(set(G.edges()) - set(curved_edges))
    # nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=straight_edges)
    # arc_rad = 0.25

    # if len(curved_edges)!=0:
    #     nx.draw_networkx_edges(G, pos, ax=ax2, edgelist=curved_edges, connectionstyle=f"arc3,rad={arc_rad}")
    # #nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle=f'arc3, rad = {arc_rad}')  
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    fig.tight_layout()

    fig.canvas.draw()
    #time.sleep(3)
    # p=nx.drawing.nx_pydot.to_pydot(G)
    # p.write_png('multi.png')
    # img2(filename='multi.png')

    return fig

if __name__ == '__main__':
    webcam()
