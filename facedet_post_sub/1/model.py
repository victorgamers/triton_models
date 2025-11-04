import math
import cv2
import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import triton_typing as pb_utils


def sigmod(x):
    return 1 / (1 + math.exp(-x))


def desigmod(x):
    return -math.log(1.0 / x - 1.0)


def initPrior(input_w, input_h):
    priors = []
    strides = [8, 16, 32]
    minsize = [[16, 32], [64, 128], [256, 512]]
    for istride in range(len(strides)):
        stride = strides[istride]
        height = input_h // stride
        width = input_w // stride
        anchors = minsize[istride]
        for y in range(height):
            for x in range(width):
                for anchor in anchors:
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    sx = anchor
                    sy = anchor
                    priors.append((cx, cy, sx, sy))
    return priors


def nms_max(predicts: np.ndarray, conf: float, threshold: float):
    outputs = [[] for i in range(predicts.shape[0])]
    conf = desigmod(conf)
    for idx, predict in enumerate(predicts):
        face_conf = predict[:, 5] - predict[:, 4]
        filter_predict = predict[face_conf > conf]
        if not filter_predict.shape[0]:
            continue
        score = filter_predict[:, 5] - filter_predict[:, 4]
        filter_predict = filter_predict[(-score).argsort()]
        deter_boxs: np.ndarray = np.concatenate(
            [filter_predict[:, :4], score.reshape(-1, 1)], axis=1
        )
        keep_boxs = []
        while deter_boxs.shape[0]:
            large_overlap = (
                bbox_iou(np.expand_dims(deter_boxs[0, :4], axis=0), deter_boxs[:, :4])
                > threshold
            )
            invalid = large_overlap
            invalid = invalid[0, :]
            keep_boxs.append(deter_boxs[0])
            deter_boxs = deter_boxs[~invalid]
        outputs[idx] = keep_boxs
    return outputs


def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    b1_x1 = b1_x1[:, np.newaxis]
    b1_y1 = b1_y1[:, np.newaxis]
    b1_x2 = b1_x2[:, np.newaxis]
    b1_y2 = b1_y2[:, np.newaxis]

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.clip(inter_x2 - inter_x1 + 1e-9, 0, None) * np.clip(
        inter_y2 - inter_y1 + 1e-9, 0, None
    )

    b1_area = (b1_x2 - b1_x1 + 1e-9) * (b1_y2 - b1_y1 + 1e-9)
    b2_area = (b2_x2 - b2_x1 + 1e-9) * (b2_y2 - b2_y1 + 1e-9)

    union_area = b1_area + b2_area - inter_area
    iou = inter_area / (union_area + 1e-16)
    return iou


def xywh2xyxy(x: np.ndarray):
    y = np.empty(x.shape, dtype=x.dtype)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def letterbox_revert(boxes, orig_w, orig_h, target_w, target_h):
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = orig_w * scale
    new_h = orig_h * scale
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2

    orig_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        orig_x1 = (x1 - pad_w) / scale
        orig_y1 = (y1 - pad_h) / scale
        orig_x2 = (x2 - pad_w) / scale
        orig_y2 = (y2 - pad_h) / scale
        orig_x1 = max(0, min(orig_x1, orig_w))
        orig_y1 = max(0, min(orig_y1, orig_h))
        orig_x2 = max(0, min(orig_x2, orig_w))
        orig_y2 = max(0, min(orig_y2, orig_h))
        orig_boxes.append([orig_x1, orig_y1, orig_x2, orig_y2])
    return np.array(orig_boxes)


def convertBoxes(bboxes, invertMat):
    a, b, c = invertMat[0]
    d, e, f = invertMat[1]
    left = bboxes[0]
    top = bboxes[1]
    right = bboxes[2]
    bottom = bboxes[3]
    src_left = left * a + top * b + c
    src_top = left * d + top * e + f
    src_right = right * a + bottom * b + c
    src_bottom = right * d + bottom * e + f
    bboxes[0] = int(src_left)
    bboxes[1] = int(src_top)
    bboxes[2] = int(src_right)
    bboxes[3] = int(src_bottom)
    return bboxes


def emotion_preporcess(image, input_w, input_h):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    image = np.ascontiguousarray(image).astype(np.float32)
    return image


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                input_image = pb_utils.get_input_tensor_by_name(request, "input")
                input_image = input_image.as_numpy()
                deter_image = pb_utils.get_input_tensor_by_name(request, "detr_image")
                deter_image = deter_image.as_numpy()
                infer_response = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("image_output", deter_image)]
                )
                responses.append(infer_response)
            except Exception as e:
                print("e================", e)
        return responses

    def finalize(self):
        print("Cleaning up===========================")
