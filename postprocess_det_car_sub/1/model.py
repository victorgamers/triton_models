import numpy as np

try:
    import triton_python_backend_utils as pb_utils
except ImportError:
    import triton_typing as pb_utils
import cv2


def numpy2img(img):
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def nms_max(predicts: np.ndarray, conf: float, threshold: float):
    predicts[..., :4] = xywh2xyxy(predicts[..., :4])
    outputs = [[] for i in range(predicts.shape[0])]
    for idx, predict in enumerate(predicts):
        filter_predict = predict[predict[:, 4] > conf]
        if not filter_predict.shape[0]:
            continue
        score = filter_predict[:, 4] * filter_predict[:, 5:].max(axis=1)
        filter_predict = filter_predict[(-score).argsort()]
        class_conf = filter_predict[:, 5:].max(axis=1).reshape(-1, 1)
        class_pred = filter_predict[:, 5:].argmax(axis=1).reshape(-1, 1)
        deter_boxs: np.ndarray = np.concatenate(
            [filter_predict[:, :5], class_conf, class_pred], axis=1
        )
        keep_boxs = []
        while deter_boxs.shape[0]:
            large_overlap = (
                bbox_iou(np.expand_dims(deter_boxs[0, :4], axis=0), deter_boxs[:, :4])
                > threshold
            )
            label_match = deter_boxs[0, -1] == deter_boxs[:, -1]
            invalid = large_overlap & label_match
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

    b1_x1 = b1_x1[:, np.newaxis]  # 形状 (N, 1)
    b1_y1 = b1_y1[:, np.newaxis]
    b1_x2 = b1_x2[:, np.newaxis]
    b1_y2 = b1_y2[:, np.newaxis]

    # 计算交集矩形的坐标
    inter_x1 = np.maximum(b1_x1, b2_x1)  # 逐元素取最大值 (N, M)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y2 = np.minimum(b1_y2, b2_y2)

    # 计算交集面积（确保非负）
    inter_area = np.clip(inter_x2 - inter_x1 + 1e-9, 0, None) * np.clip(
        inter_y2 - inter_y1 + 1e-9, 0, None
    )

    # 计算两个边界框的面积
    b1_area = (b1_x2 - b1_x1 + 1e-9) * (b1_y2 - b1_y1 + 1e-9)  # (N, 1)
    b2_area = (b2_x2 - b2_x1 + 1e-9) * (b2_y2 - b2_y1 + 1e-9)  # (M,)

    union_area = b1_area + b2_area - inter_area  # (N, M)
    iou = inter_area / (union_area + 1e-16)  # 避免除零
    return iou


def xywh2xyxy(x: np.ndarray):
    y = np.empty(x.shape, dtype=x.dtype)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def crop_objects(image, bbox):
    image = np.squeeze(image)
    h, w = image.shape[-2:]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x1 >= x2 or y1 >= y2:
        print(f"无效检测框：{bbox}，已跳过")
        return
    cropped = image[:, y1:y2, x1:x2]
    return cropped


def preprocess(img_path, img_size=640):
    # 1. 读取图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("无法读取图像")

    # 2. Letterbox缩放
    h0, w0 = img.shape[:2]
    r = min(img_size / h0, img_size / w0)
    new_unpad = (int(w0 * r), int(h0 * r))
    dw, dh = img_size - new_unpad[0], img_size - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 3. 通道转换（BGR→RGB）
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 4. 维度调整（HWC→CHW）
    img = img.transpose((2, 0, 1))

    # 5. 归一化与类型转换
    img = np.ascontiguousarray(img).astype(np.float32) / 255.0

    # 6. 添加批次维度
    img = np.expand_dims(img, axis=0)
    return img


class TritonPythonModel:
    def initialize(self, args):
        pass

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                det_input = pb_utils.get_input_tensor_by_name(request, "det_input")
                image_input = pb_utils.get_input_tensor_by_name(request, "image_input")
                det_input = det_input.as_numpy()
                image_input = image_input.as_numpy()
                bboxes = nms_max(det_input, 0.5, 0.5)
                image_inputs = []
                for i in range(len(bboxes)):
                    for j in range(len(bboxes[i])):
                        bbox = bboxes[i][j]
                        crop_img = crop_objects(image_input, bbox[:4])
                        image_inputs.append(crop_img)

                for idx, image in enumerate(image_inputs):
                    image = np.transpose(image, (1, 2, 0))
                    image = cv2.resize(image, (96, 48), interpolation=cv2.INTER_LINEAR)
                    image = np.transpose(image, (2, 0, 1))
                    image_inputs[idx] = image
                inputs = np.stack(image_inputs, 0)
                infer_response = pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("image_output", inputs)]
                )
                responses.append(infer_response)
            except Exception as e:
                print("================e", e, flush=True)

        return responses

    def finalize(self):
        print("Cleaning up===========================")
