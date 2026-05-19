import cv2
import numpy as np
import onnxruntime as ort
import time


ONNX_MODEL_PATH = "best_cards26.onnx"

CLASS_NAMES = [
    "ace",	
    "jack",	
    "king",	
    "nine",	
    "queen",
    "ten"
]

INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
CAMERA_INDEX = 0


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = image.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top


def preprocess(frame, input_size=640):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    padded, scale, pad_x, pad_y = letterbox(
        image_rgb,
        new_shape=(input_size, input_size)
    )

    input_tensor = padded.astype(np.float32) / 255.0
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor, scale, pad_x, pad_y


def xywh_to_xyxy(boxes):
    xyxy = np.zeros_like(boxes)

    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2

    return xyxy


def nms(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter_w = np.maximum(0, xx2 - xx1)
        inter_h = np.maximum(0, yy2 - yy1)

        intersection = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - intersection

        iou = intersection / np.maximum(union, 1e-6)

        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]

    return keep


def postprocess(output, scale, pad_x, pad_y, original_shape):
    predictions = output[0]

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)

    mask = scores > CONF_THRESHOLD

    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]

    if len(boxes) == 0:
        return []

    boxes = xywh_to_xyxy(boxes)

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes /= scale

    original_h, original_w = original_shape[:2]

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h - 1)

    detections = []

    unique_class_ids = np.unique(class_ids)

    for cls_id in unique_class_ids:
        cls_mask = class_ids == cls_id

        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep = nms(cls_boxes, cls_scores, IOU_THRESHOLD)

        for idx in keep:
            detections.append({
                "box": cls_boxes[idx],
                "score": float(cls_scores[idx]),
                "class_id": int(cls_id)
            })

    return detections


def draw_detections(frame, detections):
    output = frame.copy()

    for det in detections:
        box = det["box"].astype(int)
        score = det["score"]
        class_id = det["class_id"]

        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = f"class_{class_id}"

        x1, y1, x2, y2 = box

        color = (0, 255, 0)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name}: {score:.2f}"

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        text_w, text_h = text_size

        cv2.rectangle(
            output,
            (x1, y1 - text_h - baseline - 6),
            (x1 + text_w, y1),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1, y1 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output


def create_onnx_session(model_path):
    available_providers = ort.get_available_providers()

    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        model_path,
        providers=providers
    )

    print("Available providers:", available_providers)
    print("Using providers:", session.get_providers())

    return session


def main():
    session = create_onnx_session(ONNX_MODEL_PATH)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print("ONNX input name:", input_name)
    print("ONNX output name:", output_name)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read frame.")
            break

        input_tensor, scale, pad_x, pad_y = preprocess(
            frame,
            input_size=INPUT_SIZE
        )

        outputs = session.run(
            [output_name],
            {input_name: input_tensor}
        )

        detections = postprocess(
            outputs[0],
            scale,
            pad_x,
            pad_y,
            frame.shape
        )

        result_frame = draw_detections(frame, detections)

        current_time = time.time()
        fps = 1.0 / max(current_time - prev_time, 1e-6)
        prev_time = current_time

        cv2.putText(
            result_frame,
            f"FPS: {fps:.2f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            result_frame,
            f"Detections: {len(detections)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("YOLO ONNX Card Detection", result_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            print("Quitting...! My friend, see you next time!")
            break

        if key == ord("q"):
            print("Quitting...! My friend, see you next time!")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
