from itertools import product as product
from math import ceil
from pathlib import Path
from urllib.request import urlretrieve

from cv2 import INTER_LINEAR, cvtColor, COLOR_BGR2RGB, resize, imread, imwrite, rectangle, GaussianBlur
from numpy import ndarray, array, clip, float32, double, expand_dims, maximum, concatenate, minimum, where, squeeze, hstack, newaxis, exp, zeros, uint8
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession

try:
    WORK_DIR = Path(__file__).parent
except NameError:
    WORK_DIR = Path.cwd()

WEIGHTS_DIR = WORK_DIR / 'weights'
FACEDETECTOR_URL = 'https://github.com/prolapser/cheap_facerestore/releases/download/models/FaceDetector.onnx'
FACERESTORE_URL = 'https://github.com/prolapser/cheap_facerestore/releases/download/models/CodeFormer.onnx'


class PriorBox(object):
    def __init__(self, image_size: tuple[int, int] | None = None) -> None:
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self) -> ndarray:
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        output = array(anchors).reshape(-1, 4)
        if self.clip:
            output = clip(output, 0, 1)

        return output


class FaceRestore:
    def __init__(self) -> None:
        self.face_restore_path = WEIGHTS_DIR / 'CodeFormer.onnx'
        self.face_detector_path = WEIGHTS_DIR / 'FaceDetector.onnx'
        self.download_models()
        session_options = SessionOptions()
        session_options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = InferenceSession(self.face_restore_path, sess_options=session_options, providers=["CPUExecutionProvider"])
        self.resolution = self.session.get_inputs()[0].shape[-2:]
        self.face_detector = InferenceSession(self.face_detector_path)

    def download_models(self):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        model_urls = {self.face_detector_path.name: FACEDETECTOR_URL, self.face_restore_path.name: FACERESTORE_URL}
        for model_name, model_url in model_urls.items():
            model_path = WEIGHTS_DIR / model_name
            if not model_path.exists():
                print(f'скачивание {model_name}...')
                urlretrieve(model_url, model_path)

    def preprocess(self, img: ndarray, w: float) -> tuple[ndarray, ndarray]:
        img = resize(img, self.resolution, interpolation=INTER_LINEAR)
        img = img.astype(float32)[:, :, ::-1] / 255.0
        img = img.transpose((2, 0, 1))
        img = (img - 0.5) / 0.5
        img = expand_dims(img, axis=0).astype(float32)
        w = array([w], dtype=double)
        return img, w

    @staticmethod
    def postprocess(img: ndarray) -> ndarray:
        img = (img.transpose(1, 2, 0).clip(-1, 1) + 1) * 0.5
        img = (img * 255)[:, :, ::-1]
        img = img.clip(0, 255).astype('uint8')
        return img

    @staticmethod
    def decode(loc: ndarray, priors: ndarray, variances: list[float]) -> ndarray:
        boxes = concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    @staticmethod
    def py_cpu_nms(dets: ndarray, thresh: float) -> list[int]:
        x1, y1, x2, y2 = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = maximum(x1[i], x1[order[1:]])
            yy1 = maximum(y1[i], y1[order[1:]])
            xx2 = minimum(x2[i], x2[order[1:]])
            yy2 = minimum(y2[i], y2[order[1:]])
            w = maximum(0.0, xx2 - xx1 + 1)
            h = maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def detect_and_crop_faces(self, img: ndarray, padding_percentage: float = 0.5, shift_up_percentage: float = 0.2) -> list[tuple[ndarray, tuple[int, int, int, int]]]:
        img_rgb = cvtColor(img, COLOR_BGR2RGB)
        img_resized = float32(img_rgb)
        scale = array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img_resized -= (104, 117, 123)
        img_resized = img_resized.transpose(2, 0, 1)
        img_resized = expand_dims(img_resized, axis=0)

        inputs = {"input": img_resized}
        loc, conf, landms = self.face_detector.run(None, inputs)

        im_height, im_width, _ = img.shape
        priorbox = PriorBox(image_size=(im_height, im_width))
        priors = priorbox.forward()
        prior_data = priors

        boxes = self.decode(squeeze(loc, axis=0), prior_data, [0.1, 0.2])
        boxes = boxes * scale
        scores = squeeze(conf, axis=0)[:, 1]

        inds = where(scores > 0.02)[0]
        boxes = boxes[inds]
        scores = scores[inds]
        keep = self.py_cpu_nms(hstack((boxes, scores[:, newaxis])).astype(float32, copy=False), 0.4)
        boxes = boxes[keep]

        faces = []
        for box in boxes:
            x, y, x2, y2 = box[:4].astype(int)
            w, h = x2 - x, y2 - y

            padding_x = int(w * padding_percentage)
            padding_y = int(h * padding_percentage)
            shift_up = int(h * shift_up_percentage)

            x_padded = max(x - padding_x, 0)
            y_padded = max(y - padding_y - shift_up, 0)
            x2_padded = min(x2 + padding_x, img.shape[1])
            y2_padded = min(y2 + padding_y - shift_up, img.shape[0])

            face = img[y_padded:y2_padded, x_padded:x2_padded]
            faces.append((face, (x_padded, y_padded, x2_padded - x_padded, y2_padded - y_padded)))

        return faces

    @staticmethod
    def insert_face_back(original_img: ndarray, enhanced_face: ndarray, face_coords: tuple[int, int, int, int], feather_percent: int = 10) -> ndarray:
        x, y, w, h = face_coords
        face_resized = resize(enhanced_face, (w, h))
        blend_width = int(w * (feather_percent / 100))
        mask = zeros((h, w), dtype=uint8)
        rectangle(mask, (blend_width, blend_width), (w - blend_width, h - blend_width), 255, thickness=-1)
        mask_blurred = GaussianBlur(mask, (2 * blend_width + 1, 2 * blend_width + 1), 0) / 255
        mask_blurred = mask_blurred[:, :, newaxis]
        roi = original_img[y:y + h, x:x + w]
        face_with_mask = face_resized * mask_blurred
        roi_with_mask = roi * (1 - mask_blurred)
        face_blended = face_with_mask + roi_with_mask
        original_img[y:y + h, x:x + w] = face_blended.astype(uint8)

        return original_img

    def enhance(self, image_path: str | Path, effect: int | float) -> ndarray:
        if effect > 1 or effect < 0:
            raise ValueError('параметр `effect` должен быть в диапазоне от 0 до 1')
        img = imread(str(image_path))
        faces = self.detect_and_crop_faces(img)
        if faces:
            for face, face_coords in faces:
                img_prep, w_prep = self.preprocess(face, 1-effect)
                output = self.session.run(None, {'x': img_prep, 'w': w_prep})[0][0]
                enhanced_face = self.postprocess(output)
                img = self.insert_face_back(img, enhanced_face, face_coords)
            return img
        else:
            print("лица не найдены.")
            return img


if __name__ == '__main__':
    input_image = Path('/content/image.jpg')
    fr = FaceRestore()
    imwrite(str(input_image.with_name(input_image.stem + '_fr.png')), fr.enhance(input_image, 0.9))
