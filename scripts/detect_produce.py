import torch
import torchvision
from torchvision import transforms
from PIL import Image
import io

# Load a pre-trained object detection model (Faster R-CNN with ResNet-50 backbone)
# This model is class‑agnostic; we will treat every detection with a score above a threshold as a produce item.
_DETECTOR = None
_DETECTOR_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SCORE_THRESHOLD = 0.1  # further lowered to capture low‑confidence detections

def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        # Using a COCO‑pretrained model – it knows many common objects including "apple", "banana", …
        # For full produce coverage you can fine‑tune on a custom dataset later.
        _DETECTOR = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        _DETECTOR.to(_DETECTOR_DEVICE)
        _DETECTOR.eval()
    return _DETECTOR

def detect_items(pil_image: Image.Image):
    """Detect produce items in an image.

    Returns a list of bounding boxes ``[left, top, right, bottom]`` in pixel coordinates.
    Only boxes whose confidence score exceeds ``SCORE_THRESHOLD`` are kept.
    """
    model = _load_detector()
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(pil_image).to(_DETECTOR_DEVICE)
    with torch.no_grad():
        outputs = model([tensor])[0]
    boxes = []
    for box, score in zip(outputs["boxes"], outputs["scores"]):
        if score.item() >= SCORE_THRESHOLD:
            # Convert to Python ints for JSON‑serialisation later
            boxes.append([int(coord) for coord in box.tolist()])
    return boxes

