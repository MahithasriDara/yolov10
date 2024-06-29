from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
import torch

class YOLOv10DetectionValidator(DetectionValidator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.args.save_json |= self.is_coco

    def postprocess(self, preds):
        if isinstance(preds, dict):
            preds = preds["one2one"]

        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        
        # Acknowledgement: Thanks to sanha9999 in #190 and #181!
        if preds.shape[-1] == 6:
            return preds
        else:
            preds = preds.transpose(-1, -2)
            boxes, scores, labels = ops.v10postprocess(preds, self.args.max_det, self.nc)
            bboxes = ops.xywh2xyxy(boxes)
            return torch.cat([bboxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=-1)



    def calculate_metrics(self, true_labels, pred_labels):
        precision = precision_score(true_labels, pred_labels, average='weighted')
        recall = recall_score(true_labels, pred_labels, average='weighted')
        accuracy = accuracy_score(true_labels, pred_labels)
        return precision, recall, accuracy

    def evaluate(self, preds, targets):
        # Evaluate and compute metrics
        true_labels = []
        pred_labels = []

        for target in targets:
            true_labels.extend(target["labels"].cpu().numpy())

        for pred in preds:
            pred_labels.extend(pred[..., -1].cpu().numpy())

        precision, recall, accuracy = self.calculate_metrics(true_labels, pred_labels)
        map_val = self.map(preds, targets)
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "map": map_val
        }
        
        return metrics
