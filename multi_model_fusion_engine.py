{\rtf1\ansi\ansicpg950\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from __future__ import annotations\
\
from dataclasses import dataclass, field\
from typing import Dict, List, Tuple, Optional, Any\
import math\
import json\
from collections import Counter, defaultdict\
\
\
# =========================\
# Data structures\
# =========================\
\
@dataclass\
class Detection:\
    """\
    Single detection result from one model.\
    bbox format: (x1, y1, x2, y2)\
    logits: optional raw class logits before softmax\
    score: optional confidence score if logits are not available\
    """\
    model_name: str\
    class_name: str\
    bbox: Tuple[float, float, float, float]\
    score: float\
    logits: Optional[Dict[str, float]] = None\
\
\
@dataclass\
class FusionConfig:\
    """\
    Configuration for multi-model fusion.\
    """\
    class_names: List[str]\
    target_classes: List[str] = field(default_factory=lambda: ["hole", "leak", "white", "normal"])\
\
    # Temperature scaling per model\
    temperature: Dict[str, float] = field(default_factory=dict)\
\
    # Per-class AP from validation set: ap_table[model_name][class_name] = AP\
    ap_table: Dict[str, Dict[str, float]] = field(default_factory=dict)\
\
    # Fusion exponent alpha in paper\
    alpha: float = 1.0\
\
    # Confidence threshold before fusion\
    conf_threshold: float = 0.25\
\
    # Class-wise NMS IoU thresholds\
    nms_iou_thresholds: Dict[str, float] = field(default_factory=lambda: \{\
        "hole": 0.60,\
        "leak": 0.60,\
        "white": 0.60,\
        "normal": 0.60,\
    \})\
\
    # Normal-veto thresholds\
    tau_def: float = 0.35\
    tau_norm: float = 0.55\
\
    # Rule library: Top-1 / Top-2 priority by class\
    rule_library: Dict[str, List[str]] = field(default_factory=lambda: \{\
        "white": ["YOLOv7", "YOLOv5"],\
        "hole": ["YOLOv5", "YOLOv11"],\
        "leak": ["YOLOv5", "YOLOv11"],\
        "normal": []\
    \})\
\
    # Tie threshold for near-tied fused scores\
    tie_margin: float = 0.03\
\
    # Optional view-aware weights multiplier\
    view_weight_multiplier: Dict[str, Dict[str, float]] = field(default_factory=lambda: \{\
        "front": \{\},\
        "back": \{\
            "YOLOv5": 1.05,\
            "YOLOv8": 1.05,\
            "YOLOv11": 1.05\
        \}\
    \})\
\
\
@dataclass\
class FusionResult:\
    fused_class: str\
    fused_score: float\
    fused_bbox: Optional[Tuple[float, float, float, float]]\
    fused_scores_by_class: Dict[str, float]\
    kept_detections: List[Detection]\
    metadata: Dict[str, Any]\
\
\
# =========================\
# Utility functions\
# =========================\
\
def softmax(logits: Dict[str, float], temperature: float = 1.0) -> Dict[str, float]:\
    """\
    Temperature-scaled softmax over class logits.\
    """\
    if temperature <= 0:\
        raise ValueError("Temperature must be > 0.")\
\
    scaled = \{k: v / temperature for k, v in logits.items()\}\
    max_logit = max(scaled.values())\
    exp_vals = \{k: math.exp(v - max_logit) for k, v in scaled.items()\}\
    total = sum(exp_vals.values())\
    return \{k: v / total for k, v in exp_vals.items()\}\
\
\
def iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:\
    """\
    Compute IoU between two boxes.\
    """\
    x1 = max(box1[0], box2[0])\
    y1 = max(box1[1], box2[1])\
    x2 = min(box1[2], box2[2])\
    y2 = min(box1[3], box2[3])\
\
    inter_w = max(0.0, x2 - x1)\
    inter_h = max(0.0, y2 - y1)\
    inter_area = inter_w * inter_h\
\
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])\
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])\
\
    union_area = area1 + area2 - inter_area\
    if union_area <= 0:\
        return 0.0\
    return inter_area / union_area\
\
\
def class_wise_nms(\
    detections: List[Detection],\
    iou_thresholds: Dict[str, float]\
) -> List[Detection]:\
    """\
    Perform class-wise NMS.\
    """\
    grouped: Dict[str, List[Detection]] = defaultdict(list)\
    for det in detections:\
        grouped[det.class_name].append(det)\
\
    kept: List[Detection] = []\
    for cls, cls_dets in grouped.items():\
        cls_dets = sorted(cls_dets, key=lambda d: d.score, reverse=True)\
        threshold = iou_thresholds.get(cls, 0.60)\
\
        selected: List[Detection] = []\
        while cls_dets:\
            best = cls_dets.pop(0)\
            selected.append(best)\
            cls_dets = [\
                d for d in cls_dets\
                if iou(best.bbox, d.bbox) < threshold\
            ]\
        kept.extend(selected)\
\
    return kept\
\
\
# =========================\
# Fusion engine\
# =========================\
\
class MultiModelFusionEngine:\
    """\
    Multi-model fusion engine matching the paper logic:\
    - temperature scaling\
    - AP-based confidence weighting\
    - rule-library selection\
    - tie fallback with majority vote\
    - class-wise NMS\
    - normal-veto\
    """\
\
    def __init__(self, config: FusionConfig):\
        self.config = config\
\
    def _get_calibrated_probs(self, det: Detection) -> Dict[str, float]:\
        """\
        Convert logits to calibrated probabilities.\
        If logits are unavailable, fall back to the detection score on predicted class.\
        """\
        if det.logits:\
            tau = self.config.temperature.get(det.model_name, 1.0)\
            return softmax(det.logits, tau)\
\
        probs = \{cls: 0.0 for cls in self.config.class_names\}\
        probs[det.class_name] = det.score\
        return probs\
\
    def _compute_weights(self, view: str) -> Dict[str, Dict[str, float]]:\
        """\
        Compute class-wise model weights from AP table:\
        w_(i,c) = AP(i,c)^alpha / sum_j AP(j,c)^alpha\
        Optionally apply view-aware multipliers.\
        """\
        weights: Dict[str, Dict[str, float]] = defaultdict(dict)\
\
        for cls in self.config.target_classes:\
            raw_scores = \{\}\
            for model_name, class_ap in self.config.ap_table.items():\
                ap_val = class_ap.get(cls, 0.0)\
                score = max(ap_val, 1e-8) ** self.config.alpha\
\
                # view-aware multiplier\
                score *= self.config.view_weight_multiplier.get(view, \{\}).get(model_name, 1.0)\
                raw_scores[model_name] = score\
\
            denom = sum(raw_scores.values()) or 1.0\
            for model_name, score in raw_scores.items():\
                weights[cls][model_name] = score / denom\
\
        return weights\
\
    def _apply_normal_veto(self, fused_scores: Dict[str, float]) -> Optional[str]:\
        """\
        If all defect classes are weak and normal is strong, output normal.\
        """\
        defect_classes = [c for c in self.config.target_classes if c != "normal"]\
\
        all_defect_low = all(fused_scores.get(c, 0.0) < self.config.tau_def for c in defect_classes)\
        normal_high = fused_scores.get("normal", 0.0) >= self.config.tau_norm\
\
        if all_defect_low and normal_high:\
            return "normal"\
        return None\
\
    def _majority_vote_fallback(self, detections: List[Detection]) -> Optional[str]:\
        """\
        Majority vote fallback when fused scores are near-tied.\
        """\
        if not detections:\
            return None\
\
        votes = Counter(d.class_name for d in detections)\
        return votes.most_common(1)[0][0]\
\
    def _rule_library_resolve(\
        self,\
        near_tied_classes: List[str],\
        detections: List[Detection]\
    ) -> Optional[str]:\
        """\
        Use rule-library Top-1 / Top-2 models if class scores are near-tied.\
        """\
        for cls in near_tied_classes:\
            preferred_models = self.config.rule_library.get(cls, [])\
            if not preferred_models:\
                continue\
\
            cls_dets = [d for d in detections if d.class_name == cls and d.model_name in preferred_models]\
            if cls_dets:\
                # Choose the class if its preferred detectors support it strongly\
                best_score = max(d.score for d in cls_dets)\
                if best_score >= self.config.conf_threshold:\
                    return cls\
\
        return None\
\
    def fuse_sample(\
        self,\
        detections: List[Detection],\
        view: str = "front"\
    ) -> FusionResult:\
        """\
        Fuse detections from multiple models for one sample / one side.\
        """\
        # Step 1: filter low-confidence detections\
        filtered = [d for d in detections if d.score >= self.config.conf_threshold]\
\
        # Step 2: class-wise NMS\
        kept = class_wise_nms(filtered, self.config.nms_iou_thresholds)\
\
        # Step 3: class-wise weights\
        weights = self._compute_weights(view)\
\
        # Step 4: fused scores\
        fused_scores = \{cls: 0.0 for cls in self.config.class_names\}\
\
        for det in kept:\
            calibrated_probs = self._get_calibrated_probs(det)\
            for cls in self.config.target_classes:\
                fused_scores[cls] += weights[cls].get(det.model_name, 0.0) * calibrated_probs.get(cls, 0.0)\
\
        # Step 5: normal veto\
        veto_class = self._apply_normal_veto(fused_scores)\
        if veto_class is not None:\
            fused_class = veto_class\
        else:\
            sorted_scores = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)\
            top_class, top_score = sorted_scores[0]\
            second_class, second_score = sorted_scores[1]\
\
            # Step 6: near-tie resolution via rule library / majority vote\
            if abs(top_score - second_score) <= self.config.tie_margin:\
                near_tied = [top_class, second_class]\
                resolved = self._rule_library_resolve(near_tied, kept)\
                if resolved is None:\
                    resolved = self._majority_vote_fallback(kept)\
                fused_class = resolved if resolved is not None else top_class\
            else:\
                fused_class = top_class\
\
        # Step 7: choose fused bbox from best kept detection in fused class\
        fused_bbox = None\
        fused_score = fused_scores.get(fused_class, 0.0)\
\
        fused_class_dets = [d for d in kept if d.class_name == fused_class]\
        if fused_class_dets:\
            best_det = max(fused_class_dets, key=lambda d: d.score)\
            fused_bbox = best_det.bbox\
\
        return FusionResult(\
            fused_class=fused_class,\
            fused_score=fused_score,\
            fused_bbox=fused_bbox,\
            fused_scores_by_class=fused_scores,\
            kept_detections=kept,\
            metadata=\{\
                "view": view,\
                "weights": weights,\
                "num_input_detections": len(detections),\
                "num_kept_after_nms": len(kept),\
            \}\
        )\
\
    def fuse_double_sided(\
        self,\
        front_detections: List[Detection],\
        back_detections: List[Detection]\
    ) -> Dict[str, FusionResult]:\
        """\
        Fuse front-side and back-side detections separately.\
        Final decision can be combined by set-union rule outside.\
        """\
        front_result = self.fuse_sample(front_detections, view="front")\
        back_result = self.fuse_sample(back_detections, view="back")\
\
        return \{\
            "front": front_result,\
            "back": back_result\
        \}\
\
    def union_decision(\
        self,\
        front_result: FusionResult,\
        back_result: FusionResult\
    ) -> Dict[str, Any]:\
        """\
        Apply set-union rule for two-sided inspection.\
        If either side detects a defect, mark sample as defective.\
        """\
        defect_classes = [c for c in self.config.target_classes if c != "normal"]\
\
        front_is_defect = front_result.fused_class in defect_classes\
        back_is_defect = back_result.fused_class in defect_classes\
\
        if front_is_defect and back_is_defect:\
            # choose higher fused score\
            if front_result.fused_score >= back_result.fused_score:\
                final_class = front_result.fused_class\
                source = "front"\
            else:\
                final_class = back_result.fused_class\
                source = "back"\
        elif front_is_defect:\
            final_class = front_result.fused_class\
            source = "front"\
        elif back_is_defect:\
            final_class = back_result.fused_class\
            source = "back"\
        else:\
            final_class = "normal"\
            source = "both"\
\
        return \{\
            "final_class": final_class,\
            "source_side": source,\
            "front_class": front_result.fused_class,\
            "back_class": back_result.fused_class,\
            "front_score": front_result.fused_score,\
            "back_score": back_result.fused_score,\
        \}\
\
    def log_result(\
        self,\
        sample_id: str,\
        front_result: FusionResult,\
        back_result: FusionResult,\
        final_decision: Dict[str, Any],\
        filepath: str\
    ) -> None:\
        """\
        Append one fused result to a JSONL log file.\
        """\
        record = \{\
            "sample_id": sample_id,\
            "front": \{\
                "class": front_result.fused_class,\
                "score": front_result.fused_score,\
                "bbox": front_result.fused_bbox,\
                "scores_by_class": front_result.fused_scores_by_class,\
                "metadata": front_result.metadata,\
            \},\
            "back": \{\
                "class": back_result.fused_class,\
                "score": back_result.fused_score,\
                "bbox": back_result.fused_bbox,\
                "scores_by_class": back_result.fused_scores_by_class,\
                "metadata": back_result.metadata,\
            \},\
            "final_decision": final_decision,\
        \}\
\
        with open(filepath, "a", encoding="utf-8") as f:\
            f.write(json.dumps(record, ensure_ascii=False) + "\\n")\
\
\
# =========================\
# Example usage\
# =========================\
\
if __name__ == "__main__":\
    config = FusionConfig(\
        class_names=["hole", "leak", "white", "normal"],\
        ap_table=\{\
            "YOLOv4": \{"hole": 0.980, "leak": 0.970, "white": 0.990, "normal": 0.990\},\
            "YOLOv5": \{"hole": 0.995, "leak": 0.995, "white": 0.995, "normal": 0.994\},\
            "YOLOv7": \{"hole": 0.992, "leak": 0.995, "white": 0.996, "normal": 0.994\},\
            "YOLOv8": \{"hole": 0.988, "leak": 0.994, "white": 0.995, "normal": 0.992\},\
            "YOLOv11": \{"hole": 0.994, "leak": 0.995, "white": 0.995, "normal": 0.994\},\
            "SSD": \{"hole": 0.980, "leak": 0.980, "white": 0.990, "normal": 0.990\},\
        \},\
        temperature=\{\
            "YOLOv4": 1.1,\
            "YOLOv5": 1.0,\
            "YOLOv7": 1.0,\
            "YOLOv8": 1.0,\
            "YOLOv11": 0.95,\
            "SSD": 1.1,\
        \},\
        rule_library=\{\
            "white": ["YOLOv7", "YOLOv5"],\
            "hole": ["YOLOv5", "YOLOv11"],\
            "leak": ["YOLOv5", "YOLOv11"],\
            "normal": []\
        \}\
    )\
\
    engine = MultiModelFusionEngine(config)\
\
    front_detections = [\
        Detection("YOLOv5", "hole", (10, 10, 50, 50), 0.92, \{"hole": 3.0, "leak": 0.2, "white": 0.1, "normal": 0.4\}),\
        Detection("YOLOv8", "hole", (11, 12, 51, 49), 0.90, \{"hole": 2.7, "leak": 0.1, "white": 0.2, "normal": 0.5\}),\
        Detection("YOLOv11", "hole", (9, 11, 48, 52), 0.91, \{"hole": 2.9, "leak": 0.2, "white": 0.1, "normal": 0.3\}),\
        Detection("SSD", "normal", (8, 8, 52, 52), 0.35, \{"hole": 0.7, "leak": 0.3, "white": 0.2, "normal": 1.8\}),\
    ]\
\
    back_detections = [\
        Detection("YOLOv5", "normal", (10, 10, 50, 50), 0.70, \{"hole": 0.2, "leak": 0.2, "white": 0.1, "normal": 2.5\}),\
        Detection("YOLOv8", "normal", (11, 10, 50, 51), 0.72, \{"hole": 0.3, "leak": 0.2, "white": 0.1, "normal": 2.7\}),\
        Detection("YOLOv11", "normal", (9, 9, 49, 50), 0.75, \{"hole": 0.2, "leak": 0.1, "white": 0.1, "normal": 2.9\}),\
    ]\
\
    results = engine.fuse_double_sided(front_detections, back_detections)\
    final_decision = engine.union_decision(results["front"], results["back"])\
\
    print("Front fused:", results["front"].fused_class, results["front"].fused_score)\
    print("Back fused:", results["back"].fused_class, results["back"].fused_score)\
    print("Final decision:", final_decision)}