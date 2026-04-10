{\rtf1\ansi\ansicpg950\cocoartf2868
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # main_production_pipeline.py\
\
import time\
import random\
from typing import List, Dict\
from multi_model_fusion_engine import (\
    MultiModelFusionEngine,\
    FusionConfig,\
    Detection\
)\
\
# =========================\
# \uc0\u27169 \u25836 \u36664 \u36865 \u24118 \u31995 \u32113 \
# =========================\
\
class ConveyorSystem:\
    def __init__(self):\
        self.sample_id = 0\
\
    def get_next_sample(self) -> Dict:\
        """\uc0\u27169 \u25836 \u36914 \u26009 """\
        self.sample_id += 1\
        return \{\
            "id": f"sample_\{self.sample_id\}",\
            "orientation": random.choice(["front", "back"])  # \uc0\u38568 \u27231 \u36914 \u20837 \u26041 \u21521 \
        \}\
\
\
# =========================\
# \uc0\u27169 \u25836 \u30456 \u27231 \
# =========================\
\
class Camera:\
    def capture(self, sample):\
        """\uc0\u27169 \u25836 \u24433 \u20687 \u25847 \u21462 """\
        return \{\
            "sample_id": sample["id"],\
            "image": f"image_of_\{sample['id']\}"\
        \}\
\
\
# =========================\
# \uc0\u27169 \u25836 \u27169 \u22411 \u25512 \u35542 \u65288 \u20320 \u20043 \u24460 \u21487 \u25563 \u25104  YOLO/SSD\u65289 \
# =========================\
\
class ModelInference:\
    def __init__(self, model_names):\
        self.model_names = model_names\
\
    def infer(self, image_data) -> List[Detection]:\
        detections = []\
\
        for model in self.model_names:\
            # \uc0\u27169 \u25836 \u38928 \u28204 \
            cls = random.choice(["hole", "leak", "white", "normal"])\
            score = random.uniform(0.3, 0.99)\
\
            detections.append(\
                Detection(\
                    model_name=model,\
                    class_name=cls,\
                    bbox=(10, 10, 50, 50),\
                    score=score,\
                    logits=None\
                )\
            )\
\
        return detections\
\
\
# =========================\
# \uc0\u21076 \u38500 \u25511 \u21046 \u65288 Air Blow\u65289 \
# =========================\
\
class Actuator:\
    def reject(self, sample_id):\
        print(f"[REJECT] \{sample_id\} -> blown off conveyor")\
\
    def accept(self, sample_id):\
        print(f"[PASS] \{sample_id\} -> continue")\
\
\
# =========================\
# \uc0\u32763 \u38754 \u27231 \u27083 \
# =========================\
\
class Flipper:\
    def flip(self, sample):\
        """\uc0\u27169 \u25836 \u32763 \u38754 """\
        sample["orientation"] = "back"\
        return sample\
\
\
# =========================\
# \uc0\u20027 \u27969 \u31243 \u65288 Figure 8 \u23565 \u25033 \u65289 \
# =========================\
\
class ProductionPipeline:\
    def __init__(self, fusion_engine: MultiModelFusionEngine):\
        self.conveyor = ConveyorSystem()\
        self.camera = Camera()\
        self.models = ModelInference([\
            "YOLOv4", "YOLOv5", "YOLOv7", "YOLOv8", "YOLOv11", "SSD"\
        ])\
        self.actuator = Actuator()\
        self.flipper = Flipper()\
        self.engine = fusion_engine\
\
    def process_one_sample(self):\
\
        # =========================\
        # (1) \uc0\u36914 \u26009 \
        # =========================\
        sample = self.conveyor.get_next_sample()\
        sample_id = sample["id"]\
\
        print(f"\\n=== Processing \{sample_id\} ===")\
\
        # =========================\
        # (2) \uc0\u31532 \u19968 \u36650 \u24433 \u20687 \u25847 \u21462 \
        # =========================\
        img_front = self.camera.capture(sample)\
\
        # =========================\
        # (3) \uc0\u24179 \u34892 \u27169 \u22411 \u25512 \u35542 \
        # =========================\
        detections_front = self.models.infer(img_front)\
\
        # =========================\
        # (4) Fusion decision\
        # =========================\
        front_result = self.engine.fuse_sample(detections_front, view="front")\
\
        print(f"[Front] \{front_result.fused_class\} (\{front_result.fused_score:.3f\})")\
\
        # =========================\
        # (5) \uc0\u31532 \u19968 \u36650 \u21076 \u38500 \
        # =========================\
        if front_result.fused_class != "normal":\
            self.actuator.reject(sample_id)\
\
            # logging\
            self.engine.log_result(\
                sample_id,\
                front_result,\
                front_result,  # \uc0\u27794 \u26377  back\
                \{"final_class": front_result.fused_class, "stage": "first"\},\
                "log.jsonl"\
            )\
            return\
\
        self.actuator.accept(sample_id)\
\
        # =========================\
        # (6) \uc0\u32763 \u38754 \
        # =========================\
        sample = self.flipper.flip(sample)\
\
        # =========================\
        # (7) \uc0\u31532 \u20108 \u36650 \u24433 \u20687 \u25847 \u21462 \
        # =========================\
        img_back = self.camera.capture(sample)\
\
        # =========================\
        # (8) \uc0\u31532 \u20108 \u36650 \u25512 \u35542 \
        # =========================\
        detections_back = self.models.infer(img_back)\
\
        back_result = self.engine.fuse_sample(detections_back, view="back")\
\
        print(f"[Back] \{back_result.fused_class\} (\{back_result.fused_score:.3f\})")\
\
        # =========================\
        # (9) \uc0\u38617 \u38754 \u34701 \u21512 \u65288 union rule\u65289 \
        # =========================\
        final_decision = self.engine.union_decision(front_result, back_result)\
\
        print(f"[Final] \{final_decision['final_class']\}")\
\
        # =========================\
        # (10) \uc0\u26368 \u32066 \u21076 \u38500 \
        # =========================\
        if final_decision["final_class"] != "normal":\
            self.actuator.reject(sample_id)\
        else:\
            self.actuator.accept(sample_id)\
\
        # =========================\
        # (11) logging\
        # =========================\
        self.engine.log_result(\
            sample_id,\
            front_result,\
            back_result,\
            final_decision,\
            "log.jsonl"\
        )\
\
\
# =========================\
# \uc0\u21021 \u22987 \u21270 \u31995 \u32113 \
# =========================\
\
def create_fusion_engine():\
    config = FusionConfig(\
        class_names=["hole", "leak", "white", "normal"],\
        ap_table=\{\
            "YOLOv4": \{"hole": 0.98, "leak": 0.97, "white": 0.99, "normal": 0.99\},\
            "YOLOv5": \{"hole": 0.995, "leak": 0.995, "white": 0.995, "normal": 0.994\},\
            "YOLOv7": \{"hole": 0.992, "leak": 0.995, "white": 0.996, "normal": 0.994\},\
            "YOLOv8": \{"hole": 0.988, "leak": 0.994, "white": 0.995, "normal": 0.992\},\
            "YOLOv11": \{"hole": 0.994, "leak": 0.995, "white": 0.995, "normal": 0.994\},\
            "SSD": \{"hole": 0.98, "leak": 0.98, "white": 0.99, "normal": 0.99\},\
        \}\
    )\
\
    return MultiModelFusionEngine(config)\
\
\
# =========================\
# \uc0\u20027 \u31243 \u24335 \u20837 \u21475 \
# =========================\
\
if __name__ == "__main__":\
\
    engine = create_fusion_engine()\
    pipeline = ProductionPipeline(engine)\
\
    # \uc0\u27169 \u25836 \u36899 \u32396 \u29983 \u29986 \
    for _ in range(10):\
        pipeline.process_one_sample()\
        time.sleep(0.5)  # \uc0\u27169 \u25836 \u36664 \u36865 \u24118 \u36895 \u24230 }