import os
from dataclasses import dataclass

import cv2
import numpy as np
import supervision as sv
import torch
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                   DetectionTargetModel)
from autodistill.helpers import load_image
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AdamW, AutoModelForCausalLM, AutoProcessor,
                          get_scheduler)

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_example(task_prompt, processor, model, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


@dataclass
class Florence2(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        model_id = "microsoft/Florence-2-base"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, device_map="cuda"
        )
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input, return_format="PIL")
        ontology_classes = self.ontology.classes()
        for i, class_name in enumerate(ontology_classes):
            result = run_example(
                "<OPEN_VOCABULARY_DETECTION>",
                self.processor,
                self.model,
                image,
                class_name,
            )
            if i == 0:
                results = result["<OPEN_VOCABULARY_DETECTION>"]
            else:
                results["bboxes"].append(result["<OPEN_VOCABULARY_DETECTION>"]["bboxes"][0])
                results["bboxes_labels"].append(result["<OPEN_VOCABULARY_DETECTION>"]["bboxes_labels"][0])

        boxes_and_labels = list(zip(results["bboxes"], results["bboxes_labels"]))

        if (
            len(
                [
                    box
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            )
            == 0
        ):
            return sv.Detections.empty()

        detections = sv.Detections(
            xyxy=np.array(
                [
                    box
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
            class_id=np.array(
                [
                    ontology_classes.index(label)
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
            confidence=np.array(
                [
                    1.0
                    for box, label in boxes_and_labels
                    if label in ontology_classes and ontology_classes
                ]
            ),
        )

        detections = detections[detections.confidence > confidence]

        return detections
    