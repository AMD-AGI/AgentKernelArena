"""Soft targets for the distillation loss."""
import torch

# Teacher logits have the spread of the student logits (`input`, std 8 in task.yaml).
_TEACHER_STD = 8.0


def make_inputs(case, inputs, generator, device):
    shape = tuple(inputs["input"].shape)
    teacher = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    teacher.mul_(_TEACHER_STD)
    return {"target": torch.softmax(teacher / float(inputs["temp_factor"]), dim=1)}
