"""Экспорт D-FINE-S в ONNX (вход 1x3x640x640 → logits, pred_boxes). Запуск один раз: python export_onnx.py"""
import pathlib, torch
from transformers import AutoModelForObjectDetection
from pipeline import DET_REPO
OUT = pathlib.Path(__file__).resolve().parent / "assets" / "onnx" / "dfine_s.onnx"
OUT.parent.mkdir(parents=True, exist_ok=True)
class W(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): o = self.m(pixel_values=x); return o.logits, o.pred_boxes
m = AutoModelForObjectDetection.from_pretrained(DET_REPO).eval()
torch.onnx.export(W(m), torch.zeros(1, 3, 640, 640), str(OUT), input_names=["pixel_values"], output_names=["logits", "pred_boxes"], opset_version=17, dynamo=False)
print("ok", OUT, OUT.stat().st_size // 2**20, "MB")
