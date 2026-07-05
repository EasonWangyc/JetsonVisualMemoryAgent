from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export CLIP ViT image encoder to ONNX.")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32")
    parser.add_argument("--out", default="artifacts/onnx/clip_image_encoder.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--report", default="reports/export_clip_onnx.json")
    return parser.parse_args()


def write_report(path: str | Path, payload: dict) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = {
        "tool": "export_clip_onnx",
        "model_name": args.model_name,
        "out": args.out,
        "status": "skipped",
        "reason": "",
    }
    try:
        import torch
        from transformers import CLIPModel
    except ImportError as exc:
        report["reason"] = f"missing dependency: {exc.name}"
        write_report(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    class ClipImageWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            features = self.model.get_image_features(pixel_values=pixel_values)
            return features / features.norm(dim=-1, keepdim=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = CLIPModel.from_pretrained(args.model_name).eval()
    wrapper = ClipImageWrapper(model).eval()
    dummy = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out),
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch"}, "image_embeds": {0: "batch"}},
        opset_version=args.opset,
    )
    report.update({"status": "ready", "reason": "exported", "bytes": out.stat().st_size})
    write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
