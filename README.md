# SolarScanner 🌞🛰️ – Integrated Deep‑Learning Pipeline for Post‑Disaster Urban Damage Assessment

SolarScanner (short **solars**) is a two‑stage deep‑learning system that **segments buildings** in high‑resolution satellite/drone imagery and then **classifies the damage level** of every detected building.

| Stage                        | Model                         | Dataset      | Metric                  |
| ---------------------------- | ----------------------------- | ------------ | ----------------------- |
| **1. Building Segmentation** | U‑Net + ResNet‑50 encoder     | SpaceNet v2  | **IoU 0.766**           |
| **2. Damage Classification** | Vision Transformer (ViT‑B/16) | xBD (xView2) | **Acc 0.856 / F1 0.84** |

> ⚠️ Decision‑support only – predictions must be verified by an expert. Use with caution. This model does not provide exact solution.

## 📂 Repository Layout

```
solars/
├── notebooks/
│   ├── DataDownload.ipynb
│   ├── Project.ipynb
│   └── Project_XBD_Only.ipynb
```

## 🚀 Quick Start

Use cells in the given notebooks. For download data use DataDownload. To training and inference you can use Project.ipynb or only xbd training and inference Project_XBD_Only.ipynb.

## 🏆 Results

| Task         | Metric   | Score     |
| ------------ | -------- | --------- |
| Segmentation | IoU      | **0.766** |
| Damage CLS   | Accuracy | **0.856** |

## Citation

```bibtex
@unpublished{topaloglu2025solars,
  author  = {Tuğcan Topaloğlu},
  title   = {{SolarScanner}: Two‑Stage Deep Learning for Post‑Disaster Building Damage Assessment},
  year    = {2025},
  url     = {https://github.com/tugcantopaloglu/solars}
}
```

## License

MIT
