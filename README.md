# SolarScanner

Research notebooks for a two-stage building damage assessment pipeline: U-Net building segmentation followed by Vision Transformer classification of cropped building patches.

## Repository contents

| File | Purpose |
| --- | --- |
| `DataDownload.ipynb` | Historical Colab download recipes for SpaceNet and xBD/xView2. Review provider access, paths, storage and commands before use. |
| `Project.ipynb` | SpaceNet preprocessing and segmentation training, xBD classification training and combined image inference. |
| `Project_XBD_only.ipynb` | xBD classification training and combined inference using an existing segmentation checkpoint. It still contains optional SpaceNet preprocessing cells. |
| [Paper.pdf](Paper.pdf) | Author's research paper and reported experiments. |
| `tests/test_notebooks.py` | Small CPU regression fixtures without dataset or model downloads. |
| `LICENSE` | MIT license for the repository code. |

The notebooks are at the repository root. Data and trained `.pth` files are not included. The [Hugging Face model repository](https://huggingface.co/tugcantopaloglu/solarscanner-solars) is external to this checkout.

## Running the notebooks

```sh
git clone https://github.com/tugcantopaloglu/solarscanner-solars-paper-deep-learning.git
cd solarscanner-solars-paper-deep-learning
```

Open a notebook in Google Colab with a GPU runtime. Setup cells install PyTorch, Transformers, timm, Albumentations, segmentation-models-pytorch, OpenCV, Rasterio, GeoPandas, Shapely, Matplotlib and scikit-learn. They mount Google Drive and use `/content/drive/MyDrive/RemoteSensing/` paths. Adjust paths and installation commands for other environments. There is no locked full-training environment in this repository. ViT model/processor downloads use a pinned Hugging Face revision in the configuration cell; this maintenance pin does not identify the original training revision.

Configure `ROOT_DIR`, `OUTPUT_DIR`, `XBD_IMAGE_DIR`, `XBD_LABEL_DIR` and model paths before processing data. The expected structure is:

```text
datasets/
  spacenet/
    AOI..._Train/
      RGB-PanSharpen/*.tif
      geojson/buildings/buildings_*.geojson
  spacenet_preprocessed/
    images/*.png
    masks/*.png
  xbd/
    images/*_post_disaster.png
    labels/*_post_disaster.json
```

SpaceNet preprocessing writes color-stretched PNGs and corresponding masks. PNG dataset loading pairs image and mask filenames. xBD loading reads building polygons from `features.xy` and maps their `properties.subtype` to `no-damage`, `minor-damage`, `major-damage` or `destroyed`.

`DataDownload.ipynb` offers alternative historical recipes, not one validated download workflow. Obtain data through the providers' current access process and inspect archive layouts before extracting. Requester-pays S3 commands may require authenticated access and incur charges. Kaggle copies use the path returned by `kagglehub`; the obsolete signed download link and blanket process-termination command have been removed. No dataset downloads were performed during maintenance.

For `Project.ipynb`, run preprocessing only when needed, then dataset/model cells in order. Both xBD training and validation loaders are defined from a seeded 80/20 split. The full notebook trains segmentation for up to 500 epochs and classification for up to 100; review these settings before running. `MAX_FILES_SPACENET` and `MAX_FILES_XBD` are available for bounded data inspection in the full notebook. The xBD-only training cell currently requests the full dataset explicitly.

For `Project_XBD_only.ipynb`, skip the SpaceNet preprocessing and loader cells when working only on damage classification. Combined inference still requires a compatible trained building segmentation model. Load matching ResNet-50 U-Net and four-class ViT state dictionaries from `MODEL_BUILDING_PATH` and `MODEL_DAMAGE_PATH`. State-dict loads use `weights_only=True`.

The segmentation model retains sigmoid probability output for existing checkpoints and inference. Its focal loss now receives converted logits, and Dice loss is configured for probabilities. This correction affects future training and does not revalidate historical results or existing weights.

The inference cells may use a generated dummy image when no input file is available. Such output is a structural demonstration and does not validate damage assessment. Predictions require expert review before operational use.

## Paper-reported results

The following values come from `Paper.pdf`. They were not reproduced during maintenance. Historical saved notebook outputs and the PDF are retained.

| Task | Metric | Paper value and context |
| --- | --- | --- |
| SpaceNet building segmentation | IoU | 0.7664, best value from training logs at epoch 382 |
| xBD damage classification | Accuracy | 85.63%, validation result described in the paper |
| xBD damage classification | Macro F1 | 0.84, paper test classification table with 2,000 samples |

These values use different evaluation contexts. The checked-in segmentation loop selects checkpoints using training IoU. Classification splits are made at building-patch level, so buildings from one scene can appear in both train and validation sets; these splits do not establish unseen-scene generalization.

## Local regression checks

Maintenance fixtures use Python 3.13.1, PyTorch 2.13.0, torchvision 0.28.0 and segmentation-models-pytorch 0.5.0. With suitable CPU PyTorch/torchvision installations:

Create and activate a virtual environment for your shell, then install the fixture dependencies.

```sh
python -m venv .venv
python -m pip install torch torchvision numpy Pillow opencv-python-headless shapely segmentation-models-pytorch nbformat ipython
python -m unittest discover -s tests -v
```

The fixtures check notebook syntax, paired PNG loading, binary masks with and without a transform, xBD polygon crops, seeded nonempty loader creation in both training notebooks, compatible learning-rate scheduler construction, probability-based segmentation loss, checkpoint loading and inference report structure. Processor/model stubs keep external model and training boundaries offline.

## Acceptance requirements

- Data: authorized SpaceNet and xBD imagery/labels, matching archive layout, storage and dataset version are external prerequisites. Dataset creation eagerly stores cropped patches in memory; full-data RAM use has not been measured.
- Models: compatible learned U-Net/ViT checkpoints and required pretrained model caches are external. No published weights were downloaded or evaluated.
- Hardware: GPU training, CUDA compatibility, RAM/VRAM limits and drone or satellite inference quality require a separate hardware run. CPU fixture checks do not establish these.
- Experiments: seed 42 makes the new splits repeatable for fixed input order and versions, but does not recover historical splits or ensure deterministic GPU kernels. Record split indices, versions, data provenance and separate scene-level held-out results before claiming reproduction.

## Citation

```bibtex
@unpublished{topaloglu2025solars,
  author = {Tuğcan Topaloğlu},
  title = {{SolarScanner}: Two-Stage Deep Learning for Post-Disaster Building Damage Assessment},
  year = {2025},
  url = {https://github.com/tugcantopaloglu/solarscanner-solars-paper-deep-learning}
}
```

## License

Code is MIT licensed. SpaceNet, xBD and external models have their own access and license terms.
