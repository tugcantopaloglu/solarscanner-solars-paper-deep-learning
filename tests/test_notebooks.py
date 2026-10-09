import ast
import contextlib
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import cv2
import nbformat
import numpy as np
from IPython.core.inputtransformer2 import TransformerManager
from PIL import Image
import segmentation_models_pytorch.losses as smp_losses
from shapely.wkt import loads as wkt_loads
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as tv_transforms


ROOT = Path(__file__).resolve().parents[1]
TRAINING_NOTEBOOKS = ("Project.ipynb", "Project_XBD_only.ipynb")


def tree_for(cell):
    return ast.parse(TransformerManager().transform_cell(cell.source))


def execute_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "notebook-fixture", "exec"), namespace)


class Processor:
    @classmethod
    def from_pretrained(cls, name, **kwargs):
        return cls()

    def __call__(self, images, return_tensors):
        values = torch.from_numpy(np.array(images).copy()).permute(2, 0, 1).float() / 255
        return SimpleNamespace(pixel_values=values.unsqueeze(0))


class SolarNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebooks = {name: nbformat.read(ROOT / name, as_version=4) for name in TRAINING_NOTEBOOKS}
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def namespace(self):
        import glob
        return {"os": os, "json": json, "glob": glob, "Image": Image, "torch": torch,
                "np": np, "Dataset": Dataset, "random": random, "cv2": cv2,
                "tv_transforms": tv_transforms, "wkt_loads": wkt_loads,
                "tqdm": lambda items, **kwargs: items, "SEED": 42,
                "DataLoader": lambda dataset, **kwargs: DataLoader(dataset, **dict(kwargs, num_workers=0, pin_memory=False)),
                "random_split": random_split, "ViTImageProcessor": Processor,
                "IMG_SIZE_SEG": (16, 16), "IMG_SIZE_CLS": (16, 16), "plt": MagicMock(),
                "MAX_FILES_XBD": None, "VIT_MODEL_REVISION": "offline-vit-revision", "smp_losses": smp_losses, "nn": torch.nn,
                "damage_to_int": {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3},
                "int_to_damage": {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"},
                "XBD_DAMAGE_CLASSES": ["no-damage", "minor-damage", "major-damage", "destroyed"]}

    def load_definition(self, notebook, name, namespace):
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            for node in ast.walk(tree_for(cell)):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
                    execute_nodes([node], namespace)
                    return namespace[name]
        self.fail(f"Definition missing: {name}")

    def create_xbd_fixture(self, patches):
        images = self.root / "xbd-images"
        labels = self.root / "xbd-labels"
        images.mkdir(exist_ok=True)
        labels.mkdir(exist_ok=True)
        Image.new("RGB", (16, 16), (20, 80, 150)).save(images / "scene_post_disaster.png")
        features = [{"wkt": "POLYGON ((2 2, 10 2, 10 10, 2 10, 2 2))",
                     "properties": {"subtype": "major-damage"}} for _ in range(patches)]
        (labels / "scene_post_disaster.json").write_text(json.dumps({"features": {"xy": features}}), encoding="utf-8")
        return images, labels

    def test_all_notebook_cells_compile(self):
        for path in ROOT.glob("*.ipynb"):
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    compile(TransformerManager().transform_cell(cell.source), str(path), "exec")

    def test_png_pairs_and_mask_tensors_with_optional_transform(self):
        images = self.root / "images"
        masks = self.root / "masks"
        images.mkdir()
        masks.mkdir()
        pixels = np.zeros((8, 8), dtype=np.uint8)
        pixels[2:6, 2:6] = 255
        Image.new("RGB", (8, 8), (40, 80, 120)).save(images / "paired.png")
        Image.fromarray(pixels).save(masks / "paired.png")
        Image.new("RGB", (8, 8)).save(images / "unpaired.png")
        (images / "notes.txt").write_text("not an image", encoding="utf-8")
        (images / "subdirectory.png").mkdir()
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                namespace = self.namespace()
                dataset_class = self.load_definition(notebook, "SpaceNetPreprocessedDataset", namespace)
                dataset = dataset_class(str(images), str(masks))
                self.assertEqual(dataset.image_files, ["paired.png"])
                image, mask = dataset[0]
                self.assertEqual(tuple(image.shape), (3, 8, 8))
                self.assertEqual(tuple(mask.shape), (1, 8, 8))
                self.assertEqual(mask.dtype, torch.float32)
                self.assertEqual(set(mask.unique().tolist()), {0.0, 1.0})
                def transform(image, mask):
                    return {"image": torch.from_numpy(image.transpose(2, 0, 1).copy()), "mask": torch.from_numpy(mask)}
                dataset = dataset_class(str(images), str(masks), transform=transform)
                self.assertEqual(tuple(dataset[0][1].shape), (1, 8, 8))
                (masks / "paired.png").write_text("corrupt", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Could not read mask"):
                    dataset[0]
                Image.fromarray(pixels).save(masks / "paired.png")

    def test_xbd_crop_and_processor_contract(self):
        images, labels = self.create_xbd_fixture(2)
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                namespace = self.namespace()
                dataset_class = self.load_definition(notebook, "XBDDamageDataset", namespace)
                dataset = dataset_class(str(images), str(labels), namespace["damage_to_int"], processor=Processor())
                self.assertEqual(len(dataset), 2)
                image, label = dataset[0]
                self.assertEqual(tuple(image.shape), (3, 8, 8))
                self.assertEqual(label.item(), 2)
                self.assertEqual(label.dtype, torch.long)

    def test_xbd_crops_clip_to_image_and_reject_outside_polygons(self):
        images, labels = self.create_xbd_fixture(0)
        features = [
            {"wkt": "POLYGON ((-4 2, 10 2, 10 10, -4 10, -4 2))", "properties": {"subtype": "minor-damage"}},
            {"wkt": "POLYGON ((20 2, 28 2, 28 10, 20 10, 20 2))", "properties": {"subtype": "destroyed"}},
        ]
        (labels / "scene_post_disaster.json").write_text(json.dumps({"features": {"xy": features}}), encoding="utf-8")
        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                namespace = self.namespace()
                dataset_class = self.load_definition(notebook, "XBDDamageDataset", namespace)
                dataset = dataset_class(str(images), str(labels), namespace["damage_to_int"], processor=Processor())
                self.assertEqual(len(dataset), 1)
                image, label = dataset[0]
                self.assertEqual(tuple(image.shape), (3, 8, 10))
                self.assertEqual(label.item(), 1)

    def test_both_notebooks_create_repeatable_nonempty_xbd_loaders(self):
        for patches in (1, 2, 6):
            images, labels = self.create_xbd_fixture(patches)
            for name, notebook in self.notebooks.items():
                with self.subTest(notebook=name, patches=patches):
                    namespace = self.namespace()
                    namespace.update(XBD_IMAGE_DIR=str(images), XBD_LABEL_DIR=str(labels))
                    self.load_definition(notebook, "XBDDamageDataset", namespace)
                    marker = "class XBDDamageDataset" if name == "Project.ipynb" else "full_xbd_dataset = None"
                    cell = next(c for c in notebook.cells if marker in c.source)
                    nodes = [n for n in tree_for(cell).body if not isinstance(n, (ast.Import, ast.ImportFrom))]
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        execute_nodes(nodes, namespace)
                    loaders = [namespace["xbd_train_loader"], namespace["xbd_val_loader"]]
                    if patches == 1:
                        self.assertEqual(loaders, [None, None])
                        continue
                    indices = [loader.dataset.indices for loader in loaders]
                    self.assertTrue(all(indices))
                    self.assertEqual(sorted(sum(indices, [])), list(range(patches)))
                    with contextlib.redirect_stdout(io.StringIO()):
                        execute_nodes(nodes, namespace)
                    repeated = [namespace[k].dataset.indices for k in ("xbd_train_loader", "xbd_val_loader")]
                    self.assertEqual(indices, repeated)

    def test_schedulers_construct_with_current_torch(self):
        optimizer = torch.optim.AdamW(torch.nn.Linear(2, 2).parameters())
        namespace = {"optim": torch.optim, "optimizer_seg": optimizer, "optimizer_cls": optimizer}
        found = 0
        for notebook in self.notebooks.values():
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                for node in ast.walk(tree_for(cell)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ReduceLROnPlateau":
                        scheduler = eval(compile(ast.Expression(node), "scheduler", "eval"), namespace)
                        scheduler.step(0.5)
                        found += 1
        self.assertEqual(found, 3)

    def test_combined_loss_accepts_probability_outputs(self):
        namespace = self.namespace()
        loss_class = self.load_definition(self.notebooks["Project.ipynb"], "CombinedLoss", namespace)
        criterion = loss_class()
        target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        correct = torch.tensor([[[[0.99, 0.01], [0.99, 0.01]]]], requires_grad=True)
        wrong = torch.tensor([[[[0.01, 0.99], [0.01, 0.99]]]])
        self.assertLess(criterion(correct, target).item(), 0.003)
        self.assertGreater(criterion(wrong, target).item(), 3.0)
        criterion(correct, target).backward()
        self.assertTrue(torch.isfinite(correct.grad).all())
        perfect = target.clone().requires_grad_(True)
        self.assertTrue(torch.isfinite(criterion(perfect, target)))

    def test_state_dict_loads_are_restricted(self):
        found = 0
        for notebook in self.notebooks.values():
            for cell in notebook.cells:
                if cell.cell_type != "code":
                    continue
                for node in ast.walk(tree_for(cell)):
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load":
                        if isinstance(node.func.value, ast.Name) and node.func.value.id == "torch":
                            values = {kw.arg: kw.value for kw in node.keywords}
                            self.assertIs(values["weights_only"].value, True)
                            found += 1
        self.assertEqual(found, 6)
        path = self.root / "fixture.pth"
        torch.save({"weight": torch.ones(2)}, path)
        self.assertTrue(torch.equal(torch.load(path, weights_only=True)["weight"], torch.ones(2)))

    def test_inference_reports_for_detected_and_empty_masks(self):
        class Segmenter(torch.nn.Module):
            def __init__(self, detect):
                super().__init__()
                self.detect = detect

            def forward(self, images):
                mask = torch.zeros(1, 1, 16, 16)
                if self.detect:
                    mask[:, :, 2:10, 2:10] = 1
                return mask

        class Classifier(torch.nn.Module):
            def forward(self, pixel_values):
                return SimpleNamespace(logits=torch.tensor([[0.0, 0.0, 1.0, 0.0]]))

        for name, notebook in self.notebooks.items():
            with self.subTest(notebook=name):
                namespace = self.namespace()
                for definition in ("mask_to_bounding_boxes", "crop_patches_from_image_pil", "run_inference_pipeline"):
                    self.load_definition(notebook, definition, namespace)
                transform = lambda image: {"image": torch.from_numpy(image.transpose(2, 0, 1).copy()).float()}
                pipeline = namespace["run_inference_pipeline"]
                image = Image.new("RGB", (16, 16), (40, 80, 120))
                with contextlib.redirect_stdout(io.StringIO()):
                    visualization, report = pipeline(image, Segmenter(True), Classifier(), transform, Processor(), torch.device("cpu"), min_building_area_px=1)
                    empty_visualization, empty_report = pipeline(image, Segmenter(False), Classifier(), transform, Processor(), torch.device("cpu"), min_building_area_px=1)
                self.assertEqual(visualization.shape, (16, 16, 3))
                self.assertEqual(report["classified_patches"], 1)
                self.assertEqual(report["damage_counts_by_class_str"]["major-damage"], 1)
                self.assertEqual(report["percentage_heavily_damaged_of_classified"], 100)
                self.assertEqual(empty_visualization.shape, (16, 16, 3))
                self.assertEqual(empty_report["total_buildings"], 0)


if __name__ == "__main__":
    unittest.main()
