import os
import copy
from pathlib import Path
from typing import List, Optional

import click
import torch
import tqdm

from dataclasses import dataclass, field
from jinja2 import Environment, FileSystemLoader, select_autoescape
from PIL import Image

from surya.detection import batch_text_detection
from surya.layout import batch_layout_detection
from surya.model.detection import segformer
from surya.model.recognition.model import load_model
from surya.model.recognition.processor import load_processor
from surya.model.detection.segformer import load_model as seg_load_model, load_processor as seg_load_processor
from surya.input.processing import open_pdf, get_page_images, slice_polys_from_image
from surya.postprocessing.text import sort_text_lines
from surya.recognition import batch_recognition
from surya.settings import settings
from surya.schema import TextLine, OCRResult, PolygonBox

_env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape()
)
TEMPLATE = _env.get_template("alto.xml")


@dataclass
class KrakenLine:
    content: str
    _obj: PolygonBox
    confidence: float

    def __hash__(self):
        return hash(tuple([tuple(point) for point in self._obj.polygon]))

    @property
    def polygon(self) -> Optional[List[List[float]]]:
        return self._obj.polygon if self._obj else None

    @property
    def h(self) -> Optional[int]:
        return int(self._obj.bbox[0]) if self._obj else None

    @property
    def v(self) -> Optional[int]:
        return int(self._obj.bbox[1]) if self._obj else None

    @property
    def width(self) -> Optional[int]:
        return int(self._obj.width) if self._obj else None

    @property
    def height(self) -> Optional[int]:
        return int(self._obj.height) if self._obj else None

    @property
    def points(self) -> Optional[str]:
        return " ".join([f"{int(a)} {int(b)}" for a, b in self.polygon]) if self.polygon else ""

    @property
    def baseline(self) -> Optional[str]:
        if not self._obj:
            return
        reduction: int = self.height // 10  # factor
        return f"{int(self.polygon[3][0])} {int(self.polygon[3][1]-reduction)} " \
               f"{int(self.polygon[2][0])} {int(self.polygon[2][1]-reduction)}"


@dataclass
class KrakenRegion:
    name: str
    idx: int
    _obj: Optional[PolygonBox] = None
    lines: List[KrakenLine] = field(default_factory=list)

    @property
    def polygon(self) -> Optional[List[List[float]]]:
        return self._obj.polygon if self._obj else None

    @property
    def h(self) -> Optional[int]:
        return int(self._obj.bbox[0]) if self._obj else None

    @property
    def v(self) -> Optional[int]:
        return int(self._obj.bbox[1]) if self._obj else None

    @property
    def width(self) -> Optional[int]:
        return int(self._obj.width) if self._obj else None

    @property
    def height(self) -> Optional[int]:
        return int(self._obj.height) if self._obj else None

    @property
    def points(self):
        return " ".join([f"{int(a)} {int(b)}" for a, b in self.polygon]) if self.polygon else ""


@dataclass
class KrakenPage:
    width: int
    height: int
    name: Optional[str] = None
    regions: List[KrakenRegion] = field(default_factory=list)


def _custom_ocr(images, det_predictions, langs, rec_model, rec_processor) -> List[OCRResult]:
    slice_map = []
    all_slices = []
    all_langs = []

    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        polygons = [p.polygon for p in det_pred.bboxes]
        slices = slice_polys_from_image(image, polygons)
        slice_map.append(len(slices))
        all_slices.extend(slices)
        all_langs.extend([lang] * len(slices))

    rec_predictions, confidence_scores = batch_recognition(all_slices, all_langs, rec_model, rec_processor)

    predictions_by_image = []
    slice_start = 0
    for idx, (image, det_pred, lang) in enumerate(zip(images, det_predictions, langs)):
        slice_end = slice_start + slice_map[idx]
        image_lines = rec_predictions[slice_start:slice_end]
        line_confidences = confidence_scores[slice_start:slice_end]
        slice_start = slice_end

        assert len(image_lines) == len(det_pred.bboxes)

        lines = []
        for text_line, confidence, bbox in zip(image_lines, line_confidences, det_pred.bboxes):
            lines.append(TextLine(
                text=text_line,
                polygon=bbox.polygon,
                bbox=bbox.bbox,
                confidence=confidence
            ))

        lines = sort_text_lines(lines)

        predictions_by_image.append(OCRResult(
            text_lines=lines,
            languages=lang,
            image_bbox=det_pred.image_bbox
        ))

    return predictions_by_image


def on_image(images: List[Image.Image], langs: List[List[str]]) -> List[KrakenPage]:
    # Load models
    det_processor, det_model = segformer.load_processor(), segformer.load_model()
    rec_model, rec_processor = load_model(), load_processor()
    lay_model = seg_load_model(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)
    lay_processor = seg_load_processor(checkpoint=settings.LAYOUT_MODEL_CHECKPOINT)

    # Do line detections
    line_predictions = batch_text_detection(images, det_model, det_processor)

    if det_model.device == "cuda":
        torch.cuda.empty_cache()  # Empty cache from first model run

    # Do OCR
    predictions = _custom_ocr(
        images,
        det_predictions=line_predictions, langs=langs, rec_model=rec_model, rec_processor=rec_processor
    )

    # Do layout predictions
    layout_predictions = batch_layout_detection(images, lay_model, lay_processor, copy.deepcopy(line_predictions))
    # from collections import namedtuple
    # x = namedtuple("x", ["bboxes"])
    # layout_predictions = [x("")] * len(images)

    # And now we need to merge predictions within layout :)
    out = []
    for lay_pred, text_pred, image in zip(layout_predictions, predictions, images):
        page = KrakenPage(width=image.width, height=image.height)
        unused_lines = [
            KrakenLine(line.text, _obj=line, confidence=line.confidence)
            for line in text_pred.text_lines
            if line.text.strip()
        ]
        used_line = set()
        for idx, region in enumerate(lay_pred.bboxes):
            kreg = KrakenRegion(name=region.label, _obj=region, idx=idx)
            for line in unused_lines:
                if line not in used_line and line._obj.intersection_pct(region) >= .8:
                    kreg.lines.append(line)
                    used_line.add(line)
            if kreg.lines:
                page.regions.append(kreg)
        # Deal with undispatched

        unused_lines = [line for line in unused_lines if line not in used_line]
        if unused_lines:
            page.regions.append(KrakenRegion("empty", _obj=None, lines=unused_lines, idx=-1))

        out.append(page)

    return out


@click.command("run")
@click.argument("source", type=click.Path(exists=True, file_okay=True, dir_okay=False), nargs=-1)
@click.option("-d", "--destination", type=click.Path(file_okay=False, dir_okay=True), help="Output directory",
              default="ocr-output")
@click.option("-f", "--format", default="image", type=click.Choice(["image", "pdf"]))
@click.option("-l", "--langs", default=("en", ), help="Lang that needs to be recognized", multiple=True)
def run(source, destination, format, langs: List[str]):
    if format == "image":
        images = [Image.open(img) for img in source]
    else:
        if len(source) > 1:
            raise ValueError("Unable to process more than one PDF at the moment")
        doc = open_pdf(source[0])
        page_count = len(doc)
        page_indices = list(range(page_count))

        images = get_page_images(doc, page_indices)
        doc.close()
        source = list(source) * len(images)

    results = on_image(images, [langs] * len(images))

    os.makedirs(destination, exist_ok=True)

    for idx, (img, page, src) in tqdm.tqdm(enumerate(zip(images, results, source)), desc="Exporting results", total=len(source)):
        base = f"{destination}/{Path(src).stem}-{idx:04}"
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert("RGB")
        img.save(f"{base}.jpg")
        with open(f"{base}.xml", "w") as f:
            f.write(
                TEMPLATE.render(
                    regions=page.regions,
                    page=page,
                    labels=sorted(list(set([region.name for region in page.regions]))),
                    filename=f"{os.path.basename(base)}.jpg"
                )
            )


if __name__ == "__main__":
    run()
