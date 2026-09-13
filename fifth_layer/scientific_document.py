"""Local PDF structural ingestion only; document extraction is not understanding."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import sha256
import importlib
import json
from pathlib import Path
import re

from .world_model._structured import freeze, geometry, identifier, number
from .world_model.evidence import stable_id
from .world_model.physical_state import plain
from .world_model.common_evidence_state import bounded_plain


ELEMENT_TYPES = ('text_block', 'title', 'section_heading', 'equation', 'figure',
                 'plot', 'table', 'caption', 'reference', 'footnote', 'image_region', 'unknown')


def _metadata(value):
    if not isinstance(value, Mapping):
        raise ValueError('metadata must be a mapping')
    data = bounded_plain(value)
    if len(json.dumps(data, sort_keys=True, allow_nan=False)) > 16384:
        raise ValueError('metadata exceeds 16 KiB limit')
    return freeze(data)


def _index(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')


@dataclass(frozen=True)
class ExtractionWarning:
    code: str
    page_index: int | None = None

    def __post_init__(self):
        identifier(self.code, 'warning code')
        if len(self.code) > 128:
            raise ValueError('warning code too long')
        if self.page_index is not None:
            _index(self.page_index, 'page_index')


class DocumentInputError(ValueError):
    """Structured failure codes without arbitrary native parser error messages."""
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DocumentElement:
    element_id: str
    element_type: str
    page_index: int
    bbox: tuple[float, float, float, float] | None
    text_content: str | None = None
    confidence: float | None = None
    source: str = 'external'
    classification_status: str = 'structural'
    metadata: Mapping = field(default_factory=dict)
    table_cells: tuple = ()

    def __post_init__(self):
        for name in ('element_id', 'source'):
            identifier(getattr(self, name), name)
        _index(self.page_index, 'page_index')
        if self.element_type not in ELEMENT_TYPES:
            raise ValueError('unsupported structural element type')
        if self.bbox is not None:
            geometry(self.bbox, 'bbox', 4)
            object.__setattr__(self, 'bbox', tuple(self.bbox))
        if self.text_content is not None and (type(self.text_content) is not str or len(self.text_content) > 16384):
            raise ValueError('text_content must be a bounded string or None')
        if self.confidence is not None:
            number(self.confidence, 'confidence', unit=True)
        if self.classification_status not in ('structural', 'heuristic', 'external', 'unknown'):
            raise ValueError('invalid structural classification status')
        object.__setattr__(self, 'metadata', _metadata(self.metadata))
        if self.classification_status == 'heuristic' and not self.metadata.get('classification_rule'):
            raise ValueError('heuristic classification requires a rule reference')
        if not isinstance(self.table_cells, (tuple, list)) or len(self.table_cells) > 200:
            raise ValueError('table requires bounded ordered rows')
        rows = []
        for row in self.table_cells:
            if not isinstance(row, (tuple, list)) or len(row) > 100:
                raise ValueError('table requires bounded ordered columns')
            if any(cell is not None and (type(cell) is not str or len(cell) > 1024) for cell in row):
                raise ValueError('table cells must be bounded strings or None')
            rows.append(tuple(row))
        if rows and (self.element_type != 'table' or len({len(row) for row in rows}) != 1):
            raise ValueError('table cells require rectangular table structure')
        if sum(len(cell or '') for row in rows for cell in row) > 16384:
            raise ValueError('table content too large')
        object.__setattr__(self, 'table_cells', tuple(rows))


@dataclass(frozen=True)
class ScientificPage:
    page_index: int
    width: float | None
    height: float | None
    elements: tuple[DocumentElement, ...] = ()
    extraction_status: str = 'complete'
    rotation: int = 0

    def __post_init__(self):
        _index(self.page_index, 'page_index')
        if self.extraction_status not in ('complete', 'empty', 'partial', 'failed'):
            raise ValueError('invalid page extraction status')
        if (self.width is None or self.height is None) and not (
                self.width is None and self.height is None and self.extraction_status == 'failed'):
            raise ValueError('dimensions may be unknown only for failed pages')
        for value in (self.width, self.height):
            if value is not None:
                number(value, 'page dimension', nonnegative=True)
                if value == 0:
                    raise ValueError('page dimension must be positive')
        if type(self.rotation) is not int or self.rotation not in (0, 90, 180, 270):
            raise ValueError('invalid page rotation')
        elements = tuple(self.elements)
        if any(not isinstance(e, DocumentElement) or e.page_index != self.page_index for e in elements):
            raise ValueError('elements must belong to the page')
        if len({e.element_id for e in elements}) != len(elements):
            raise ValueError('duplicate element ID')
        if self.extraction_status in ('empty', 'failed') and elements:
            raise ValueError('empty/failed page cannot carry elements')
        object.__setattr__(self, 'elements', elements)

    @property
    def text(self):
        return ''.join(e.text_content or '' for e in self.elements)


@dataclass(frozen=True)
class ScientificDocumentState:
    document_id: str
    source_metadata: Mapping
    pages: tuple[ScientificPage, ...]
    extraction_warnings: tuple[ExtractionWarning, ...] = ()
    schema_version: str = field(default='scientific-document-input-0.1', init=False)

    def __post_init__(self):
        identifier(self.document_id, 'document_id')
        object.__setattr__(self, 'source_metadata', _metadata(self.source_metadata))
        pages = tuple(self.pages)
        if not pages or any(not isinstance(p, ScientificPage) for p in pages):
            raise ValueError('document requires pages')
        pages = tuple(sorted(pages, key=lambda p: p.page_index))
        if tuple(p.page_index for p in pages) != tuple(range(len(pages))):
            raise ValueError('page indices must be contiguous and unique')
        ids = [e.element_id for p in pages for e in p.elements]
        if len(ids) != len(set(ids)):
            raise ValueError('duplicate document element ID')
        warnings = tuple(self.extraction_warnings)
        if any(not isinstance(w, ExtractionWarning) or (w.page_index is not None and w.page_index >= len(pages)) for w in warnings):
            raise ValueError('invalid document warnings')
        if any(p.extraction_status in ('empty', 'partial', 'failed') and not any(w.page_index == p.page_index for w in warnings) for p in pages):
            raise ValueError('incomplete/empty pages require explicit warnings')
        object.__setattr__(self, 'pages', pages)
        object.__setattr__(self, 'extraction_warnings', tuple(sorted(set(warnings), key=lambda w: (-1 if w.page_index is None else w.page_index, w.code))))

    @property
    def page_count(self):
        return len(self.pages)

    @property
    def elements(self):
        return tuple(e for p in self.pages for e in p.elements)

    @property
    def warnings(self):
        return self.extraction_warnings

    def to_dict(self):
        return {**plain(self), 'page_count': self.page_count}

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class ScientificDocumentConfig:
    max_file_bytes: int = 32 * 1024 * 1024
    max_pages: int = 200
    max_text_per_page: int = 100000
    max_total_text: int = 1000000
    max_elements_per_page: int = 2000
    caption_heuristic: bool = False

    def __post_init__(self):
        for name in ('max_file_bytes', 'max_pages', 'max_text_per_page', 'max_total_text', 'max_elements_per_page'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('resource limits must be positive integers')
        if type(self.caption_heuristic) is not bool:
            raise ValueError('caption_heuristic must be boolean')


class ScientificDocumentParser:
    def __init__(self, config=None):
        self.config = ScientificDocumentConfig() if config is None else config
        if not isinstance(self.config, ScientificDocumentConfig):
            raise ValueError('expected ScientificDocumentConfig')

    def parse(self, path):
        try:
            source = Path(path).resolve()
        except (TypeError, ValueError, OSError):
            raise DocumentInputError('invalid_source_path') from None
        if source.suffix.lower() != '.pdf':
            raise DocumentInputError('unsupported_extension')
        try:
            if source.stat().st_size > self.config.max_file_bytes:
                raise DocumentInputError('file_size_limit')
            with source.open('rb') as stream:
                payload = stream.read(self.config.max_file_bytes + 1)
        except FileNotFoundError:
            raise DocumentInputError('file_not_found') from None
        except OSError:
            raise DocumentInputError('source_unreadable') from None
        if len(payload) > self.config.max_file_bytes:
            raise DocumentInputError('file_size_limit')
        try:
            pdf = importlib.import_module('pymupdf')
        except ImportError:
            raise DocumentInputError('pdf_dependency_unavailable') from None
        digest = sha256(payload).hexdigest()
        identity = 'document:' + digest
        try:
            document = pdf.open(stream=payload, filetype='pdf')
        except Exception:
            raise DocumentInputError('corrupted_pdf') from None
        with document:
            if document.needs_pass or document.is_encrypted:
                raise DocumentInputError('encrypted_pdf')
            try:
                count = document.page_count
            except Exception:
                raise DocumentInputError('invalid_page_tree') from None
            if count == 0:
                raise DocumentInputError('zero_page_pdf')
            if count > self.config.max_pages:
                raise DocumentInputError('page_count_limit')
            pages, warnings, total = [], [], 0
            if document.is_repaired:
                warnings.append(ExtractionWarning('pdf_repaired_by_parser'))
            version = str(pdf.VersionBind)
            extraction_id = stable_id('pdf-extraction', identity, version, plain(self.config))
            for index in range(count):
                try:
                    page = document.load_page(index)
                    result, page_warnings = self._page(page, index, extraction_id, pdf)
                except Exception:
                    result = ScientificPage(index, None, None, extraction_status='failed')
                    page_warnings = [ExtractionWarning('page_unreadable', index)]
                pages.append(result)
                warnings.extend(page_warnings)
                total += len(result.text)
                if total > self.config.max_total_text:
                    raise DocumentInputError('total_text_limit')
            if all(p.extraction_status == 'failed' for p in pages):
                raise DocumentInputError('no_readable_pages')
        return ScientificDocumentState(identity, {
            'source_reference': str(source), 'filename': source.name, 'extension': source.suffix.lower(),
            'file_size': len(payload), 'sha256': digest, 'parser': 'PyMuPDF', 'parser_version': version,
            'extraction_id': extraction_id, 'config': plain(self.config),
            'coordinate_system': 'unrotated_page_points_top_left_origin',
            'role': 'structural_input_not_scientific_analysis'}, tuple(pages), tuple(warnings))

    def _page(self, page, index, extraction_id, pdf):
        warnings, records = [], []
        width, height, rotation = page.cropbox.width, page.cropbox.height, page.rotation
        text_failed = image_failed = False
        try:
            blocks = page.get_text('blocks', flags=pdf.TEXTFLAGS_BLOCKS & ~pdf.TEXT_PRESERVE_IMAGES, sort=True)
            remaining = self.config.max_text_per_page
            for block in blocks:
                if len(records) >= self.config.max_elements_per_page:
                    warnings.append(ExtractionWarning('element_limit_truncated', index))
                    break
                if len(block) < 7:
                    warnings.append(ExtractionWarning('malformed_text_block_skipped', index))
                    continue
                if block[6] != 0:
                    warnings.append(ExtractionWarning('unsupported_block_type_skipped', index))
                    continue
                text = block[4]
                if not text:
                    continue
                kept = text[:min(remaining, 16384)]
                if len(kept) < len(text):
                    warnings.append(ExtractionWarning('text_limit_truncated', index))
                if not kept:
                    continue
                remaining -= len(kept)
                kind, classification, metadata = 'text_block', 'structural', {'parser_block_index': block[5]}
                if self.config.caption_heuristic and re.match(r'^\s*(?:Fig(?:ure)?\.?|Table)\s+\d+\b', kept, re.IGNORECASE):
                    kind, classification = 'caption', 'heuristic'
                    metadata['classification_rule'] = 'caption_prefix_v0.1'
                records.append((tuple(block[:4]), kind, kept, classification, metadata))
        except Exception:
            text_failed = True
            warnings.append(ExtractionWarning('text_extraction_failed', index))
        try:
            for info in page.get_image_info(hashes=False, xrefs=False):
                if len(records) >= self.config.max_elements_per_page:
                    warnings.append(ExtractionWarning('element_limit_truncated', index))
                    break
                records.append((tuple(info['bbox']), 'image_region', None, 'structural',
                    {'parser_image_index': info['number'], 'pixel_width': info['width'], 'pixel_height': info['height']}))
        except Exception:
            image_failed = True
            warnings.append(ExtractionWarning('image_region_extraction_failed', index))
        records.sort(key=lambda r: (r[0][1], r[0][0], r[1], json.dumps(r[4], sort_keys=True)))
        elements = []
        for i, (bbox, kind, text, classification, metadata) in enumerate(records):
            try:
                elements.append(DocumentElement(stable_id('document-element', extraction_id, index, i, kind),
                    kind, index, bbox, text, source='PyMuPDF', classification_status=classification, metadata=metadata))
            except ValueError:
                warnings.append(ExtractionWarning('malformed_element_skipped', index))
        elements = tuple(elements)
        if not any(e.text_content for e in elements):
            warnings.append(ExtractionWarning('no_text_extracted_ocr_not_run', index))
        if not elements:
            warnings.append(ExtractionWarning('no_structural_elements_extracted', index))
        status = 'partial' if warnings else 'complete'
        if not elements:
            status = 'failed' if text_failed and image_failed else ('partial' if text_failed or image_failed else 'empty')
        return ScientificPage(index, width, height, elements, status, rotation), warnings
