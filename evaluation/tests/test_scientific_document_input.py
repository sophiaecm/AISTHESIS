"""Offline synthetic PDFs; structural correctness, not scientific interpretation."""
from dataclasses import FrozenInstanceError, replace
from hashlib import sha256
import json
from pathlib import Path

import pymupdf
import pytest

from fifth_layer.scientific_document import (ScientificDocumentParser, ScientificDocumentConfig,
    ScientificDocumentState, ScientificPage, DocumentElement, ExtractionWarning, DocumentInputError)


def pdf_file(tmp_path, texts=('First page text', 'Second page text'), *, image=False, encrypted=False, rotation=0):
    path = tmp_path / 'paper.pdf'
    with pymupdf.open() as doc:
        for text in texts:
            page = doc.new_page(width=300, height=400)
            if text:
                page.insert_text((25, 40), text)
            if image:
                page.insert_image(pymupdf.Rect(20, 80, 70, 130), stream=b'P6\n2 2\n255\n' + b'\xff\x00\x00' * 4)
            page.set_rotation(rotation)
        options = {'encryption': pymupdf.PDF_ENCRYPT_AES_256, 'owner_pw': 'owner', 'user_pw': 'secret'} if encrypted else {}
        doc.save(path, no_new_id=True, **options)
    return path


def parse(path, **config):
    return ScientificDocumentParser(ScientificDocumentConfig(**config)).parse(path)


def test_valid_pdf_page_count_boundaries_and_text(tmp_path):
    state = parse(pdf_file(tmp_path))
    assert state.page_count == 2
    assert tuple(p.page_index for p in state.pages) == (0, 1)
    assert state.pages[0].text == 'First page text\n'
    assert state.pages[1].text == 'Second page text\n'
    assert all(e.page_index == i for i, page in enumerate(state.pages) for e in page.elements)


def test_deterministic_ids_and_serialization(tmp_path):
    path = pdf_file(tmp_path)
    a, b = parse(path), parse(path)
    assert a == b and a.document_id == b.document_id
    assert [e.element_id for e in a.elements] == [e.element_id for e in b.elements]
    assert a.to_json() == b.to_json()
    assert json.loads(a.to_json()) == a.to_dict()


def test_text_blocks_and_bbox(tmp_path):
    state = parse(pdf_file(tmp_path, ('Some text',)))
    element = state.elements[0]
    assert element.element_type == 'text_block'
    assert element.bbox[0] == 25 and element.bbox[2] > 25
    assert state.pages[0].width == 300 and state.pages[0].height == 400
    assert element.source == 'PyMuPDF' and element.confidence is None


def test_empty_page_is_explicit(tmp_path):
    state = parse(pdf_file(tmp_path, ('',)))
    assert state.pages[0].extraction_status == 'empty'
    assert state.pages[0].text == '' and not state.elements
    assert 'no_text_extracted_ocr_not_run' in {w.code for w in state.warnings}


@pytest.mark.parametrize('suffix', ['.txt', '.png', '.docx'])
def test_unsupported_extension(tmp_path, suffix):
    with pytest.raises(DocumentInputError, match='unsupported_extension'):
        parse(tmp_path / ('paper' + suffix))


def test_missing_file(tmp_path):
    with pytest.raises(DocumentInputError) as error:
        parse(tmp_path / 'missing.pdf')
    assert error.value.code == 'file_not_found'


def test_corrupted_pdf(tmp_path):
    path = tmp_path / 'corrupt.pdf'
    path.write_bytes(b'not a PDF')
    with pytest.raises(DocumentInputError, match='corrupted_pdf'):
        parse(path)


def test_encrypted_pdf(tmp_path):
    with pytest.raises(DocumentInputError, match='encrypted_pdf'):
        parse(pdf_file(tmp_path, encrypted=True))


def test_zero_page_pdf(tmp_path):
    path = tmp_path / 'empty.pdf'
    path.write_bytes(b'%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n'
        b'2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n'
        b'trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n')
    with pytest.raises(DocumentInputError, match='zero_page_pdf'):
        parse(path)


def test_immutable_state_and_no_unsafe_serialization(tmp_path):
    state = parse(pdf_file(tmp_path))
    with pytest.raises(FrozenInstanceError):
        state.document_id = 'changed'
    with pytest.raises(TypeError):
        state.source_metadata['filename'] = 'changed'
    exported = state.to_dict()
    exported['pages'][0]['elements'][0]['text_content'] = 'changed'
    assert state.pages[0].text == 'First page text\n'
    assert '%PDF' not in state.to_json()
    assert 'Document(' not in state.to_json()


def test_caller_metadata_and_table_detached():
    metadata = {'nested': {'detector': 'synthetic'}}
    cells = [['a', None], ['b', '1']]
    element = DocumentElement('e', 'table', 0, (0, 0, 20, 20),
                              metadata=metadata, table_cells=cells, classification_status='external')
    metadata['nested']['detector'] = 'changed'
    cells[0][0] = 'changed'
    assert element.metadata['nested']['detector'] == 'synthetic'
    assert element.table_cells == (('a', None), ('b', '1'))


def test_source_metadata_and_hash(tmp_path):
    path = pdf_file(tmp_path)
    state = parse(path)
    source = state.source_metadata
    assert source['filename'] == 'paper.pdf' and source['extension'] == '.pdf'
    assert source['source_reference'] == str(path.resolve())
    assert source['file_size'] == path.stat().st_size
    assert source['sha256'] == sha256(path.read_bytes()).hexdigest()
    assert state.document_id == 'document:' + source['sha256']
    assert source['parser'] == 'PyMuPDF' and source['parser_version'] == pymupdf.VersionBind


def test_image_region_extracted_without_figure_claim(tmp_path):
    state = parse(pdf_file(tmp_path, ('',), image=True))
    element = state.elements[0]
    assert element.element_type == 'image_region' and element.text_content is None
    assert element.bbox == (20, 80, 70, 130)
    assert element.metadata['pixel_width'] == 2
    assert element.confidence is None
    assert all(e.element_type != 'figure' for e in state.elements)


@pytest.mark.parametrize('kind', ['equation', 'figure', 'plot', 'table', 'caption', 'unknown',
    'title', 'section_heading', 'reference', 'footnote'])
def test_future_structural_types_representable(kind):
    e = DocumentElement('e', kind, 0, None, 'raw text', classification_status='external')
    state = ScientificDocumentState('doc', {}, (ScientificPage(0, 100, 100, (e,)),))
    assert state.to_dict()['pages'][0]['elements'][0]['element_type'] == kind
    assert 'epistemic_status' not in state.to_json()


def test_caption_heuristic_opt_in_and_labelled(tmp_path):
    path = pdf_file(tmp_path, ('Figure 1. Synthetic caption',))
    assert parse(path).elements[0].element_type == 'text_block'
    element = parse(path, caption_heuristic=True).elements[0]
    assert element.element_type == 'caption'
    assert element.classification_status == 'heuristic'
    assert element.metadata['classification_rule'] == 'caption_prefix_v0.1'
    assert 'figure_id' not in element.metadata


def test_equation_and_claims_not_interpreted(tmp_path):
    state = parse(pdf_file(tmp_path, ('E = mc^2\nWe claim a result.',)))
    assert all(e.element_type == 'text_block' for e in state.elements)
    assert 'We claim a result.' in state.pages[0].text
    assert all(e.confidence is None for e in state.elements)


@pytest.mark.parametrize('config,code', [({'max_file_bytes': 10}, 'file_size_limit'),
    ({'max_pages': 1}, 'page_count_limit'), ({'max_total_text': 1}, 'total_text_limit')])
def test_resource_rejection(tmp_path, config, code):
    with pytest.raises(DocumentInputError, match=code):
        parse(pdf_file(tmp_path), **config)


def test_page_text_limit_is_visible(tmp_path):
    state = parse(pdf_file(tmp_path), max_text_per_page=5)
    assert all(len(p.text) <= 5 for p in state.pages)
    assert all(p.extraction_status == 'partial' for p in state.pages)
    assert 'text_limit_truncated' in {w.code for w in state.warnings}


def test_element_limit_is_visible(tmp_path):
    state = parse(pdf_file(tmp_path, ('text',), image=True), max_elements_per_page=1)
    assert len(state.elements) == 1
    assert 'element_limit_truncated' in {w.code for w in state.warnings}


@pytest.mark.parametrize('value', [object(), b'pdf', {'tensor': [1.]}, {'text': 'x' * 17000}, ['not mapping']])
def test_unsafe_or_malformed_metadata_rejected(value):
    with pytest.raises(ValueError):
        DocumentElement('e', 'unknown', 0, None, metadata=value)


def test_partial_text_failure_preserves_other_structure(tmp_path, monkeypatch):
    path = pdf_file(tmp_path, ('text',), image=True)
    def fail(*args, **kwargs):
        raise RuntimeError('unsafe native message')
    monkeypatch.setattr(pymupdf.Page, 'get_text', fail)
    state = parse(path)
    assert state.pages[0].extraction_status == 'partial'
    assert state.elements[0].element_type == 'image_region'
    assert 'text_extraction_failed' in {w.code for w in state.warnings}
    assert 'unsafe native message' not in state.to_json()


def test_unreadable_page_keeps_page_boundary(tmp_path, monkeypatch):
    path = pdf_file(tmp_path)
    original = pymupdf.Document.load_page
    def load(doc, index):
        if index == 1:
            raise RuntimeError('unreadable')
        return original(doc, index)
    monkeypatch.setattr(pymupdf.Document, 'load_page', load)
    state = parse(path)
    assert state.page_count == 2
    assert state.pages[1].extraction_status == 'failed'
    assert state.pages[1].width is None
    assert ExtractionWarning('page_unreadable', 1) in state.warnings


def test_all_pages_failed_does_not_silently_succeed(tmp_path, monkeypatch):
    path = pdf_file(tmp_path)
    def fail(*args):
        raise RuntimeError('bad page')
    monkeypatch.setattr(pymupdf.Document, 'load_page', fail)
    with pytest.raises(DocumentInputError, match='no_readable_pages'):
        parse(path)


def test_different_pdf_different_identity(tmp_path):
    path = pdf_file(tmp_path, ('First',))
    first = parse(path)
    path.unlink()
    second = parse(pdf_file(tmp_path, ('Other',)))
    assert first.document_id != second.document_id


def test_relative_absolute_paths_and_config_stable(tmp_path, monkeypatch):
    path = pdf_file(tmp_path)
    monkeypatch.chdir(tmp_path)
    config = ScientificDocumentConfig()
    parser = ScientificDocumentParser(config)
    assert parser.parse('paper.pdf') == parser.parse(path.resolve())
    assert parser.config is config and config == ScientificDocumentConfig()


def test_rotated_page_coordinates_use_unrotated_dimensions(tmp_path):
    state = parse(pdf_file(tmp_path, ('text',), rotation=90))
    assert state.pages[0].rotation == 90
    assert (state.pages[0].width, state.pages[0].height) == (300, 400)
    assert state.elements[0].bbox[0] == 25


def test_warnings_and_page_order_are_canonical():
    a = ScientificPage(0, 100, 100, extraction_status='empty')
    b = ScientificPage(1, 100, 100, extraction_status='empty')
    warnings = [ExtractionWarning('empty', 1), ExtractionWarning('empty', 0)]
    state = ScientificDocumentState('doc', {}, (b, a), warnings)
    warnings.clear()
    assert tuple(p.page_index for p in state.pages) == (0, 1)
    assert len(state.warnings) == 2


def test_missing_dependency_is_structured(tmp_path, monkeypatch):
    path = pdf_file(tmp_path)
    import fifth_layer.scientific_document as module
    def missing(name):
        raise ImportError('dependency unavailable')
    monkeypatch.setattr(module.importlib, 'import_module', missing)
    with pytest.raises(DocumentInputError, match='pdf_dependency_unavailable'):
        parse(path)


@pytest.mark.parametrize('changes', [{'bbox': (2, 0, 1, 1)}, {'page_index': -1},
    {'classification_status': 'verified_fact'}, {'classification_status': 'heuristic'},
    {'confidence': 2}, {'table_cells': [['a'], ['b', 'c']]}])
def test_invalid_element_contracts(changes):
    values = dict(element_id='e', element_type='table', page_index=0, bbox=None)
    values.update(changes)
    with pytest.raises(ValueError):
        DocumentElement(**values)


def test_ingestion_does_not_mutate_existing_state(tmp_path):
    from evaluation.tests.test_common_evidence_state import build, item
    common = build(item())
    before = common.to_json()
    parse(pdf_file(tmp_path))
    assert common.to_json() == before


def test_source_handle_is_closed_and_ids_are_path_independent(tmp_path):
    path = pdf_file(tmp_path)
    first = parse(path)
    other = path.with_name('renamed.PDF')
    path.rename(other)
    second = parse(other)
    assert first.document_id == second.document_id
    assert tuple(e.element_id for e in first.elements) == tuple(e.element_id for e in second.elements)
    other.unlink()
    assert first.pages[0].text == 'First page text\n'


def test_bad_element_keeps_valid_page_content(tmp_path, monkeypatch):
    path = pdf_file(tmp_path, ('text',))
    original = pymupdf.Page.get_text
    def blocks(page, *args, **kwargs):
        return original(page, *args, **kwargs) + [(50, 0, 10, 20, 'invalid bbox', 99, 0)]
    monkeypatch.setattr(pymupdf.Page, 'get_text', blocks)
    state = parse(path)
    assert state.pages[0].extraction_status == 'partial'
    assert state.pages[0].text == 'text\n'
    assert ExtractionWarning('malformed_element_skipped', 0) in state.warnings


def test_image_extraction_failure_keeps_text(tmp_path, monkeypatch):
    path = pdf_file(tmp_path, ('text',))
    def fail(*args, **kwargs):
        raise RuntimeError('image decoder failure')
    monkeypatch.setattr(pymupdf.Page, 'get_image_info', fail)
    state = parse(path)
    assert state.pages[0].text == 'text\n'
    assert ExtractionWarning('image_region_extraction_failed', 0) in state.warnings


@pytest.mark.parametrize('changes', [{'max_pages': 0}, {'max_file_bytes': True}, {'caption_heuristic': 'yes'}])
def test_invalid_config_rejected(changes):
    with pytest.raises(ValueError):
        ScientificDocumentConfig(**changes)
