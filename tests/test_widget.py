import pytest

from ellipse_nuclei_segmentation._plugin import NucleiAnnotatorWidget


@pytest.fixture
def viewer(make_napari_viewer):
    return make_napari_viewer()


def test_widget_creates(viewer):
    widget = NucleiAnnotatorWidget(viewer)
    assert widget is not None


def test_widget_initial_state(viewer):
    widget = NucleiAnnotatorWidget(viewer)
    assert widget.loader is None
    assert widget.current_frame == 0
    assert widget._drawing_mode is None


def test_annotation_list_empty_on_start(viewer):
    widget = NucleiAnnotatorWidget(viewer)
    assert widget.list_annotations.count() == 0


def test_start_and_cancel_annotation(viewer):
    widget = NucleiAnnotatorWidget(viewer)
    widget.btn_2d.setChecked(True)
    widget._start_annotation("ellipse_2d")
    assert widget._drawing_mode == "ellipse_2d"
    widget._cancel_annotation()
    assert widget._drawing_mode is None
