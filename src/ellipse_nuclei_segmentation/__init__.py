try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._annotations import AnnotationManager, Ellipse2D
from ._plugin import NucleiAnnotatorWidget
from ._nd2_loader import ND2Loader

__all__ = [
    "NucleiAnnotatorWidget",
    "ND2Loader",
    "AnnotationManager",
    "Ellipse2D",
]
