import os
# Must be set before any Qt imports so tests run headlessly
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope='session', autouse=True)
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app
