import sys
from pathlib import Path
import importlib
from PyQt5.QtCore import (Qt, QObject, QSize, pyqtSlot,
                          pyqtSignal, QCoreApplication)
from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QRadioButton,
                             QCheckBox, QMessageBox, QApplication, QDialog,
                             QFileDialog, QDialogButtonBox, QLabel, QFrame,
                             QAction, QVBoxLayout, QShortcut, QTextEdit,
                             QDesktopWidget, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QHBoxLayout, QVBoxLayout,
                             QFormLayout, QGridLayout, QInputDialog, QLineEdit)
from PyQt5.QtGui import (QImage, QPixmap, QPalette, QBrush,
                         QColor, QFont, QKeySequence)
add_paths = [str(Path(__file__).resolve().parent.parent.parent.parent/'source'),
             str(Path(__file__).resolve().parent.parent.parent/'source')]
for p in add_paths:
    if p not in sys.path:
        sys.path.append(p)

class XyloInfModel(QMainWindow):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowStaysOnTopHint)
        self.selection = '*'
        spec_file = Path(__file__).resolve().parent/'model_specification.txt'
        with open(spec_file, 'r') as fh:
            first = fh.readlines()[0]
            if first.strip():
                self.selection = first.strip()
        print(f"self.selection: {self.selection}")
