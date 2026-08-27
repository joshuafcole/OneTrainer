from modules.ui.BaseBucketTierParamsWindowView import (
    MAX_SIZE_TOOLTIP,
    BaseBucketTierListView,
    BaseBucketTierParamsWindowView,
)
from modules.ui.BucketTierParamsWindowController import BucketTierListController, BucketTierParamsWindowController
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.util.bucket_tiers import MODE_VALUES, STRATEGY_VALUES
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState
from modules.util.ui.validation_helpers import check_range

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QScrollArea, QWidget


class PySide6BucketTierListView(PySide6ConfigListView, BaseBucketTierListView):
    def __init__(self, master, controller: BucketTierListController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="aspect_ratio_bucket_min_tiers",
            from_external_file=False,
            add_button_text="add tier",
            is_full_width=True,
        )
        BaseBucketTierListView.__init__(self, pyside6_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6BucketTierWidget(master, element, i, open_command, remove_command, clone_command, save_command)


class PySide6BucketTierWidget(QWidget):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master)
        self.element = element
        self.ui_state = PySide6UIState(element)
        self.i = i
        self.save_command = save_command

        lo = pyside6_components._layout(self)
        lo.setColumnStretch(1, 1)
        lo.setColumnStretch(2, 1)
        lo.setColumnStretch(3, 1)

        pyside6_components.colored_icon_button(self, 0, 0, "X", "#C00000", lambda: remove_command(self.i))

        # Max Size
        self.max_size = pyside6_components.entry(self, 0, 1, self.ui_state, "max_size",
                                                 tooltip=MAX_SIZE_TOOLTIP, wide_tooltip=True, width=50,
                                                 extra_validate=check_range(lower=0))
        self.max_size.editingFinished.connect(save_command)

        # Strategy
        pyside6_components.options(self, 0, 2, STRATEGY_VALUES, self.ui_state, "strategy",
                                   command=lambda _: save_command())

        # Mode
        pyside6_components.options(self, 0, 3, MODE_VALUES, self.ui_state, "mode",
                                   command=lambda _: save_command())

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()


class PySide6BucketTierParamsWindowView(BaseBucketTierParamsWindowView, QDialog):
    def __init__(self, parent, controller: BucketTierParamsWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseBucketTierParamsWindowView.__init__(self, pyside6_components)

        # delete on close so entry widgets and the field validators they register globally are freed, not leaked
        self.finished.connect(self.deleteLater)

        self.setWindowTitle("Aspect Bucket Min-Size Tiers")
        self.resize(800, 620)

        outer = QGridLayout(self)
        outer.setRowStretch(0, 1)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        inner = QWidget()
        scroll.setWidget(inner)
        inner_lo = pyside6_components._layout(inner)
        inner_lo.setColumnStretch(0, 1)

        self.build_content(inner, controller, ui_state)

        expand_frame = QWidget(inner)
        inner_lo.addWidget(expand_frame, inner_lo.rowCount(), 0, 1, 2)
        # Must be assigned to an instance variable — PySide6ConfigListView is not a QWidget,
        # so Qt won't keep it alive. Without this, the GC collects it and the button's
        # clicked signal loses its connection to __add_element.
        self._tier_list_view = PySide6BucketTierListView(expand_frame, BucketTierListController(controller.config), ui_state)

        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
