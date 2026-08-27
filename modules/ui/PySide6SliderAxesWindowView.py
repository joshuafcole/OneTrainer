from modules.ui.BaseSliderAxesWindowView import (
    GAIN_TOOLTIP,
    NAME_TOOLTIP,
    TARGET_TOOLTIP,
    BaseSliderAxesWindowView,
    BaseSliderAxisListView,
)
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.ui.SliderAxesWindowController import SliderAxesWindowController, SliderAxisListController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QDialog, QGridLayout, QPushButton, QScrollArea, QWidget


class PySide6SliderAxisListView(PySide6ConfigListView, BaseSliderAxisListView):
    def __init__(self, master, controller: SliderAxisListController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="slider_axes",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add axis",
            is_full_width=True,
        )
        BaseSliderAxisListView.__init__(self, pyside6_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6SliderAxisWidget(
            master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class PySide6SliderAxisWidget(QWidget):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        super().__init__(master)
        self.element = element
        self.ui_state = PySide6UIState(element)
        self.i = i
        self.save_command = save_command

        pyside6_components._layout(self).setColumnStretch(5, 1)

        pyside6_components.colored_icon_button(self, 0, 0, "X", "#C00000", lambda: remove_command(self.i))
        pyside6_components.colored_icon_button(
            self, 0, 1, "+", "#00C000", lambda: clone_command(self.i, controller.randomize_uuid))

        pyside6_components.label(self, 0, 2, "enabled:")
        pyside6_components.switch(self, 0, 3, self.ui_state, "enabled", command=save_command)

        pyside6_components.label(self, 0, 4, "axis:", tooltip=NAME_TOOLTIP, wide_tooltip=True)
        self.name_entry = pyside6_components.entry(self, 0, 5, self.ui_state, "name")
        self.name_entry.editingFinished.connect(save_command)

        pyside6_components.label(self, 0, 6, "gain:", tooltip=GAIN_TOOLTIP, wide_tooltip=True)
        self.gain_entry = pyside6_components.entry(self, 0, 7, self.ui_state, "gain_k", width=70)
        self.gain_entry.editingFinished.connect(save_command)

        pyside6_components.label(self, 0, 8, "target:", tooltip=TARGET_TOOLTIP, wide_tooltip=True)
        pyside6_components.switch(self, 0, 9, self.ui_state, "is_target", command=save_command)

    def configure_element(self):
        pass

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()


class PySide6SliderAxesWindowView(BaseSliderAxesWindowView, QDialog):
    def __init__(self, parent, controller: SliderAxesWindowController, ui_state):
        QDialog.__init__(self, parent)
        BaseSliderAxesWindowView.__init__(self, pyside6_components)

        # delete on close so entry widgets and the field validators they register globally are freed, not leaked
        self.finished.connect(self.deleteLater)

        self.setWindowTitle("Slider Coordinate Axes")
        self.resize(900, 660)

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
        # Must be assigned to an instance variable -- PySide6ConfigListView is not a QWidget,
        # so Qt won't keep it alive. Without this, the GC collects it and the button's
        # clicked signal loses its connection to __add_element.
        self._axis_list_view = PySide6SliderAxisListView(
            expand_frame, SliderAxisListController(controller.config), ui_state)

        outer.addWidget(scroll, 0, 0)

        ok = QPushButton("ok", self)
        ok.clicked.connect(self.accept)
        outer.addWidget(ok, 1, 0)
