from modules.ui.BaseSliderTabView import BaseSliderPromptWidgetView, BaseSliderTabView
from modules.ui.PySide6ConfigListView import PySide6ConfigListView
from modules.ui.SliderTabController import SliderTabController
from modules.util.ui import pyside6_components
from modules.util.ui.PySide6UIState import PySide6UIState

from PySide6.QtWidgets import QWidget


class PySide6SliderTabView(PySide6ConfigListView, BaseSliderTabView):

    def __init__(self, master, controller: SliderTabController, ui_state):
        PySide6ConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="slider_prompts",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add prompt pair",
            is_full_width=True,
            show_toggle_button=True,
        )

        settings_frame = QWidget(self.top_frame)
        pyside6_components._layout(self.top_frame).addWidget(settings_frame, 1, 0, 1, 6)
        pyside6_components._layout(settings_frame).setColumnStretch(0, 1)
        self.build_slider_settings(settings_frame)

    def _set_block_visible(self, widget, visible: bool) -> None:
        if widget is None:
            return
        # the element list lives inside a scroll area; hiding the widget alone
        # would leave the empty scroller behind
        target = getattr(widget, "_scroll_area", widget)
        target.setVisible(visible)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return PySide6SliderPromptWidgetView(
            master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class PySide6SliderPromptWidgetView(BaseSliderPromptWidgetView, QWidget):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        QWidget.__init__(self, master)
        BaseSliderPromptWidgetView.__init__(self, pyside6_components)

        self.element = element
        ui_state = PySide6UIState(element)

        pyside6_components._layout(self).setColumnStretch(0, 1)

        top_frame = QWidget(self)
        pyside6_components._layout(top_frame).setColumnStretch(3, 1)
        pyside6_components._layout(self).addWidget(top_frame, 0, 0)

        bottom_frame = QWidget(self)
        pyside6_components._layout(bottom_frame).setColumnStretch(1, 1)
        pyside6_components._layout(bottom_frame).setColumnStretch(3, 1)
        pyside6_components._layout(self).addWidget(bottom_frame, 1, 0)

        self.build_content(top_frame, bottom_frame, ui_state, i, save_command, remove_command,
                           clone_command, controller)

    def place_in_list(self):
        pyside6_components._layout(self.parent()).addWidget(self, getattr(self, 'visible_index', self.i), 0)
        self.show()

    def destroy(self):
        self.deleteLater()
