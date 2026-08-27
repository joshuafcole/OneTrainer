from modules.ui.BaseSliderTabView import BaseSliderPromptWidgetView, BaseSliderTabView
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.ui.CtkSliderAxesWindowView import CtkSliderAxesWindowView
from modules.ui.SliderTabController import SliderTabController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState

import customtkinter as ctk


class CtkSliderTabView(CtkConfigListView, BaseSliderTabView):

    def __init__(self, master, controller: SliderTabController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="slider_prompts",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add prompt pair",
            is_full_width=True,
            show_toggle_button=True,
        )

        settings_frame = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        settings_frame.grid(row=1, column=0, columnspan=6, sticky="new", pady=(8, 0))
        settings_frame.grid_columnconfigure(0, weight=1)
        self.build_slider_settings(settings_frame)

    def _open_axes_window(self) -> None:
        window = self.controller.open_axes_window(self, self.ui_state, CtkSliderAxesWindowView)
        self.wait_window(window)
        self.refresh_ui()

    def _set_block_visible(self, widget, visible: bool) -> None:
        if widget is None or not widget.winfo_exists():
            return
        if visible:
            widget.grid()
        else:
            widget.grid_remove()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkSliderPromptWidgetView(
            master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class CtkSliderPromptWidgetView(BaseSliderPromptWidgetView, ctk.CTkFrame):

    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        ctk.CTkFrame.__init__(self, master=master, corner_radius=10, bg_color="transparent")
        BaseSliderPromptWidgetView.__init__(self, ctk_components)

        self.element = element
        ui_state = CtkUIState(self, element)

        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(3, weight=1)

        bottom_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(3, weight=1)

        self.build_content(top_frame, bottom_frame, ui_state, i, save_command, remove_command,
                           clone_command, controller)

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
