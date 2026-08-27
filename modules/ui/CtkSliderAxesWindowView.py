from modules.ui.BaseSliderAxesWindowView import (
    GAIN_TOOLTIP,
    NAME_TOOLTIP,
    TARGET_TOOLTIP,
    BaseSliderAxesWindowView,
    BaseSliderAxisListView,
)
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.ui.SliderAxesWindowController import SliderAxesWindowController, SliderAxisListController
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon

import customtkinter as ctk


class CtkSliderAxisListView(CtkConfigListView, BaseSliderAxisListView):
    def __init__(self, master, controller: SliderAxisListController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="slider_axes",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add axis",
            is_full_width=True,
        )
        BaseSliderAxisListView.__init__(self, ctk_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkSliderAxisWidget(
            master, element, i, open_command, remove_command, clone_command, save_command, self.controller)


class CtkSliderAxisWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command, controller):
        super().__init__(master=master, corner_radius=10, bg_color="transparent")

        self.element = element
        self.ui_state = CtkUIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(5, weight=1)

        ctk_components.colored_icon_button(
            self, 0, 0, "X", "#C00000", lambda: remove_command(self.i))
        ctk_components.colored_icon_button(
            self, 0, 1, "+", "#00C000", lambda: clone_command(self.i, controller.randomize_uuid), padx=5)

        ctk_components.label(self, 0, 2, "enabled:")
        ctk_components.switch(self, 0, 3, self.ui_state, "enabled", command=save_command, width=40)

        ctk_components.label(self, 0, 4, "axis:", tooltip=NAME_TOOLTIP, wide_tooltip=True)
        self.name_entry = ctk_components.entry(self, 0, 5, self.ui_state, "name")
        self.name_entry.bind("<FocusOut>", lambda _: save_command())

        ctk_components.label(self, 0, 6, "gain:", tooltip=GAIN_TOOLTIP, wide_tooltip=True)
        self.gain_entry = ctk_components.entry(self, 0, 7, self.ui_state, "gain_k", width=70)
        self.gain_entry.bind("<FocusOut>", lambda _: save_command())

        ctk_components.label(self, 0, 8, "target:", tooltip=TARGET_TOOLTIP, wide_tooltip=True)
        ctk_components.switch(self, 0, 9, self.ui_state, "is_target", command=save_command, width=40)

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")


class CtkSliderAxesWindowView(BaseSliderAxesWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: SliderAxesWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseSliderAxesWindowView.__init__(self, ctk_components)

        self.title("Slider Coordinate Axes")
        self.geometry("900x620")
        self.resizable(True, True)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        frame.grid_columnconfigure(0, weight=1)

        expand_frame = ctk.CTkFrame(frame, bg_color="transparent")
        expand_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

        self.components.button(self, 1, 0, "ok", command=self.destroy)
        self.build_content(frame, controller, ui_state)
        CtkSliderAxisListView(expand_frame, SliderAxisListController(controller.config), ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
