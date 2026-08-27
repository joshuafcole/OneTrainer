from modules.ui.BaseBucketTierParamsWindowView import (
    MAX_SIZE_TOOLTIP,
    BaseBucketTierListView,
    BaseBucketTierParamsWindowView,
)
from modules.ui.BucketTierParamsWindowController import BucketTierListController, BucketTierParamsWindowController
from modules.ui.CtkConfigListView import CtkConfigListView
from modules.util.bucket_tiers import MODE_VALUES, STRATEGY_VALUES
from modules.util.ui import ctk_components
from modules.util.ui.CtkUIState import CtkUIState
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.validation_helpers import check_range

import customtkinter as ctk


class CtkBucketTierListView(CtkConfigListView, BaseBucketTierListView):
    def __init__(self, master, controller: BucketTierListController, ui_state):
        CtkConfigListView.__init__(
            self, master, controller, ui_state,
            attr_name="aspect_ratio_bucket_min_tiers",
            from_external_file=False,
            add_button_text="add tier",
            is_full_width=True,
        )
        BaseBucketTierListView.__init__(self, ctk_components)

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return CtkBucketTierWidget(master, element, i, open_command, remove_command, clone_command, save_command)


class CtkBucketTierWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master=master, bg_color="transparent")
        self.element = element
        self.ui_state = CtkUIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1, uniform=1)
        self.grid_columnconfigure(2, weight=1, uniform=1)
        self.grid_columnconfigure(3, weight=1, uniform=1)

        close_button = ctk.CTkButton(
            master=self,
            width=20,
            height=20,
            text="X",
            corner_radius=2,
            fg_color="#C00000",
            command=lambda: remove_command(self.i))
        close_button.grid(row=0, column=0)

        # Max Size
        self.max_size = ctk_components.entry(self, 0, 1, self.ui_state, "max_size",
                                             tooltip=MAX_SIZE_TOOLTIP, wide_tooltip=True,
                                             extra_validate=check_range(lower=0))
        self.max_size.bind("<FocusOut>", lambda _: save_command())
        self.max_size.configure(width=50)

        # Strategy
        ctk_components.options(self, 0, 2, STRATEGY_VALUES, self.ui_state, "strategy",
                               command=lambda _: save_command())

        # Mode
        ctk_components.options(self, 0, 3, MODE_VALUES, self.ui_state, "mode",
                               command=lambda _: save_command())

    def place_in_list(self):
        self.grid(row=self.i, column=0, padx=5, pady=5, sticky="new")


class CtkBucketTierParamsWindowView(BaseBucketTierParamsWindowView, ctk.CTkToplevel):
    def __init__(self, parent, controller: BucketTierParamsWindowController, ui_state, *args, **kwargs):
        ctk.CTkToplevel.__init__(self, parent, *args, **kwargs)
        BaseBucketTierParamsWindowView.__init__(self, ctk_components)

        self.title("Aspect Bucket Min-Size Tiers")
        self.geometry("800x560")
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
        CtkBucketTierListView(expand_frame, BucketTierListController(controller.config), ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))
