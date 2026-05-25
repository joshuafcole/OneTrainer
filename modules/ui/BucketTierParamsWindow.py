from modules.ui.ConfigList import ConfigList
from modules.util.config.TrainConfig import TrainConfig
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk

# Strategies a tier can apply. Mirrors mgds.util.bucketRebalancing; "keep" is the
# implicit behavior for buckets above all tiers and is not selectable here.
STRATEGY_VALUES = ["drop", "donate", "borrow", "repeat"]
MODE_VALUES = ["move", "copy"]


class BucketTierParams(ConfigList):
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="aspect_ratio_bucket_min_tiers",
            from_external_file=False,
            add_button_text="add tier",
            is_full_width=True,
        )

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return BucketTierWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict[str, str]:
        return {"max_size": "", "strategy": "drop", "mode": "move"}

    def open_element_window(self, i, ui_state):
        pass


class BucketTierWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master=master, bg_color="transparent")
        self.element = element
        self.ui_state = UIState(self, element)
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
        close_button.grid(row=0, column=0, padx=5)

        # Max size: this tier applies to buckets whose population is below it.
        self.max_size = components.entry(
            self, 0, 1, self.ui_state, "max_size",
            tooltip="Apply this tier to aspect buckets with fewer than this many images. "
                    "Tiers are evaluated smallest-first; the first match wins.",
            wide_tooltip=True)
        self.max_size.bind("<FocusOut>", lambda _: save_command())
        self.max_size.configure(width=50)

        # Strategy for buckets below max_size.
        components.options(
            self, 0, 2, STRATEGY_VALUES, self.ui_state, "strategy",
            command=lambda _: save_command())

        # Mode (borrow only): move steals neighbor images; copy shares them.
        components.options(
            self, 0, 3, MODE_VALUES, self.ui_state, "mode",
            command=lambda _: save_command())

    def place_in_list(self):
        self.grid(row=self.i, column=0, padx=5, pady=5, sticky="new")


class BucketTierParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

        self.title("Aspect Bucket Min-Size Tiers")
        self.geometry("700x400")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        components.label(
            self, 0, 0,
            "Rebalance aspect buckets too small to fill a batch. Each tier maps a "
            "population threshold to a strategy:\n"
            "  drop = discard the images;  donate = merge into the nearest kept bucket;\n"
            "  borrow = pull neighbor images in to fill a batch;  repeat = oversample the bucket.\n"
            "Empty list = upstream behavior (sparse buckets are silently dropped).",
            wide_tooltip=True)

        self.frame = ctk.CTkFrame(self)
        self.frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=1)

        components.button(self, 2, 0, "ok", command=self.on_window_close)

        self.params = BucketTierParams(self.frame, self.train_config, self.ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def on_window_close(self):
        self.destroy()
