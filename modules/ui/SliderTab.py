from modules.ui.ConfigList import ConfigList
from modules.util.config.SliderConfig import SliderAxisConfig, SliderPromptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.ui import components
from modules.util.ui.ui_utils import set_window_icon
from modules.util.ui.UIState import UIState

import customtkinter as ctk


class SliderTab(ConfigList):
    """Concept-Sliders training config.

    The trained adapter is a plain LoRA/LoKr (configured on the LoRA tab); this
    tab drives the slider *objective*: the prompt triples that supply the frozen
    base's guidance direction, the guidance scale eta, and the synthetic x_t
    settings (prompt-pair has no image dataset).
    """

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="slider_prompts",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add prompt pair",
            is_full_width=True,
            show_toggle_button=True,
        )

        self.__build_settings_frame()

    def __build_settings_frame(self):
        frame = ctk.CTkFrame(self.top_frame, fg_color="transparent")
        frame.grid(row=1, column=0, columnspan=6, sticky="nsew", pady=(8, 0))
        frame.grid_columnconfigure(0, weight=1)

        # --- common (both regimes) ------------------------------------------
        # sigma range is sampled by both objectives; regime + a per-regime hint
        # sit above it. The two regime-specific blocks below are shown/hidden by
        # the regime selector's callback so only the relevant fields appear.
        common = ctk.CTkFrame(frame, fg_color="transparent")
        common.grid(row=0, column=0, sticky="new")
        common.grid_columnconfigure(1, weight=1)
        common.grid_columnconfigure(4, weight=1)

        components.label(common, 0, 0, "Regime",
                         tooltip="Prompt pair: text-only Concept Sliders (no image dataset); the frozen base supplies the guidance direction and x_t is generated on-manifold. Image (coordinate): a real image dataset (configure it on the Concepts tab) whose captions carry a declared-axis coordinate token like (distance:-2); the coordinate is stripped from the conditioning and becomes the training-time adapter multiplier m=k*value.")
        # created here so the callback (which fires on init) finds the blocks below
        self.__regime_blocks_built = False
        self.__prompt_frame = None
        self.__image_frame = None

        components.label(common, 1, 3, "Sigma min",
                         tooltip="Lower bound of the noise level sampled for the trained timestep.")
        components.entry(common, 1, 4, self.ui_state, "slider_sigma_min")
        components.label(common, 2, 3, "Sigma max",
                         tooltip="Upper bound of the noise level sampled for the trained timestep. Higher sigma (more noise) concentrates the attribute signal.")
        components.entry(common, 2, 4, self.ui_state, "slider_sigma_max")

        # per-regime hint line
        self.__regime_hint = components.label(common, 3, 0, "")
        self.__regime_hint.grid(row=3, column=0, columnspan=5, sticky="w", pady=(6, 0))

        # --- prompt-pair-only block -----------------------------------------
        prompt_frame = ctk.CTkFrame(frame, fg_color="transparent")
        prompt_frame.grid(row=1, column=0, sticky="new", pady=(6, 0))
        prompt_frame.grid_columnconfigure(1, weight=1)
        prompt_frame.grid_columnconfigure(4, weight=1)
        self.__prompt_frame = prompt_frame

        components.label(prompt_frame, 0, 0, "Guidance scale (eta)",
                         tooltip="Training-time guidance scale applied to v(c+) - v(c-). Paper uses ~3-4. Independent of the inference slider strength (the adapter multiplier).")
        components.entry(prompt_frame, 0, 1, self.ui_state, "slider_eta")
        components.label(prompt_frame, 1, 0, "Training strength",
                         tooltip="Adapter multiplier magnitude used for the trained passes (+/- this). The inference strength range is decoupled from this.")
        components.entry(prompt_frame, 1, 1, self.ui_state, "slider_strength")
        components.label(prompt_frame, 2, 0, "Symmetric",
                         tooltip="Also train the -strength pole toward v(c_t) - eta*delta, so the slider is linear around 0 (both poles learned).")
        components.switch(prompt_frame, 2, 1, self.ui_state, "slider_symmetric")
        components.label(prompt_frame, 0, 3, "Steps per epoch",
                         tooltip="Prompt-pair sliders have no dataset, so this sets how many synthetic training steps make up one epoch.")
        components.entry(prompt_frame, 0, 4, self.ui_state, "slider_steps_per_epoch")
        components.label(prompt_frame, 1, 3, "x_t anchor steps",
                         tooltip="Euler steps used to denoise random noise down to the sampled sigma, producing an on-manifold x_t (SDEdit). More steps = more on-manifold but slower per training step. 0 falls back to off-manifold Gaussian x_t.")
        components.entry(prompt_frame, 1, 4, self.ui_state, "slider_anchor_steps")
        components.label(prompt_frame, 3, 0, "Preservation prompts",
                         tooltip="Optional disentanglement set P (CS Eq. 8), pipe-delimited (a|b|c). The guidance direction is averaged over the preservation-augmented pairs to keep the slider from entangling unrelated attributes. Empty = bare positive/negative pair.")
        entry = components.entry(prompt_frame, 3, 1, self.ui_state, "slider_preservation_prompts")
        entry.grid(row=3, column=1, columnspan=4, sticky="ew")

        # --- image-coordinate-only block ------------------------------------
        image_frame = ctk.CTkFrame(frame, fg_color="transparent")
        image_frame.grid(row=1, column=0, sticky="new", pady=(6, 0))
        image_frame.grid_columnconfigure(2, weight=1)
        self.__image_frame = image_frame

        components.label(image_frame, 0, 0, "Coordinate axes",
                         tooltip="Declared axes for the coordinate-labeled image regime. Each axis name is the caption token key (e.g. 'distance' for (distance:-2)); it is stripped from the conditioning. Exactly one enabled axis must be the target axis -- its coordinate drives the multiplier m=k*value (k = the axis gain).")
        components.button(image_frame, 0, 1, "edit slider axes...", command=self.__open_axes_window)

        self.__regime_blocks_built = True
        # regime selector last, so its init callback finds the blocks above
        components.options_kv(common, 0, 1, [
            ("Prompt pair", SliderRegime.PROMPT_PAIR),
            ("Image (coordinate)", SliderRegime.IMAGE),
        ], self.ui_state, "slider_regime", command=self.__on_regime_change)

    __PROMPT_HINT = ("Prompt-pair regime: add positive/negative prompt triples below; no image dataset is used. "
                     "The Concepts tab is ignored.")
    __IMAGE_HINT = ("Image (coordinate) regime: configure a normal dataset on the Concepts tab whose captions carry "
                    "an axis token like (distance:-2). The prompt-pair list below is unused.")

    def __on_regime_change(self, regime):
        if not getattr(self, "_SliderTab__regime_blocks_built", False):
            return
        is_image = (regime == SliderRegime.IMAGE)

        if self.__prompt_frame is not None:
            self.__prompt_frame.grid_remove() if is_image else self.__prompt_frame.grid()
        if self.__image_frame is not None:
            self.__image_frame.grid() if is_image else self.__image_frame.grid_remove()
        # the tab body is the prompt-triple list; it is meaningless in IMAGE mode
        if getattr(self, "element_list", None) is not None and self.element_list.winfo_exists():
            self.element_list.grid_remove() if is_image else self.element_list.grid()
        if self.__regime_hint is not None and self.__regime_hint.winfo_exists():
            self.__regime_hint.configure(text=self.__IMAGE_HINT if is_image else self.__PROMPT_HINT)

    def __open_axes_window(self):
        window = SliderAxisWindow(self.master, self.train_config, self.ui_state)
        self.master.wait_window(window)

    def refresh_ui(self):
        if self.element_list is not None:
            self.element_list.destroy()
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()
        # _create_element_list re-grids the prompt list at row 1; re-apply the
        # regime-conditional visibility so IMAGE mode keeps it hidden.
        self.__on_regime_change(self.train_config.slider_regime)

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SliderPromptWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return SliderPromptConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass


class SliderPromptWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master=master, corner_radius=10, bg_color="transparent")

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(0, weight=1)

        top_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(3, weight=1)

        bottom_frame = ctk.CTkFrame(master=self, corner_radius=0, fg_color="transparent")
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.grid_columnconfigure(1, weight=1)
        bottom_frame.grid_columnconfigure(3, weight=1)

        # close button
        close_button = ctk.CTkButton(
            master=top_frame, width=20, height=20, text="X", corner_radius=2,
            fg_color="#C00000", command=lambda: remove_command(self.i),
        )
        close_button.grid(row=0, column=0)

        # clone button
        clone_button = ctk.CTkButton(
            master=top_frame, width=20, height=20, text="+", corner_radius=2,
            fg_color="#00C000", command=lambda: clone_command(self.i, self.__randomize_uuid),
        )
        clone_button.grid(row=0, column=1, padx=5)

        # enabled
        components.label(top_frame, 0, 2, "enabled:")
        enabled_switch = components.switch(top_frame, 0, 3, self.ui_state, "enabled", command=save_command)
        enabled_switch.configure(width=40)

        # target (neutral concept c_t)
        components.label(bottom_frame, 0, 0, "target (c_t):",
                         tooltip="The neutral concept the slider centers on. The base predicts v(c_t); the adapter shifts it toward the guidance direction.")
        components.entry(bottom_frame, 0, 1, self.ui_state, "target")

        # weight
        components.label(bottom_frame, 0, 2, "weight:",
                         tooltip="Relative sampling weight when several triples share one slider.")
        weight_entry = components.entry(bottom_frame, 0, 3, self.ui_state, "weight")
        weight_entry.configure(width=60)

        # positive (c+)
        components.label(bottom_frame, 1, 0, "positive (c+):",
                         tooltip="The enhanced-attribute conditioning. +strength pushes the slider toward this pole.")
        components.entry(bottom_frame, 1, 1, self.ui_state, "positive")

        # negative (c-)
        components.label(bottom_frame, 2, 0, "negative (c-):",
                         tooltip="The suppressed-attribute conditioning. -strength pushes the slider toward this pole.")
        components.entry(bottom_frame, 2, 1, self.ui_state, "negative")

    def __randomize_uuid(self, slider_config: SliderPromptConfig):
        slider_config.uuid = SliderPromptConfig.default_values().uuid
        return slider_config

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")


class SliderAxisList(ConfigList):
    """List editor for the coordinate-labeled image regime's declared axes."""

    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="slider_axes",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add axis",
            is_full_width=True,
        )

    def refresh_ui(self):
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SliderAxisWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return SliderAxisConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass


class SliderAxisWidget(ctk.CTkFrame):
    def __init__(self, master, element, i, open_command, remove_command, clone_command, save_command):
        super().__init__(master=master, corner_radius=10, bg_color="transparent")

        self.element = element
        self.ui_state = UIState(self, element)
        self.i = i
        self.save_command = save_command

        self.grid_columnconfigure(8, weight=1)

        # close
        close_button = ctk.CTkButton(
            master=self, width=20, height=20, text="X", corner_radius=2,
            fg_color="#C00000", command=lambda: remove_command(self.i),
        )
        close_button.grid(row=0, column=0, padx=5, pady=5)

        # clone
        clone_button = ctk.CTkButton(
            master=self, width=20, height=20, text="+", corner_radius=2,
            fg_color="#00C000", command=lambda: clone_command(self.i, self.__randomize_uuid),
        )
        clone_button.grid(row=0, column=1, padx=5)

        # enabled
        components.label(self, 0, 2, "enabled:")
        components.switch(self, 0, 3, self.ui_state, "enabled", command=save_command)

        # name (the caption token key)
        components.label(self, 0, 4, "axis:",
                         tooltip="The caption token key matched and stripped, e.g. 'distance' for (distance:-2).")
        name_entry = components.entry(self, 0, 5, self.ui_state, "name")
        name_entry.configure(width=110)

        # gain k
        components.label(self, 0, 6, "gain (k):",
                         tooltip="Global gain mapping the raw coordinate value onto the adapter multiplier: m = k*value.")
        gain_entry = components.entry(self, 0, 7, self.ui_state, "gain_k")
        gain_entry.configure(width=60)

        # is_target
        components.label(self, 0, 9, "target:",
                         tooltip="Exactly one enabled axis must be the target: its coordinate drives the slider multiplier. Other declared axes are only stripped from the caption (and reserved for future stratified sampling).")
        components.switch(self, 0, 10, self.ui_state, "is_target", command=save_command)

        # stratify (reserved; sampler is a fast-follow)
        components.label(self, 0, 11, "stratify:",
                         tooltip="Reserved for the balanced sampler (a fast-follow): when set, this axis is used to stratify sampling to cancel its confound. Unused in v1.")
        components.switch(self, 0, 12, self.ui_state, "stratify", command=save_command)

    def __randomize_uuid(self, axis_config: SliderAxisConfig):
        axis_config.uuid = SliderAxisConfig.default_values().uuid
        return axis_config

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")


class SliderAxisWindow(ctk.CTkToplevel):
    def __init__(self, parent, train_config: TrainConfig, ui_state, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.parent = parent
        self.train_config = train_config
        self.ui_state = ui_state

        self.title("Slider Coordinate Axes")
        self.geometry("900x400")
        self.resizable(True, True)

        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=0)
        self.grid_columnconfigure(0, weight=1)

        components.label(
            self, 0, 0,
            "Coordinate-labeled image sliders (docs §10). Annotate each image's caption with the "
            "axis coordinate as an a1111-style token, e.g. '(distance:-2)'. Declare the axis name(s) "
            "here so they are stripped from the conditioning; flag exactly one enabled axis as the "
            "target -- its coordinate drives the multiplier m = k*value.",
            wide_tooltip=True)

        self.frame = ctk.CTkFrame(self)
        self.frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.frame.grid_columnconfigure(0, weight=1)

        components.button(self, 2, 0, "ok", command=self.on_window_close)

        self.axes = SliderAxisList(self.frame, self.train_config, self.ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()
        self.after(200, lambda: set_window_icon(self))

    def on_window_close(self):
        self.destroy()
