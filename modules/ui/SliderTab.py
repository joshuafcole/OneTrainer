from modules.ui.ConfigList import ConfigList
from modules.util.config.SliderConfig import SliderImagePairConfig, SliderPromptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.SliderRegime import SliderRegime
from modules.util.ui import components
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
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(4, weight=1)

        # regime
        components.label(frame, 0, 0, "Regime",
                         tooltip="Prompt pair: text-only Concept Sliders (no image dataset); the frozen base supplies the guidance direction and x_t is generated on-manifold. Image pair: before/after image pairs supply the target (not yet wired for Anima).")
        components.options_kv(frame, 0, 1, [
            ("Prompt pair", SliderRegime.PROMPT_PAIR),
            ("Image pair", SliderRegime.IMAGE_PAIR),
        ], self.ui_state, "slider_regime")

        # eta
        components.label(frame, 1, 0, "Guidance scale (eta)",
                         tooltip="Training-time guidance scale applied to v(c+) - v(c-). Paper uses ~3-4. Independent of the inference slider strength (the adapter multiplier).")
        components.entry(frame, 1, 1, self.ui_state, "slider_eta")

        # strength
        components.label(frame, 2, 0, "Training strength",
                         tooltip="Adapter multiplier magnitude used for the trained passes (+/- this). The inference strength range is decoupled from this.")
        components.entry(frame, 2, 1, self.ui_state, "slider_strength")

        # symmetric
        components.label(frame, 3, 0, "Symmetric",
                         tooltip="Also train the -strength pole toward v(c_t) - eta*delta, so the slider is linear around 0 (both poles learned).")
        components.switch(frame, 3, 1, self.ui_state, "slider_symmetric")

        # steps per epoch
        components.label(frame, 0, 3, "Steps per epoch",
                         tooltip="Prompt-pair sliders have no dataset, so this sets how many synthetic training steps make up one epoch.")
        components.entry(frame, 0, 4, self.ui_state, "slider_steps_per_epoch")

        # anchor steps
        components.label(frame, 1, 3, "x_t anchor steps",
                         tooltip="Euler steps used to denoise random noise down to the sampled sigma, producing an on-manifold x_t (SDEdit). More steps = more on-manifold but slower per training step. 0 falls back to off-manifold Gaussian x_t.")
        components.entry(frame, 1, 4, self.ui_state, "slider_anchor_steps")

        # sigma range
        components.label(frame, 2, 3, "Sigma min",
                         tooltip="Lower bound of the noise level sampled for x_t / the trained timestep.")
        components.entry(frame, 2, 4, self.ui_state, "slider_sigma_min")
        components.label(frame, 3, 3, "Sigma max",
                         tooltip="Upper bound of the noise level sampled for x_t / the trained timestep. Higher sigma (more noise) concentrates the attribute signal.")
        components.entry(frame, 3, 4, self.ui_state, "slider_sigma_max")

        # preservation prompts (disentanglement, CS Eq. 8). Pipe-delimited so it
        # fits a single-line entry; empty falls back to the bare pair (Eq. 7).
        components.label(frame, 4, 0, "Preservation prompts",
                         tooltip="Optional disentanglement set P (CS Eq. 8), pipe-delimited (a|b|c). The guidance direction is averaged over the preservation-augmented pairs to keep the slider from entangling unrelated attributes. Empty = bare positive/negative pair.")
        entry = components.entry(frame, 4, 1, self.ui_state, "slider_preservation_prompts")
        entry.grid(row=4, column=1, columnspan=4, sticky="ew")

        # image-pair set (used when Regime = Image pair). Edited in a popup so it
        # doesn't fight the prompt-triple list for vertical space on this tab.
        components.label(frame, 5, 0, "Image pairs",
                         tooltip="Before/after image pairs for the Image-pair regime (CS Eq. 9). The -strength adapter reconstructs 'before', +strength reconstructs 'after'. 3-6 pairs recommended. Ignored in the Prompt-pair regime.")
        components.button(frame, 5, 1, "edit image pairs...", self.__open_image_pairs_window,
                          tooltip="Open the before/after image-pair editor.")

    def __open_image_pairs_window(self):
        window = SliderImagePairWindow(self.master.winfo_toplevel(), self.train_config, self.ui_state)
        self.master.wait_window(window)

    def refresh_ui(self):
        if self.element_list is not None:
            self.element_list.destroy()
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

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


class SliderImagePairWindow(ctk.CTkToplevel):
    """Popup hosting the before/after image-pair list (IMAGE_PAIR regime)."""

    def __init__(self, parent, train_config: TrainConfig, ui_state: UIState):
        super().__init__(parent)
        self.title("Slider image pairs")
        self.geometry("760x520")
        self.transient(parent)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")

        self.list = SliderImagePairList(frame, train_config, ui_state)

        self.wait_visibility()
        self.grab_set()
        self.focus_set()


class SliderImagePairList(ConfigList):
    def __init__(self, master, train_config: TrainConfig, ui_state: UIState):
        super().__init__(
            master,
            train_config,
            ui_state,
            attr_name="slider_image_pairs",
            enable_key="enabled",
            from_external_file=False,
            add_button_text="add image pair",
            is_full_width=True,
            show_toggle_button=True,
        )

    def refresh_ui(self):
        if self.element_list is not None:
            self.element_list.destroy()
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()

    def create_widget(self, master, element, i, open_command, remove_command, clone_command, save_command):
        return SliderImagePairWidget(master, element, i, open_command, remove_command, clone_command, save_command)

    def create_new_element(self) -> dict:
        return SliderImagePairConfig.default_values()

    def open_element_window(self, i, ui_state) -> ctk.CTkToplevel:
        pass


class SliderImagePairWidget(ctk.CTkFrame):
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

        # before (A / negative pole)
        components.label(bottom_frame, 0, 0, "before (A):",
                         tooltip="The 'before' image. The -strength adapter is trained to reconstruct this (the A / negative pole).")
        components.path_entry(bottom_frame, 0, 1, self.ui_state, "before",
                              mode="file", allow_model_files=False, allow_image_files=True)

        # after (B / positive pole)
        components.label(bottom_frame, 1, 0, "after (B):",
                         tooltip="The 'after' image. The +strength adapter is trained to reconstruct this (the B / positive pole).")
        components.path_entry(bottom_frame, 1, 1, self.ui_state, "after",
                              mode="file", allow_model_files=False, allow_image_files=True)

        # prompt (empty = pure image-pair; non-empty = combined prompt-anchored)
        components.label(bottom_frame, 2, 0, "prompt:",
                         tooltip="Conditioning for both reconstructions. Empty = pure empty-prompt image-pair slider (CS Eq. 9). A prompt makes it a combined prompt-anchored example.")
        components.entry(bottom_frame, 2, 1, self.ui_state, "prompt")

        # weight
        components.label(bottom_frame, 3, 0, "weight:",
                         tooltip="Relative sampling weight among the pairs.")
        weight_entry = components.entry(bottom_frame, 3, 1, self.ui_state, "weight")
        weight_entry.configure(width=60)

    def __randomize_uuid(self, pair_config: SliderImagePairConfig):
        pair_config.uuid = SliderImagePairConfig.default_values().uuid
        return pair_config

    def configure_element(self):
        pass

    def place_in_list(self):
        self.grid(row=self.i, column=0, pady=5, padx=5, sticky="new")
