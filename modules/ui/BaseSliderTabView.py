from abc import abstractmethod

from modules.ui.BaseConfigListView import BaseConfigListView
from modules.util.enum.SliderRegime import SliderRegime

# Which settings blocks each regime shows. The tab is a list of prompt triples
# plus a settings panel, and a regime uses only part of it -- so the regime
# selector's job is to hide what does not apply, and this mapping is the whole
# of that decision. A new regime adds a row here and a block builder; it does not
# add another branch to a callback.
_BLOCKS_BY_REGIME: dict[SliderRegime, frozenset[str]] = {
    SliderRegime.PROMPT_PAIR: frozenset({"prompt_pair", "prompt_list"}),
    # No prompt_list: the coordinate regime's input is the Concepts tab, and
    # leaving the triples on screen would suggest they were being read.
    SliderRegime.IMAGE: frozenset({"image"}),
}

_HINT_BY_REGIME: dict[SliderRegime, str] = {
    SliderRegime.PROMPT_PAIR: (
        "Prompt-pair regime: add positive/negative prompt triples below. There is no image "
        "dataset -- the frozen base supplies the guidance direction and the training latents "
        "are generated. The Concepts tab is not used."
    ),
    SliderRegime.IMAGE: (
        "Image (coordinate) regime: an ordinary dataset from the Concepts tab, whose captions "
        "say where each image sits on the axis -- \"a photo of a car on a road, (distance:-2)\". "
        "That token is taken out of the caption and used as the adapter multiplier instead. "
        "Declare the axis name under Coordinate axes below. Prompt triples are not used."
    ),
}


class BaseSliderTabView(BaseConfigListView):
    """Concept-Sliders training config.

    The trained adapter is a plain LoRA/LoKr, configured on the LoRA tab. This tab
    drives the slider *objective*: the prompt triples that supply the frozen
    base's guidance direction, the guidance scale, and the synthetic-latent
    settings.
    """

    @abstractmethod
    def _set_block_visible(self, widget, visible: bool) -> None:
        """Show or hide one settings block. The only toolkit-specific part of the
        regime machinery."""

    @abstractmethod
    def _open_axes_window(self) -> None:
        """Open the coordinate-axes editor. Toolkit-specific only because the
        window class is."""

    @staticmethod
    def blocks_for_regime(regime: SliderRegime) -> frozenset[str]:
        return _BLOCKS_BY_REGIME.get(regime, frozenset())

    @staticmethod
    def hint_for_regime(regime: SliderRegime) -> str:
        return _HINT_BY_REGIME.get(regime, "")

    def build_slider_settings(self, master):
        """Build the settings panel under the prompt-triple list's toolbar.

        Order matters: every block must exist before the regime selector is
        created, because the selector fires its command on construction.
        """
        self._blocks = {}
        self._regime_hint = None

        common = self.components.inline_frame(master, 0, 0)
        self.components.label(
            common, 0, 0, "Regime",
            tooltip="Which kind of slider to train. Prompt pair is text-only: no image "
                    "dataset, the frozen base supplies the guidance direction, and the "
                    "training latents are generated on the model's own trajectory.")
        self.components.label(
            common, 1, 3, "Sigma min",
            tooltip="Lower bound of the noise level sampled for the trained timestep, as a "
                    "fraction of the model's noise schedule.")
        self.components.entry(common, 1, 4, self.ui_state, "slider_sigma_min")
        self.components.label(
            common, 2, 3, "Sigma max",
            tooltip="Upper bound of the noise level sampled for the trained timestep. Higher "
                    "noise concentrates the attribute signal; very low noise mostly trains "
                    "texture. Min and max may be given in either order.")
        self.components.entry(common, 2, 4, self.ui_state, "slider_sigma_max")

        hint_row = self.components.inline_frame(master, 1, 0)
        self._regime_hint = self.components.label(hint_row, 0, 0, "", wraplength=900)

        prompt_pair = self.components.inline_frame(master, 2, 0)
        self._blocks["prompt_pair"] = prompt_pair
        self.components.label(
            prompt_pair, 0, 0, "Guidance scale",
            tooltip="How far past the base model the target is pushed: the training target is "
                    "v(target) + this * (v(positive) - v(negative)). Larger means a stronger "
                    "effect per step and a more aggressive slider. This is a TRAINING-time "
                    "scale and is independent of the strength you set at inference.")
        self.components.entry(prompt_pair, 0, 1, self.ui_state, "slider_eta")
        self.components.label(
            prompt_pair, 1, 0, "Training strength",
            tooltip="The adapter multiplier the trained passes run at (+/- this). Leave at 1.0 "
                    "unless you want the slider calibrated so that a different multiplier is "
                    "its nominal full effect.")
        self.components.entry(prompt_pair, 1, 1, self.ui_state, "slider_strength")
        self.components.label(
            prompt_pair, 2, 0, "Symmetric",
            tooltip="Also train the -strength pole toward v(target) - scale * delta, so the "
                    "slider is linear around 0 and negative values undo the effect rather "
                    "than doing something unrelated. Costs one extra forward per step.")
        self.components.switch(prompt_pair, 2, 1, self.ui_state, "slider_symmetric")
        self.components.label(
            prompt_pair, 0, 3, "Steps per epoch",
            tooltip="A prompt-pair slider has no dataset, so there is no natural epoch length. "
                    "This sets how many training steps make one epoch -- it is what backups, "
                    "sampling and the learning-rate schedule count against.")
        self.components.entry(prompt_pair, 0, 4, self.ui_state, "slider_steps_per_epoch")
        self.components.label(
            prompt_pair, 1, 3, "Latent anchor steps",
            tooltip="Denoising steps used to walk from pure noise down to the sampled noise "
                    "level under the target prompt, so the training latent sits on the model's "
                    "own trajectory. More steps is more faithful and slower -- every step is a "
                    "full forward. 0 uses plain Gaussian noise instead, which is much cheaper "
                    "but measures the guidance direction somewhere the model never visits.")
        self.components.entry(prompt_pair, 1, 4, self.ui_state, "slider_anchor_steps")
        # Own frame, own columns: this field holds a delimited *list*, so it needs
        # to be much wider than the numeric entries above -- and widening column 1
        # of the grid they share would stretch those too.
        preservation = self.components.inline_frame(prompt_pair, 3, 0, 5)
        self.components.label(
            preservation, 0, 0, "Preservation prompts",
            tooltip="Optional. Contexts in which the attribute should change and nothing else "
                    "should, separated by | (or by new lines). Each one is appended to both "
                    "poles and the guidance direction is averaged over all of them, which keeps "
                    "the slider from dragging unrelated attributes along. Leave empty to use "
                    "the bare positive/negative pair.")
        self.components.entry(
            preservation, 0, 1, self.ui_state, "slider_preservation_prompts", width=560)

        image = self.components.inline_frame(master, 3, 0)
        self._blocks["image"] = image
        self.components.label(
            image, 0, 0, "Coordinate axes",
            tooltip="The axis names your captions use, e.g. 'distance' for a caption reading "
                    "'(distance:-2)'. Declared axes are stripped from the caption before "
                    "training; the one flagged as the target supplies the multiplier.")
        self.components.button(image, 0, 1, "edit axes...", command=self._open_axes_window)

        # last, so its construction-time callback finds every block above
        self.components.options_kv(
            common, 0, 1, self.controller.get_slider_regimes(),
            self.ui_state, "slider_regime", command=self.apply_regime,
        )

    def apply_regime(self, regime: SliderRegime) -> None:
        blocks = getattr(self, "_blocks", None)
        if blocks is None:
            return  # still building; the selector's own callback will run again

        visible = self.blocks_for_regime(regime)
        for name, widget in blocks.items():
            self._set_block_visible(widget, name in visible)
        if self.element_list is not None:
            self._set_block_visible(self.element_list, "prompt_list" in visible)
        if self._regime_hint is not None:
            self.components.set_label_text(self._regime_hint, self.hint_for_regime(regime))

    def refresh_ui(self):
        if self.element_list is not None:
            self._destroy_frame(self.element_list)
            self.element_list = None
        self.widgets_initialized = False
        self._create_element_list()
        # _create_element_list re-creates the prompt list from scratch, so the
        # regime-conditional visibility has to be re-applied -- otherwise loading a
        # config for a regime that hides the list brings it back.
        self.apply_regime(self.controller.train_config.slider_regime)

    def open_element_window(self, i, ui_state):
        pass


class BaseSliderPromptWidgetView:
    """One prompt triple: the neutral concept and the two poles."""

    def __init__(self, components):
        self.components = components

    def build_content(self, top_frame, bottom_frame, ui_state, i, save_command, remove_command,
                      clone_command, controller):
        self.ui_state = ui_state
        self.i = i
        self.save_command = save_command

        self.components.colored_icon_button(top_frame, 0, 0, "X", "#C00000", lambda: remove_command(self.i))
        self.components.colored_icon_button(
            top_frame, 0, 1, "+", "#00C000", lambda: clone_command(self.i, controller.randomize_uuid), padx=5)

        self.components.label(top_frame, 0, 2, "enabled:")
        self.components.switch(top_frame, 0, 3, self.ui_state, "enabled", command=save_command, width=40)

        self.components.label(
            bottom_frame, 0, 0, "target:",
            tooltip="The neutral concept the slider is centred on -- what you will be "
                    "generating when you move the slider. The base model's prediction for this "
                    "prompt is the point the adapter learns to shift away from.")
        self.components.entry(bottom_frame, 0, 1, self.ui_state, "target")

        self.components.label(
            bottom_frame, 0, 2, "weight:",
            tooltip="Relative sampling weight when several triples share one slider. 0 excludes "
                    "a triple without deleting it.")
        self.components.entry(bottom_frame, 0, 3, self.ui_state, "weight", width=60)

        self.components.label(
            bottom_frame, 1, 0, "positive:",
            tooltip="The same concept with MORE of the attribute. A positive slider value moves "
                    "toward this.")
        self.components.entry(bottom_frame, 1, 1, self.ui_state, "positive")

        self.components.label(
            bottom_frame, 2, 0, "negative:",
            tooltip="The same concept with LESS of the attribute. A negative slider value moves "
                    "toward this.")
        self.components.entry(bottom_frame, 2, 1, self.ui_state, "negative")

    def configure_element(self):
        pass
