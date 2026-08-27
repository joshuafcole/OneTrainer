from modules.util.config.SliderConfig import SliderPromptConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.SliderRegime import SliderRegime


class SliderTabController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_slider_regimes(self) -> list[tuple[str, SliderRegime]]:
        return [
            ("Prompt pair", SliderRegime.PROMPT_PAIR),
        ]

    def create_new_element(self) -> SliderPromptConfig:
        return SliderPromptConfig.default_values()

    def randomize_uuid(self, prompt_config: SliderPromptConfig) -> SliderPromptConfig:
        prompt_config.uuid = SliderPromptConfig.default_values().uuid
        return prompt_config
