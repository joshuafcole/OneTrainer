from modules.util.config.SliderConfig import SliderAxisConfig
from modules.util.config.TrainConfig import TrainConfig


class SliderAxesWindowController:
    def __init__(self, config: TrainConfig):
        self.config = config


class SliderAxisListController:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def create_new_element(self) -> SliderAxisConfig:
        # SliderAxisConfig's default name is blank rather than a plausible-looking
        # "distance". An axis name that does not appear in the captions parses no
        # coordinate at all, so a pre-filled name is a row that looks configured
        # and trains nothing; a blank one declares nothing, and a config whose
        # only axes are blank is refused by resolve_target_axis.
        return SliderAxisConfig.default_values()

    def randomize_uuid(self, axis_config: SliderAxisConfig) -> SliderAxisConfig:
        axis_config.uuid = SliderAxisConfig.default_values().uuid
        return axis_config
