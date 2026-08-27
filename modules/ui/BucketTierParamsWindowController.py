from modules.util.config.TrainConfig import TrainConfig


class BucketTierParamsWindowController:
    def __init__(self, config: TrainConfig):
        self.config = config


class BucketTierListController:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def create_new_element(self) -> dict[str, str]:
        # A new row starts on the least destructive rule there is: "drop" is what
        # already happens to a sparse bucket today, so adding a row and forgetting to
        # finish it cannot change how the dataset trains. max_size is left blank so
        # the row is inert until the user says which buckets it applies to.
        return {"max_size": "", "strategy": "drop", "mode": "move"}
