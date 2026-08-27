from modules.ui.BaseConfigListView import BaseConfigListView

TIER_EXPLANATION = (
    "An aspect bucket holding fewer images than one batch is discarded by the batch sorter, and "
    "nothing tells you those images never trained. Each tier below maps a bucket population to "
    "something to do about it instead:\n"
    "\n"
    "    drop     discard the bucket (what already happens today, stated explicitly)\n"
    "    donate   merge its images into the nearest surviving bucket\n"
    "    borrow   pull images from neighbouring buckets in until this one fills a batch\n"
    "    repeat   oversample this bucket until it fills a batch\n"
    "\n"
    "A tier applies to buckets holding FEWER than its Max Size images. Tiers are evaluated "
    "smallest Max Size first and the first match wins, so [2 drop, 6 donate, 10 borrow] means: "
    "1 image drops, 2-5 donate, 6-9 borrow, 10 or more are left alone.\n"
    "\n"
    "Each row below is one tier, and its three fields are Max Size, Strategy, and Mode. "
    "Mode applies to borrow only: move hands a neighbour's image over so it trains at this "
    "bucket's crop instead of its own, while copy leaves the original where it is and adds a "
    "second, re-cropped copy here, so the image trains at both crops -- more coverage, and a "
    "longer epoch.\n"
    "\n"
    "An empty list is the previous behaviour, where every sparse bucket is silently dropped."
)

MAX_SIZE_TOOLTIP = (
    "Apply this tier to aspect buckets holding fewer than this many images. "
    "Tiers are evaluated smallest first and the first match wins."
)


class BaseBucketTierParamsWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
        self.components.label(master, 0, 0, TIER_EXPLANATION, wraplength=640)


class BaseBucketTierListView(BaseConfigListView):
    def __init__(self, components):
        self.components = components

    def open_element_window(self, i, ui_state):
        pass
