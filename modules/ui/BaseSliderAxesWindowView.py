from modules.ui.BaseConfigListView import BaseConfigListView

AXES_EXPLANATION = (
    "A coordinate-labeled image slider trains on an ordinary dataset from the Concepts tab. "
    "What makes it a slider is the captions: each one says where that image sits on the axis, "
    "written the way a1111 writes emphasis.\n"
    "\n"
    "    a photo of a car on a road, (distance:-2)\n"
    "    a photo of a car on a road, (distance:2)\n"
    "\n"
    "Declare the axis name below and that token is taken out of the caption before training and "
    "used as the adapter multiplier instead. Removing it is the point: the caption has to "
    "describe everything about the image EXCEPT the axis, so that the axis is the only thing "
    "left for the adapter to explain. Emphasis tokens you did not declare, like (red car:1.2), "
    "are left alone.\n"
    "\n"
    "Values are read as written, so label the dataset in whatever units you measured -- the gain "
    "is what brings the extremes to roughly +/-1. Spreading the labels either side of 0 is what "
    "gives you a slider that runs both ways; -1/+1 alone gives you two poles, which is the same "
    "thing with two labels.\n"
    "\n"
    "Exactly one enabled axis is the target: its coordinate drives this run. Declaring a second "
    "axis without making it the target still strips it from the captions, which is how you keep "
    "a confounder you have labelled out of the conditioning."
)

NAME_TOOLTIP = (
    "The caption token key, matched case-insensitively: 'distance' for a caption reading "
    "'(distance:-2)'. This token is removed from every caption before training."
)

GAIN_TOOLTIP = (
    "Maps a raw coordinate onto the adapter multiplier: m = gain * value. With coordinates "
    "labelled -2..2, a gain of 0.5 trains the extremes at the multiplier 1.0 that inference "
    "applies by default. Read at training time, so changing it does not invalidate the latent "
    "cache."
)

TARGET_TOOLTIP = (
    "The axis this run trains -- exactly one. Other declared axes are still stripped from the "
    "captions, but their coordinates are not used."
)


class BaseSliderAxesWindowView:
    def __init__(self, components):
        self.components = components

    def build_content(self, master, controller, ui_state):
        self.components.label(master, 0, 0, AXES_EXPLANATION, wraplength=640)


class BaseSliderAxisListView(BaseConfigListView):
    def __init__(self, components):
        self.components = components

    def open_element_window(self, i, ui_state):
        pass
