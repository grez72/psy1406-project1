from fastai2.vision.all import *
from fastai2.vision.widgets import *
from torchvision import models

from IPython.core.debugger import set_trace

from .data import *
from .similarity import *
from .plotting import *

btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()


def show_upload(folder='images'):
    if not os.path.exists(folder):
        os.makedirs(folder)

    def on_click(change):
        image_name = btn_upload.metadata[0]['name']
        img = PILImage.create(btn_upload.data[-1])
        out_pl.clear_output()
        with out_pl:
            display(img.to_thumb(128, 128))
        img.save(os.path.join(folder, image_name))
    btn_upload.observe(on_click, names=['data'])

    return display(VBox([widgets.Label('Upload an image'), btn_upload, out_pl, lbl_pred]))
