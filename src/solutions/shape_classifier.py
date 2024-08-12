import os.path

from fastbook import *
from fastai.vision.widgets import *
from fastai.data.block import DataBlock, CategoryBlock
from pprint import pprint
from src import paths

block = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


dls = block.dataloaders(paths.IMAGES)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)

print(learn.predict(os.path.join(paths.IMAGES,'circles','circle_4.png')))
print(learn.predict(os.path.join(paths.IMAGES,'rectangles','rectangle_4.png')))

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()

plt.show()