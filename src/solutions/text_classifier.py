from fastai.text.all import *
from src import paths


path = paths.TEXT
dls_lm = DataBlock(
    blocks=TextBlock.from_folder(path, is_lm=True),
    get_items=get_text_files,
    splitter=RandomSplitter(0.1),
).dataloaders(path, path=path, bs=128,sqe_len=80)


learn = language_model_learner(
    dls_lm, AWD_LSTM, drop_mult=0.3,
    metrics=[accuracy, Perplexity()]).to_fp16()

learn.fit_one_cycle(1, 2e-2)

learn.save('1epoch')

learn.unfreeze()
learn.fit_one_cycle(10, 2e-3)


learn.save_encoder('finetuned')

dls_clas = DataBlock(
    blocks=(TextBlock.from_folder(path, vocab=dls_lm.vocab),CategoryBlock),
    get_y=parent_label,
    get_items=get_text_files,
    splitter=RandomSplitter(0.1),
).dataloaders(path, path=path, bs=128, seq_len=72)


learn = text_classifier_learner(dls_clas, AWD_LSTM, drop_mult=0.5,
                                metrics=accuracy).to_fp16()

learn = learn.load_encoder('finetuned')

learn.fit_one_cycle(1, 2e-2)

learn.freeze_to(-2)
learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2))



learn.freeze_to(-3)
learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3))


learn.unfreeze()
learn.fit_one_cycle(2, slice(1e-3/(2.6**4),1e-3))

# Try a few predictions
print(learn.predict("This is a fantastic happenstance"))


print(learn.predict("tragic  sad"))


print(learn.predict("terrible stupid "))


print(learn.predict("happy go lucky"))