import tensorflow as tf
from model import GAPNetModel
import configs


if __name__ == '__main__':
    opts = configs.opts

    gapnet_model = GAPNetModel(opts)
    gapnet_model.run(opts['epoch'], use_multiprocessing=True, workers=1, verbose=1)
    gapnet_model.predict()
