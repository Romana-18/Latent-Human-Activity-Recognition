from src.models.cnn1d import CNN1D
from src.models.deepconvlstm import DeepConvLSTM
from src.models.lstm import RNNClassifier
from src.models.transformer import TSTransformer

REGISTRY = {"cnn1d": CNN1D, "rnn": RNNClassifier, "transformer": TSTransformer,
            "deepconvlstm": DeepConvLSTM}


def build_model(name, in_channels, n_classes, seq_len, **kwargs):
    return REGISTRY[name](in_channels, n_classes, seq_len, **kwargs)
