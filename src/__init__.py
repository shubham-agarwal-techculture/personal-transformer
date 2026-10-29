"""
Educational Vanilla Transformer implementation.

A pure NumPy implementation of the Transformer architecture for learning purposes.
"""

from .utils import softmax, relu, gelu, get_positional_encoding, initialize_weights, initialize_bias
from .attention import ScaledDotProductAttention, MultiHeadAttention
from .ffn import PositionWiseFFN
from .normalization import LayerNormalization, Dropout
from .encoder import EncoderLayer, Encoder
from .decoder import DecoderLayer, Decoder, create_look_ahead_mask
from .transformer import Transformer
from .training import cross_entropy_loss, accuracy, TrainingLoop

__version__ = "0.1.0"

