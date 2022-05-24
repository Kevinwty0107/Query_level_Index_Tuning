from rlgraph.components.layers.strings.embedding_lookup import EmbeddingLookup
from rlgraph.components.layers.preprocessing.reshape import ReShape
from rlgraph.components.layers.nn import DenseLayer, ConcatLayer
from rlgraph.components.neural_networks import NeuralNetwork


def build_index_net(state_mode, states_spec, embed_dim, vocab_size, layer_size):
    if state_mode == 'index_net':
        embedding_out = EmbeddingLookup(embed_dim=embed_dim,
                                        vocab_size=vocab_size)(states_spec['sequence'])
        embedding_flat = ReShape(flatten=True, scope="flatten-embedding")(embedding_out)

        dense_out = DenseLayer(units=layer_size,
                               activation='relu',
                               scope='selectivity-dense')(states_spec['selectivity'])
        concat_out = ConcatLayer()(embedding_flat, dense_out)
        final_dense = DenseLayer(units=layer_size,
                                 activation='relu',
                                 scope='final_dense')(concat_out)

        return NeuralNetwork(outputs=final_dense)
    else:
        return [
            dict(type='embedding', embed_dim=embed_dim, vocab_size=vocab_size),
            dict(type="reshape", flatten=True),
            dict(type='dense', units=layer_size, activation='relu', scope="dense_1")
        ]
