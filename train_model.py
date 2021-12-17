from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Reader
from pipeline.component import HomoNN
from pipeline.interface import Data
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
import argparse
import wandb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--neurons', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    homo_nn_0 = HomoNN(
        name="homo_nn_0",
        max_iter=args.epochs,
        batch_size=-1,
        early_stop={"early_stop": "diff", "eps": 0.0001}
    )

    # set up network
    shape = (10, )
    homo_nn_0.add(
        Dense(units=args.neurons, input_shape=shape)
    )
    for i in range(args.layers):
        homo_nn_0.add(
            Dense(units=args.neurons, activation="sigmoid")
        )
    homo_nn_0.add(
        Dense(units=1, input_shape=(shape, ), activation="sigmoid")
    )
    homo_nn_0.compile(
        optimizer=optimizers.Adam(learning_rate=args.lr),
        metrics=["accuracy", "AUC"],
        loss="binary_crossentropy"
    )

    # set up pipeline
    pipeline = PipeLine() \
        .set_initiator(role='guest', party_id=9999) \
        .set_roles(guest=9999, host=[10000], arbiter=10000)
    reader_0 = Reader(name="reader_0")
    # set guest parameter
    reader_0.get_party_instance(role='guest', party_id=9999).component_param(
                        table={"name": "breast_homo_guest", "namespace": "experiment"})
    # set host parameter
    reader_0.get_party_instance(role='host', party_id=10000).component_param(
        table={"name": "breast_homo_host", "namespace": "experiment"})
    data_transform_0 = DataTransform(name="data_transform_0", with_label=True)
    # set guest parameter
    data_transform_0.get_party_instance(role='guest', party_id=9999).component_param(
        with_label=True)
    data_transform_0.get_party_instance(role='host', party_id=[10000]).component_param(
        with_label=True)
    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(homo_nn_0, data=Data(train_data=data_transform_0.output.data))
    pipeline.compile()
    pipeline.fit()
    summary = pipeline.get_component("homo_nn_0").get_summary()
    for loss_value in summary['loss_history']:
        pass #wandb