import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    By default it gives an embedding of the partial NCI1 graph dataset.
    The default hyperparameters give a good quality representation without grid search.
    Representations are sorted by ID.
    """
    parser = argparse.ArgumentParser(description="Run Graph2Vec.")

    parser.add_argument("--feature-type",
                        nargs="?",
                        default="time",#time, ratio, status
	                help="Embeddings feature type.")

    parser.add_argument("--input-path",
                        nargs="?",
                        default=r"./dataset",
	                help="Input folder with jsons.")

    parser.add_argument("--output-path",
                        nargs="?",
                        default="./linkfea",
	                help="Embeddings path.")

    parser.add_argument("--dimensions",
                        type=int,
                        default=64,
	                help="Number of dimensions. Default is 64.")

    parser.add_argument("--workers",
                        type=int,
                        default=4,
	                help="Number of workers. Default is 4.")

    parser.add_argument("--epochs",
                        type=int,
                        default=20,
	                help="Number of epochs. Default is 20.")

    parser.add_argument("--min-count",
                        type=int,
                        default=5,
	                help="Minimal structural feature count. Default is 5.")

    parser.add_argument("--wl-iterations",
                        type=int,
                        default=2,
	                help="Number of Weisfeiler-Lehman iterations. Default is 2.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.025,
	                help="Initial learning rate. Default is 0.025.")

    parser.add_argument("--down-sampling",
                        type=float,
                        default=0.0001,
	                help="Down sampling rate of features. Default is 0.0001.")

    return parser.parse_args()