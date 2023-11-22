def build_subparser(subparsers):
    subparsers
    parser_building = subparsers.add_parser("build")
    parser_building.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="WILDRECEIPT",
        choices=["FUNSD", "SROIE", "CORD", "WILDRECEIPT"],
        help="Selecting the dataset for your model's training.",
    )
    parser_building.add_argument(
        "-n",
        "--max_node",
        type=int,
        help="Max nodes per node (edges per node)",
        default=6,
    )
    return parser_building


def train_subparser(subparsers):
    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="WILDRECEIPT",
        choices=["FUNSD", "SROIE", "CORD", "WILDRECEIPT"],
        help="Selecting the dataset for your model's training.",
    )
    parser_train.add_argument(
        "-p",
        "--path",
        type=str,
        default="data/",
        help="Selecting the dataset path for the model's training.",
    )
    parser_train.add_argument(
        "-hs", "--hidden_size", type=int, default=32, help="GCN hidden size."
    )
    parser_train.add_argument(
        "-hl",
        "--hidden_layers",
        type=int,
        default=20,
        help="Number of GCN hidden Layers.",
    )
    parser_train.add_argument(
        "-lr", "--learning_rate", type=float, default=0.01, help="The learning rate."
    )
    parser_train.add_argument(
        "-e", "--epochs", type=int, default=200, help="The number of epochs."
    )
    return subparsers  # Return the subparsers object for further use
