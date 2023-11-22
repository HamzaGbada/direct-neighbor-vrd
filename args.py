import argparse


def build_parser():
    parser = argparse.ArgumentParser(description='This command creates a graph-based dataset for node classification for '
                                                 'a specific dataset in order to extract entities from Visually Rich '
                                                 'Documents. The default is: "./data/<DATASET_NAME>/<Train||Test>/"')
    parser.add_argument('-d', '--dataset', help='Choose the dataset to use. It can be "FUNSD", "CORD", "SROIE" or "WILDRECEIPT" ',
                        default="FUNSD")
    parser.add_argument('-n', '--max_node', type=int,
                        help='Max nodes per node (edges per node)', default=6)
    return parser

def build_subparser(subparsers):
    parser_building = subparsers.add_parser('build')
    parser_building.add_argument('-t', '--train', action='store_true',
                                  help='Boolean to choose between the train or test dataset')
    return parser_building

def train_subparser(subparsers):
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('-d', "--dataname", type=str, default='WILDRECEIPT',
                              choices=['FUNSD', 'SROIE', 'CORD', 'WILDRECEIPT'],
                              help="Selecting the dataset for your model's training.")
    parser_train.add_argument('-p', "--path", type=str, default='data/',
                              help="Selecting the dataset path for the model's training.")
    parser_train.add_argument('-hs', "--hidden_size", type=int, default=32,
                              help="GCN hidden size.")
    parser_train.add_argument('-hl', "--hidden_layers", type=int, default=20,
                              help="Number of GCN hidden Layers.")
    parser_train.add_argument('-lr', "--learning_rate", type=float, default=0.01,
                              help="The learning rate.")
    parser_train.add_argument('-e', "--epochs", type=int, default=200,
                              help="The number of epochs.")
    return parser_train

# Main parser
main_parser = build_parser()

# Create subparsers
subparser = main_parser.add_subparsers(dest="subcommand", help="Choose subcommand")
build_subparser(subparser)
train_subparser(subparser)

# Parse arguments
args = main_parser.parse_args()

if args.subcommand == "build":
    # Access build arguments
    print(args.dataset)
    print(args.max_node)
elif args.subcommand == "train":
    # Access train arguments
    print(args.dataname)
    print(args.path)
    print(args.hidden_size)
    print(args.hidden_layers)
    print(args.learning_rate)
    print(args.epochs)
else:
    # Handle invalid subcommands
    main_parser.print_help()
