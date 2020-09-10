from argparse import Namespace, ArgumentParser

def get_parser(ptype: str) -> Namespace:
    parser = ArgumentParser()
    if ptype == "TRAIN":
        train_parse(parser)
    elif ptype == "HPO":
        train_parse(parser)
        hpo_parse(parser)
    
    return parser.parse_args()

def train_parse(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--dropout", default=0.25, type=float, help="Dropout to apply to the output heads")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N', help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--weight-decay', type=float, default=0.0, metavar='LR', help='Weight decay (default: 0.0)')
    parser.add_argument('--warup-steps', type=float, default=100, metavar='LR', help='Warmup Steps (default: 100)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False, help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--lr-step', type=int, default=1, metavar='S', help='StepLR stepsize (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    

def hpo_parse(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--hpo_num_samples", default=1, type=int, help="number of HPO trials to do")
    parser.add_argument("--hpo_min_steps", default=1, type=int, help="number of HPO trial steps to do at minimum")
    parser.add_argument("--hpo_max_steps", default=10, type=int, help="number of HPO trial steps to do at maximum")
    parser.add_argument("--hpo_hp_initial_points", default=10, type=int, help="how many random trials to do before starting proper hpo")
    parser.add_argument("--hpo_hyperband_brackets", default=3, type=int, help="how many hyperband brackets to run")