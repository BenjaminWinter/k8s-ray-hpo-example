
import ray
import os
from argparse import Namespace
from main import run
from parser import get_parser
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.suggest.hyperopt import HyperOptSearch

from hyperopt import hp

def run_wrapper(config, *args, **kwargs):
    params                        = config["run_args"]
    params.lr                     = config["lr"]
    params.lr_step                = config["lr-step"]
    params.dropout                = config["dropout"]
    params.weight_decay           = config["weight-decay"]
    params.do_hpo                 = True
    params.epochs                 = 999

    run(params)

def run_hpo(args):
    space = {
        "lr": hp.uniform("lr", 3e-7, 3e-3),
        "dropout": hp.uniform("dropout", 0.1, 0.9),
        "lr-step": hp.quniform("lr-step", 1, 3, 1),
        "weight-decay": hp.loguniform("weight-decay", -5, -2)
    }
    defaults = [{
        "lr": 3e-5,
        "dropout": 0.1,
        "lr-step": 1,
        "weight-decay": 0.0
    },{
        "lr": 5e-5,
        "dropout": 0.25,
        "lr-step": 1,
        "weight-decay": 0.1
    }]

    search = HyperOptSearch(
        space,
        metric="test_loss",
        mode="min",
        points_to_evaluate=defaults,
        n_initial_points=args.hpo_hp_initial_points
    )

    scheduler = AsyncHyperBandScheduler(
        metric="test_loss",
        mode="min",
        brackets=args.hpo_hyperband_brackets,
        grace_period=args.hpo_min_steps,
        max_t=args.hpo_max_steps
    )

    config = {
        "num_samples" : args.hpo_num_samples,
        "resources_per_trial": {"cpu": 1, "gpu": 1},
        "config":{
            "run_args": args
        }
    }
    ray.tune.run(
        run_wrapper,
        search_alg=search,
        scheduler=scheduler,
        **config
    )



if __name__ == "__main__":
    args = get_parser("HPO")

    if ("RAY_HEAD_SERVICE_HOST" not in os.environ
        or os.environ["RAY_HEAD_SERVICE_HOST"] == ""):
        raise ValueError("RAY_HEAD_SERVICE_HOST environment variable empty."
                            "Is there a ray cluster running?")
    ray.init(address=os.environ["RAY_HEAD_SERVICE_HOST"] + ":6379")

    run_hpo(args)
