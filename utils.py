import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True, help="The workload to be evaluated")
    parser.add_argument("--algo", type=str, required=True, help="The algorithm to be evaluated")
    parser.add_argument("--seed", type=int, default=42, help="The seed to be evaluated")
    return parser.parse_args()

