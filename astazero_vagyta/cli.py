import argparse

from .preprocess.app import main as preprocess
from .slope_across.app import main as across
from .slope_along.app import main as along
from .track_depth.app import main as depth
from .track_edge.one_section import main as edge

VERSION = "0.1.0"


def main():
    parser = argparse.ArgumentParser(description="AstaZero Vägyta")
    parser.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    subparser = parser.add_subparsers(dest="command", required=True)

    parser1 = subparser.add_parser("preprocess", help="Preprocess the raw data")
    parser1.add_argument("--all", default="True", help="Preprocess all data")

    parser2 = subparser.add_parser("across", help="Run the across analysis")
    parser2.add_argument(
        "--all", default="False", help="Run the across analysis on all data"
    )

    parser3 = subparser.add_parser("along", help="Run the along analysis")
    parser3.add_argument(
        "--all", default="False", help="Run the along analysis on all data"
    )

    parser4 = subparser.add_parser("depth", help="Run the depth analysis")
    parser4.add_argument(
        "--all", default="False", help="Run the depth analysis on all data"
    )

    parser5 = subparser.add_parser("edge", help="Run the edge analysis")
    parser5.add_argument(
        "--all", default="False", help="Run the edge analysis on all data"
    )

    subparser.add_parser("full", help="Plot the data")

    args = parser.parse_args()

    if args.command == "preprocess":

        preprocess(args.all)

    elif args.command == "across":

        across(args.all)

    elif args.command == "along":

        along(args.all)

    elif args.command == "depth":

        depth(args.all)

    elif args.command == "edge":

        edge(args.all)

    elif args.command == "full":
        print("Running full analysis")
        preprocess("True")
        across("True")
        along("True")
        depth("True")
        edge("True")

    else:
        raise ValueError(f"Invalid command: {args.command}")
