from argparse import ArgumentParser

from .eval_verification import EvalVerifiacation
from .eval_megaface import EvalMegaface
from .eval_ijbc import EvalIjbc

def main():
    parser = ArgumentParser("face test CLI tool", usage="face_eval <command> [<args>]")
    commands_parser = parser.add_subparsers(help="face test - face_eval command-line helpers")

    EvalVerifiacation.register_subcommand(commands_parser)
    EvalMegaface.register_subcommand(commands_parser)
    EvalIjbc.register_subcommand(commands_parser)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    service = args.func(args)
    service.run()

if __name__ == "__main__":
    main()