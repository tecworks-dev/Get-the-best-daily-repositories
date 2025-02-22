import argparse
from namo.api.openai import start_server


def handle_chat(args):
    print(f"Starting chat with model: {args.model}")
    print("Chat functionality is under development.")


def handle_server(args):
    print("Starting the server...")
    start_server(ip=args.ip, port=args.port, model=args.model)


def main():

    parser = argparse.ArgumentParser(
        description="Namo CLI: A tool for chat and server management."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    chat_parser = subparsers.add_parser("chat", help="Start a chat session")
    chat_parser.add_argument(
        "--model", type=str, help="Type of model or model local path."
    )
    chat_parser.set_defaults(func=handle_chat)

    server_parser = subparsers.add_parser("server", help="Start the server")
    server_parser.add_argument(
        "--ip",
        type=str,
        default="127.0.0.1",
        help="The IP address to bind the server to",
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="The port to run the server on"
    )
    server_parser.add_argument(
        "--model", type=str, help="Type of model or model local path."
    )
    server_parser.set_defaults(func=handle_server)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
