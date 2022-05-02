import argparse
import sys
import ruamel.yaml
from yaml import YAMLError
from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from config.build import load_model
from config.build import build_model

import config.utils

base_choices = ["unet", "recurrent"]
config_dir = Path(__file__).parent / "config"
session_logs_file = config_dir / "session_logs.yaml"
yaml_parser = ruamel.yaml.YAML()


def activate_session_handler(**activate_args):
    """Activates a training session."""
    session_id = activate_args["session-id"]
    try:
        logs = config.utils.get_session_logs(session_logs_file)
    except (IOError, YAMLError) as e:
        print(str(e))
        sys.exit(1)
    if session_id not in logs:
        print(f"Error: no such session exists: {session_id}")
    else:
        session_dir = logs[session_id]["location"]
        current_session = logs.get("current", {})
        new_session = {"session-id": session_id, "location": session_dir}
        if not current_session or current_session["session-id"] != session_id:
            if current_session:
                logs.pop("current")
            logs["current"] = new_session
            try:
                yaml_parser.dump(logs, session_logs_file)
                print(f"Success: {session_id} activated.")
            except YAMLError:
                print(f"Error: cannot activate session: {session_id}.")
                sys.exit(1)
        else:
            print(f"Error: {session_id} is already activated.")


def model_config_handler(**config_args):
    """Initializes a model in the current session directory."""
    try:
        logs = config.utils.get_session_logs(session_logs_file)
    except (IOError, YAMLError) as e:
        print(str(e))
        sys.exit(1)
    session_id = logs["current"]["session-id"]
    session_dir = Path(logs["current"]["location"])

    base_model, model_name = config_args["model"], config_args["name"]
    new_model_dir = session_dir / model_name
    template_model_config = config_dir / base_model / f"{base_model}_base.yaml"
    template_data_config = config_dir / "data_config.yaml"
    template_training_config = config_dir / "training_config.yaml"

    try:
        new_model_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"FileExistsError: {model_name} already exists.")
        sys.exit(1)
    try:
        config.utils.clone_config_file(template_model_config, new_model_dir)
        logs[session_id]["models"].append(model_name)
    except YAMLError as e:
        print(str(e))
        sys.exit(1)
    try:
        config.utils.log_session(session_logs_file, **logs[session_id])
        print(
            f"Success: saved model configuration file for {model_name}",
            f"({base_model}) to {session_id}.",
        )
    except YAMLError as e:
        print(str(e))
        sys.exit(1)
    try:
        config.utils.clone_config_file(template_data_config, new_model_dir)
        print(
            f"Success: saved audio data configuration file for {model_name}",
            f"({base_model}) to {session_id}.",
        )
    except YAMLError as e:
        print(str(e))
        sys.exit(1)
    try:
        config.utils.clone_config_file(template_training_config, new_model_dir)
        print(
            f"Success: saved training settings configuration file for",
            f"{model_name} ({base_model}) to {session_id}.",
        )
    except YAMLError as e:
        print(str(e))
        sys.exit(1)


def new_session_handler(**session_args):
    """Handles creation of a new training session."""
    session_id = session_args["session-id"]
    save_dir = Path(session_args["save"])
    if not save_dir.is_dir():
        print(
            f"FileNotFoundError: no such parent directory:",
            f"{save_dir.absolute()}",
        )
    else:
        session_dir = save_dir / session_id
        # Handle session folder creation and logging.
        try:
            session_dir.mkdir(exist_ok=False)
        except FileExistsError:
            print(
                f"FileExistsError: session already exists:",
                f"{session_dir.name}",
            )
            sys.exit(1)
        try:
            session_data = OrderedDict(
                {
                    "session-id": session_id,
                    "location": str(session_dir),
                    "date-created": datetime.now().strftime(
                        "%d/%m/%Y %H:%M:%S"
                    ),
                    "models": [],
                }
            )
            config.utils.log_session(session_logs_file, **session_data)
            print(f"Success: created a new session: {session_dir.name}.")
            activate_session_handler(**session_args)
        except YAMLError:
            print(f"Error: session: {session_dir.name} cannot be created.")
            session_dir.rmdir()
            sys.exit(1)


def clear_session_handler(**clear_args):
    """Handles session clearing."""
    session_id = clear_args["session-id"]
    try:
        logs = config.utils.get_session_logs(session_logs_file)
    except YAMLError as e:
        print(str(e))
        sys.exit(1)
    if session_id not in logs:
        print(f"Error: must specify a valid session to clear.")
    else:
        try:
            config.utils.clear_session(session_logs_file, session_id)
        except (IOError, YAMLError) as e:
            print(str(e))
            sys.exit(1)


def list_handler(**optional_args):
    """Handles list command."""
    if optional_args["list"]:
        try:
            config.utils.list_models(session_logs_file)
        except (IOError, YAMLError) as e:
            print(str(e))
            sys.exit(1)


def visualize_handler(**visualize_args):
    """Visualizes a model with torchinfo."""
    try:
        logs = config.utils.get_session_logs(session_logs_file)
        session_dir = Path(logs["current"]["location"])
        model_name = visualize_args["name"]
        model_dir = session_dir / model_name
        load_model(model_dir=model_dir, visualize=True)
    except (IOError, YAMLError) as e:
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "A program to initialize training sessions and configuration files."
    )

    subparsers = parser.add_subparsers(
        help="Different session actions. Use {action} -h for help."
    )

    create_parser = subparsers.add_parser("create")
    activate_parser = subparsers.add_parser("activate")
    config_parser = subparsers.add_parser("config")
    clear_parser = subparsers.add_parser("clear")
    visualize_parser = subparsers.add_parser("visualize")

    create_parser.add_argument(
        "session-id",
        type=str,
        help="Custom session id to name session folder.",
        metavar="session-id",
    )

    create_parser.add_argument(
        "--save",
        type=str,
        help="Path to store the session folder. Default: cwd.",
        metavar="",
        default=str(Path.cwd()),
    )

    activate_parser.add_argument(
        "session-id",
        type=str,
        help="Session id to activate.",
        metavar="session-id",
    )

    config_parser.add_argument(
        "--model",
        type=str,
        help=f"Base model architecture to configure. Choices: {base_choices}",
        metavar="",
        required=True,
    )

    config_parser.add_argument(
        "--name",
        "-n",
        type=str,
        help="Name of your model.",
        metavar="",
        required=True,
    )

    parser.add_argument(
        "--list",
        "-l",
        help="Display the list of trained models.",
        action="store_true",
    )

    clear_parser.add_argument(
        "session-id", type=str, help="Training session to delete."
    )

    visualize_parser.add_argument(
        "--name",
        type=str,
        help="The name of the model to visualize.",
        metavar="",
        required=True,
    )

    create_parser.set_defaults(func=new_session_handler)
    activate_parser.set_defaults(func=activate_session_handler)
    config_parser.set_defaults(func=model_config_handler)
    clear_parser.set_defaults(func=clear_session_handler)
    visualize_parser.set_defaults(func=visualize_handler)
    parser.set_defaults(func=list_handler)

    # Parse and handle the command arguments.
    args = parser.parse_args()
    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args.func(**vars(args))
