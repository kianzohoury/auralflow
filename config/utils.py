import sys
import ruamel.yaml
from yaml import YAMLError
from tabulate import tabulate
from pathlib import Path
from typing import Optional
from datetime import datetime
from collections import OrderedDict


yaml_parser = ruamel.yaml.YAML()

config_dir = Path(__file__) / "config"


def get_session_logs(logs_file: Path) -> Optional[dict]:
    """Reads session logs audio.

    Raises:
        IOError: If the session logs file cannot be read from.
        YAMLError: If the session logs file cannot be read from or written to.
    """
    if not logs_file.is_file():
        try:
            logs_file.open("w")
        except IOError as e:
            raise IOError(
                f"Cannot create session logs file", "{str(logs_file)}."
            ) from e
    try:
        return yaml_parser.load(logs_file) or {}
    except YAMLError as e:
        raise YAMLError(f"Cannot load audio from {str(logs_file)}.") from e


def log_session(logs_file: Path, **data):
    """Logs a new session or updates it in the logs file.

    Raises:
        YAMLError: If the session logs file cannot be read from or written to.
    """
    try:
        current_logs = get_session_logs(logs_file)
        session_id = data["session-id"]
        current_logs[session_id] = data
        yaml_parser.dump(current_logs, logs_file)
    except YAMLError as e:
        raise YAMLError("Cannot log the current session.") from e


def clone_config_file(template_config: Path, model_dir: Path):
    """Clones a configuration file into a session's model folder.

    Args:
        template_config: Template file to clone from.
        model_dir: Destination path for the clone file.

    raises:
        YAMLError: If the configuration file cannot be read from or written to.
    """
    try:
        config_data = yaml_parser.load(template_config)
        destination_file = model_dir / template_config.name
        yaml_parser.dump(config_data, destination_file)
    except YAMLError as e:
        raise YAMLError(
            f"Cannot clone the configuration file",
            "{str(template_config)} into {str(model_dir)}.",
        ) from e


def remove_directory(folder: Path):
    """Removes all session files and folders safely."""
    if folder.is_dir():
        for sub_folder in folder.iterdir():
            remove_directory(sub_folder)
        try:
            folder.rmdir()
        except IOError as e:
            raise IOError(f"Cannot remove directory: {str(folder)}") from e
    else:
        folder.unlink(missing_ok=True)


def clear_session(logs_file: Path, session_id: str):
    """Clears a session from the session logs and erases its contents."""
    try:
        logs = get_session_logs(logs_file)
        if session_id not in logs:
            print(f"Error: no such session: {session_id} exists.")
        else:
            try:
                session_dir = Path(logs[session_id]["location"])
                remove_directory(session_dir)
                logs.pop(session_id)
                if len(logs) == 1:
                    logs.pop("current")
                print(f"Success: {session_id} was erased.")
                yaml_parser.dump(logs, logs_file)
            except (IOError, YAMLError) as e:
                raise Exception(f"Cannot clear session {session_id}.") from e
    except (IOError, YAMLError) as e:
        raise Exception(f"Cannot clear session {session_id}.") from e


def list_models(logs_file: Path):
    """Displays the full list of active training sessions/models."""
    try:
        logs = get_session_logs(logs_file)
    except (IOError, YAMLError) as e:
        raise Exception(f"Cannot list session logs.") from e
    if not logs:
        print("Info: No models to list.")
    else:
        table_entries = []
        for session_name, info in logs.items():
            if session_name == "current":
                continue
            location = info["location"]
            date = info["date-created"]
            models = info["models"]
            for model in models:
                table_entries.append(
                    [session_name, model, date, location + f"/{model}"]
                )
        table_string = tabulate(
            table_entries,
            headers=["Session", "Model", "Created", "Location"],
            tablefmt="rst",
        )
        print(table_string)
