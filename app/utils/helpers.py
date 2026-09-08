"""
helpers.py

Reusable helper functions for CampusGuide AI.
"""

import json
import pickle
from pathlib import Path
from typing import Any


def read_json(file_path: Path) -> Any:
    """
    Read a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON object
    """

    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(file_path: Path, data: Any) -> None:
    """
    Write data to JSON.
    """

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def save_pickle(file_path: Path, data: Any) -> None:
    """
    Save Python object as pickle.
    """

    with open(file_path, "wb") as file:
        pickle.dump(data, file)


def load_pickle(file_path: Path) -> Any:
    """
    Load pickle object.
    """

    with open(file_path, "rb") as file:
        return pickle.load(file)


def clean_text(text: str) -> str:
    """
    Clean text before embedding.

    Removes unnecessary spaces and newlines.
    """

    text = text.replace("\n", " ")
    text = text.replace("\t", " ")

    text = " ".join(text.split())

    return text.strip()


def file_exists(file_path: Path) -> bool:
    """
    Check if file exists.
    """

    return file_path.exists()


def create_directory(directory: Path) -> None:
    """
    Create directory if it doesn't exist.
    """

    directory.mkdir(parents=True, exist_ok=True)


def flatten_dict(data: dict, parent_key: str = "") -> dict:
    """
    Flatten nested dictionary.

    Example:

    {
        "department": {
            "name": "Computer Science"
        }
    }

    becomes

    {
        "department.name": "Computer Science"
    }
    """

    items = {}

    for key, value in data.items():

        new_key = (
            f"{parent_key}.{key}"
            if parent_key
            else key
        )

        if isinstance(value, dict):
            items.update(
                flatten_dict(value, new_key)
            )
        else:
            items[new_key] = value

    return items


def format_metadata(source: str, topic: str) -> dict:
    """
    Create metadata for LangChain Documents.
    """

    return {
        "source": source,
        "topic": topic
    }