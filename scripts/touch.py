"""Create a tagged YAML configuration with selected values overridden.

Examples
--------
Create ``config.0.yml`` from ``config.yml``::
    python scripts/touch.py config.yml 0 --lmaxT 3200 --quick false

YAML containers are accepted as values when quoted for the shell::
    python scripts/touch.py local/config.yml 0 \
        --profile '{class: ProfileBeta, theta_c: 1.0, n: 1000, num_points: 3}' \
        --outdir "output/wide44/wide44p1.0_{field}"

Remove the tagged configuration::
    python scripts/touch.py config.yml 0
"""

import argparse
import os

import yaml


class TouchError(ValueError):
    """Raised for invalid touch.py input that should be reported by argparse."""


def tagged_path(source_file: str, config_id: int) -> str:
    """Return the path used for a configuration tagged with ``config_id``."""
    stem, extension = os.path.splitext(source_file)
    if extension not in {'.yml', '.yaml'}:
        raise TouchError(f"configuration file must end in '.yml' or '.yaml': {source_file}")
    return f'{stem}.{config_id}{extension}'


def parse_overrides(arguments: list[str]) -> dict[str, object]:
    """Parse ``--key value`` arguments, decoding values as YAML."""
    overrides = {}
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if not option.startswith('--') or option == '--':
            raise TouchError(f"expected an override in the form '--key value', got: {option}")

        if '=' in option:
            key, value_text = option[2:].split('=', maxsplit=1)
            index += 1
        else:
            if index + 1 == len(arguments):
                raise TouchError(f"missing value for override '{option}'")
            key = option[2:]
            value_text = arguments[index + 1]
            if value_text.startswith('--'):
                raise TouchError(f"missing value for override '{option}'")
            index += 2

        if not key:
            raise TouchError('override keys cannot be empty')
        if key in overrides:
            raise TouchError(f"override '{key}' was supplied more than once")

        try:
            overrides[key] = yaml.safe_load(value_text)
        except yaml.YAMLError as error:
            raise TouchError(f"invalid YAML value for override '{key}': {error}") from error

    return overrides


def load_yaml(source: str) -> dict:
    """Load a top-level YAML mapping from ``source``."""
    try:
        with open(source, encoding='utf-8') as file:
            config = yaml.safe_load(file)
    except OSError as error:
        raise TouchError(f"could not read configuration file '{source}': {error}") from error
    except yaml.YAMLError as error:
        raise TouchError(f"invalid YAML in configuration file '{source}': {error}") from error

    if not isinstance(config, dict):
        raise TouchError(f"configuration file '{source}' must contain a top-level mapping")
    return config


def update_yaml(source_file: str, config_id: int, overrides: dict[str, object]) -> str:
    """Write a tagged copy of ``source_file`` with second-level values updated."""
    destination = tagged_path(source_file, config_id)
    config = load_yaml(source_file)

    for key, value in overrides.items():
        matches = [group for group in config.values() if isinstance(group, dict) and key in group]
        if not matches:
            raise TouchError(f"override '{key}' was not found in any second-level configuration group")
        if len(matches) > 1:
            raise TouchError(f"override '{key}' appears in more than one second-level configuration group")
        matches[0][key] = value

    try:
        with open(destination, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, sort_keys=False)
    except OSError as error:
        raise TouchError(f"could not write configuration file '{destination}': {error}") from error
    return destination


def cleanup(source_file: str, config_id: int) -> str:
    """Remove the tagged configuration if it exists."""
    destination = tagged_path(source_file, config_id)
    try:
        os.remove(destination)
    except FileNotFoundError:
        pass
    except OSError as error:
        raise TouchError(f"could not remove configuration file '{destination}': {error}") from error
    return destination


def main(argv: list[str] | None = None) -> str:
    """Run the command-line interface and return the affected path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', help='existing .yml or .yaml configuration file')
    parser.add_argument('id', type=int, help='numeric tag included in the generated filename')
    args, unknown_args = parser.parse_known_args(argv)

    try:
        overrides = parse_overrides(unknown_args)
        if overrides:
            return update_yaml(args.file, args.id, overrides)
        return cleanup(args.file, args.id)
    except TouchError as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
