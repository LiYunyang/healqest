#!/usr/bin/env python3
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


class Defaults(list):
    """Keep Cobaya's !defaults directive when copying a run YAML."""


class Loader(yaml.SafeLoader):
    pass


class Dumper(yaml.SafeDumper):
    pass


Loader.add_constructor('!defaults', lambda loader, node: Defaults(loader.construct_sequence(node)))
Dumper.add_representer(
    Defaults, lambda dumper, value: dumper.represent_sequence('!defaults', value, flow_style=True)
)


def tagged_path(source, tag):
    stem, extension = os.path.splitext(source)
    return f'{stem}.{tag}{extension}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file')
    parser.add_argument('tag')
    args, overrides = parser.parse_known_args()
    destination = tagged_path(args.file, args.tag)

    if not overrides:
        try:
            os.remove(destination)
        except FileNotFoundError:
            pass
        return

    if len(overrides) % 2 or any(not key.startswith('--') for key in overrides[::2]):
        parser.error('overrides must have the form --A/B/C value')

    with open(args.file, encoding='utf-8') as file:
        config = yaml.load(file, Loader=Loader)

    for option, value in zip(overrides[::2], overrides[1::2]):
        keys = option[2:].split('/')
        target = config
        for key in keys[:-1]:
            target = target[key]
        # Treat overrides as substitutions so a typo cannot create a new field.
        target[keys[-1]]
        target[keys[-1]] = yaml.safe_load(value)

    with open(destination, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, Dumper=Dumper, sort_keys=False)
    print(destination)


if __name__ == '__main__':
    main()
