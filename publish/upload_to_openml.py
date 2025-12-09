#!/usr/bin/env python3
"""Upload a dataset to OpenML using the `openml` Python client.

Requires `OPENML_API_KEY` in the environment. The input dataset should be a CSV file.
"""
import os
import argparse
import openml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True, help='CSV file to upload')
    parser.add_argument('--name', required=True)
    parser.add_argument('--description', default='NeuroCHIMERA dataset')
    args = parser.parse_args()

    api_key = os.getenv('OPENML_API_KEY')
    if not api_key:
        raise SystemExit('OPENML_API_KEY not found in environment')

    openml.config.apikey = api_key
    dataset = openml.datasets.create_dataset(name=args.name,
                                             description=args.description,
                                             creator='Veselov; Angulo de Lafuente',
                                             file=args.file)
    dataset.publish()
    print('Published dataset:', dataset)


if __name__ == '__main__':
    main()
