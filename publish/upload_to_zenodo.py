#!/usr/bin/env python3
"""Upload a file to Zenodo via the REST API.

Requires an environment variable `ZENDO_TOKEN` with a Zenodo personal access token.
This script performs a single-file upload to a new deposition (draft). For production,
consider creating metadata and uploading multiple files per deposition.
"""
import os
import requests
import argparse


ZENODO_API = 'https://zenodo.org/api'


def create_deposition(token):
    r = requests.post(f'{ZENODO_API}/deposit/depositions', params={'access_token': token}, json={})
    r.raise_for_status()
    return r.json()


def upload_file(deposition, token, filepath):
    bucket_url = deposition['links']['bucket']
    fname = os.path.basename(filepath)
    with open(filepath, 'rb') as fp:
        r = requests.put(f"{bucket_url}/{fname}", params={'access_token': token}, data=fp)
        r.raise_for_status()
    return r


def set_metadata(deposition_id, token, metadata):
    r = requests.put(f'{ZENODO_API}/deposit/depositions/{deposition_id}', params={'access_token': token}, json={'metadata': metadata})
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--title', default='NeuroCHIMERA release')
    parser.add_argument('--creators', nargs='+', default=['Veselov, Vladimir F.', 'Angulo de Lafuente, Francisco'])
    args = parser.parse_args()

    token = os.getenv('ZENDO_TOKEN')
    if not token:
        raise SystemExit('ZENDO_TOKEN not found in environment')

    deposition = create_deposition(token)
    print('Created deposition:', deposition['id'])
    upload_file(deposition, token, args.file)
    metadata = {
        'title': args.title,
        'upload_type': 'publication',
        'creators': [{'name': c} for c in args.creators],
        'access_right': 'open'
    }
    result = set_metadata(deposition['id'], token, metadata)
    print('Metadata set; deposition id:', result['id'])
    print('When ready, visit Zenodo to publish the deposition and mint a DOI.')


if __name__ == '__main__':
    main()
