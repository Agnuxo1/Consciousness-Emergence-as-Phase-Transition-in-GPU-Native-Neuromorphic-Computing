#!/usr/bin/env python3
"""Upload files to Figshare using the Figshare API v2.

Requires `FIGSHARE_TOKEN` in the environment.
This script creates a new article and uploads a file to it (draft state).
"""
import os
import requests
import argparse


FIGSHARE_API = 'https://api.figshare.com/v2'


def create_article(token, title):
    headers = {'Authorization': f'token {token}'}
    payload = {'title': title, 'description': title, 'defined_type': 'dataset'}
    r = requests.post(f'{FIGSHARE_API}/account/articles', headers=headers, json=payload)
    r.raise_for_status()
    return r.json()


def upload_file(article_id, token, filepath):
    headers = {'Authorization': f'token {token}'}
    r = requests.post(f'{FIGSHARE_API}/account/articles/{article_id}/files', headers=headers, files={'file': open(filepath, 'rb')})
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', required=True)
    parser.add_argument('--title', default='NeuroCHIMERA dataset')
    args = parser.parse_args()

    token = os.getenv('FIGSHARE_TOKEN')
    if not token:
        raise SystemExit('FIGSHARE_TOKEN not found in environment')

    article = create_article(token, args.title)
    print('Created article:', article['location'])
    uploaded = upload_file(article['id'], token, args.file)
    print('Uploaded file metadata:', uploaded)
    print('Visit Figshare to publish the article and add metadata.')


if __name__ == '__main__':
    main()
