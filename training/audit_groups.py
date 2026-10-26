"""Audit a PRIVATE JSON array with user-assigned id/group/split fields."""
import argparse
import json
from pathlib import Path

from common import check_groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    args = parser.parse_args()
    counts = check_groups(json.loads(args.manifest.read_text(encoding='utf-8')))
    print(json.dumps({'counts': counts, 'random_split_performed': False}))
