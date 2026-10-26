"""Create/verify PRIVATE input hashes. Original MIT candidate; no downloads."""
import argparse
import json
from pathlib import Path

from common import sha256, verify_freeze, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('create')
    create.add_argument('--files-from', type=Path, required=True,
                        help='Private JSON array of existing absolute file paths')
    create.add_argument('--output', type=Path, required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('freeze', type=Path)
    args = parser.parse_args()
    if args.command == 'verify':
        verify_freeze(args.freeze)
        return
    files = json.loads(args.files_from.read_text(encoding='utf-8'))
    if not isinstance(files, list) or not files:
        raise ValueError('nonempty_file_list_required')
    paths = [Path(name) for name in files]
    if any(not path.is_absolute() or not path.is_file() for path in paths):
        raise ValueError('existing_absolute_files_required')
    if args.output.exists():
        raise FileExistsError('freeze_output_exists')
    write_json(args.output, {'schema': 1, 'files': {
        str(path.resolve()): sha256(path) for path in paths}})


if __name__ == '__main__':
    main()
