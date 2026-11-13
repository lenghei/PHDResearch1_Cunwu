"""Use the unchanged research evaluator under the v3 authorization ticket."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
import evaluate
from portable_v3 import check_evaluation_authorization


def execute(bundle, data_root, parent, protocol):
    check_evaluation_authorization(parent, bundle, protocol)
    with patch.object(evaluate, 'check_worker_authorization', check_evaluation_authorization):
        return evaluate.evaluate(bundle, data_root, parent, protocol)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--training-parent', type=Path, required=True)
    parser.add_argument('--protocol', choices=('pilot20', 'formal100'), required=True)
    args = parser.parse_args()
    execute(args.bundle.resolve(strict=True), args.data_root.resolve(strict=True),
            args.training_parent.resolve(strict=True), args.protocol)
