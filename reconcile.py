"""Compare actual SAS exports against the frozen Python bridge."""
import argparse
from src.reconciliation import compare_bridge

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='outputs/validation/reconciliation')
    parser.add_argument('--sas-predictions', required=True)
    parser.add_argument('--sas-coefficients', required=True)
    args = parser.parse_args()
    result = compare_bridge(args.directory, args.sas_predictions, args.sas_coefficients)
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'PASS_NUMERICAL_CHECKS' else 1)
