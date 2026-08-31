import os
import argparse
from pathlib import Path
import sys

# Add project root to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from database.datasets.builder import DatasetBuilder
from analysis.technical.configs.standard import StandardTAConfig


def main():
    parser = argparse.ArgumentParser(description="Build datasets from downloaded Binance data")
    parser.add_argument("--data_dir", default="data", help="Directory containing downloaded CSV files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist.")
        return

    builder = DatasetBuilder()
    ta_config = StandardTAConfig()

    # Create output directory if it doesn't exist
    output_dir = Path(os.path.dirname(config.dataset_save_filepath.format('DUMMY')))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over CSV files in data_dir
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    print(f"Found {len(csv_files)} files in {data_dir}")

    for csv_file in csv_files:
        # Try to parse symbol from filename (assuming format symbol_interval_...)
        filename = csv_file.name
        parts = filename.split('_')
        if len(parts) < 2:
            print(f"Skipping {filename}: cannot parse symbol")
            continue

        symbol = parts[0]
        print(f"Processing {symbol} from {filename}...")

        # Construct output path
        output_path = config.dataset_save_filepath.format(symbol)

        try:
            builder.build_dataset(
                ohlcv_history_filepath=str(csv_file),
                dataset_save_filepath=output_path,
                ta_config=ta_config,
            )
            print(f"Successfully built dataset for {symbol} at {output_path}")
        except Exception as e:
            print(f"Failed to build dataset for {symbol}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
