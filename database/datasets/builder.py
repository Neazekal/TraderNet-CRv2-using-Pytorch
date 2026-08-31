from typing import Optional
import numpy as np
import pandas as pd
from functools import reduce
from analysis.technical.configs.config import TAConfig
from analysis.technical.configs.standard import StandardTAConfig
from analysis.technical.technical import TechnicalAnalysis
from database.preprocessing.binance.ohlcv import OHLCVPreprocessing
from database.preprocessing.ta.ta import TechnicalAnalysisPreprocessing


class DatasetBuilder:
    def __init__(self):
        self._key_column = 'date'

    @staticmethod
    def _import_ohlcv_dataset(ohlcv_history_filepath: str) -> pd.DataFrame:
        ohlcv_df = pd.read_csv(ohlcv_history_filepath)
        ohlcv_df = OHLCVPreprocessing().preprocess(ohlcv_df=ohlcv_df)
        return ohlcv_df

    @staticmethod
    def _compute_technical_indicators(
            ohlcv_df: pd.DataFrame,
            ta_config: TAConfig,
    ) -> pd.DataFrame:
        ta_config_dict = ta_config.get_technical_analysis_config_dict(
            opens=ohlcv_df['open'],
            highs=ohlcv_df['high'],
            lows=ohlcv_df['low'],
            closes=ohlcv_df['close'],
            volumes=ohlcv_df['volume']
        )
        ta_df = TechnicalAnalysis(dates=ohlcv_df['date']).compute_technical_indicators(ta_config_dict=ta_config_dict)
        ta_df = TechnicalAnalysisPreprocessing(closes=ohlcv_df['close']).preprocess(ta_df=ta_df)
        return ta_df

    def _merge_datasets(self, dataset_df_list: list) -> pd.DataFrame:
        for df in dataset_df_list:
            assert self._key_column in df.columns, \
                f'AssertionError: Key column: "{self._key_column}" is missing from a dataset. Cannot merge datasets'

        return dataset_df_list[0] if len(dataset_df_list) == 1 else \
            reduce(lambda left, right: pd.merge(left, right, on=self._key_column, how='left'), dataset_df_list)

    @staticmethod
    def _handle_missing_values(dataset_df: pd.DataFrame) -> pd.DataFrame:
        dataset_df['hour'] = dataset_df['date'].apply(lambda date: int(str(date).split(' ')[1].split(':')[0]))

        dataset_df.dropna(inplace=True)

        assert not dataset_df.isna().any().any(), \
            f'AssertionError: Imputation failed or incomplete. ' \
            f'Found missing values at columns: {dataset_df.columns[dataset_df.isna().any()]}'

        return dataset_df

    def build_dataset(
            self,
            ohlcv_history_filepath: str,
            dataset_save_filepath: str,
            ta_config: Optional[TAConfig] = None,
    ):
        if ta_config is None:
            ta_config = StandardTAConfig()
        ohlcv_df = self._import_ohlcv_dataset(ohlcv_history_filepath=ohlcv_history_filepath)
        ta_df = self._compute_technical_indicators(ohlcv_df=ohlcv_df, ta_config=ta_config)

        num_expected_samples = ohlcv_df.shape[0]
        merged_dataset_df = self._merge_datasets(dataset_df_list=[ohlcv_df, ta_df])

        assert num_expected_samples == merged_dataset_df.shape[0], \
            'AssertionError: Merged dataset size mismatch: ' \
            f'Expected {num_expected_samples} samples, got {merged_dataset_df.shape[0]}'

        dataset_df = self._handle_missing_values(dataset_df=merged_dataset_df)
        dataset_df.to_csv(dataset_save_filepath, index=False)
