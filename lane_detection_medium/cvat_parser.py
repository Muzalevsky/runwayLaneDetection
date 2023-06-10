from typing import Optional

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class AnnotationFieldParser(ABC):
    label_key = "label"
    frame_key = "frame"
    point_key = "points"

    @abstractmethod
    def parse(self, data: dict, **kwargs) -> pd.DataFrame:
        """Convert json-data into DataFrame."""


class TagsFieldParser(AnnotationFieldParser):
    """Class Implementation for CVAT 'tags' field parsing."""

    field_name = "tags"

    @property
    def attr_names(self):
        return [self.frame_key, self.label_key]

    @staticmethod
    def _oh_encode(df: pd.DataFrame) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse_output=True)
        encoded_df = pd.DataFrame(
            encoder.fit_transform(df).toarray(),
            # NOTE: return list of arrays
            columns=encoder.categories_[0],
            dtype=int,
        )
        return encoded_df

    def parse(self, data: dict, **kwargs) -> pd.DataFrame:
        raw_df = pd.json_normalize(data, record_path=[self.field_name])

        if raw_df.empty:
            return pd.DataFrame()

        # One-Hot Encoding of label column
        encoded_df = self._oh_encode(raw_df[[self.label_key]])
        encoded_df[self.frame_key] = raw_df[self.frame_key]
        # The same frame rows merging
        df = encoded_df.groupby(self.frame_key, as_index=False).sum()

        # tags counts into bool encodings conversion
        cols = df.columns[df.columns != self.frame_key]
        df[cols] = df[cols].astype(bool).astype(int)

        return df


class ShapesFieldParser(AnnotationFieldParser):
    """Class Implementation for CVAT 'shapes' field parsing."""

    field_name = "shapes"

    @property
    def attr_names(self) -> list[str]:
        return [self.frame_key, self.label_key, "type", "occluded", "z_order", self.point_key]

    def parse(self, data: dict, meta: Optional[str] = None) -> pd.DataFrame:
        df = pd.json_normalize(data, record_path=[self.field_name], meta=meta)

        if df.empty:
            return pd.DataFrame()

        df = df.loc[:, df.columns.isin(self.attr_names)]
        # List into hashable tuple conversion
        df[self.point_key] = df[self.point_key].apply(tuple)

        # NOTE: in case of duplicates - the first observation per group will be used
        df = df.pivot_table(
            values=self.point_key, index=self.frame_key, columns=self.label_key, aggfunc="first"
        ).reset_index()

        return df


class TracksFieldParser(AnnotationFieldParser):
    """Class Implementation for CVAT 'tracks' field parsing."""

    parent_field_name = "tracks"
    field_name = "shapes"

    @property
    def parent_label_key(self) -> str:
        return f"{self.parent_field_name}.label"

    @property
    def attr_names(self) -> list[str]:
        return [
            self.point_key,
            self.frame_key,
            self.parent_label_key,
            f"{self.parent_field_name}.frame",
        ]

    def _distribute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distribute delta points data."""

        distr_df = df.copy()

        if len(df) <= 1:
            # Nothing to distribute
            distr_df.drop(columns=f"{self.parent_field_name}.frame", inplace=True)
            distr_df = distr_df.rename(columns={self.parent_label_key: self.label_key})
            return distr_df

        group_name_key = "group_id"
        group_key = f"{self.parent_field_name}.frame"

        distr_df[group_name_key] = (distr_df[group_key] != distr_df[group_key].shift()).cumsum()
        # List into hashable tuple conversion
        distr_df[self.point_key] = distr_df[self.point_key].apply(tuple)
        grouped = distr_df.groupby([group_name_key, self.parent_label_key], as_index=False)

        def agg_func(x):
            # NOTE: drop extra rectangle in case of multiple rectangles
            x = x.drop_duplicates(subset=[self.frame_key], keep="first")

            transformed_x = (
                x.set_index(self.frame_key)
                .reindex(np.arange(x[self.frame_key].min(), x[self.frame_key].max() + 1))
                .drop(columns=[group_name_key, f"{self.parent_field_name}.frame"])
                .reset_index()
            )

            transformed_x[self.parent_label_key] = x[self.parent_label_key].unique()[0]
            transformed_x[self.point_key] = transformed_x[self.point_key].fillna(method="pad")
            return transformed_x

        # reset multi indices
        distr_df = grouped.apply(agg_func).reset_index(level=1, drop=True).reset_index(drop=True)
        distr_df = distr_df.rename(columns={self.parent_label_key: self.label_key})
        return distr_df

    def parse(self, data: dict, **kwargs) -> pd.DataFrame:
        df = pd.json_normalize(
            data,
            record_path=[self.parent_field_name, self.field_name],
            meta=[
                [self.parent_field_name, self.label_key],
                [self.parent_field_name, self.frame_key],
            ],
        )

        if df.empty:
            return pd.DataFrame()

        df = df.loc[:, df.columns.isin(self.attr_names)]
        df = self._distribute(df)

        # NOTE: in case of duplicates - the first observation per group will be used
        df = df.pivot_table(
            values=self.point_key, index=self.frame_key, columns=self.label_key, aggfunc="first"
        ).reset_index()

        return df


class CvatAnnotationParser:
    """CVAT Annotations Parser Class Implementation."""

    error_key = "error"

    def __init__(self, tags: Optional[list[str]] = None, shapes: Optional[list[str]] = None):
        self._logger = logging.getLogger(self.__class__.__name__)

        self._tags_parser = TagsFieldParser()
        self._shapes_parser = ShapesFieldParser()
        self._tracks_parser = TracksFieldParser()

        self._tags_labels = tags or []
        self._shapes_labels = shapes or []

    @property
    def field_names(self) -> list[str]:
        return [
            self._tags_parser.field_name,
            self._shapes_parser.field_name,
            self._tracks_parser.field_name,
        ]

    @staticmethod
    def _merge_dataframes(left_df: pd.DataFrame, right_df: pd.DataFrame):
        if left_df.empty and right_df.empty:
            return pd.DataFrame()

        if left_df.empty:
            return right_df

        if right_df.empty:
            return left_df

        common_cols = left_df.columns.intersection(right_df.columns).values.tolist()
        merged_df = pd.merge(left_df, right_df, on=common_cols, how="outer")
        # frame group-wise row merging
        merged_df = merged_df.groupby(AnnotationFieldParser.frame_key, as_index=False).first()

        return merged_df

    def _filter_columns(self, df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
        cols_to_drop = df.columns[
            (~df.columns.isin(labels)) & (df.columns != AnnotationFieldParser.frame_key)
        ]
        filtered_df = df.copy()

        if len(cols_to_drop):
            self._logger.warning(f"Extra tags are detected: {cols_to_drop.values}")

        filtered_df[self.error_key] = 0

        tags_mask = (df[cols_to_drop] == 1).any(axis=1)
        shapes_mask = (df[cols_to_drop].notnull()).any(axis=1)
        filtered_df.loc[tags_mask | shapes_mask, self.error_key] = 1

        filtered_df.drop(columns=cols_to_drop, inplace=True)

        return filtered_df

    def parse(self, data: dict) -> pd.DataFrame:
        """Parse CVAT json-like dictionary into DataFrame."""

        tags_df = self._tags_parser.parse(data)
        tags_df = self._filter_columns(tags_df, self._tags_labels)

        shapes_df = self._shapes_parser.parse(data)
        shapes_df = self._filter_columns(shapes_df, self._shapes_labels)

        tracks_df = self._tracks_parser.parse(data)
        tracks_df = self._filter_columns(tracks_df, self._shapes_labels)

        merged_df = self._merge_dataframes(shapes_df, tracks_df)

        if merged_df.empty:
            self._logger.debug("'shapes' and 'tracks' are empty.")
            return tags_df

        if tags_df.empty:
            self._logger.debug("'tags' are empty.")
            return merged_df

        merged_df = self._merge_dataframes(tags_df, merged_df)
        # replace None values in tags with 0
        merged_df[tags_df.columns] = merged_df[tags_df.columns].fillna(0)
        return merged_df
