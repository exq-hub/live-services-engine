"""Search utilities for filtering, indexing, and temporal overlap operations.

This module provides utility functions and classes for:
- Active filter validation against item metadata
- IVF (Inverted File) index searching with clustering
- Temporal shot overlap mapping for video/audio content
- Shot boundary detection and time range calculations
"""

import bisect
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from app.schemas import ActiveFilters, Filter, ItemInfo

# centroids = []
# cluster_datapoint_idxs = []
# clusters = []
# cluster_search_scope = 64


def check_active_filters(
    item: ItemInfo, active_filters: ActiveFilters, filters: dict[str, Filter]
) -> bool:
    """Check if an item passes all active filter criteria.
    
    Validates an item against various filter types including single selection,
    multi-selection, numeric ranges, and count-based filters.
    
    Args:
        item: The item to check against filters
        active_filters: The currently active filter configuration
        filters: Dictionary of available filter definitions
        
    Returns:
        True if the item passes all active filters, False otherwise
    """
    for idx, name in enumerate(active_filters.names):
        name = name.replace(" ", "_").lower()
        if filters[name]["type"] == 0: # Single
            if filters[name]["values"][val] != item["metadata"][name]:
                return False
        elif filters[name]["type"] == 1:  # Multi
            objs = set(item["metadata"][name])
            if name in active_filters.treat_values_as_and:
                objs = set(item["metadata"][name])
                for val in active_filters.values[idx]:
                    if filters[name]["values"][val] not in objs:
                        return False
            else:
                check = False
                for val in active_filters.values[idx]:
                    if filters[name]["values"][val] in objs:
                        check = True
                        break
                if not check:
                    return False
        elif (
            filters[name]["type"] == 3 or filters[name]["type"] == 4
        ):  # RangeNumber : list[list[int]]
            check = False
            for val in active_filters.values[idx]:
                min_val: int = val[0]
                max_val: int = val[1]
                if (
                    item["metadata"][name] >= min_val
                    and item["metadata"][name] <= max_val
                ):
                    check = True
            if not check:
                return False
        elif (
            filters[name]["type"] == 6 or filters[name]["type"] == 7
        ):  # Count : list[list[int]]
            for val in active_filters.values[idx]:
                name: int = val[0]
                count: int = val[1]
                for obj in item["metadata"][name]:
                    if count != obj[1]:
                        return False
            name = active_filters.values[idx][0]
            count = active_filters.values[idx][1]
            for obj in item["metadata"][name]:
                if count != obj[1]:
                    return False
    return True


def index_search(
    query_vec: np.ndarray,
    n: int,
    seen: set[int],
    active_filters: ActiveFilters,
    excluded: set[int] | None,
    initial_expansion: int,
    centroids: np.ndarray,
    clusters: np.ndarray,
    cluster_datapoint_idxs: np.ndarray,
    total_items: int,
    metadata: list[ItemInfo],
    filters: dict[str, Filter],
    related: dict[str, list[int]],
) -> List[int]:
    """Search an Inverted File (IVF) index with clustering for efficient retrieval.
    
    This function performs approximate nearest neighbor search using an IVF index
    structure with clustering. It progressively expands the search scope until
    enough results are found.
    
    Args:
        query_vec: Query vector for similarity search
        n: Number of results to return
        seen: Set of item IDs that have already been seen
        active_filters: Filters to apply during search
        excluded: Set of item IDs to exclude from results
        initial_expansion: Initial number of clusters to search
        centroids: Cluster centroid vectors for coarse search
        clusters: Clustered item vectors for fine search
        cluster_datapoint_idxs: Mapping from cluster items to global indices
        total_items: Total number of items in the collection
        metadata: Item metadata for filter validation
        filters: Filter definitions for validation
        related: Related items mapping for exclusion expansion
        
    Returns:
        List of item IDs ranked by similarity to the query vector
    """
    items_checked = 0

    start = 0
    end = initial_expansion

    dist_clusters = np.dot(query_vec, centroids.T).flatten()
    items = []
    suggestions = np.array([])

    excluded_items = set()
    if len(excluded) > 0:
        temp = []
        for exc in excluded:
            temp += related[metadata["items"][exc]["item_id"].split("_")[0]]
        excluded_items = set(temp)

    while True:
        # TODO: Check if argsort is the fastest to do this.
        top_clusters = np.argsort(dist_clusters)[::-1][start:end]
        for t in top_clusters:
            items_checked += len(clusters[t])
            for idx, item in enumerate(clusters[t]):
                item_id = cluster_datapoint_idxs[t][idx]
                if item_id in seen:
                    continue
                if active_filters is not None:
                    if not check_active_filters(
                        metadata["items"][item_id], active_filters, filters
                    ):
                        continue
                if item_id in excluded_items:
                    continue
                items.append((np.dot(query_vec, item), item_id))
        if len(suggestions) + len(items) < n and items_checked < total_items:
            suggestions = np.concatenate(
                (suggestions, [it[1] for it in sorted(items, reverse=True)])
            )
            start = end
            end = end + end if end + end < len(clusters) else len(clusters)
        elif len(suggestions) + len(items) > n:
            suggestions = np.concatenate(
                (
                    suggestions,
                    [
                        it[1]
                        for it in sorted(items, reverse=True)[: (n - len(suggestions))]
                    ],
                )
            )
            break
        else:
            suggestions = np.concatenate(
                (suggestions, [it[1] for it in sorted(items, reverse=True)])
            )
            break
    return suggestions.tolist()


@dataclass
class Shot:
    """Represents a temporal shot or segment within a video.
    
    Attributes:
        item_id: Unique identifier for the shot/item
        video_id: Identifier of the parent video
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    item_id: str
    video_id: str
    start_time: float
    end_time: float


@dataclass
class TimeRange:
    """Represents a temporal range for overlap calculations.
    
    Attributes:
        identifier: Unique identifier for this time range
        video_id: Identifier of the associated video
        start_time: Range start time in seconds
        end_time: Range end time in seconds
    """
    identifier: str
    video_id: str
    start_time: float
    end_time: float


@dataclass
class OverlapResult:
    """Result of temporal overlap calculation between shots and time ranges.
    
    Attributes:
        item_id: Identifier of the overlapping shot
        overlap_duration: Duration of overlap in seconds
        overlap_ratio: Ratio of overlap to shot duration (0.0 to 1.0)
    """
    item_id: str
    overlap_duration: float
    overlap_ratio: float  # overlap / shot_duration


class ShotOverlapMapper:
    """Utility class for mapping temporal overlaps between shots and time ranges.
    
    This class efficiently finds overlapping shots within specified time ranges
    using binary search on sorted shot boundaries. It's designed for video/audio
    content where precise temporal alignment is needed.
    """
    
    def __init__(self):
        """Initialize the mapper with empty shot collections."""
        # Store shots grouped by video for efficient lookup
        self.video_to_shots: Dict[str, List[Shot]] = defaultdict(list)
        # Sorted shots by start time for binary search
        self.video_to_sorted_shots: Dict[str, List[Shot]] = defaultdict(list)

    def _convert_to_float(self, value) -> float:
        """Convert time value to float, handling strings and other types."""
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert time value '{value}' to float: {e}")

    def add_shot_boundaries(self, item_to_shot_mapping: Dict[str, Dict]):
        for item_id, shot_data in item_to_shot_mapping.items():
            try:
                shot = Shot(
                    item_id=item_id,
                    video_id=str(shot_data["video_id"]),
                    start_time=self._convert_to_float(shot_data["start_time"]),
                    end_time=self._convert_to_float(shot_data["end_time"]),
                )

                # Validate that end_time > start_time
                if shot.end_time <= shot.start_time:
                    print(
                        f"Warning: Invalid shot duration for {item_id}: start={shot.start_time}, end={shot.end_time}"
                    )
                    continue

                self.video_to_shots[shot.video_id].append(shot)

            except (KeyError, ValueError) as e:
                print(f"Warning: Skipping invalid shot data for {item_id}: {e}")
                continue

        # Sort shots by start time for efficient binary search
        for video_id, shots in self.video_to_shots.items():
            self.video_to_sorted_shots[video_id] = sorted(
                shots, key=lambda s: s.start_time
            )

    def clear_shot_boundaries(self):
        """Clear all stored shot boundaries."""
        self.video_to_shots.clear()
        self.video_to_sorted_shots.clear()

    def find_overlapping_shots(
        self,
        time_range: TimeRange,
        min_overlap_duration: float = 0.0,
        min_overlap_ratio: float = 0.0,
    ) -> List[OverlapResult]:
        if time_range.video_id not in self.video_to_sorted_shots:
            return []

        overlapping_shots = []
        sorted_shots = self.video_to_sorted_shots[time_range.video_id]

        # Binary search to find first shot that could potentially overlap
        # We want shots where shot.start < time_range.end
        # Use bisect_left with a custom key to find insertion point
        left_idx = bisect.bisect_left(
            [s.start_time for s in sorted_shots], time_range.end_time
        )

        # Check all shots that start before time_range ends
        for i in range(left_idx):
            shot = sorted_shots[i]

            # Skip if shot ends before time_range starts (no overlap)
            if shot.end_time <= time_range.start_time:
                continue

            # Calculate overlap
            overlap_start = max(shot.start_time, time_range.start_time)
            overlap_end = min(shot.end_time, time_range.end_time)
            overlap_duration = overlap_end - overlap_start

            # Skip if no actual overlap (shouldn't happen given our logic, but safety check)
            if overlap_duration <= 0:
                continue

            # Calculate overlap ratio relative to shot duration
            shot_duration = shot.end_time - shot.start_time
            overlap_ratio = overlap_duration / shot_duration if shot_duration > 0 else 0

            # Apply filters
            if (
                overlap_duration >= min_overlap_duration
                and overlap_ratio >= min_overlap_ratio
            ):
                overlapping_shots.append(
                    OverlapResult(
                        item_id=shot.item_id,
                        overlap_duration=overlap_duration,
                        overlap_ratio=overlap_ratio,
                    )
                )

        # Sort by overlap duration (descending) for better relevance
        overlapping_shots.sort(key=lambda x: x.overlap_duration, reverse=True)
        return overlapping_shots

    def map_ranges_to_item_ids(
        self,
        second_mapping: Dict[str, Dict],
        min_overlap_duration: float = 0.0,
        min_overlap_ratio: float = 0.0,
        return_details: bool = False,
    ) -> Dict[str, List]:
        result = {}

        for identifier, range_data in second_mapping.items():
            try:
                time_range = TimeRange(
                    identifier=identifier,
                    video_id=str(range_data["video_id"]),
                    start_time=self._convert_to_float(range_data["start_time"]),
                    end_time=self._convert_to_float(range_data["end_time"]),
                )

                # Validate that end_time > start_time
                if time_range.end_time <= time_range.start_time:
                    print(
                        f"Warning: Invalid time range for {identifier}: start={time_range.start_time}, end={time_range.end_time}"
                    )
                    result[identifier] = []
                    continue

                overlapping_shots = self.find_overlapping_shots(
                    time_range, min_overlap_duration, min_overlap_ratio
                )

                if return_details:
                    result[identifier] = overlapping_shots
                else:
                    result[identifier] = [shot.item_id for shot in overlapping_shots]

            except (KeyError, ValueError) as e:
                print(
                    f"Warning: Skipping invalid time range data for {identifier}: {e}"
                )
                result[identifier] = []
                continue

        return result

    def create_overlap_mapping(
        self,
        item_to_shot_mapping: Dict[str, Dict],
        second_mapping: Dict[str, Dict],
        min_overlap_duration: float = 0.0,
        min_overlap_ratio: float = 0.0,
        return_details: bool = False,
        clear_existing: bool = True,
    ) -> Dict[str, List]:
        if clear_existing:
            self.clear_shot_boundaries()

        self.add_shot_boundaries(item_to_shot_mapping)
        return self.map_ranges_to_item_ids(
            second_mapping, min_overlap_duration, min_overlap_ratio, return_details
        )
