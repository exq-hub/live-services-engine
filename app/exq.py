import contextlib
from collections import defaultdict
from typing import List, Optional

import numpy as np
import torch
import zarr
from fastapi import APIRouter, BackgroundTasks, Depends
from numpy.random import default_rng
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

import app.search_utils as search_utils
from app.schemas import (
    AddOrRemoveModelRequest,
    AggregationSearchRequest,
    AppliedFilter,
    ClearItemSetRequest,
    Filter,
    IsExcludedRequest,
    ItemRequest,
    SessionInfo,
    TextSearchRequest,
    RFSearchRequest,
)
from app.shared_resources import SharedResources
from app.utils import dump_log_msgpack, get_current_timestamp, get_shared_resources

##### ROUTER

router = APIRouter()


@router.get("/exq/init/{session}")
def init(
    session: str,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Initial request, logs session and returns total number of items in collection"""
    print(session)
    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "data": {"session": session, "collections": shared.collections},
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)
    return {"session": session, "collections": shared.collections}


@router.post("exq/info/totalItems")
def get_total_items(
    body: SessionInfo,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    total_items = shared.total_items[body.collection]
    log_message = {
        "timestamp": get_current_timestamp(),
        "action": "Initialize Exquisitor LSE Session",
        "data": {
            "session": body.session,
            "collection": body.collection,
            "total_items": total_items,
        },
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)
    return {"total_items": total_items}


@router.post("/exq/log/addModel")
def log_add_model(
    body: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Logging for adding a model"""
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Log Add Model",
        "display_attrs": {
            "session": body.session,
            "modelId": body.modelId,
            "collection": body.collection,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)


@router.post("/exq/log/removeModel")
def log_remove_model(
    body: AddOrRemoveModelRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Logging for removing a model"""
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Log Remove Model",
        "display_attrs": {
            "session": body.session,
            "modelId": body.modelId,
            "collection": body.collection,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)


@router.post("/exq/log/applyFilters")
def log_apply_filters(
    body: AppliedFilter,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Logging for resetting filters of a given model"""
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session_info.session,
        "action": "Log Apply Filters",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "collection": body.session_info.collection,
            "FilterName": body.name,
            "FilterValues": body.values,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)


@router.post("/exq/log/resetFilters")
def log_reset_filters(
    body: SessionInfo,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Logging for resetting filters of a given model"""
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Log Reset Filters",
        "display_attrs": {
            "session": body.session,
            "modelId": body.modelId,
            "collection": body.collection,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)


@router.get("/exq/filters/{session}__{collection}")
def get_filters(
    session: str,
    collection: str,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Return all filters and their values"""
    filters = shared.filters[collection]
    filter_objects = []
    for idx, k in enumerate(filters):
        obj: Filter = {}
        obj["id"] = idx
        obj["collectionId"] = 0
        obj["name"] = k.replace("_", " ").capitalize()
        obj["filterType"] = filters[k]["type"]
        if (
            obj["filterType"] == 2 or obj["filterType"] == 3
        ):  # NumberRange / NumberRangeMulti
            obj["values"] = [
                min(list(filters[k]["values"])),
                max(list(filters[k]["values"])),
            ]
        if (
            obj["filterType"] == 4 or obj["filterType"] == 5
        ):  # RangeLabel / RangeLabelMulti
            obj["values"] = [(idx, el) for el in enumerate(filters[k]["values"])]
        if obj["filterType"] == 6:  # Count
            # minmax = {}
            obj["values"] = []
            for val, cnt_min, cnt_max in filters[k]["values"]:
                obj["values"].append(val)
                obj["count"] = (cnt_min, cnt_max)
        else:  # Single / Multi
            obj["values"] = [v.replace("_", " ").capitalize() for v in list(filters[k]["values"])]
        filter_objects.append(obj)

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": session,
        "action": "Log Get Filters",
        "display_attrs": {},
        "body": {},
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)
    return {"filters": filter_objects}


@router.post("/exq/search/rf")
def rf_search(
    body: RFSearchRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Search the collection using a Linear SVM"""
    request_timestamp = get_current_timestamp()
    clf = SGDClassifier(random_state=42)
    collection = body.session_info.collection
    total_items = shared.total_items[collection]

    if collection in shared.pca:
        store = zarr.storage.ZipStore(shared.pca[collection]["embeddings_f"])
    else:
        store = zarr.storage.ZipStore(shared.embeddings_zarr[collection], mode="r")
    samples = None
    if body.query is not None:
        pseudo_rf = run_clip_search(
            text=body.query,
            seen=body.seen,
            excluded=body.excluded,
            n=10,
            filters=body.filters,
            collection=collection,
            shared=shared,
        )
        
    rng = default_rng()
    emb_arr = zarr.open(store, mode="r")["embeddings"]
    neg = np.asarray(body.neg)
    if not len(body.neg):
        # Add 5 random items
        neg = rng.choice(total_items, size=5, replace=False)

    if not len(body.pos) and body.query is None:
        # Add 5 random items
        pos = rng.choice(total_items, size=5, replace=False)
    elif body.query is None:
        pos = np.asarray(body.pos)
    else:
        pos = np.asarray(body.pos + pseudo_rf)
    samples = emb_arr[np.concatenate((pos, neg))]

    if samples is None:
        return {"suggestions": []}

    labels = np.concatenate(([1.0 for _ in pos], [-1.0 for _ in neg]))
    clf.fit(samples, labels)
    hplane = clf.coef_
    seen_set = set(body.seen)
    excluded_set = set()

    # Optimize excluded set creation using list comprehension
    if len(body.excluded) > 0:
        excluded_set = set(
            item
            for exc in body.excluded
            for item in shared.related_items[collection].get(
                shared.metadata[collection]["items"][exc]["item_id"].split("_")[0],
                [],
            )
        )

    n = body.n
    active_filters = body.filters

    # Loop to grow 'n' until we have enough suggestions
    while True:
        last = n >= shared.total_items[collection]
        _, indices = shared.clip_ann_index[collection].search(hplane, n)

        # Collect suggestions, applying filters and exclusions
        suggestions = []

        for idx in indices[0].tolist():
            if idx not in seen_set and idx not in excluded_set:
                if active_filters is None or search_utils.check_active_filters(
                    shared.metadata[collection]["items"][idx],
                    active_filters,
                    shared.filters[collection],
                ):
                    suggestions.append(idx)

        # Break when we have enough suggestions
        if len(suggestions) >= body.n:
            suggestions = suggestions[: body.n]  # Limit to requested number
            break
        elif last:
            break  # Exit loop if we have exhausted all items

        # Grow 'n' for the next iteration, limiting it to the total number of items
        n = min(n * 2, shared.total_items[collection])

    log_message = {
        "request_timestamp": request_timestamp,
        "completion_time": get_current_timestamp() - request_timestamp,
        "session": body.session_info.session,
        "action": "RF Search",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "collection": body.session_info.collection,
            "pos": body.pos,
            "neg": body.neg,
            "n_seen": len(body.seen),
            "filters": body.filters.model_dump_json() if body.filters else None,
            "excluded": body.excluded,
            "suggestions": suggestions,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)
    return {"suggestions": suggestions}


def run_caption_search(
    text: str,
    seen: List[int],
    excluded: List[str],
    n: int,
    filters: Optional[Filter],
    collection: str,
    shared: SharedResources,
) -> List[int]:
    """Core logic for caption-based text search with shot mapping."""

    # Encode and normalize text features
    with torch.inference_mode():
        text_features = shared.caption_embedding_model.encode(text)

    text_features = text_features / np.linalg.norm(
        text_features, axis=-1, keepdims=True
    )
    text_features = text_features.reshape(1, -1)

    # Process exclusions
    seen_set = set(seen)
    excluded_set = set()
    if excluded:
        excluded_set = set(
            item
            for exc in excluded
            for item in shared.related_items[collection].get(
                shared.metadata[collection]["items"][exc]["item_id"].split("_")[0],
                [],
            )
        )

    active_n = n
    active_filters = filters

    while True:
        last = active_n >= shared.total_items[collection]
        _, indices = shared.caption_ann_index[collection].search(
            text_features, active_n
        )

        # Map caption indices to shot indices (core caption-specific logic)
        caption_ids = [
            shared.caption_shot_ids_list[collection][idx] for idx in indices[0].tolist()
        ]
        base_shots = [
            shared.shot_overlap_mapper[collection][cap_id][0] for cap_id in caption_ids
        ]
        indices = [shared.item_to_datapoint[collection][shot] for shot in base_shots]

        # Collect valid suggestions
        suggestions = []
        for idx in indices:
            if idx not in seen_set and idx not in excluded_set:
                if active_filters is None or search_utils.check_active_filters(
                    shared.metadata[collection]["items"][idx],
                    active_filters,
                    shared.filters,
                ):
                    suggestions.append(idx)

        # Return conditions
        if len(suggestions) >= n:
            return suggestions[:n]
        elif last:
            return suggestions

        # Grow search radius
        active_n = min(active_n * 2, shared.total_items[collection])


@router.post("/exq/search/caption")
def caption_search(
    body: TextSearchRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Search the collection using pre-computed caption embeddings"""
    request_timestamp = get_current_timestamp()
    collection = body.session_info.collection

    # TODO: Add an API endpoint to check if caption search is available instead of this hack
    if not shared.caption_ann_index:
        suggestions = []
    else:
        # Execute the core search logic
        suggestions = run_caption_search(
            text=body.text,
            seen=body.seen,
            excluded=body.excluded,
            n=body.n,
            filters=body.filters,
            collection=collection,
            shared=shared,
        )

    # Create and log the response
    log_message = {
        "request_timestamp": request_timestamp,
        "completion_time": get_current_timestamp() - request_timestamp,
        "session": body.session_info.session,
        "collection": collection,
        "action": "Caption Search",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "query": body.text,
            "n_seen": len(body.seen),
            "filters": body.filters.model_dump_json() if body.filters else None,
            "excluded": body.excluded,
            "suggestions": suggestions,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"suggestions": suggestions}


def run_clip_search(
    text: str,
    seen: List[int],
    excluded: List[str],
    n: int,
    filters: Optional[Filter],
    collection: str,
    shared: SharedResources,
) -> List[int]:
    """Core text search logic that can be reused across different endpoints"""
    # Handle text encoding with CLIP
    device = shared.device
    with (
        torch.inference_mode(),
        (
            torch.amp.autocast("cuda")
            if torch.cuda.is_available()
            else contextlib.nullcontext()
        ),
    ):
        tokenized_text = shared.clip_text_tokenizer([text]).to(device)
        text_features = shared.clip_text_model(tokenized_text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.detach().cpu().numpy()

    # Apply PCA transformation if configured
    if collection in shared.pca:
        pca_model = shared.pca[collection]["model"]
        scaler = shared.pca[collection]["scaler"]
        scaled_features = scaler.transform(text_features)
        text_features = pca_model.transform(scaled_features)

    # Process exclusions
    seen_set = set(seen)
    excluded_set = set()

    if excluded:
        excluded_set = set(
            item
            for exc in excluded
            for item in shared.related_items[collection].get(
                shared.metadata[collection]["items"][exc]["item_id"].split("_")[0], []
            )
        )

    # Find sufficient suggestions
    active_n = n
    active_filters = filters

    while True:
        last = active_n >= shared.total_items[collection]
        _, indices = shared.clip_ann_index[collection].search(text_features, active_n)

        suggestions = []
        for idx in indices[0].tolist():
            if idx not in seen_set and idx not in excluded_set:
                if active_filters is None or search_utils.check_active_filters(
                    shared.metadata[collection]["items"][idx],
                    active_filters,
                    shared.filters[collection],
                ):
                    suggestions.append(idx)

        if len(suggestions) >= n:
            return suggestions[:n]
        elif last:
            return suggestions

        active_n = min(active_n * 2, shared.total_items[collection])


@router.post("/exq/search/clip")
def clip_search(
    body: TextSearchRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Search the collection using a CLIP representation for keyframes"""
    request_timestamp = get_current_timestamp()
    collection = body.session_info.collection

    # Execute the core search logic
    suggestions = run_clip_search(
        text=body.text,
        seen=body.seen,
        excluded=body.excluded,
        n=body.n,
        filters=body.filters,
        collection=collection,
        shared=shared,
    )

    # Create and log the response
    log_message = {
        "request_timestamp": request_timestamp,
        "completion_time": get_current_timestamp() - request_timestamp,
        "session": body.session_info.session,
        "collection": collection,
        "action": "CLIP Search",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "query": body.text,
            "n_seen": len(body.seen),
            "filters": body.filters.model_dump_json() if body.filters else None,
            "excluded": body.excluded,
            "suggestions": suggestions,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"suggestions": suggestions}


@router.post("/exq/search/aggregate")
def aggregate_results(
    body: TextSearchRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """
    Aggregates results from multiple text queries using Reciprocal Rank Fusion (RRF).
    :param body: TextSearchRequest containing search parameters
    :param background_tasks: Background tasks object for logging
    :param shared: Shared resources dependency
    :return: A list of items sorted by their RRF score (descending)
    """
    request_timestamp = get_current_timestamp()
    collection = body.session_info.collection

    # RRF constant (commonly set to 60)
    k = 60

    # Track RRF scores for each item
    rrf_scores = defaultdict(float)

    clip_response = run_clip_search(
        text=body.text,
        seen=body.seen,
        excluded=body.excluded,
        n=body.n,
        filters=body.filters,
        collection=body.session_info.collection,
        shared=shared,
    )

    caption_response = run_caption_search(
        text=body.text,
        seen=body.seen,
        excluded=body.excluded,
        n=body.n,
        filters=body.filters,
        collection=body.session_info.collection,
        shared=shared,
    )

    # Process CLIP results and add to RRF scores
    for rank, item_id in enumerate(clip_response):
        rrf_scores[item_id] += 1 / (k + rank + 1)  # rank + 1 because ranks start from 1

    # Process caption results and add to RRF scores
    for rank, item_id in enumerate(caption_response):
        rrf_scores[item_id] += 1 / (k + rank + 1)  # rank + 1 because ranks start from 1

    # If no results were found, return an empty list
    if not rrf_scores:
        return {"suggestions": []}

    # Sort by RRF score (descending - higher scores are better)
    aggregated_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Create and log the response
    log_message = {
        "request_timestamp": request_timestamp,
        "completion_time": get_current_timestamp() - request_timestamp,
        "session": body.session_info.session,
        "collection": collection,
        "action": "Aggregate Text Search Results (RRF)",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "query": body.text,
            "n_seen": len(body.seen),
            "filters": body.filters.model_dump_json() if body.filters else None,
            "excluded": body.excluded,
            "suggestions": [item_id for item_id, _ in aggregated_results],
            "rrf_k": k,  # Log the k parameter used
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    # Return the list of items sorted by their RRF score
    return {"suggestions": [item_id for item_id, _ in aggregated_results]}


@router.post("/exq/item/base")
def get_item_base_info(
    body: ItemRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Get basic information for the requested item"""
    collection = body.session_info.collection
    item = shared.metadata[collection]["items"][body.itemId]
    media_type = 0
    if ".mp4" in item["media_uri"]:
        media_type = 1

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session_info.session,
        "action": "Get Info For Item",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "Item": body.itemId,
            "mediaName": item["item_id"],
            "mediaType": media_type,  # 0 = Image, 1 = Video
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    obj = {
        "id": body.itemId,
        "name": item["item_id"],  # Filename without extension
        "mediaId": body.itemId,  # Same as id if only one collection
        "mediaType": media_type,  # 0 = Image, 1 = Video
        "relatedGroupId": item["group"],
        "thumbPath": f"{shared.thumbnail_path[body.session_info.collection]}/{item['thumbnail_uri']}",
        "srcPath": f"{shared.original_path[body.session_info.collection]}/{item['media_uri']}",
    }
    if media_type == 1:
        obj["segmentInfo"] = item["metadata"]["segment_info"]
    return obj


@router.post("/exq/item/details")
def get_item_detailed_info(
    body: ItemRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Get detailed information for the requested item"""

    collection = body.session_info.collection
    item = shared.metadata[collection]["items"][body.itemId]
    group = shared.metadata[collection]["groups"][item["group"]]
    info_pairs: list[str, list[str]] = []
    for k in item["metadata"]:
        if (k == "caption" or  k == "ocr" or
            k == "utc_time" or k == "timezone" or
            k == "segment_info"):
            continue
        name = str(k).replace('_', ' ').capitalize()
        if isinstance(item["metadata"][k], list) and len(item["metadata"][k]) > 0:
            info_pairs.append((name, [str(s).replace('_', ' ').capitalize() for s in item["metadata"][k]]))
        else:
            info_pairs.append((name, [str(item["metadata"][k]).replace('_', ' ').capitalize()]))

    for k in group:
        if isinstance(group[k], list) and len(group[k]) > 0:
            if isinstance(group[k][0], str):
                info_pairs.append((k, group[k]))
            else:
                info_pairs.append((k, [str(i) for i in group[k]]))
        else:
            info_pairs.append((k, [str(group[k])]))

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session_info.session,
        "action": "Get Detailed Info For Item",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "collection": collection,
            "item": body.itemId,
            "mediaName": item["item_id"],
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {
        "infoPairs": info_pairs,
    }


@router.post("/exq/item/related")
def get_related_items(
    body: ItemRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Get related information for the requested item"""
    collection = body.session_info.collection
    item = shared.metadata[collection]["items"][body.itemId]
    name = item["item_id"]
    grp_items = shared.related_items[collection][item["group"]]

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session_info.session,
        "action": "Get Related Info For Item",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "collection": collection,
            "item": body.itemId,
            "mediaName": name,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"related": grp_items}


@router.post("/exq/item/excluded")
def is_item_excluded(
    body: IsExcludedRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    """Check if item is in an excluded group"""
    item = shared.metadata["items"][body.itemId]
    name = item["item_id"]

    excludedOrNot = False

    for exc in body.excluded_ids:
        exc_obj = {}
        exc_obj["session_info"] = body.session_info
        exc_obj["itemId"] = exc
        grp_items = set(get_related_items(exc_obj))
        if body.itemId in grp_items:
            excludedOrNot = True
            break

    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session_info.session,
        "action": "Check if item is in excluded group",
        "display_attrs": {
            "session": body.session_info.session,
            "modelId": body.session_info.modelId,
            "item": body.itemId,
            "mediaName": name,
            "excluded": excludedOrNot,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"excludedOrNot": excludedOrNot}


@router.post("/exq/log/clearExcludedGroups")
def clear_all_excluded(
    body: SessionInfo,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Cleared the excluded groups list",
        "display_attrs": {"session": body.session, "modelId": body.modelId},
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"status": "Logged succesfully."}


@router.post("/exq/log/clearItemSet")
def clear_item_set(
    body: ClearItemSetRequest,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": f"Cleared items from {body.name}",
        "display_attrs": {
            "session": body.session,
            "modelId": body.modelId,
            "name": body.name,
        },
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"status": "Logged succesfully."}


@router.post("/exq/log/clearRFModel")
def clear_rf_model(
    body: SessionInfo,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Cleared RF Model",
        "display_attrs": {"session": body.session, "modelId": body.modelId},
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"status": "Logged succesfully."}


@router.post("/exq/log/clearConversation")
def clear_conversation(
    body: SessionInfo,
    background_tasks: BackgroundTasks,
    shared=Depends(get_shared_resources),
):
    log_message = {
        "timestamp": get_current_timestamp(),
        "session": body.session,
        "action": "Cleared Conversation",
        "display_attrs": {"session": body.session, "modelId": body.modelId},
        "body": body.model_dump_json(),
    }
    background_tasks.add_task(dump_log_msgpack, log_message, shared.logfile)

    return {"status": "Logged succesfully."}
