import json
import pycocotools.mask as mt


FIELDS_TO_DELETE = [
    "seq_name", "neg_category_ids", "not_exhaustive_category_ids", "track_category_ids", "all_image_paths",
    "annotated_image_paths"
]

MIN_MASK_AREA = 100


def mask_area(rle, img_dims):
    return mt.area({
        "size": img_dims,
        "counts": rle.encode("utf-8")
    })


def map_annotations(annotations_path):
    with open(annotations_path, 'r') as fh:
        content = json.load(fh)

    sequences = []

    for seq in content["sequences"]:
        img_dims = (seq['height'], seq['width'])
        seq["id"] = f"{seq['dataset']}/{seq['seq_name']}"
        seq["image_paths"] = [f"{seq['dataset']}/{seq['seq_name']}/{fname}" for fname in seq["annotated_image_paths"]]

        updated_segmentations = []
        accepted_track_ids = set()

        for segs_t in seq['segmentations']:
            updated_segmentations.append(dict())
            for track_id, seg in segs_t.items():
                if mask_area(seg['rle'], img_dims) >= MIN_MASK_AREA:
                    updated_segmentations[-1][track_id] = seg['rle']
                    accepted_track_ids.add(track_id)

        if not accepted_track_ids:
            continue

        seq['segmentations'] = updated_segmentations
        seq["categories"] = {
            track_id: seq["track_category_ids"][track_id]
            for track_id in accepted_track_ids
        }

        for field in FIELDS_TO_DELETE:
            del seq[field]

        sequences.append(seq)

    meta_info = {
        "category_labels": {
            cat['id']: cat['name'] for cat in content["categories"]
        }
    }

    del content["categories"], content["split"]

    return {
        "sequences": sequences,
        "meta": meta_info
    }
