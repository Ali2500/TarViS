from collections import defaultdict
import json


def map_annotations(annotations_path):
    with open(annotations_path, 'r') as fh:
        content = json.load(fh)

    all_vid_annotations = defaultdict(list)

    for ann in content["annotations"]:
        all_vid_annotations[ann['video_id']].append(ann)

    sequences = []
    for vid in content['videos']:
        vid_anns = {
            track_id: ann for track_id, ann in enumerate(all_vid_annotations[vid['id']])
        }

        seq_dict = {
            "image_paths": vid['file_names'],
            "length": len(vid['file_names']),
            "categories": {track_id: ann['category_id'] for track_id, ann in vid_anns.items()},
            "segmentations": [dict() for _ in range(len(vid['file_names']))],
            "height": vid['height'],
            "width": vid["width"],
            "id": vid['id'],
            "name": vid['file_names'][0].split("/")[0]
        }

        for t in range(len(vid['file_names'])):
            for track_id, ann in vid_anns.items():
                if not ann['segmentations'][t]:
                    continue

                seq_dict["segmentations"][t][track_id] = ann['segmentations'][t]["counts"]

        sequences.append(seq_dict)

    meta_info = {
        "category_labels": {
            cat['id']: cat['name'] for cat in content["categories"]
        }
    }

    return {
        "sequences": sequences,
        "meta": meta_info
    }
