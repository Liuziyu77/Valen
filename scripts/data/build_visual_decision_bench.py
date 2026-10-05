#!/usr/bin/env python3
"""Package the evaluated image/video sets without changing questions or labels.

保留原有共享 state 分组；本地版使用真实媒体，发布版使用 ZIP。
"""
import argparse
import copy
import hashlib
import json
import math
import shutil
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, values):
    with path.open("w", encoding="utf-8") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def media_urls(record):
    for message in record["request"]["state"]["messages"]:
        for item in message["content"]:
            if isinstance(item, dict) and item.get("type") in {"image_url", "video_url"}:
                yield item[item["type"]]


def strip_media_locations(record):
    # Compare the actual evaluated inputs after normalizing media locations.
    # 只忽略媒体位置，逐项核对问题、选项、state 文本与标签。
    value = copy.deepcopy({key: record[key] for key in ("request", "targets")})
    for item in media_urls(value):
        item["url"] = "<media>"
    return value


def validate_questions(record):
    questions = record["request"]["questions"]
    assert set(questions) == set(record["targets"])
    hard = Counter()
    for qid, q in questions.items():
        if q["type"] == "choice":
            assert isinstance(q["criteria"], dict) and 1 <= len(q["criteria"]) <= 255
            keys = set(q["criteria"])
        elif q["type"] == "score":
            assert isinstance(q["criteria"], list) and 2 <= len(q["criteria"]) <= 255
            keys = {str(i) for i in range(len(q["criteria"]))}
        else:
            assert q["type"] == "noul"
            keys = {"true", "false"}
        assert isinstance(q["instructions"], str) and q["instructions"].strip()
        probs = record["targets"][qid]["probabilities"]
        assert set(probs) == keys
        assert all(isinstance(p, (int, float)) and not isinstance(p, bool)
                   and math.isfinite(p) and p >= 0 for p in probs.values())
        assert math.isclose(sum(probs.values()), 1, abs_tol=1e-6)
        if sum(p == 1 for p in probs.values()) == 1 and sum(p != 0 for p in probs.values()) == 1:
            hard[q["type"]] += 1
    return hard


def portable_metadata(value, asset_locations):
    if isinstance(value, dict):
        return {k: portable_metadata(v, asset_locations) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_metadata(v, asset_locations) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        if value in asset_locations:
            return asset_locations[value]
        # Keep useful source identifiers without publishing host directory names.
        # 保留来源文件标识，不携带机器上的目录前缀。
        for marker in ("/CapRL/video/", "/video_test/", "/seed_annotations/"):
            if marker in value:
                return value.split(marker, 1)[1]
        return Path(value).name
    return value


def copy_asset(task):
    source, destination, expected = task
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert digest(source) == expected, f"Source asset checksum mismatch: {source.name}"
    if not destination.exists() or digest(destination) != expected:
        shutil.copyfile(source, destination)
    assert digest(destination) == expected
    return destination.stat().st_size


def build(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    local, release = args.local.resolve(), args.release.resolve()
    assert local != release and local not in release.parents and release not in local.parents
    for root in (local, release):
        root.mkdir(parents=True, exist_ok=True)
        (root / "data").mkdir(exist_ok=True)
    all_records, manifest, asset_tasks = [], [], []
    statistics, sources = {}, {}
    subsets = [("visualdecisionbench_image", args.image_eval, args.image_snapshot, 2000, "image"),
               ("visualdecisionbench_video", args.video_eval, args.video_snapshot, 7143, "video")]
    for subset, source, snapshot, expected_questions, modality in subsets:
        source = source.resolve(strict=True)
        records = read_jsonl(source)
        if snapshot:
            evaluated = read_jsonl(snapshot)
            assert len(records) == len(evaluated)
            assert all(strip_media_locations(a) == strip_media_locations(b)
                       for a, b in zip(records, evaluated)), f"Evaluated inputs differ: {subset}"
        asset_map, asset_locations = {}, {}
        for record in records:
            for asset in record["assets"]:
                path = PurePosixPath(asset["path"])
                assert not path.is_absolute() and ".." not in path.parts and path.parts[0] == "assets"
                relative = (PurePosixPath("assets") / subset / PurePosixPath(*path.parts[1:])).as_posix()
                previous = asset_map.get(asset["path"])
                if previous:
                    assert previous["sha256"] == asset["sha256"]
                else:
                    asset_map[asset["path"]] = {"path": relative, "sha256": asset["sha256"],
                                              "subset": subset, "modality": modality}
                    asset_tasks.append((source.parent / asset["path"], local / relative, asset["sha256"]))
                asset_locations[str(source.parent / asset["path"])] = relative
        converted, flat, types, hard = [], [], Counter(), Counter()
        source_counts = {}
        for index, record in enumerate(records):
            hard.update(validate_questions(record))
            value = copy.deepcopy(record)
            for item in media_urls(value):
                assert item["url"] in asset_map
                item["url"] = asset_map[item["url"]]["path"]
            for asset in value["assets"]:
                asset["path"] = asset_map[asset["path"]]["path"]
            value["meta"] = portable_metadata(value.get("meta", {}), asset_locations)
            identifier = f"{subset}:{index:06d}"
            value["meta"].update(benchmark="VisualDecisionBench", benchmark_subset=subset,
                                 benchmark_record_id=identifier)
            assert strip_media_locations(value) == strip_media_locations(record)
            converted.append(value)
            origin = value["meta"].get("source", "unknown")
            entry = source_counts.setdefault(origin, {"states": 0, "questions": 0, "types": Counter(),
                                                       "licenses": set(), "source_repos": set()})
            entry["states"] += 1
            entry["licenses"].add(value["meta"].get("license", "unspecified in source export"))
            if value["meta"].get("source_repo"):
                entry["source_repos"].add(value["meta"]["source_repo"])
            for qid, question in value["request"]["questions"].items():
                types[question["type"]] += 1
                entry["questions"] += 1
                entry["types"][question["type"]] += 1
                flat.append({"id": f"{identifier}/{qid}", "subset": subset, "state_id": identifier,
                             "question_id": qid, "group_id": value["group_id"], "modality": modality,
                             "question_type": question["type"], "instructions": question["instructions"],
                             "state_json": json.dumps(value["request"]["state"], ensure_ascii=False),
                             "question_json": json.dumps(question, ensure_ascii=False),
                             "target_json": json.dumps(value["targets"][qid], ensure_ascii=False),
                             "media_paths": [a["path"] for a in value["assets"]], "source": origin})
        assert sum(types.values()) == expected_questions
        assert len({row["id"] for row in flat}) == len(flat)
        write_jsonl(local / f"{subset}.jsonl", converted)
        table = pa.Table.from_pylist(flat)
        pq.write_table(table, local / "data" / f"{subset}.parquet", compression="zstd")
        # Verify that the viewer's one-question rows reconstruct every original question.
        # 检查逐题表格可无损恢复 state、问题和标签。
        readback = pq.read_table(local / "data" / f"{subset}.parquet").to_pylist()
        assert readback == flat
        assert all(json.loads(row["state_json"]) == converted[int(row["state_id"].split(":")[-1])]["request"]["state"]
                   and json.loads(row["question_json"]) == converted[int(row["state_id"].split(":")[-1])]["request"]["questions"][row["question_id"]]
                   and json.loads(row["target_json"]) == converted[int(row["state_id"].split(":")[-1])]["targets"][row["question_id"]]
                   for row in readback)
        statistics[subset] = {"states": len(converted), "questions": len(flat), "types": dict(types),
                              "hard_label_questions": sum(hard.values()), "hard_label_types": dict(hard),
                              "assets": len(asset_map),
                              "multi_question_states": sum(len(r["request"]["questions"]) > 1 for r in converted),
                              "max_questions_per_state": max(len(r["request"]["questions"]) for r in converted)}
        for entry in source_counts.values():
            entry["types"] = dict(entry["types"])
            entry["licenses"] = sorted(entry["licenses"])
            entry["source_repos"] = sorted(entry["source_repos"])
        sources[subset] = {"original_filename": "eval.jsonl", "original_sha256": digest(source),
                           "evaluated_snapshot_sha256": digest(snapshot) if snapshot else None,
                           "sources": source_counts}
        manifest.extend(asset_map.values())
        all_records.extend(converted)
        print(json.dumps({"event": "subset_prepared", "subset": subset, **statistics[subset]}), flush=True)

    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = list(pool.map(copy_asset, asset_tasks))
    assert len({a["path"] for a in manifest}) == len(manifest)
    for asset, size in zip(manifest, sizes):
        asset["bytes"] = size
    manifest.sort(key=lambda row: row["path"])
    write_jsonl(local / "assets_manifest.jsonl", manifest)
    write_jsonl(local / "eval.jsonl", all_records)
    statistics["total"] = {key: sum(row[key] for row in statistics.values())
                           for key in ("states", "questions", "assets", "hard_label_questions", "multi_question_states")}
    statistics["total"]["types"] = dict(sum((Counter(row["types"]) for key, row in statistics.items()
                                            if key != "total"), Counter()))
    statistics["total"]["asset_bytes"] = sum(sizes)
    assert statistics["total"]["questions"] == 9143
    write_json(local / "statistics.json", statistics)
    write_json(local / "source_manifest.json", {"name": "VisualDecisionBench", "version": "1.0",
               "source_datasets": sources, "transformations": [
                   "Namespaced media paths under assets/visualdecisionbench_image and assets/visualdecisionbench_video.",
                   "Preserved state grouping, record order, questions, criteria, IDs, and target distributions.",
                   "Added benchmark subset/state IDs in meta.",
                   "Removed host directory prefixes from historical provenance paths.",
                   "Created lossless, one-question-per-row Parquet views for Hugging Face Datasets."]})
    archive = release / "assets.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as z:
        for asset in manifest:
            z.write(local / asset["path"], arcname=asset["path"])
    with zipfile.ZipFile(archive) as z:
        assert set(z.namelist()) == {asset["path"] for asset in manifest}
        for asset in manifest:
            with z.open(asset["path"]) as stream:
                h = hashlib.sha256()
                size = 0
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(chunk); size += len(chunk)
            assert size == asset["bytes"] and h.hexdigest() == asset["sha256"]
    subset_files = [f"{subset}.jsonl" for subset, *_ in subsets]
    for filename in ["eval.jsonl", *subset_files, "assets_manifest.jsonl", "statistics.json", "source_manifest.json"]:
        shutil.copyfile(local / filename, release / filename)
    for subset, *_ in subsets:
        shutil.copyfile(local / "data" / f"{subset}.parquet", release / "data" / f"{subset}.parquet")
    for filename in ("eval.jsonl", *subset_files, "source_manifest.json"):
        assert "/mnt/" not in (release / filename).read_text(encoding="utf-8")
    validation = {"success": True, "questions": 9143, "states": len(all_records),
                  "original_inputs_and_targets_preserved": True,
                  "matched_evaluated_snapshots": all(item[2] is not None for item in subsets),
                  "schema_and_target_distributions_valid": True,
                  "copied_asset_checksums_valid": True, "zip_member_names_and_checksums_valid": True,
                  "parquet_roundtrip_valid": True, "portable_media_paths": True,
                  "assets": len(manifest), "zip_bytes": archive.stat().st_size,
                  "zip_sha256": digest(archive)}
    for root in (local, release):
        write_json(root / "validation.json", validation)
    print(json.dumps({"event": "build_complete", **validation, "statistics": statistics}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-eval", type=Path, required=True)
    parser.add_argument("--video-eval", type=Path, required=True)
    parser.add_argument("--image-snapshot", type=Path)
    parser.add_argument("--video-snapshot", type=Path)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    build(parser.parse_args())
