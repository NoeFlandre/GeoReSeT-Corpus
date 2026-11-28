"""Round 1 end to end: the corpus's labels joined to the filter's generations, and data/ without its label.

build     (CPU) bucket the generations and the corpus rows by text, join each bucket, and write the round's
          files, the registry and data/ without the inline landuse, into one release-shaped folder
propose   open a pull request on the corpus with the folder's data/ and labels/ (the rest is unchanged)
finalize  merge the reviewed pull request and tag the release
"""

import json
from collections import defaultdict
from pathlib import Path

import polars as pl

from georeset.label import migrate, partition, readme, rounds

REPO = "NoeFlandre/GeoReSeT-Corpus"
TAG = "v2.0"  # the release without the inline labels
GENERATION_REPOS = {
    "description": "NoeFlandre/osm-polygon-description-tag-landuse",
    "wikipedia": "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
    "website": "NoeFlandre/osm-polygon-website-tag-landuse",
}
RECORDED = [*rounds.SINGLE, *rounds.MANY, "code_commit"]
TOTALS = ("rows", "parsed", "agree", "disagree", "missing")
KEPT_ON_PR = ("data", "labels", "README.md")


def build(work: Path, out: Path) -> None:  # pragma: no cover (Hub, GPU-sized files)
    print("round: staging generations", flush=True)
    recorded = _stage_generations(work / "generations", work / "src")
    print("round: staging corpus", flush=True)
    corpus_buckets = _stage_corpus(work, out)
    found = rounds.backfilled({key: sorted(values) for key, values in recorded.items()})
    print(f"round: joining into {found.name}", flush=True)
    _write_round(out, found, corpus_buckets, work / "generations")
    print("round: writing the card", flush=True)
    _write_card(work, out, found.name)
    print("round: verifying", flush=True)
    _verify(out, found.name)
    print("round: done", flush=True)


def _write_card(work: Path, out: Path, round_name: str) -> None:  # pragma: no cover (Hub)
    from huggingface_hub import hf_hub_download

    card = hf_hub_download(REPO, "README.md", repo_type="dataset", local_dir=work / "card")
    text = Path(card).read_text(encoding="utf-8")
    (out / "README.md").write_text(readme.migrate(text, round_name), encoding="utf-8")


def _verify(out: Path, round_name: str) -> None:  # pragma: no cover (reads the staged folder)
    data_files = sorted((out / "data").rglob("*.parquet"))
    decisions = sorted((out / "labels" / round_name).glob("decisions-*.parquet"))
    data_rows, decision_rows = migrate.rows(data_files), migrate.rows(decisions)
    migrate.check_migrated(data_rows, decision_rows, migrate.inline_labels(data_files))
    print(f"round: verified {data_rows} rows, {len(data_files)} data files", flush=True)


def _stage_generations(gen_buckets: Path, src: Path) -> dict[str, set[str]]:  # pragma: no cover
    from huggingface_hub import snapshot_download

    recorded: dict[str, set[str]] = defaultdict(set)
    for source, repo in GENERATION_REPOS.items():
        local = Path(
            snapshot_download(
                repo, repo_type="dataset", allow_patterns=["generations/**"], local_dir=src / source
            )
        )
        for path in sorted((local / "generations").rglob("*.parquet")):
            partition.write_buckets(
                partition.generation_file_rows(path, source), gen_buckets, f"{source}-{path.stem}"
            )
            _record(recorded, path)
            path.unlink()
    return recorded


def _record(recorded: dict[str, set[str]], path: Path) -> None:  # pragma: no cover
    for column in RECORDED:
        recorded[column].update(pl.read_parquet(path, columns=[column])[column].unique().to_list())


def _stage_corpus(work: Path, out: Path) -> Path:  # pragma: no cover (Hub)
    from huggingface_hub import snapshot_download

    local = Path(
        snapshot_download(
            REPO, repo_type="dataset", allow_patterns=["data/**"], local_dir=work / "src" / "corpus"
        )
    )
    buckets = work / "corpus"
    for path in sorted((local / "data").rglob("*.parquet")):
        relative = path.relative_to(local)
        partition.write_buckets(partition.corpus_file_rows(path), buckets, "-".join(relative.parts[1:]))
        migrate.strip_label(path, out / relative)
    return buckets


def _write_round(  # pragma: no cover
    out: Path, found: rounds.Round, corpus_buckets: Path, gen_buckets: Path
) -> None:
    folder = out / "labels" / found.name
    folder.mkdir(parents=True, exist_ok=True)
    totals = dict.fromkeys(TOTALS, 0)
    for bucket in partition.BUCKETS:
        decisions, raws, agreed = partition.join_bucket(bucket, corpus_buckets, gen_buckets)
        decisions.write_parquet(folder / f"decisions-{bucket}.parquet")
        raws.write_parquet(folder / f"generations-{bucket}.parquet")
        totals = {key: totals[key] + agreed[key] for key in TOTALS}
    (folder / "agreement.json").write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    rounds.write_registry(out / "labels" / "rounds.json", {found.name: found})


def propose(out: Path) -> str:  # pragma: no cover (Hub)
    from georeset import hub

    drop = [path.name for path in out.iterdir() if path.name not in KEPT_ON_PR]
    return hub.propose(
        out, REPO, "Round 1: labels moved to a sidecar; data/ without the inline landuse", drop
    )


def finalize(number: int) -> None:  # pragma: no cover (Hub)
    from georeset import hub

    hub.release(REPO, number, TAG)
