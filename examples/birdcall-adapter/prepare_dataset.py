#!/usr/bin/env python3
"""Full birdcall dataset with rich JSON responses — all 9595 samples."""

import json, os, random
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

OUTPUT_DIR = "/tmp/birdcall-full-rich"
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
SEED = 42
VALID_RATIO = 0.1

SPECIES_INFO = {
    "amerob": {"common_name": "American Robin", "scientific_name": "Turdus migratorius", "family": "Turdidae", "order": "Passeriformes", "habitat": "gardens, woodlands, parks", "call_description": "melodic whistled phrases, cheerily cheer-up cheerio"},
    "barswa": {"common_name": "Barn Swallow", "scientific_name": "Hirundo rustica", "family": "Hirundinidae", "order": "Passeriformes", "habitat": "open areas near buildings, farms", "call_description": "twittering warble with grating notes"},
    "bewwre": {"common_name": "Bewick's Wren", "scientific_name": "Thryomanes bewickii", "family": "Troglodytidae", "order": "Passeriformes", "habitat": "scrubby areas, thickets, gardens", "call_description": "complex buzzy trills and warbles"},
    "bncfly": {"common_name": "Black-capped Flycatcher", "scientific_name": "Empidonax atriceps", "family": "Tyrannidae", "order": "Passeriformes", "habitat": "highland forests, cloud forests", "call_description": "sharp pit or whit calls"},
    "carwre": {"common_name": "Carolina Wren", "scientific_name": "Thryothorus ludovicianus", "family": "Troglodytidae", "order": "Passeriformes", "habitat": "dense undergrowth, suburban gardens", "call_description": "loud rolling teakettle-teakettle-teakettle"},
    "comrav": {"common_name": "Common Raven", "scientific_name": "Corvus corax", "family": "Corvidae", "order": "Passeriformes", "habitat": "mountains, forests, coasts, deserts", "call_description": "deep resonant croaking pruk-pruk"},
    "cubthr": {"common_name": "Cuban Thrush", "scientific_name": "Turdus plumbeus", "family": "Turdidae", "order": "Passeriformes", "habitat": "tropical forests, gardens in Caribbean", "call_description": "rich fluty phrases with pauses"},
    "eursta": {"common_name": "European Starling", "scientific_name": "Sturnus vulgaris", "family": "Sturnidae", "order": "Passeriformes", "habitat": "urban areas, fields, open woodlands", "call_description": "varied mimicry, whistles, clicks, rattles"},
    "gbwwre1": {"common_name": "Great Bowerbird Wren", "scientific_name": "Malurus cyaneus", "family": "Maluridae", "order": "Passeriformes", "habitat": "dense scrub, heathlands", "call_description": "high-pitched reeling trills"},
    "houspa": {"common_name": "House Sparrow", "scientific_name": "Passer domesticus", "family": "Passeridae", "order": "Passeriformes", "habitat": "urban and suburban areas worldwide", "call_description": "persistent chirping cheep-cheep"},
    "houwre": {"common_name": "House Wren", "scientific_name": "Troglodytes aedon", "family": "Troglodytidae", "order": "Passeriformes", "habitat": "thickets, gardens, forest edges", "call_description": "energetic bubbling cascade of notes"},
    "mallar3": {"common_name": "Mallard", "scientific_name": "Anas platyrhynchos", "family": "Anatidae", "order": "Anseriformes", "habitat": "wetlands, lakes, rivers, parks", "call_description": "loud quacking, descending series"},
    "norcar": {"common_name": "Northern Cardinal", "scientific_name": "Cardinalis cardinalis", "family": "Cardinalidae", "order": "Passeriformes", "habitat": "woodlands, gardens, shrubby areas", "call_description": "clear whistled birdy-birdy-birdy"},
    "redcro": {"common_name": "Red Crossbill", "scientific_name": "Loxia curvirostra", "family": "Fringillidae", "order": "Passeriformes", "habitat": "coniferous forests", "call_description": "sharp jip-jip flight calls"},
    "rewbla": {"common_name": "Red-winged Blackbird", "scientific_name": "Agelaius phoeniceus", "family": "Icteridae", "order": "Passeriformes", "habitat": "marshes, wetlands, fields", "call_description": "conk-la-ree territorial song"},
    "rubpep1": {"common_name": "Ruby Pepper Shrike", "scientific_name": "Cyclarhis gujanensis", "family": "Vireonidae", "order": "Passeriformes", "habitat": "tropical woodlands, forest edges", "call_description": "loud repetitive whistled phrases"},
    "rucspa1": {"common_name": "Rufous-crowned Sparrow", "scientific_name": "Aimophila ruficeps", "family": "Passerellidae", "order": "Passeriformes", "habitat": "rocky hillsides, dry scrub", "call_description": "rapid stuttering trill dear-dear-dear"},
    "sonspa": {"common_name": "Song Sparrow", "scientific_name": "Melospiza melodia", "family": "Passerellidae", "order": "Passeriformes", "habitat": "brushy areas, marshes, gardens", "call_description": "variable sweet-sweet-sweet followed by trills"},
    "spotow": {"common_name": "Spotted Towhee", "scientific_name": "Pipilo maculatus", "family": "Passerellidae", "order": "Passeriformes", "habitat": "dense chaparral, thickets, forest edges", "call_description": "buzzy drink-your-teeee trill"},
    "swathr": {"common_name": "Swainson's Thrush", "scientific_name": "Catharus ustulatus", "family": "Turdidae", "order": "Passeriformes", "habitat": "dense forests, riparian thickets", "call_description": "ascending spiral of fluty notes"},
}

def main():
    os.makedirs(AUDIO_DIR, exist_ok=True)

    all_samples = []
    for shard in range(6):
        filename = f"data/train-{shard:05d}-of-00006.parquet"
        print(f"Loading shard {shard+1}/6...")
        path = hf_hub_download(repo_id="tglcourse/5s_birdcall_samples_top20", filename=filename, repo_type="dataset")
        table = pq.read_table(path)
        labels = table.column("label").to_pylist()
        audio_col = table.column("audio")
        for i in range(len(labels)):
            all_samples.append((labels[i], len(all_samples), audio_col[i].as_py()))

    print(f"Total: {len(all_samples)} samples")

    random.seed(SEED)
    random.shuffle(all_samples)
    n_valid = int(len(all_samples) * VALID_RATIO)
    valid = all_samples[:n_valid]
    train = all_samples[n_valid:]
    print(f"Train: {len(train)}, Valid: {len(valid)}")

    for split_name, samples in [("train", train), ("valid", valid)]:
        path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(path, "w") as f:
            for sp, idx, audio_struct in samples:
                fn = f"{sp}_{idx:05d}.wav"
                fp = os.path.join(AUDIO_DIR, fn)
                if not os.path.exists(fp):
                    with open(fp, "wb") as af:
                        af.write(audio_struct["bytes"])

                info = SPECIES_INFO[sp]
                response = json.dumps({
                    "species_code": sp,
                    "common_name": info["common_name"],
                    "scientific_name": info["scientific_name"],
                    "family": info["family"],
                    "order": info["order"],
                    "habitat": info["habitat"],
                    "call_description": info["call_description"],
                }, ensure_ascii=False)

                entry = {
                    "messages": [
                        {"role": "user", "content": "Listen to this bird call and identify the species. Respond with a JSON object containing species_code, common_name, scientific_name, family, order, habitat, and call_description."},
                        {"role": "assistant", "content": response},
                    ],
                    "audio": f"audio/{fn}",
                }
                f.write(json.dumps(entry) + "\n")
        print(f"Wrote {len(samples)} to {path}")

    print(f"\nDataset ready at {OUTPUT_DIR}")
    print(f"Audio files: {len(os.listdir(AUDIO_DIR))}")

if __name__ == "__main__":
    main()
