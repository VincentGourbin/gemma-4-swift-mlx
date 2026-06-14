# GUI Grounding Bench — DiffusionGemma vs Gemma 4 AR (ScreenSpot v1, 100 cas)

Comparaison de la qualité **GUI grounding** (prédiction de la position du clic à partir d'une instruction + screenshot UI) sur un échantillon stratifié de [ScreenSpot v1](https://github.com/njucckevin/SeeClick) — le standard académique des benchs de grounding pour automation d'UI.

## Setup

| Component | Specification |
|-----------|--------------|
| **Hardware** | Mac Studio M3 Max 96 GB |
| **Source** | ScreenSpot v1 (njucckevin/SeeClick) |
| **Sample** | 100 cas stratifiés (33 windows / 34 macos / 33 ios, 62 text / 38 icon) |
| **Prompt** | `CLICK: (x=0.XX, y=0.XX)` au format coordonnées normalisées [0, 1] |
| **Scoring** | Le point prédit doit tomber dans la bbox ground truth |

## Résultats globaux

| Modèle | Score | Total time | Avg/case |
|---|---|---|---|
| **Gemma 4 E4B 4-bit AR** | **31/100 (31.0%)** | 9.7 min | 5.8 s |
| **Gemma 4 26B-A4B 4-bit AR** | **58/100 (58.0%)** | 10.8 min | 6.5 s |
| **DiffusionGemma 26B-A4B bf16** | **79/100 (79.0%)** ⭐ | 43.1 min | 25.8 s |

**🎯 DiffusionGemma écrase de +21 points le 26B-A4B AR sur le même backbone** — l'avantage est encore plus marqué que sur OCR (+1.9 pts à 1000 images).

## Détail par source (windows / macos / ios)

| Source | E4B 4-bit | 26B-A4B 4-bit | DiffusionGemma bf16 |
|---|:---:|:---:|:---:|
| **windows** (33) | 13/33 (39%) | 22/33 (67%) | **27/33 (82%)** ⭐ |
| **macos** (34) | 8/34 (24%) | 18/34 (53%) | **25/34 (74%)** ⭐ |
| **ios** (33) | 10/33 (30%) | 18/33 (55%) | **27/33 (82%)** ⭐ |

DiffusionGemma gagne sur toutes les plateformes, avec un avantage particulièrement net sur **iOS** (+27 pts vs A4B AR).

## Détail par type (text / icon)

| Type | E4B 4-bit | 26B-A4B 4-bit | DiffusionGemma bf16 |
|---|:---:|:---:|:---:|
| **text** (62) | 23/62 (37%) | 43/62 (69%) | **58/62 (94%)** ⭐⭐ |
| **icon** (38) | 8/38 (21%) | 15/38 (39%) | **21/38 (55%)** ⭐ |

**DiffusionGemma sur text = 94%** — presque parfait. Trouve quasi-systématiquement le bon élément textuel. L'écart entre text et icon (94% vs 55%) reflète une difficulté générale du modèle vision sur les icônes sans label.

## Croisé source × type

| source/type | E4B 4-bit | 26B-A4B 4-bit | DiffusionGemma bf16 |
|---|:---:|:---:|:---:|
| windows/text | 8/19 (42%) | 13/19 (68%) | **18/19 (95%)** |
| windows/icon | 5/14 (36%) | 9/14 (64%) | 9/14 (64%) |
| macos/text | 7/24 (29%) | 16/24 (67%) | **21/24 (88%)** |
| macos/icon | 1/10 (10%) | 2/10 (20%) | **4/10 (40%)** |
| ios/text | 8/19 (42%) | 14/19 (74%) | **19/19 (100%)** ⭐ |
| ios/icon | 2/14 (14%) | 4/14 (29%) | **8/14 (57%)** |

🎯 **DiffusionGemma sur ios/text : 100% (19/19)** — score parfait. C'est le résultat le plus remarquable de toute la session de benchs.

## Analyse

### Pourquoi DiffusionGemma écrase autant l'AR ici ?

GUI grounding = production d'un **format structuré strict** (`CLICK: (x=0.XX, y=0.XX)`). L'AR produit parfois :
- Coordonnées non-normalisées (en pixels : `click=(40.0, 191.0)`) → fail
- Coordonnées partielles (`click=(0.87, 935.0)`) → fail
- Pas de coordonnées du tout (`click=None`)

DiffusionGemma :
- **Block-AR diffusion** : refine le canvas itérativement avant de commit → respect du format plus rigoureux
- **Self-conditioning** : le draft précédent guide la convergence vers le format demandé
- **Entropy-bound sampling** : réduit la variance sur des positions critiques

Le résultat est cohérent avec ce qu'on a vu sur OCR (DiffusionGemma > A4B sur Doc VQA +10 pts) : **les tâches qui demandent un format structuré précis bénéficient de la diffusion**.

### Compromis vitesse vs qualité

| Modèle | Acc | tok/s effectif | Use case |
|---|---|---|---|
| E4B 4-bit | 31% | ~10 t/s | Trop faible pour production |
| **26B-A4B 4-bit AR** | 58% | ~10 t/s | Compromis qualité/vitesse |
| **DiffusionGemma bf16** | **79%** | ~3 t/s | Qualité maximale, batch / offline |

DiffusionGemma est 4× plus lent en wall-clock mais 21 points de précision — vaut totalement le coût en mode automation où chaque erreur de clic = retry ou échec de la tâche.

### Limitations méthodologiques

1. **100 cas** — bonne stratification (3 plateformes × 2 types) mais petit pour des écarts < 5 pts.
2. **Format de prompt naïf** — on demande `CLICK: (x, y)` en normalisé. Les modèles AR produisent parfois du pixel-coord ; le scoring strict pénalise ce reformatage.
3. **bbox tolérance** — point doit être dans la bbox ground truth. ScreenSpot v2 et ScreenSpot-Pro utilisent une tolérance distance plus fine.
4. **Pas de chain-of-thought** — un système prompt incitant à raisonner avant de produire les coordonnées pourrait booster l'AR.

## Reproduction

```bash
# 1. Télécharger ScreenSpot v1 (depuis HuggingFace)
mkdir -p /tmp/screenspot/data && cd /tmp/screenspot/data
huggingface-cli download njucckevin/ScreenSpot --repo-type dataset --local-dir .

# 2. Extraire un sample stratifié 100 (voir prepare_sample.py)
python3 prepare_sample.py  # produit sample/meta.json + sample/img_*.png

# 3. Lancer les 3 modèles
python3 run_ss.py e4b
python3 run_ss.py a4b
python3 run_ss.py diff
```

Le `run_ss.py` construit le prompt CLICK normalisé, lance la CLI `gemma4-cli describe` (AR) ou `gemma4-cli diffusion --include-vision --image ...` (DiffusionGemma), parse les coordonnées par regex, vérifie l'inclusion dans la bbox.

## Récap des 3 benchs DiffusionGemma vs AR

| Bench | E4B 4-bit | 26B-A4B 4-bit | DiffusionGemma bf16 | Gap Diff vs A4B |
|---|---|---|---|---|
| **OCRBench** (1000 imgs) | 70.3% | 78.9% | 80.8% | +1.9 pts |
| **BFCL** (100 cas) | 95% | 95% | 95% | 0 pt (parité) |
| **ScreenSpot** (100 cas) | 31% | 58% | **79%** | **+21 pts** ⭐⭐ |

DiffusionGemma ne brille pas partout (BFCL parité, OCR léger avantage) mais **explose sur les tâches de grounding structuré** (ScreenSpot). C'est sa niche.

## Fichiers

- `diff_summary.json` / `a4b_summary.json` / `e4b_summary.json` — détail des 100 cas par modèle (click prédit + correct + elapsed + output snippet)
- `run_ss.py` — pipeline d'évaluation
- `sample/meta.json` — 100 cas stratifiés (image, bbox normalisée, instruction)
