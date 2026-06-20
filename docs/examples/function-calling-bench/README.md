# Function Calling Bench — DiffusionGemma vs Gemma 4 AR (mini BFCL)

Comparaison de la qualité function calling sur **100 cas** échantillonnés du [Berkeley Function-Calling Leaderboard (BFCL) v3](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard) — le standard de facto pour évaluer les LLMs sur l'invocation de fonctions / tool use.

## Setup

| Component | Specification |
|-----------|--------------|
| **Hardware** | Mac Studio M3 Max 96 GB |
| **Source** | BFCL v3 (gorilla-llm) |
| **Sample** | 100 cas stratifiés (30 simple + 30 multiple + 20 parallel + 20 irrelevance) |
| **Prompt format** | Injection des fonctions JSON + instructions "FUNCTION_CALL: name(args) ou NO_CALL: explanation" |
| **Scoring** | Match du nom de fonction + au moins 50% des args matchent les valeurs acceptées (BFCL convention) |

## Catégories testées

| Catégorie | N | Description |
|---|---|---|
| **simple** | 30 | 1 fonction disponible, 1 appel attendu |
| **multiple** | 30 | Plusieurs fonctions disponibles, choisir la bonne |
| **parallel** | 20 | Plusieurs appels simultanés à la même fonction |
| **irrelevance** | 20 | Question ne nécessitant PAS d'appel, doit répondre NO_CALL |

## Résultats globaux

| Modèle | Score | Total time |
|---|---|---|
| **Gemma 4 26B-A4B 4-bit AR** | **95/100 (95%)** | 8.8 min |
| **Gemma 4 E4B 4-bit AR** | **95/100 (95%)** | **8.0 min** ⚡ |
| **DiffusionGemma 26B-A4B bf16** | **95/100 (95%)** | 26.0 min |

**🎯 Parité parfaite des 3 modèles sur le score** — chacun à 95% avec 5 échecs.

## Détail par catégorie

| Catégorie | E4B 4-bit | 26B-A4B 4-bit | DiffusionGemma bf16 |
|---|:---:|:---:|:---:|
| **simple** (30) | 29/30 (97%) | 29/30 (97%) | 29/30 (97%) |
| **multiple** (30) | **30/30 (100%)** ⭐ | **30/30 (100%)** ⭐ | **30/30 (100%)** ⭐ |
| **parallel** (20) | 19/20 (95%) | **20/20 (100%)** ⭐ | **20/20 (100%)** ⭐ |
| **irrelevance** (20) | **17/20 (85%)** ⭐ | 16/20 (80%) | 16/20 (80%) |

### Stats de performance

| Modèle | Avg time/case | Tokens/case (approx) |
|---|---|---|
| E4B 4-bit AR | 4.4–5.6 s | ~30-60 tokens |
| 26B-A4B 4-bit AR | 4.8–6.6 s | ~30-60 tokens |
| **DiffusionGemma bf16** | **15–17 s** | **256 (canvas plein)** |

DiffusionGemma est ~3× plus lent par cas (5.4s baseline → 15s ici car prompt très long avec definitions JSON). Mais qualité strictement équivalente.

## Analyse des échecs

### Échecs communs aux 3 modèles (4 cas)

Cas borderline où les modèles font le même choix débattable :

| Cas ID | Catégorie | Pattern |
|---|---|---|
| `simple_13` | simple | `calculate_area_under_curve(function="x^2"...)` — ground truth strict sur le format `function=` |
| `irrelevance_68` | irrelevance | `statistics.calculate_p_value` — modèles appellent quand BFCL attend NO_CALL |
| `irrelevance_165` | irrelevance | `get_instrument_info(instrument_name="cello")` — modèles appellent (fonction semble adaptée) |
| `irrelevance_239` | irrelevance | `get_date(location_1=...)` — idem |

→ Pour les "irrelevance" : la fonction disponible **semble** raisonnable pour la question, les modèles tentent l'appel. BFCL considère que c'est faux. C'est un choix de stratégie : conservative (NO_CALL si doute) vs aggressive (call si plausible).

### Échecs uniques

**DiffusionGemma 26B-A4B** et **A4B 4-bit AR** partagent exactement les **mêmes 5 échecs** = parité parfaite de comportement.

**E4B 4-bit** :
- ❌ `parallel_49` : refuse à tort (faux NO_CALL)
- ✅ `irrelevance_118` : refuse à raison (vrai NO_CALL)

→ E4B a une **politique légèrement plus conservative** : refuse plus souvent, ce qui le sauve 1× sur irrelevance et le coûte 1× sur parallel. Score net identique.

## Verdict

### 🎯 DiffusionGemma fait du function calling AUSSI BIEN que son équivalent AR

- **Score identique** : 95% chacun
- **Mêmes 5 échecs** entre DiffusionGemma et 26B-A4B AR → comportement structurellement identique
- **Multiple** et **Parallel** : 100% des deux → le block-AR diffusion n'a aucun handicap sur les tâches structurées multi-appels

### ⚖ Compromis vitesse vs RAM

| Choix | Stratégie |
|---|---|
| **Qualité maximale + perf** | E4B 4-bit AR (95%, 8 min, 6 GB RAM) |
| **Qualité maximale équivalente** | 26B-A4B 4-bit AR (95%, 8.8 min, 16 GB) |
| **Block-AR diffusion** | DiffusionGemma bf16 (95%, 26 min, 51 GB) |

**E4B 4-bit gagne sur le rapport qualité/perf/RAM** pour le function calling pur. DiffusionGemma a tous ses avantages sur les autres tâches (OCR contextuel : +1.9 pts à 1000 images, Doc VQA +10 pts) mais pas d'avantage spécifique sur le function calling.

### 🔍 Limitations méthodologiques

1. **100 cas est limité** pour détecter des écarts de 1-2 points. BFCL complet = ~4000 cas.
2. **Scoring par substring/50%-args-match** est tolérant ; BFCL officiel utilise exact match avec normalisation.
3. **Categories manquantes** : multi_turn (conversations), live (real-world), language-specific (Java/JS/SQL).
4. **Prompt artificiel** : injection en JSON dans le user message au lieu du chat template natif Gemma 4 avec tokens `<|tool>`. Le template natif pourrait donner de meilleurs résultats.

## Reproduction

```bash
# 1. Télécharger BFCL files
mkdir -p /tmp/bfcl && cd /tmp/bfcl
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_parallel BFCL_v3_irrelevance; do
  curl -L "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/$f.json" -o "$f.json"
done
for f in BFCL_v3_simple BFCL_v3_multiple BFCL_v3_parallel; do
  curl -L "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main/possible_answer/$f.json" -o "answer_$f.json"
done

# 2. Use cases_100.json (sample stratifié) + run_bfcl100.py + run_all100.sh
cp <repo>/docs/examples/function-calling-bench/cases_100.json sample100/
./run_all100.sh
```

## Fichiers

- `cases_100.json` : 100 cas BFCL stratifiés (avec ground truth)
- `{diff,a4b,e4b}_summary.json` : 300 entrées chacun avec scoring + perf
- `run_bfcl100.py` : pipeline d'évaluation
- `run_all100.sh` : enchaîne les 3 modèles
