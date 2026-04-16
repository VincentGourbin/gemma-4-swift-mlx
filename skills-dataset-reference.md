# ToolsForge Skills — Reference pour dataset LoRA

## Architecture du routage

```
User Input
  → [1] Commande simple ? (efface, scanne X) → execution directe
  → [2] Detection domaine (LLM, 4 tokens) → R (recherche) | A (action)
  → [3] System prompt charge avec skills du domaine
  → [4] LLM genere SKILL + SQL (+ ACTION si domaine action)
  → [5] Post-corrections (extension LIKE, ajout id)
  → [6] Enrichissement WHERE (moteur combine avec activeWhereClause)
  → [7] Execution + extraction WHERE → stockage pour affinage suivant
```

## Schema SQLite

```sql
files(
  id TEXT PRIMARY KEY,
  path TEXT, name TEXT, extension TEXT,
  is_directory INTEGER, size INTEGER,
  category TEXT,  -- image|video|audio|document|code|archive|model|application|system|database|font|other
  parent_path TEXT, depth INTEGER,
  modified_at INTEGER,      -- Unix timestamp
  created_at INTEGER,
  is_local INTEGER,         -- 1=disque, 0=cloud
  cloud_status TEXT,
  is_hidden INTEGER,
  is_symlink INTEGER,
  content_type TEXT,         -- UTI
  pixel_width INTEGER,
  pixel_height INTEGER,
  duration_seconds REAL,     -- duree audio/video
  page_count INTEGER,        -- pages PDF
  author TEXT, title TEXT,
  camera_model TEXT,
  audio_artist TEXT, audio_album TEXT
)
```

## Conversions de taille

| Taille | Octets |
|--------|--------|
| 1 Ko | 1024 |
| 1 Mo | 1048576 |
| 10 Mo | 10485760 |
| 50 Mo | 52428800 |
| 100 Mo | 104857600 |
| 500 Mo | 524288000 |
| 1 Go | 1073741824 |
| 5 Go | 5368709120 |
| 10 Go | 10737418240 |
| 20 Go | 21474836480 |

---

## Etape 1 : Detection du domaine

**Prompt** : "Classe cette demande : R (recherche/analyse) ou A (action sur fichiers). Reponds UNIQUEMENT par R ou A."

### Exemples domaine R (recherche)

| Input | Output |
|-------|--------|
| les 10 plus gros fichiers | R |
| repartition par categorie | R |
| les fichiers safetensors | R |
| fichiers de plus de 1 Go | R |
| par extension | R |
| les videos | R |
| le plus gros fichier | R |
| fichiers dans Downloads | R |
| ou sont les caches | R |
| les 10 plus recents | R |
| fichiers modifies cette semaine | R |
| videos de plus de 5 minutes | R |
| photos prises avec un iPhone | R |
| PDFs de plus de 50 pages | R |
| fichiers caches | R |
| top 10 des extensions | R |
| les .app dans Applications | R |
| modeles HuggingFace | R |
| combien d'espace | R |
| quels types de fichiers | R |

### Exemples domaine A (action)

| Input | Output |
|-------|--------|
| deplace les safetensors dans Archive | A |
| supprime les fichiers de plus de 5 Go | A |
| compresse les videos en zip | A |
| trouve les doublons | A |
| exporte la liste en CSV | A |
| renomme les photos par date | A |
| nettoie les caches | A |
| extrais l'audio de la video | A |
| zippe tout dans un fichier | A |
| bouge les gros fichiers dans /tmp | A |
| jette les fichiers inutiles | A |
| exporte en JSON | A |
| archive les documents | A |
| transfere les modeles dans un autre dossier | A |
| vide le cache HuggingFace | A |

---

## Etape 2 : Skills de RECHERCHE (domaine R)

### Format de sortie

```
SKILL: nom_skill
SQL: SELECT ...
```

### Skill: top_files

**Quand** : lister des fichiers, classement, top N, "montre-moi les X"

| Input | SKILL | SQL |
|-------|-------|-----|
| les 10 plus gros fichiers | top_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 10 |
| les fichiers safetensors | top_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%safetensors%' AND is_directory = 0 ORDER BY size DESC |
| les safetensors de plus de 500 Mo | top_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%safetensors%' AND size > 524288000 AND is_directory = 0 ORDER BY size DESC |
| toutes les videos | top_files | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| les .app dans Applications | top_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%app%' AND path LIKE '%Applications%' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| le plus gros fichier | top_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 1 |
| les 10 plus recents | top_files | SELECT id, name, path, size, modified_at FROM files WHERE is_directory = 0 ORDER BY modified_at DESC LIMIT 10 |
| les 5 plus petits | top_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size ASC LIMIT 5 |
| les images de plus de 10 Mo | top_files | SELECT id, name, path, size FROM files WHERE category = 'image' AND size > 10485760 AND is_directory = 0 ORDER BY size DESC |
| photos prises avec un iPhone | top_files | SELECT id, name, path, size, camera_model FROM files WHERE category = 'image' AND camera_model LIKE '%iPhone%' AND is_directory = 0 ORDER BY size DESC |
| les fichiers audio | top_files | SELECT id, name, path, size FROM files WHERE category = 'audio' AND is_directory = 0 ORDER BY size DESC |
| les modeles IA | top_files | SELECT id, name, path, size FROM files WHERE category = 'model' AND is_directory = 0 ORDER BY size DESC |
| les archives | top_files | SELECT id, name, path, size FROM files WHERE category = 'archive' AND is_directory = 0 ORDER BY size DESC |
| les fichiers python | top_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%py%' AND is_directory = 0 ORDER BY size DESC |
| les PDFs | top_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%pdf%' AND is_directory = 0 ORDER BY size DESC |

### Skill: category_breakdown

**Quand** : repartition par categorie, par type

| Input | SKILL | SQL |
|-------|-------|-----|
| repartition par categorie | category_breakdown | SELECT category, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 GROUP BY category ORDER BY taille_totale DESC |
| par type | category_breakdown | SELECT category, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 GROUP BY category ORDER BY taille_totale DESC |
| quels types de fichiers | category_breakdown | SELECT category, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 GROUP BY category ORDER BY taille_totale DESC |

### Skill: extension_breakdown

**Quand** : repartition par extension, les N plus grosses extensions

| Input | SKILL | SQL |
|-------|-------|-----|
| par extension | extension_breakdown | SELECT extension, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 AND extension IS NOT NULL GROUP BY extension ORDER BY taille_totale DESC LIMIT 30 |
| les 10 plus grosses extensions | extension_breakdown | SELECT extension, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 AND extension IS NOT NULL GROUP BY extension ORDER BY taille_totale DESC LIMIT 10 |
| quelles extensions | extension_breakdown | SELECT extension, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 AND extension IS NOT NULL GROUP BY extension ORDER BY taille_totale DESC LIMIT 30 |

### Skill: filter_size

**Quand** : filtrer par taille

| Input | SKILL | SQL |
|-------|-------|-----|
| fichiers de plus de 1 Go | filter_size | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 1073741824 ORDER BY size DESC |
| moins de 100 Mo | filter_size | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size < 104857600 ORDER BY size DESC |
| entre 500 Mo et 5 Go | filter_size | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 524288000 AND size < 5368709120 ORDER BY size DESC |
| fichiers modifies cette semaine | filter_size | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND modified_at > strftime('%s', 'now', '-7 days') ORDER BY modified_at DESC |

### Skill: filter_path

**Quand** : filtrer par chemin/dossier

| Input | SKILL | SQL |
|-------|-------|-----|
| fichiers dans Downloads | filter_path | SELECT id, name, path, size FROM files WHERE path LIKE '%Downloads%' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| tout dans Documents | filter_path | SELECT id, name, path, size FROM files WHERE path LIKE '%Documents%' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| ou sont les caches | filter_path | SELECT id, name, path, size FROM files WHERE path LIKE '%Caches%' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| modeles HuggingFace | filter_path | SELECT id, name, path, size FROM files WHERE path LIKE '%huggingface%' AND is_directory = 0 ORDER BY size DESC LIMIT 500 |
| fichiers dans le cloud | filter_path | SELECT id, name, path, size FROM files WHERE is_local = 0 AND is_directory = 0 ORDER BY size DESC |

### Skill: filter_name

**Quand** : chercher par nom de fichier

| Input | SKILL | SQL |
|-------|-------|-----|
| fichiers contenant backup | filter_name | SELECT id, name, path, size FROM files WHERE name LIKE '%backup%' AND is_directory = 0 ORDER BY size DESC |
| les README | filter_name | SELECT id, name, path, size FROM files WHERE name LIKE '%README%' AND is_directory = 0 ORDER BY size DESC |
| fichiers caches | filter_name | SELECT id, name, path, size FROM files WHERE is_hidden = 1 AND is_directory = 0 ORDER BY size DESC |

### Skill: filter_extension

**Quand** : FILTRER la treemap (pas lister) sur une extension

| Input | SKILL | SQL |
|-------|-------|-----|
| filtre les .py | filter_extension | SELECT id, name, path, size FROM files WHERE extension LIKE '%py%' AND is_directory = 0 ORDER BY size DESC |
| filtre sur les videos | filter_extension | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 ORDER BY size DESC |
| videos de plus de 5 minutes | filter_extension | SELECT id, name, path, size, duration_seconds FROM files WHERE category = 'video' AND is_directory = 0 AND duration_seconds > 300 ORDER BY duration_seconds DESC |

### Skill: navigate

**Quand** : naviguer dans un dossier

| Input | SKILL | SQL |
|-------|-------|-----|
| va dans Downloads | navigate | SELECT id, name, path, size FROM files WHERE is_directory = 1 AND path LIKE '%Downloads%' LIMIT 1 |
| ouvre Documents | navigate | SELECT id, name, path, size FROM files WHERE is_directory = 1 AND path LIKE '%Documents%' LIMIT 1 |

---

## Etape 2 : Skills d'ACTION (domaine A)

### Format de sortie

```
SKILL: nom_skill
SQL: SELECT id, name, path, size FROM files WHERE ...
ACTION: {"type": "...", ...}
```

### Skill: move_files

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| deplace les safetensors dans /tmp/archive | move_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%safetensors%' AND is_directory = 0 ORDER BY size DESC | {"type":"move","destination":"/tmp/archive"} |
| bouge les gros fichiers dans /Volumes/Externe | move_files | SELECT id, name, path, size FROM files WHERE size > 1073741824 AND is_directory = 0 ORDER BY size DESC | {"type":"move","destination":"/Volumes/Externe"} |
| transfere les modeles dans Archive | move_files | SELECT id, name, path, size FROM files WHERE category = 'model' AND is_directory = 0 ORDER BY size DESC | {"type":"move","destination":"/Users/vincent/Archive"} |

### Skill: delete_files

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| supprime les fichiers de plus de 5 Go | delete_files | SELECT id, name, path, size FROM files WHERE size > 5368709120 AND is_directory = 0 ORDER BY size DESC | {"type":"delete"} |
| jette les fichiers temporaires | delete_files | SELECT id, name, path, size FROM files WHERE path LIKE '%tmp%' AND is_directory = 0 ORDER BY size DESC | {"type":"delete"} |
| efface les .log | delete_files | SELECT id, name, path, size FROM files WHERE extension LIKE '%log%' AND is_directory = 0 ORDER BY size DESC | {"type":"delete"} |

### Skill: rename_files

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| renomme les photos avec la date | rename_files | SELECT id, name, path, size FROM files WHERE category = 'image' AND is_directory = 0 ORDER BY name | {"type":"rename","pattern":"{date}_{name}.{ext}"} |
| ajoute _backup au nom | rename_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY name | {"type":"rename","pattern":"{name}_backup.{ext}"} |

### Skill: compress_files

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| compresse les videos dans /tmp/backup.zip | compress_files | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 ORDER BY size DESC | {"type":"compress","destination":"/tmp/backup.zip"} |
| zippe les documents | compress_files | SELECT id, name, path, size FROM files WHERE category = 'document' AND is_directory = 0 ORDER BY size DESC | {"type":"compress","destination":"/tmp/documents.zip"} |

### Skill: duplicate_files

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| trouve les doublons | duplicate_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size DESC | {"type":"duplicates"} |
| fichiers en double | duplicate_files | SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size DESC | {"type":"duplicates"} |

### Skill: export_list

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| exporte la liste en CSV | export_list | SELECT name, path, size, extension, category FROM files WHERE is_directory = 0 ORDER BY size DESC | {"type":"export","format":"csv"} |
| exporte en JSON | export_list | SELECT name, path, size, extension, category FROM files WHERE is_directory = 0 ORDER BY size DESC | {"type":"export","format":"json"} |

### Skill: clean_caches

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| nettoie les caches | clean_caches | SELECT id, name, path, size FROM files WHERE path LIKE '%Cache%' AND is_directory = 0 ORDER BY size DESC | {"type":"cleanCaches"} |
| vide le cache HuggingFace | clean_caches | SELECT id, name, path, size FROM files WHERE path LIKE '%huggingface%' AND is_directory = 0 ORDER BY size DESC | {"type":"cleanCaches"} |

### Skill: rename_by_metadata

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| renomme les photos par date EXIF | rename_by_metadata | SELECT id, name, path, size, modified_at FROM files WHERE category = 'image' AND is_directory = 0 | {"type":"renameByMetadata","pattern":"{date}_{name}.{ext}"} |
| organise par date de prise de vue | rename_by_metadata | SELECT id, name, path, size, modified_at FROM files WHERE category = 'image' AND is_directory = 0 | {"type":"renameByMetadata","pattern":"{date}_{name}.{ext}"} |

### Skill: split_audio_video

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| extrais l'audio de la video | split_audio_video | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 | {"type":"splitMedia"} |
| separe audio et video | split_audio_video | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 | {"type":"splitMedia"} |

### Skill: audit_codecs

| Input | SKILL | SQL | ACTION |
|-------|-------|-----|--------|
| quels videos ne sont pas en HEVC | audit_codecs | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 | {"type":"auditCodecs"} |
| audit des codecs video | audit_codecs | SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 | {"type":"auditCodecs"} |

---

## Moteur d'affinage (combinaison WHERE)

Le moteur combine programmatiquement le WHERE actif avec le nouveau SQL du LLM.
Le LLM ne doit PAS reprendre les filtres precedents — il genere juste sa nouvelle intention.

### Exemples d'affinage

| Contexte actif | Input utilisateur | SQL LLM | SQL final apres enrichissement |
|----------------|-------------------|---------|-------------------------------|
| (aucun) | les videos | WHERE category='video' AND is_directory=0 LIMIT 10 | WHERE category='video' AND is_directory=0 LIMIT 10 |
| category='video' | le plus gros | WHERE is_directory=0 LIMIT 1 | WHERE category='video' AND is_directory=0 LIMIT 1 |
| category='video' | de plus de 100 Mo | WHERE size > 104857600 AND is_directory=0 | WHERE category='video' AND size > 104857600 AND is_directory=0 |
| category='video' AND size > 104857600 | exporte en CSV | WHERE is_directory=0 | WHERE category='video' AND size > 104857600 AND is_directory=0 |
| (aucun) | les safetensors | WHERE extension LIKE '%safetensors%' AND is_directory=0 | WHERE extension LIKE '%safetensors%' AND is_directory=0 |
| extension LIKE '%safetensors%' | de moins de 20 Go | WHERE size < 21474836480 AND is_directory=0 | WHERE extension LIKE '%safetensors%' AND size < 21474836480 AND is_directory=0 |
| extension LIKE '%safetensors%' AND size < 21474836480 | supprime-les | WHERE is_directory=0 | WHERE extension LIKE '%safetensors%' AND size < 21474836480 AND is_directory=0 |

### Reset du contexte

| Input | Effet |
|-------|-------|
| efface | activeWhereClause = nil |
| reset | activeWhereClause = nil |
| clear | activeWhereClause = nil |
| nouveau scan | activeWhereClause = nil |

---

## Regles importantes pour le LLM

1. Toujours utiliser `LIKE '%xxx%'` pour les extensions (pas `=`)
2. Toujours inclure `is_directory = 0` pour les fichiers (pas les dossiers)
3. Les aggregations (category_breakdown, extension_breakdown) n'ont PAS de colonne `id`
4. Les conversions de taille doivent etre EXACTES (voir table)
5. Les dates utilisent `strftime('%s', 'now', '-N days')` pour les periodes relatives
6. Le LLM genere la NOUVELLE intention seulement — le moteur combine avec le contexte
7. Pour les actions, toujours inclure la ligne ACTION avec le JSON
8. Les destinations doivent etre des chemins ABSOLUS
