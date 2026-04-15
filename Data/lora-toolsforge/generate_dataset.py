#!/usr/bin/env python3
"""Generateur v2 du dataset ToolsForge — base sur le code source reel"""

import json, random, os, itertools
random.seed(42)

samples = []

def add(user_input, skill, sql, action=None):
    assistant = f"SKILL: {skill}\nSQL: {sql}"
    if action:
        assistant += f"\nACTION: {json.dumps(action, ensure_ascii=False)}"
    samples.append({"messages": [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": assistant}
    ]})

# ============================================================
# CONSTANTES (du code source ToolsForge)
# ============================================================

SIZE_MAP = {
    "1 Ko": 1024, "10 Ko": 10240, "100 Ko": 102400,
    "1 Mo": 1048576, "5 Mo": 5242880, "10 Mo": 10485760,
    "50 Mo": 52428800, "100 Mo": 104857600, "200 Mo": 209715200,
    "500 Mo": 524288000, "1 Go": 1073741824, "2 Go": 2147483648,
    "5 Go": 5368709120, "10 Go": 10737418240, "20 Go": 21474836480,
}

CATEGORIES = {
    "image": ["images", "photos", "les images", "les photos", "fichiers image"],
    "video": ["videos", "les videos", "les films", "fichiers video"],
    "audio": ["fichiers audio", "la musique", "les audios", "les mp3", "les sons"],
    "document": ["documents", "les documents", "les docs", "les textes"],
    "code": ["le code", "les scripts", "les sources", "fichiers de code"],
    "archive": ["archives", "les archives", "les zip", "fichiers compresses"],
    "model": ["modeles", "les modeles", "modeles IA", "les modeles ML", "fichiers modele"],
    "application": ["applications", "les apps", "les .app"],
    "system": ["fichiers systeme", "les plist", "les configs", "fichiers de config"],
    "database": ["les bases de donnees", "fichiers base de donnees", "les sqlite"],
    "font": ["les polices", "fichiers de police", "les fonts"],
}

EXTENSIONS = {
    # image
    "jpg": "jpg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp",
    "heic": "heic", "heif": "heif", "tiff": "tiff", "svg": "svg", "bmp": "bmp",
    "raw": "raw", "cr2": "cr2", "ico": "ico", "icns": "icns",
    # video
    "mp4": "mp4", "mov": "mov", "avi": "avi", "mkv": "mkv", "m4v": "m4v", "webm": "webm",
    # audio
    "mp3": "mp3", "m4a": "m4a", "flac": "flac", "wav": "wav", "aac": "aac", "ogg": "ogg",
    # document
    "pdf": "pdf", "doc": "doc", "docx": "docx", "xls": "xls", "xlsx": "xlsx",
    "ppt": "ppt", "pptx": "pptx", "txt": "txt", "csv": "csv", "epub": "epub", "rtf": "rtf",
    # code
    "py": "py", "swift": "swift", "js": "js", "ts": "ts", "jsx": "jsx", "tsx": "tsx",
    "c": "c", "cpp": "cpp", "h": "h", "java": "java", "go": "go", "rs": "rs",
    "rb": "rb", "php": "php", "css": "css", "html": "html", "sh": "sh", "md": "md",
    # archive
    "zip": "zip", "tar": "tar", "gz": "gz", "rar": "rar", "7z": "7z", "dmg": "dmg",
    # model
    "safetensors": "safetensors", "gguf": "gguf", "onnx": "onnx",
    "mlmodel": "mlmodel", "mlpackage": "mlpackage", "pt": "pt", "bin": "bin",
    # system
    "plist": "plist", "log": "log", "json": "json", "xml": "xml",
    "yaml": "yaml", "yml": "yml", "toml": "toml", "ini": "ini", "env": "env",
    # database
    "db": "db", "sqlite": "sqlite", "sqlite3": "sqlite3", "sql": "sql",
    # font
    "ttf": "ttf", "otf": "otf", "woff": "woff", "woff2": "woff2",
    # application
    "app": "app", "pkg": "pkg", "dylib": "dylib",
    # other
    "DS_Store": "DS_Store",
}

PATHS = {
    "Documents": "Documents", "Downloads": "Downloads", "Desktop": "Desktop",
    "Applications": "Applications", "Library": "Library",
    "Pictures": "Pictures", "Movies": "Movies", "Music": "Music",
    "le Bureau": "Desktop", "les telechargements": "Downloads",
    "le dossier Documents": "Documents", "le dossier Music": "Music",
    "le dossier Images": "Pictures", "le dossier Films": "Movies",
    "Developpements": "Developpements", "le dossier dev": "Developpements",
}

CACHE_PATHS = {
    "les caches": "Cache", "le cache": "Cache", "le cache systeme": "Cache",
    "le cache HuggingFace": "huggingface", "les caches Xcode": "Xcode",
    "le cache npm": "npm", "le cache pip": "pip", "le cache brew": "homebrew",
    "le cache de HuggingFace": "huggingface",
}

DURATIONS = [(1, 60), (2, 120), (5, 300), (10, 600), (30, 1800), (60, 3600)]
PAGES = [5, 10, 20, 50, 100, 200, 500]
CAMERAS = ["iPhone", "Canon", "Sony", "Fujifilm", "Nikon", "GoPro", "Samsung"]

DESTINATIONS = [
    "/tmp/archive", "/tmp/backup", "/Volumes/Externe", "/Volumes/SSD",
    "~/Archive", "~/Documents/old", "~/Desktop/tri", "/Volumes/NAS",
]

# ============================================================
# SKILL: top_files (le plus frequent)
# ============================================================

# Top N par taille
for n in [1, 3, 5, 10, 15, 20, 50, 100, 200, 500]:
    phrases = [
        f"les {n} plus gros fichiers", f"top {n} par taille",
        f"montre les {n} plus gros", f"les {n} fichiers les plus lourds",
        f"affiche les {n} plus volumineux", f"les {n} plus gros",
    ]
    if n == 1:
        phrases += ["le plus gros fichier", "le fichier le plus lourd", "le plus gros", "le plus volumineux"]
    for p in phrases:
        add(p, "top_files",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT {n}")

# Top N par date
for n in [5, 10, 20, 50]:
    for p in [f"les {n} plus recents", f"les {n} derniers modifies",
              f"les {n} derniers fichiers", f"fichiers recents (top {n})"]:
        add(p, "top_files",
            f"SELECT id, name, path, size, modified_at FROM files WHERE is_directory = 0 ORDER BY modified_at DESC LIMIT {n}")

# Plus petits
for n in [5, 10, 20]:
    for p in [f"les {n} plus petits", f"les {n} plus legers", f"les {n} fichiers les plus petits"]:
        add(p, "top_files",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size ASC LIMIT {n}")

# Par categorie
for cat, synonyms in CATEGORIES.items():
    for syn in synonyms:
        for verb in ["", "tous ", "toutes ", "liste ", "montre ", "affiche "]:
            phrase = f"{verb}{syn}".strip()
            add(phrase, "top_files",
                f"SELECT id, name, path, size FROM files WHERE category = '{cat}' AND is_directory = 0 ORDER BY size DESC LIMIT 500")

# Par extension (variantes)
for ext_name, ext in list(EXTENSIONS.items()):
    phrases = [f"les .{ext}", f"les fichiers .{ext}", f"les {ext}", f"fichiers {ext}", f"tous les .{ext}"]
    for p in random.sample(phrases, min(3, len(phrases))):
        add(p, "top_files",
            f"SELECT id, name, path, size FROM files WHERE extension LIKE '%{ext}%' AND is_directory = 0 ORDER BY size DESC")

# Categorie + taille
for cat in ["image", "video", "audio", "model", "document", "code", "archive"]:
    for size_l, size_b in random.sample(list(SIZE_MAP.items()), min(4, len(SIZE_MAP))):
        add(f"les {cat}s de plus de {size_l}", "top_files",
            f"SELECT id, name, path, size FROM files WHERE category = '{cat}' AND size > {size_b} AND is_directory = 0 ORDER BY size DESC")

# Extension + taille
for ext in ["safetensors", "gguf", "mp4", "mov", "zip", "pdf", "py", "swift"]:
    for size_l, size_b in [("10 Mo", 10485760), ("100 Mo", 104857600), ("500 Mo", 524288000), ("1 Go", 1073741824), ("5 Go", 5368709120)]:
        add(f"les .{ext} de plus de {size_l}", "top_files",
            f"SELECT id, name, path, size FROM files WHERE extension LIKE '%{ext}%' AND size > {size_b} AND is_directory = 0 ORDER BY size DESC")

# Extension + path
for ext in ["py", "swift", "json", "pdf", "md", "plist"]:
    for path_name, path in [("Documents", "Documents"), ("Desktop", "Desktop"), ("Downloads", "Downloads"), ("Developpements", "Developpements")]:
        add(f"les .{ext} dans {path_name}", "top_files",
            f"SELECT id, name, path, size FROM files WHERE extension LIKE '%{ext}%' AND path LIKE '%{path}%' AND is_directory = 0 ORDER BY size DESC")

# Photos par appareil
for device in CAMERAS:
    for p in [f"photos prises avec un {device}", f"les photos {device}", f"images {device}"]:
        add(p, "top_files",
            f"SELECT id, name, path, size, camera_model FROM files WHERE category = 'image' AND camera_model LIKE '%{device}%' AND is_directory = 0 ORDER BY size DESC")

# Videos par duree
for minutes, seconds in DURATIONS:
    add(f"videos de plus de {minutes} minutes", "top_files",
        f"SELECT id, name, path, size, duration_seconds FROM files WHERE category = 'video' AND duration_seconds > {seconds} AND is_directory = 0 ORDER BY duration_seconds DESC")
    add(f"videos de moins de {minutes} minutes", "top_files",
        f"SELECT id, name, path, size, duration_seconds FROM files WHERE category = 'video' AND duration_seconds < {seconds} AND is_directory = 0 ORDER BY duration_seconds ASC")

# PDFs par pages
for pages in PAGES:
    add(f"PDFs de plus de {pages} pages", "top_files",
        f"SELECT id, name, path, size, page_count FROM files WHERE extension LIKE '%pdf%' AND page_count > {pages} AND is_directory = 0 ORDER BY page_count DESC")

# Audio par artiste/album
for p in ["musique de Taylor Swift", "chansons de Daft Punk", "albums de Radiohead"]:
    artist = p.split("de ")[-1]
    add(p, "top_files",
        f"SELECT id, name, path, size, audio_artist, audio_album FROM files WHERE category = 'audio' AND audio_artist LIKE '%{artist}%' AND is_directory = 0 ORDER BY name")

# ============================================================
# SKILL: category_breakdown
# ============================================================
for p in ["repartition par categorie", "par type", "par categorie",
          "quels types de fichiers", "distribution par categorie",
          "resume par type", "statistiques par categorie", "combien par type",
          "ventilation par categorie", "categories", "types de fichiers",
          "combien d'espace utilise", "espace disque total",
          "taille totale des fichiers", "combien de Go en tout",
          "combien de place ca prend", "c'est quoi la repartition",
          "qu'est-ce qui prend de la place", "ou va l'espace disque"]:
    add(p, "category_breakdown",
        "SELECT category, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 GROUP BY category ORDER BY taille_totale DESC")

# ============================================================
# SKILL: extension_breakdown
# ============================================================
for n in [5, 10, 15, 20, 30, 50]:
    for p in [f"top {n} extensions", f"les {n} plus grosses extensions",
              f"repartition par extension (top {n})", f"les {n} extensions les plus lourdes"]:
        add(p, "extension_breakdown",
            f"SELECT extension, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 AND extension IS NOT NULL GROUP BY extension ORDER BY taille_totale DESC LIMIT {n}")

for p in ["par extension", "quelles extensions", "repartition par extension",
          "les extensions", "statistiques par extension", "extensions par taille",
          "quelles sont les extensions presentes", "types d'extensions"]:
    add(p, "extension_breakdown",
        "SELECT extension, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 AND extension IS NOT NULL GROUP BY extension ORDER BY taille_totale DESC LIMIT 30")

# ============================================================
# SKILL: filter_size
# ============================================================
for label, bytes_val in SIZE_MAP.items():
    for p in [f"fichiers de plus de {label}", f"plus de {label}", f"superieur a {label}",
              f"plus gros que {label}", f"au dessus de {label}"]:
        add(p, "filter_size",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > {bytes_val} ORDER BY size DESC")
    for p in [f"fichiers de moins de {label}", f"moins de {label}", f"inferieur a {label}",
              f"plus petit que {label}", f"en dessous de {label}"]:
        add(p, "filter_size",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size < {bytes_val} ORDER BY size DESC")

# Plages
for (low_l, low_b), (high_l, high_b) in [
    (("100 Mo", 104857600), ("1 Go", 1073741824)),
    (("500 Mo", 524288000), ("5 Go", 5368709120)),
    (("1 Go", 1073741824), ("10 Go", 10737418240)),
    (("10 Mo", 10485760), ("100 Mo", 104857600)),
]:
    for p in [f"entre {low_l} et {high_l}", f"de {low_l} a {high_l}",
              f"fichiers entre {low_l} et {high_l}"]:
        add(p, "filter_size",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > {low_b} AND size < {high_b} ORDER BY size DESC")

# Temporel
periods = [
    ("aujourd'hui", 1), ("hier", 1), ("cette semaine", 7), ("ces 3 derniers jours", 3),
    ("ces 2 dernieres semaines", 14), ("ce mois", 30), ("ce mois-ci", 30),
    ("les 90 derniers jours", 90), ("cette annee", 365), ("la semaine derniere", 7),
]
for period, days in periods:
    for p in [f"modifies {period}", f"fichiers modifies {period}", f"changes {period}",
              f"mis a jour {period}"]:
        add(p, "filter_size",
            f"SELECT id, name, path, size, modified_at FROM files WHERE is_directory = 0 AND modified_at > strftime('%s', 'now', '-{days} days') ORDER BY modified_at DESC")

# ============================================================
# SKILL: filter_path
# ============================================================
for path_name, path_val in PATHS.items():
    for p in [f"fichiers dans {path_name}", f"dans {path_name}", f"tout dans {path_name}",
              f"contenu de {path_name}", f"qu'est-ce qu'il y a dans {path_name}"]:
        add(p, "filter_path",
            f"SELECT id, name, path, size FROM files WHERE path LIKE '%{path_val}%' AND is_directory = 0 ORDER BY size DESC LIMIT 500")

for cache_name, cache_path in CACHE_PATHS.items():
    for p in [f"ou sont {cache_name}", f"montre {cache_name}", f"taille de {cache_name}"]:
        add(p, "filter_path",
            f"SELECT id, name, path, size FROM files WHERE path LIKE '%{cache_path}%' AND is_directory = 0 ORDER BY size DESC LIMIT 500")

# Chemins specifiques
special_paths = [
    ("modeles HuggingFace", "huggingface"), ("fichiers Xcode", "Xcode"),
    ("node_modules", "node_modules"), ("le dossier .git", ".git"),
    ("fichiers homebrew", "homebrew"), ("fichiers dans opt", "opt"),
    ("dans Library", "Library"), ("dans usr", "usr"),
]
for phrase, path in special_paths:
    add(phrase, "filter_path",
        f"SELECT id, name, path, size FROM files WHERE path LIKE '%{path}%' AND is_directory = 0 ORDER BY size DESC LIMIT 500")

# Cloud
for p in ["fichiers dans le cloud", "fichiers cloud", "fichiers non locaux", "fichiers iCloud"]:
    add(p, "filter_path",
        "SELECT id, name, path, size FROM files WHERE is_local = 0 AND is_directory = 0 ORDER BY size DESC")

# ============================================================
# SKILL: filter_name
# ============================================================
name_patterns = [
    "backup", "old", "temp", "cache", "log", "test", "README", "config",
    "Dockerfile", "Makefile", ".env", "package", "index", "todo",
    "LICENSE", "CHANGELOG", "setup", "requirements", "Podfile",
    "Cartfile", ".gitignore", "webpack", "tsconfig", "eslint",
]
for name in name_patterns:
    for p in [f"fichiers contenant {name}", f"cherche {name}", f"les {name}"]:
        add(p, "filter_name",
            f"SELECT id, name, path, size FROM files WHERE name LIKE '%{name}%' AND is_directory = 0 ORDER BY size DESC")

for p in ["fichiers caches", "les fichiers invisibles", "fichiers masques", "les dotfiles"]:
    add(p, "filter_name",
        "SELECT id, name, path, size FROM files WHERE is_hidden = 1 AND is_directory = 0 ORDER BY size DESC")

for p in ["les liens symboliques", "les symlinks"]:
    add(p, "filter_name",
        "SELECT id, name, path, size FROM files WHERE is_symlink = 1 AND is_directory = 0 ORDER BY size DESC")

# ============================================================
# SKILL: filter_extension
# ============================================================
for ext in ["py", "swift", "json", "mp4", "pdf", "safetensors", "log", "plist", "csv", "html"]:
    for p in [f"filtre les .{ext}", f"filtre sur les {ext}", f"uniquement les .{ext}",
              f"affiche seulement les .{ext}"]:
        add(p, "filter_extension",
            f"SELECT id, name, path, size FROM files WHERE extension LIKE '%{ext}%' AND is_directory = 0 ORDER BY size DESC")

for cat in CATEGORIES:
    for p in [f"filtre sur les {cat}s", f"filtre {cat}", f"uniquement les {cat}s"]:
        add(p, "filter_extension",
            f"SELECT id, name, path, size FROM files WHERE category = '{cat}' AND is_directory = 0 ORDER BY size DESC")

# Metadata filters via filter_extension
for minutes, seconds in [(5, 300), (10, 600), (30, 1800)]:
    add(f"videos de plus de {minutes} minutes", "filter_extension",
        f"SELECT id, name, path, size, duration_seconds FROM files WHERE category = 'video' AND is_directory = 0 AND duration_seconds > {seconds} ORDER BY duration_seconds DESC")

# ============================================================
# SKILL: navigate
# ============================================================
nav_paths = [
    ("Downloads", "Downloads"), ("Documents", "Documents"), ("Desktop", "Desktop"),
    ("Applications", "Applications"), ("Pictures", "Pictures"), ("Movies", "Movies"),
    ("Music", "Music"), ("Library", "Library"), ("Developpements", "Developpements"),
]
for path_name, path_val in nav_paths:
    for p in [f"va dans {path_name}", f"ouvre {path_name}", f"navigue vers {path_name}",
              f"entre dans {path_name}", f"cd {path_name}", f"aller dans {path_name}"]:
        add(p, "navigate",
            f"SELECT id, name, path, size FROM files WHERE is_directory = 1 AND path LIKE '%{path_val}%' LIMIT 1")

# ============================================================
# ACTIONS
# ============================================================

# move_files
for dest in DESTINATIONS:
    items = [
        ("les safetensors", "extension LIKE '%safetensors%'"),
        ("les gros fichiers", "size > 1073741824"),
        ("les videos", "category = 'video'"),
        ("les modeles", "category = 'model'"),
        ("les archives", "category = 'archive'"),
        ("les images", "category = 'image'"),
        ("les documents", "category = 'document'"),
        ("les fichiers audio", "category = 'audio'"),
    ]
    for what, where in random.sample(items, min(4, len(items))):
        for verb in ["deplace", "bouge", "transfere", "mets"]:
            add(f"{verb} {what} dans {dest}", "move_files",
                f"SELECT id, name, path, size FROM files WHERE {where} AND is_directory = 0 ORDER BY size DESC",
                {"type": "move", "destination": dest})

# delete_files
delete_items = [
    ("les fichiers de plus de 5 Go", "size > 5368709120"),
    ("les fichiers de plus de 10 Go", "size > 10737418240"),
    ("les fichiers de plus de 20 Go", "size > 21474836480"),
    ("les fichiers temporaires", "path LIKE '%tmp%'"),
    ("les .log", "extension LIKE '%log%'"),
    ("les .DS_Store", "name LIKE '%DS_Store%'"),
    ("les .pyc", "extension LIKE '%pyc%'"),
    ("les .o", "extension LIKE '%o%'"),
    ("les fichiers de cache", "path LIKE '%Cache%'"),
    ("les vieux fichiers", "modified_at < strftime('%s', 'now', '-365 days')"),
    ("les doublons potentiels", "is_directory = 0 AND size > 0"),
]
for what, where in delete_items:
    for verb in ["supprime", "efface", "jette", "vire", "retire", "degage"]:
        add(f"{verb} {what}", "delete_files",
            f"SELECT id, name, path, size FROM files WHERE {where} AND is_directory = 0 ORDER BY size DESC",
            {"type": "delete"})

# compress_files
for cat in ["video", "document", "image", "code", "model", "audio"]:
    for dest in [f"/tmp/{cat}s.zip", f"~/Archive/{cat}s.zip", f"~/Desktop/{cat}s.zip"]:
        add(f"compresse les {cat}s dans {dest}", "compress_files",
            f"SELECT id, name, path, size FROM files WHERE category = '{cat}' AND is_directory = 0 ORDER BY size DESC",
            {"type": "compress", "destination": dest})
    for p in [f"zippe les {cat}s", f"archive les {cat}s", f"compresse les {cat}s"]:
        add(p, "compress_files",
            f"SELECT id, name, path, size FROM files WHERE category = '{cat}' AND is_directory = 0 ORDER BY size DESC",
            {"type": "compress", "destination": f"/tmp/{cat}s.zip"})

# duplicate_files
for p in ["trouve les doublons", "fichiers en double", "cherche les duplicatas",
          "y a-t-il des doublons", "detecte les fichiers identiques",
          "quels fichiers sont en double", "scan les doublons",
          "fichiers dupliques", "doublons", "les doublons"]:
    add(p, "duplicate_files",
        "SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size DESC",
        {"type": "duplicates"})

# export_list
for fmt in ["csv", "json"]:
    for p in [f"exporte en {fmt.upper()}", f"exporte la liste en {fmt}",
              f"export {fmt}", f"genere un {fmt}", f"sauvegarde en {fmt}",
              f"telecharge en {fmt}", f"extraction {fmt}"]:
        add(p, "export_list",
            "SELECT name, path, size, extension, category FROM files WHERE is_directory = 0 ORDER BY size DESC",
            {"type": "export", "format": fmt})

# clean_caches
for cache_name, cache_path in CACHE_PATHS.items():
    for verb in ["nettoie", "vide", "purge", "supprime", "efface"]:
        add(f"{verb} {cache_name}", "clean_caches",
            f"SELECT id, name, path, size FROM files WHERE path LIKE '%{cache_path}%' AND is_directory = 0 ORDER BY size DESC",
            {"type": "cleanCaches"})

for p in ["fais le menage dans les caches", "libere de l'espace cache",
          "nettoyage des caches", "vider tous les caches"]:
    add(p, "clean_caches",
        "SELECT id, name, path, size FROM files WHERE path LIKE '%Cache%' AND is_directory = 0 ORDER BY size DESC",
        {"type": "cleanCaches"})

# rename_files
rename_phrases = [
    ("renomme les photos par date", "image", "{date}_{name}.{ext}"),
    ("renomme par date de creation", "image", "{date}_{name}.{ext}"),
    ("organise les photos par date", "image", "{date}_{name}.{ext}"),
    ("ajoute la date au nom des images", "image", "{date}_{name}.{ext}"),
    ("numerote les fichiers", None, "{index}_{name}.{ext}"),
]
for phrase, cat, pattern in rename_phrases:
    where = f"category = '{cat}' AND is_directory = 0" if cat else "is_directory = 0"
    add(phrase, "rename_files",
        f"SELECT id, name, path, size FROM files WHERE {where} ORDER BY name",
        {"type": "rename", "pattern": pattern})

for suffix in ["_backup", "_old", "_archived", "_v2", "_copy"]:
    add(f"ajoute {suffix} au nom", "rename_files",
        "SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY name",
        {"type": "rename", "pattern": "{name}" + suffix + ".{ext}"})

# rename_by_metadata
for p in ["renomme les photos par date EXIF", "organise par date de prise de vue",
          "trie les photos par metadata", "renomme par EXIF",
          "renomme les images par date EXIF", "classe les photos par date"]:
    add(p, "rename_by_metadata",
        "SELECT id, name, path, size, modified_at FROM files WHERE category = 'image' AND is_directory = 0",
        {"type": "renameByMetadata", "pattern": "{date}_{name}.{ext}"})

# split_audio_video
for p in ["extrais l'audio de la video", "separe audio et video",
          "extrais le son des videos", "audio seulement des videos",
          "extrais la bande son", "separe les pistes audio et video"]:
    add(p, "split_audio_video",
        "SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0",
        {"type": "splitMedia"})

# audit_codecs
for p in ["quelles videos ne sont pas en HEVC", "audit des codecs video",
          "verifie les codecs", "quels codecs pour les videos",
          "videos pas en H.265", "optimisation codecs video",
          "quelles videos peut-on reencoder"]:
    add(p, "audit_codecs",
        "SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0",
        {"type": "auditCodecs"})

# ============================================================
# FORMULATIONS INFORMELLES / NATURELLES
# ============================================================

informal = [
    ("c'est quoi les plus gros trucs", "top_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 10", None),
    ("y a des trucs enormes?", "top_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 10", None),
    ("qu'est-ce qui bouffe de la place", "top_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 20", None),
    ("montre moi les fichiers recents", "top_files",
     "SELECT id, name, path, size, modified_at FROM files WHERE is_directory = 0 ORDER BY modified_at DESC LIMIT 20", None),
    ("les trucs dans Downloads", "filter_path",
     "SELECT id, name, path, size FROM files WHERE path LIKE '%Downloads%' AND is_directory = 0 ORDER BY size DESC LIMIT 500", None),
    ("fais le menage", "clean_caches",
     "SELECT id, name, path, size FROM files WHERE path LIKE '%Cache%' AND is_directory = 0 ORDER BY size DESC",
     {"type": "cleanCaches"}),
    ("degage les vieux trucs", "delete_files",
     f"SELECT id, name, path, size FROM files WHERE modified_at < strftime('%s', 'now', '-365 days') AND is_directory = 0 ORDER BY size DESC",
     {"type": "delete"}),
    ("range les modeles quelque part", "move_files",
     "SELECT id, name, path, size FROM files WHERE category = 'model' AND is_directory = 0 ORDER BY size DESC",
     {"type": "move", "destination": "/tmp/archive"}),
    ("scanne les doublons", "duplicate_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 0 ORDER BY size DESC",
     {"type": "duplicates"}),
    ("combien de fichiers au total", "category_breakdown",
     "SELECT category, COUNT(*) as nombre, SUM(size) as taille_totale FROM files WHERE is_directory = 0 GROUP BY category ORDER BY taille_totale DESC", None),
    ("qu'est-ce que j'ai comme videos", "top_files",
     "SELECT id, name, path, size FROM files WHERE category = 'video' AND is_directory = 0 ORDER BY size DESC LIMIT 500", None),
    ("ou sont mes gros fichiers", "top_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 ORDER BY size DESC LIMIT 20", None),
    ("quels sont les dossiers les plus lourds", "top_files",
     "SELECT id, name, path, size FROM files WHERE is_directory = 1 ORDER BY size DESC LIMIT 20", None),
    ("les fichiers que j'ai pas touches depuis longtemps", "filter_size",
     f"SELECT id, name, path, size, modified_at FROM files WHERE is_directory = 0 AND modified_at < strftime('%s', 'now', '-180 days') ORDER BY modified_at ASC LIMIT 100", None),
    ("tout ce qui fait plus d'1 Go", "filter_size",
     "SELECT id, name, path, size FROM files WHERE is_directory = 0 AND size > 1073741824 ORDER BY size DESC", None),
    ("les fichiers inutiles", "filter_name",
     "SELECT id, name, path, size FROM files WHERE (name LIKE '%DS_Store%' OR name LIKE '%Thumbs%' OR name LIKE '%.pyc') AND is_directory = 0 ORDER BY size DESC", None),
]
for phrase, skill, sql, action in informal:
    add(phrase, skill, sql, action)

# ============================================================
# SPLIT ET EXPORT
# ============================================================

random.shuffle(samples)
print(f"Total samples generes: {len(samples)}")

from collections import Counter
skills = Counter()
for s in samples:
    skill = s["messages"][1]["content"].split("\n")[0].replace("SKILL: ", "")
    skills[skill] += 1
print("\nRepartition par skill:")
for skill, count in skills.most_common():
    print(f"  {skill}: {count}")

n = len(samples)
train_end = int(n * 0.85)
valid_end = int(n * 0.93)

train = samples[:train_end]
valid = samples[train_end:valid_end]
test = samples[valid_end:]

os.makedirs("/tmp/toolsforge-v2", exist_ok=True)
for name, data in [("train", train), ("valid", valid), ("test", test)]:
    with open(f"/tmp/toolsforge-v2/{name}.jsonl", "w") as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"{name}: {len(data)} samples")
