#!/bin/bash
# Temoin (K-37) : un consommateur qui depend de la bibliotheque ne doit resoudre ni
# Hummingbird ni swift-nio (le serveur vit dans le paquet imbrique Server/).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/Sources/Temoin"
cat > "$TMP/Package.swift" <<SWIFT
// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "Temoin", platforms: [.macOS(.v15)],
    dependencies: [.package(path: "$REPO")],
    targets: [.target(name: "Temoin", dependencies: [.product(name: "Gemma4Swift", package: "$(basename "$REPO")")])])
SWIFT
echo "import Gemma4Swift" > "$TMP/Sources/Temoin/Temoin.swift"
(cd "$TMP" && swift package resolve >/dev/null)
if grep -E '"identity" : "(hummingbird|swift-nio[a-z-]*)"' "$TMP/Package.resolved"; then
    echo "ECHEC : le serveur fuit dans le graphe des consommateurs" >&2
    exit 1
fi
echo "OK : $(grep -c '"identity"' "$TMP/Package.resolved") paquets resolus, ni hummingbird ni swift-nio"
