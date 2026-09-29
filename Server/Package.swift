// swift-tools-version: 6.0
//
// Serveur d'inference Gemma 4 (lot G), en paquet imbrique : les consommateurs de la
// bibliotheque (`../Package.swift`) ne resolvent ni Hummingbird ni swift-nio
// (audit-annexes-serveur.md § 2.5, experience E3). Construire avec xcodebuild (Metal) :
//   cd Server && xcodebuild -scheme gemma4-server -destination "platform=macOS" \
//     -derivedDataPath ../.build/xcode-server -skipMacroValidation build
import PackageDescription

let package = Package(
    name: "Gemma4Server",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "gemma4-server", targets: ["Gemma4ServerCLI"]),
        .library(name: "Gemma4Server", targets: ["Gemma4Server"]),
    ],
    dependencies: [
        .package(path: ".."),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0"),
        // Plancher explicite : aucune resolution ne redescend sous les correctifs des
        // deux CVE de swift-nio (audit § 2.5).
        .package(url: "https://github.com/apple/swift-nio.git", from: "2.100.0"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.2.0"),
    ],
    targets: [
        .target(
            name: "Gemma4Server",
            dependencies: [
                .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
                .product(name: "Hummingbird", package: "hummingbird"),
                .product(name: "NIOCore", package: "swift-nio"),
            ]
        ),
        .executableTarget(
            name: "Gemma4ServerCLI",
            dependencies: [
                "Gemma4Server",
                .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ]
        ),
        .testTarget(
            name: "Gemma4ServerTests",
            dependencies: [
                "Gemma4Server",
                .product(name: "Gemma4Swift", package: "gemma-4-swift-mlx"),
                .product(name: "HummingbirdTesting", package: "hummingbird"),
            ]
        ),
    ]
)
