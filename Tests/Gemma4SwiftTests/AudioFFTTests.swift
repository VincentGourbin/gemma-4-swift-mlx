import Testing
import Foundation
import Accelerate
@testable import Gemma4Swift

/// Non-regression numerique de la FFT audio (S-09 : pointeurs de DSPSplitComplex
/// reecrits). Reference : |numpy.fft.rfft(cos(2πkn/N))| vaut N/2 au bin k.
@Suite("FFT audio")
struct AudioFFTTests {

    @Test("cosinus pur : pic N/2 au bin k, bruit negligeable ailleurs")
    func testCosinePeak() throws {
        let n = 512
        let k = 10
        let log2n = vDSP_Length(9)
        let setup = try #require(vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)))
        defer { vDSP_destroy_fftsetup(setup) }

        let signal = (0 ..< n).map { Float(cos(2 * Double.pi * Double(k * $0) / Double(n))) }
        let magnitudes = Gemma4AudioProcessor.computeFFTMagnitude(signal, fftSetup: setup, log2n: log2n)

        #expect(magnitudes.count == n / 2 + 1)
        #expect(abs(magnitudes[k] - Float(n / 2)) < 1e-2, "bin \(k) : \(magnitudes[k])")
        let leakage = magnitudes.enumerated().filter { $0.offset != k }.map(\.element).max() ?? 0
        #expect(leakage < 1e-2, "fuite hors du bin \(k) : \(leakage)")
    }
}
