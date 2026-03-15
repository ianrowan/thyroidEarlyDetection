import Foundation
import CoreML

class ThyroidModel {
    private let model: ThyroidEarlyDetection

    init() {
        do {
            let config = MLModelConfiguration()
            self.model = try ThyroidEarlyDetection(configuration: config)
        } catch {
            fatalError("Failed to load ThyroidEarlyDetection model: \(error)")
        }
    }

    func predict(features: ThyroidFeatures) throws -> Double {
        let input = ThyroidEarlyDetectionInput(
            rhr_deviation_14d: features.rhr_deviation_14d,
            rhr_deviation_30d: features.rhr_deviation_30d,
            rhr_delta: features.rhr_delta
        )

        let output = try model.prediction(input: input)

        guard let probability = output.classProbability[1] else {
            return 0.0
        }

        return probability
    }
}
