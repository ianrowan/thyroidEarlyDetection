import Foundation

struct FeatureComputer {
    static func computeFeatures(
        from samples: [RHRSample],
        excluding vacations: [VacationPeriod] = []
    ) -> ThyroidFeatures? {
        let filteredSamples = samples.filter { sample in
            !vacations.contains { $0.contains(sample.date) }
        }

        guard filteredSamples.count >= 10 else { return nil }

        let sortedSamples = filteredSamples.sorted { $0.date < $1.date }
        let dailyAverages = aggregateByDay(samples: sortedSamples)

        guard dailyAverages.count >= 10 else { return nil }

        let currentWindow = Array(dailyAverages.suffix(5))
        let currentWindowMean = mean(currentWindow)

        guard dailyAverages.count >= 10 else { return nil }
        let priorWindow = Array(dailyAverages.dropLast(5).suffix(5))
        let priorWindowMean = mean(priorWindow)

        let rhr_delta = currentWindowMean - priorWindowMean

        let baseline14d = Array(dailyAverages.dropLast(5).suffix(14))
        let baseline14Mean = mean(baseline14d)
        let baseline14Std = standardDeviation(baseline14d, mean: baseline14Mean)

        let baseline30d = Array(dailyAverages.dropLast(5).suffix(30))
        let baseline30Mean = mean(baseline30d)
        let baseline30Std = standardDeviation(baseline30d, mean: baseline30Mean)

        let epsilon = 0.01
        let rhr_deviation_14d = (currentWindowMean - baseline14Mean) / max(baseline14Std, epsilon)
        let rhr_deviation_30d = (currentWindowMean - baseline30Mean) / max(baseline30Std, epsilon)

        return ThyroidFeatures(
            rhr_deviation_14d: rhr_deviation_14d,
            rhr_deviation_30d: rhr_deviation_30d,
            rhr_delta: rhr_delta
        )
    }

    private static func aggregateByDay(samples: [RHRSample]) -> [Double] {
        let calendar = Calendar.current
        var dailyGroups: [Date: [Double]] = [:]

        for sample in samples {
            let dayStart = calendar.startOfDay(for: sample.date)
            dailyGroups[dayStart, default: []].append(sample.bpm)
        }

        let sortedDays = dailyGroups.keys.sorted()
        return sortedDays.map { day in
            let values = dailyGroups[day]!
            return values.reduce(0, +) / Double(values.count)
        }
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func standardDeviation(_ values: [Double], mean: Double) -> Double {
        guard values.count > 1 else { return 0 }
        let variance = values.map { pow($0 - mean, 2) }.reduce(0, +) / Double(values.count - 1)
        return sqrt(variance)
    }
}

struct ThyroidFeatures {
    let rhr_deviation_14d: Double
    let rhr_deviation_30d: Double
    let rhr_delta: Double
}
