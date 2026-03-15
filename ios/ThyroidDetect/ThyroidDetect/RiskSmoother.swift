import Foundation
import OSLog

private let logger = Logger(subsystem: "com.thyroid.detect", category: "RiskSmoother")

@Observable
class RiskSmoother {
    static let shared = RiskSmoother()

    private let windowSize = 4
    private var riskHistory: [Double] = []

    var smoothingEnabled: Bool {
        get { UserDefaults.standard.bool(forKey: "smoothingEnabled") }
        set {
            UserDefaults.standard.set(newValue, forKey: "smoothingEnabled")
            logger.info("Smoothing \(newValue ? "enabled" : "disabled")")
        }
    }

    private static let historyKey = "risk_history"

    init() {
        load()
        if !UserDefaults.standard.contains(key: "smoothingEnabled") {
            UserDefaults.standard.set(true, forKey: "smoothingEnabled")
        }
    }

    func addRisk(_ rawRisk: Double) -> Double {
        riskHistory.append(rawRisk)

        if riskHistory.count > windowSize {
            riskHistory.removeFirst()
        }

        save()
        logger.debug("Added risk \(rawRisk), history: \(self.riskHistory)")

        return smoothingEnabled ? smoothedRisk : rawRisk
    }

    var smoothedRisk: Double {
        guard !riskHistory.isEmpty else { return 0 }
        return riskHistory.reduce(0, +) / Double(riskHistory.count)
    }

    var rawRisk: Double {
        riskHistory.last ?? 0
    }

    var historyCount: Int {
        riskHistory.count
    }

    func save() {
        UserDefaults.standard.set(riskHistory, forKey: Self.historyKey)
    }

    func load() {
        if let history = UserDefaults.standard.array(forKey: Self.historyKey) as? [Double] {
            riskHistory = history
            logger.info("Loaded \(history.count) risk history entries")
        }
    }

    func clear() {
        riskHistory = []
        save()
        logger.info("Cleared risk history")
    }

    func getRiskForDisplay(_ rawRisk: Double) -> (displayed: Double, raw: Double) {
        let displayed = smoothingEnabled ? addRisk(rawRisk) : rawRisk
        return (displayed, rawRisk)
    }
}

extension UserDefaults {
    func contains(key: String) -> Bool {
        return object(forKey: key) != nil
    }
}
