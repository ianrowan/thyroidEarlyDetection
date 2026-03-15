import Foundation
import SwiftUI

struct RiskResult {
    let probability: Double
    let rawProbability: Double
    let level: RiskLevel
    let trend: Trend?
    let consecutiveElevatedDays: Int
    let isSmoothed: Bool

    var displayProbability: String {
        "\(Int(probability * 100))%"
    }

    var rawDisplayProbability: String {
        "\(Int(rawProbability * 100))%"
    }

    var statusMessage: String {
        switch level {
        case .normal:
            return "Looking good"
        case .elevated:
            return "Monitor closely"
        case .high:
            return "Schedule labs"
        }
    }
}

enum RiskLevel {
    case normal
    case elevated
    case high

    var color: Color {
        switch self {
        case .normal: return .green
        case .elevated: return .yellow
        case .high: return .red
        }
    }

    static func from(probability: Double) -> RiskLevel {
        if probability >= 0.50 {
            return .high
        } else if probability >= 0.35 {
            return .elevated
        } else {
            return .normal
        }
    }
}

enum Trend {
    case rising
    case stable
    case falling

    var icon: String {
        switch self {
        case .rising: return "arrow.up.right"
        case .stable: return "arrow.right"
        case .falling: return "arrow.down.right"
        }
    }
}

struct DailyRisk: Identifiable {
    let id = UUID()
    let date: Date
    let probability: Double
    let rawProbability: Double
    let level: RiskLevel
    let isVacation: Bool

    init(date: Date, probability: Double, rawProbability: Double, level: RiskLevel, isVacation: Bool = false) {
        self.date = date
        self.probability = probability
        self.rawProbability = rawProbability
        self.level = level
        self.isVacation = isVacation
    }

    var dayLabel: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEE"
        return formatter.string(from: date)
    }

    var shortDate: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "M/d"
        return formatter.string(from: date)
    }
}

struct VacationPeriod: Codable, Identifiable, Equatable {
    let id: UUID
    var startDate: Date
    var endDate: Date

    init(id: UUID = UUID(), startDate: Date, endDate: Date) {
        self.id = id
        self.startDate = startDate
        self.endDate = endDate
    }

    var displayRange: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d, yyyy"
        let start = formatter.string(from: startDate)
        let end = formatter.string(from: endDate)
        if Calendar.current.isDate(startDate, inSameDayAs: endDate) {
            return start
        }
        return "\(start) - \(end)"
    }

    func contains(_ date: Date) -> Bool {
        let calendar = Calendar.current
        let dayStart = calendar.startOfDay(for: date)
        let rangeStart = calendar.startOfDay(for: startDate)
        let rangeEnd = calendar.startOfDay(for: endDate)
        return dayStart >= rangeStart && dayStart <= rangeEnd
    }
}
