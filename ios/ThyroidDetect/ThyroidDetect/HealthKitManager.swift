import Foundation
import HealthKit
import OSLog

private let logger = Logger(subsystem: "com.thyroid.detect", category: "HealthKit")

@Observable
class HealthKitManager {
    var rhrSamples: [RHRSample] = []
    var isLoading = false
    var error: String?
    var riskResult: RiskResult?
    var riskHistory: [DailyRisk] = []
    var excludedVacationDays: Int = 0

    private let healthStore = HKHealthStore()
    private let smoother = RiskSmoother.shared
    private let vacationManager = VacationManager.shared

    func requestAuthorization() async throws {
        guard HKHealthStore.isHealthDataAvailable() else {
            throw HealthKitError.notAvailable
        }

        let rhrType = HKQuantityType(.restingHeartRate)

        try await healthStore.requestAuthorization(toShare: [], read: [rhrType])
        logger.info("HealthKit authorization granted")
    }

    func refresh(forDate targetDate: Date = Date()) async {
        isLoading = true
        error = nil

        do {
            try await requestAuthorization()

            let samples = try await queryRHRSamples(days: 40, endingOn: targetDate)

            let vacations = vacationManager.vacations
            let vacationDayCount = countVacationDays(in: samples, vacations: vacations)

            await MainActor.run {
                self.rhrSamples = samples
                self.excludedVacationDays = vacationDayCount
                logger.info("Fetched \(samples.count) RHR samples, \(vacationDayCount) vacation days excluded")
            }

            if samples.count >= 10 {
                let history = await computeRiskHistory(forDate: targetDate, days: 7)

                if let todayRisk = history.first(where: { Calendar.current.isDate($0.date, inSameDayAs: targetDate) }),
                   !todayRisk.isVacation {
                    let result = RiskResult(
                        probability: todayRisk.probability,
                        rawProbability: todayRisk.rawProbability,
                        level: todayRisk.level,
                        trend: nil,
                        consecutiveElevatedDays: 0,
                        isSmoothed: smoother.smoothingEnabled
                    )

                    await MainActor.run {
                        self.riskResult = result
                        self.riskHistory = history
                        logger.info("Risk computed: raw=\(todayRisk.rawProbability), displayed=\(todayRisk.probability), smoothed=\(self.smoother.smoothingEnabled)")
                    }
                } else if history.isEmpty {
                    await MainActor.run {
                        self.error = "Insufficient data for analysis. Need at least 35 days of RHR data."
                    }
                } else {
                    await MainActor.run {
                        self.riskHistory = history
                        self.error = "Selected date is a vacation day."
                    }
                }
            } else {
                await MainActor.run {
                    self.error = "Need more data. Currently have \(samples.count) days, need at least 10."
                }
            }

        } catch {
            await MainActor.run {
                self.error = error.localizedDescription
                logger.error("HealthKit error: \(error.localizedDescription)")
            }
        }

        await MainActor.run {
            self.isLoading = false
        }
    }

    private func queryRHRSamples(days: Int, endingOn endDate: Date = Date()) async throws -> [RHRSample] {
        let rhrType = HKQuantityType(.restingHeartRate)
        let calendar = Calendar.current
        let dayEnd = calendar.startOfDay(for: calendar.date(byAdding: .day, value: 1, to: endDate)!)
        let startDate = calendar.date(byAdding: .day, value: -days, to: dayEnd)!

        let predicate = HKQuery.predicateForSamples(
            withStart: startDate,
            end: dayEnd,
            options: .strictStartDate
        )

        let sortDescriptor = SortDescriptor(\HKSample.startDate, order: .forward)

        let samples = try await healthStore.queryAsync(
            type: rhrType,
            predicate: predicate,
            sortDescriptors: [sortDescriptor]
        )

        return samples.compactMap { sample in
            guard let quantitySample = sample as? HKQuantitySample else { return nil }
            let bpm = quantitySample.quantity.doubleValue(for: HKUnit(from: "count/min"))
            return RHRSample(date: quantitySample.startDate, bpm: bpm)
        }
    }
    
    private func computeRiskHistory(forDate endDate: Date, days: Int) async -> [DailyRisk] {
        let calendar = Calendar.current
        let model = ThyroidModel()
        let vacations = vacationManager.vacations
        var rawRisks: [(date: Date, probability: Double, isVacation: Bool)] = []

        for dayOffset in (0..<days).reversed() {
            guard let date = calendar.date(byAdding: .day, value: -dayOffset, to: endDate) else { continue }

            let isVacation = vacationManager.isVacationDay(date)

            if isVacation {
                rawRisks.append((date: date, probability: 0, isVacation: true))
            } else {
                do {
                    let samples = try await queryRHRSamples(days: 40, endingOn: date)
                    if let features = FeatureComputer.computeFeatures(from: samples, excluding: vacations) {
                        let probability = try model.predict(features: features)
                        rawRisks.append((date: date, probability: probability, isVacation: false))
                    }
                } catch {
                    logger.warning("Failed to compute risk for \(date): \(error.localizedDescription)")
                }
            }
        }

        rawRisks.sort { $0.date < $1.date }

        var history: [DailyRisk] = []
        let nonVacationRisks = rawRisks.filter { !$0.isVacation }

        for entry in rawRisks {
            if entry.isVacation {
                let risk = DailyRisk(
                    date: entry.date,
                    probability: 0,
                    rawProbability: 0,
                    level: .normal,
                    isVacation: true
                )
                history.append(risk)
            } else {
                let nonVacationIndex = nonVacationRisks.firstIndex { $0.date == entry.date } ?? 0
                let smoothedProbability: Double
                if smoother.smoothingEnabled {
                    let windowStart = max(0, nonVacationIndex - 3)
                    let window = nonVacationRisks[windowStart...nonVacationIndex].map { $0.probability }
                    smoothedProbability = window.reduce(0, +) / Double(window.count)
                } else {
                    smoothedProbability = entry.probability
                }

                let risk = DailyRisk(
                    date: entry.date,
                    probability: smoothedProbability,
                    rawProbability: entry.probability,
                    level: RiskLevel.from(probability: smoothedProbability),
                    isVacation: false
                )
                history.append(risk)
            }
        }

        return history
    }

    private func countVacationDays(in samples: [RHRSample], vacations: [VacationPeriod]) -> Int {
        let calendar = Calendar.current
        var uniqueDays = Set<Date>()

        for sample in samples {
            let dayStart = calendar.startOfDay(for: sample.date)
            if vacations.contains(where: { $0.contains(dayStart) }) {
                uniqueDays.insert(dayStart)
            }
        }

        return uniqueDays.count
    }
}

extension HKHealthStore {
    func queryAsync(
        type: HKQuantityType,
        predicate: NSPredicate?,
        sortDescriptors: [SortDescriptor<HKSample>]
    ) async throws -> [HKSample] {
        try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: type,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: sortDescriptors.map {
                    NSSortDescriptor(key: "startDate", ascending: $0.order == .forward)
                }
            ) { _, samples, error in
                if let error = error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: samples ?? [])
                }
            }
            execute(query)
        }
    }
}

struct RHRSample {
    let date: Date
    let bpm: Double
}

enum HealthKitError: Error {
    case notAvailable
}
