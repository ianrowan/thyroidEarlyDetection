import Foundation
import os

@Observable
final class VacationManager {
    static let shared = VacationManager()

    private let logger = Logger(subsystem: "com.thyroiddetect", category: "VacationManager")
    private let storageKey = "vacationPeriods"

    private(set) var vacations: [VacationPeriod] = []

    private init() {
        loadVacations()
    }

    func addVacation(start: Date, end: Date) {
        let vacation = VacationPeriod(startDate: start, endDate: end)
        vacations.append(vacation)
        vacations.sort { $0.startDate > $1.startDate }
        saveVacations()
        logger.info("Added vacation: \(vacation.displayRange)")
    }

    func deleteVacation(id: UUID) {
        vacations.removeAll { $0.id == id }
        saveVacations()
        logger.info("Deleted vacation with id: \(id)")
    }

    func isVacationDay(_ date: Date) -> Bool {
        vacations.contains { $0.contains(date) }
    }

    func vacationDaysCount(in samples: [Date]) -> Int {
        samples.filter { isVacationDay($0) }.count
    }

    private func loadVacations() {
        guard let data = UserDefaults.standard.data(forKey: storageKey) else {
            logger.info("No saved vacations found")
            return
        }

        do {
            vacations = try JSONDecoder().decode([VacationPeriod].self, from: data)
            vacations.sort { $0.startDate > $1.startDate }
            logger.info("Loaded \(self.vacations.count) vacations")
        } catch {
            logger.error("Failed to decode vacations: \(error.localizedDescription)")
        }
    }

    private func saveVacations() {
        do {
            let data = try JSONEncoder().encode(vacations)
            UserDefaults.standard.set(data, forKey: storageKey)
            logger.info("Saved \(self.vacations.count) vacations")
        } catch {
            logger.error("Failed to encode vacations: \(error.localizedDescription)")
        }
    }
}
