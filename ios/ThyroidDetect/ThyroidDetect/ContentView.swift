import SwiftUI
import Charts

struct ContentView: View {
    @State private var healthManager = HealthKitManager()
    @State private var selectedDate = Date()
    @State private var showDatePicker = false
    @State private var showAddVacation = false
    @State private var vacationExpanded = false
    @State private var vacationStartDate = Date()
    @State private var vacationEndDate = Date()
    private let smoother = RiskSmoother.shared
    private let vacationManager = VacationManager.shared

    private var isToday: Bool {
        Calendar.current.isDateInToday(selectedDate)
    }

    private var canGoForward: Bool {
        !isToday
    }

    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                dateSelectorBar
                
                if healthManager.isLoading {
                    loadingView
                } else if let error = healthManager.error {
                    errorView(error: error)
                } else if let result = healthManager.riskResult {
                    riskCardView(result: result)
                    contextBar
                    if !healthManager.riskHistory.isEmpty {
                        riskTrendChart
                    }
                    if !healthManager.rhrSamples.isEmpty {
                        rhrTrendChart
                    }
                    vacationSection
                } else {
                    emptyStateView
                }
            }
            .padding()
        }
        .background(Color.black.ignoresSafeArea())
        .preferredColorScheme(.dark)
        .task {
            await healthManager.refresh(forDate: selectedDate)
        }
        .refreshable {
            await healthManager.refresh(forDate: selectedDate)
        }
        .onChange(of: selectedDate) {
            Task {
                await healthManager.refresh(forDate: selectedDate)
            }
        }
        .sheet(isPresented: $showDatePicker) {
            datePickerSheet
        }
        .sheet(isPresented: $showAddVacation) {
            addVacationSheet
        }
    }
    
    private var dateSelectorBar: some View {
        HStack(spacing: 16) {
            Button(action: { moveDate(by: -1) }) {
                Image(systemName: "chevron.left")
                    .font(.title3)
                    .foregroundStyle(.white)
            }
            
            Button(action: { showDatePicker = true }) {
                VStack(spacing: 2) {
                    Text(selectedDate.formatted(date: .abbreviated, time: .omitted))
                        .font(.headline)
                    if isToday {
                        Text("Today")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .foregroundStyle(.white)
            
            Button(action: { moveDate(by: 1) }) {
                Image(systemName: "chevron.right")
                    .font(.title3)
                    .foregroundStyle(canGoForward ? .white : .gray.opacity(0.3))
            }
            .disabled(!canGoForward)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
    
    private var datePickerSheet: some View {
        NavigationStack {
            DatePicker(
                "Select Date",
                selection: $selectedDate,
                in: ...Date(),
                displayedComponents: .date
            )
            .datePickerStyle(.graphical)
            .padding()
            .navigationTitle("Select Date")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") {
                        showDatePicker = false
                    }
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Today") {
                        selectedDate = Date()
                        showDatePicker = false
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
    
    private func moveDate(by days: Int) {
        if let newDate = Calendar.current.date(byAdding: .day, value: days, to: selectedDate) {
            if newDate <= Date() {
                selectedDate = newDate
            }
        }
    }

    private var loadingView: some View {
        VStack(spacing: 20) {
            ProgressView()
                .scaleEffect(1.5)
            Text("Analyzing your heart rate data...")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(.top, 100)
    }

    private func errorView(error: String) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 50))
                .foregroundStyle(.orange)
            Text(error)
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
            Button("Try Again") {
                Task {
                    await healthManager.refresh()
                }
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .frame(maxWidth: .infinity)
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "heart.text.square")
                .font(.system(size: 50))
                .foregroundStyle(.red)
            Text("Welcome to ThyroidDetect")
                .font(.title2)
                .bold()
            Text("Pull to refresh to analyze your resting heart rate data from Apple Health.")
                .font(.body)
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(maxWidth: .infinity)
    }

    private func riskCardView(result: RiskResult) -> some View {
        VStack(spacing: 16) {
            ZStack {
                Circle()
                    .stroke(result.level.color.opacity(0.3), lineWidth: 12)
                    .frame(width: 180, height: 180)

                Circle()
                    .trim(from: 0, to: result.probability)
                    .stroke(result.level.color, style: StrokeStyle(lineWidth: 12, lineCap: .round))
                    .frame(width: 180, height: 180)
                    .rotationEffect(.degrees(-90))

                VStack(spacing: 4) {
                    Text(result.displayProbability)
                        .font(.system(size: 48, weight: .bold))
                        .foregroundStyle(result.level.color)
                    Text("risk")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            }

            Text(result.statusMessage)
                .font(.title2)
                .bold()
                .foregroundStyle(result.level.color)

            if result.isSmoothed && result.probability != result.rawProbability {
                HStack(spacing: 6) {
                    Image(systemName: "chart.line.downtrend.xyaxis")
                    Text("Raw: \(result.rawDisplayProbability)")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 4)
                .background(.ultraThinMaterial, in: Capsule())
            }

            if let trend = result.trend {
                HStack(spacing: 8) {
                    Image(systemName: trend.icon)
                    Text("Trend")
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
        }
        .padding(32)
        .background {
            RoundedRectangle(cornerRadius: 24)
                .fill(.ultraThinMaterial)
                .overlay {
                    RoundedRectangle(cornerRadius: 24)
                        .stroke(result.level.color.opacity(0.3), lineWidth: 1)
                }
                .shadow(color: result.level.color.opacity(0.3), radius: 20)
        }
    }

    private var contextBar: some View {
        VStack(spacing: 12) {
            smoothingToggle

            VStack(spacing: 4) {
                if healthManager.excludedVacationDays > 0 {
                    Text("Based on 40 days of data (\(healthManager.excludedVacationDays) vacation days excluded)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Text("Based on 40 days of data ending \(selectedDate.formatted(date: .abbreviated, time: .omitted))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text("Last updated: \(Date.now.formatted(date: .abbreviated, time: .shortened))")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    @ViewBuilder
    private var smoothingToggle: some View {
        HStack(spacing: 12) {
            Image(systemName: "waveform.path.ecg")
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 2) {
                Text("Trend Smoothing")
                    .font(.subheadline)
                Text("SMA-4 reduces noise")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }

            Spacer()

            Toggle("", isOn: Binding(
                get: { smoother.smoothingEnabled },
                set: { newValue in
                    smoother.smoothingEnabled = newValue
                    Task {
                        await healthManager.refresh(forDate: selectedDate)
                    }
                }
            ))
            .labelsHidden()
            .tint(.blue)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
    }
    
    private var riskTrendChart: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("7-Day Risk Trend")
                    .font(.headline)
                Spacer()
                if let trend = computeTrendDirection() {
                    HStack(spacing: 4) {
                        Image(systemName: trend.icon)
                        Text(trend.label)
                    }
                    .font(.caption)
                    .foregroundStyle(trend.color)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(trend.color.opacity(0.2), in: Capsule())
                }
            }
            
            HStack(alignment: .bottom, spacing: 8) {
                ForEach(healthManager.riskHistory) { risk in
                    VStack(spacing: 8) {
                        ZStack(alignment: .bottom) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.gray.opacity(0.2))
                                .frame(width: 36, height: 120)

                            if risk.isVacation {
                                ZStack {
                                    RoundedRectangle(cornerRadius: 6)
                                        .fill(Color.gray.opacity(0.15))
                                        .frame(width: 36, height: 120)
                                    VStack(spacing: 4) {
                                        Image(systemName: "airplane")
                                            .font(.system(size: 14))
                                            .foregroundStyle(.gray.opacity(0.6))
                                    }
                                }
                            } else {
                                RoundedRectangle(cornerRadius: 6)
                                    .fill(
                                        LinearGradient(
                                            colors: [risk.level.color.opacity(0.6), risk.level.color],
                                            startPoint: .bottom,
                                            endPoint: .top
                                        )
                                    )
                                    .frame(width: 36, height: max(8, 120 * risk.probability))
                            }

                            if Calendar.current.isDate(risk.date, inSameDayAs: selectedDate) {
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(Color.white, lineWidth: 2)
                                    .frame(width: 36, height: 120)
                            }
                        }

                        if risk.isVacation {
                            Text("—")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.gray)
                        } else {
                            Text("\(Int(risk.probability * 100))%")
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(risk.level.color)
                        }

                        VStack(spacing: 2) {
                            Text(risk.dayLabel)
                                .font(.system(size: 11, weight: .medium))
                                .foregroundStyle(risk.isVacation ? .gray : .primary)
                            Text(risk.shortDate)
                                .font(.system(size: 9))
                                .foregroundStyle(.secondary)
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
            }
            
            HStack(spacing: 16) {
                ForEach([(RiskLevel.normal, "Normal"), (.elevated, "Elevated"), (.high, "High")], id: \.0) { level, label in
                    HStack(spacing: 4) {
                        Circle()
                            .fill(level.color)
                            .frame(width: 8, height: 8)
                        Text(label)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .frame(maxWidth: .infinity)
        }
        .padding()
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        }
    }
    
    private func computeTrendDirection() -> (icon: String, label: String, color: Color)? {
        guard healthManager.riskHistory.count >= 2 else { return nil }
        let recent = healthManager.riskHistory.suffix(3)
        let first = recent.first?.probability ?? 0
        let last = recent.last?.probability ?? 0
        let diff = last - first
        
        if diff > 0.05 {
            return ("arrow.up.right", "Rising", .red)
        } else if diff < -0.05 {
            return ("arrow.down.right", "Falling", .green)
        } else {
            return ("arrow.right", "Stable", .gray)
        }
    }

    private var rhrTrendChart: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Resting Heart Rate Trend")
                .font(.headline)
                .foregroundStyle(.primary)

            let dailyData = aggregateToDaily(samples: healthManager.rhrSamples)

            Chart {
                ForEach(dailyData, id: \.date) { point in
                    LineMark(
                        x: .value("Date", point.date),
                        y: .value("BPM", point.bpm)
                    )
                    .foregroundStyle(.red.gradient)
                }

                if let baseline = calculateBaseline(dailyData: dailyData) {
                    RuleMark(y: .value("Baseline", baseline))
                        .foregroundStyle(.gray.opacity(0.5))
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                }
            }
            .frame(height: 200)
            .chartYScale(domain: .automatic(includesZero: false))
        }
        .padding()
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        }
    }

    private func aggregateToDaily(samples: [RHRSample]) -> [DailyRHR] {
        let calendar = Calendar.current
        var dailyGroups: [Date: [Double]] = [:]

        for sample in samples {
            let dayStart = calendar.startOfDay(for: sample.date)
            dailyGroups[dayStart, default: []].append(sample.bpm)
        }

        return dailyGroups.map { date, values in
            let average = values.reduce(0, +) / Double(values.count)
            return DailyRHR(date: date, bpm: average)
        }
        .sorted { $0.date < $1.date }
    }

    private func calculateBaseline(dailyData: [DailyRHR]) -> Double? {
        guard dailyData.count > 5 else { return nil }
        let recent30 = Array(dailyData.suffix(30))
        return recent30.map(\.bpm).reduce(0, +) / Double(recent30.count)
    }

    private var vacationSection: some View {
        VStack(spacing: 0) {
            Button(action: { withAnimation { vacationExpanded.toggle() } }) {
                HStack {
                    Image(systemName: "airplane.departure")
                        .foregroundStyle(.secondary)
                    Text("Vacation Periods")
                        .font(.headline)
                    if !vacationManager.vacations.isEmpty {
                        Text("(\(vacationManager.vacations.count))")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    Image(systemName: vacationExpanded ? "chevron.up" : "chevron.down")
                        .foregroundStyle(.secondary)
                        .font(.caption)
                }
                .padding()
            }
            .buttonStyle(.plain)

            if vacationExpanded {
                VStack(spacing: 12) {
                    if vacationManager.vacations.isEmpty {
                        Text("No vacation periods set")
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(vacationManager.vacations) { vacation in
                            HStack {
                                Text(vacation.displayRange)
                                    .font(.subheadline)
                                Spacer()
                                Button(action: {
                                    vacationManager.deleteVacation(id: vacation.id)
                                    Task {
                                        await healthManager.refresh(forDate: selectedDate)
                                    }
                                }) {
                                    Image(systemName: "trash")
                                        .foregroundStyle(.red)
                                }
                            }
                            .padding(.horizontal)
                            .padding(.vertical, 8)
                            .background(Color.gray.opacity(0.1), in: RoundedRectangle(cornerRadius: 8))
                        }
                    }

                    Button(action: {
                        vacationStartDate = Date()
                        vacationEndDate = Date()
                        showAddVacation = true
                    }) {
                        HStack {
                            Image(systemName: "plus.circle.fill")
                            Text("Add Vacation Period")
                        }
                        .font(.subheadline)
                        .foregroundStyle(.blue)
                    }
                    .padding(.top, 4)
                }
                .padding(.horizontal)
                .padding(.bottom)
            }
        }
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(.ultraThinMaterial)
        }
    }

    private var addVacationSheet: some View {
        NavigationStack {
            Form {
                Section {
                    DatePicker(
                        "Start Date",
                        selection: $vacationStartDate,
                        in: ...Date(),
                        displayedComponents: .date
                    )
                    DatePicker(
                        "End Date",
                        selection: $vacationEndDate,
                        in: vacationStartDate...Date(),
                        displayedComponents: .date
                    )
                }

                Section {
                    Text("Vacation days will be excluded from all heart rate calculations.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Add Vacation")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        showAddVacation = false
                    }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        vacationManager.addVacation(start: vacationStartDate, end: vacationEndDate)
                        showAddVacation = false
                        Task {
                            await healthManager.refresh(forDate: selectedDate)
                        }
                    }
                }
            }
        }
        .presentationDetents([.medium])
    }
}

struct DailyRHR {
    let date: Date
    let bpm: Double
}

#Preview {
    ContentView()
}
