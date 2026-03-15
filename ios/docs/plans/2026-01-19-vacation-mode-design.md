# Vacation Mode Design

## Overview

Allow users to define date ranges for vacations that are excluded from all RHR calculations. Vacation days are visually indicated on charts.

## Data Model

```swift
struct VacationPeriod: Codable, Identifiable {
    let id: UUID
    var startDate: Date
    var endDate: Date

    var displayRange: String
    func contains(_ date: Date) -> Bool
}
```

**Storage:** `VacationManager` class using UserDefaults with Codable array.

Methods:
- `vacations: [VacationPeriod]` - all stored vacations
- `addVacation(start:end:)` - create new vacation period
- `deleteVacation(id:)` - remove vacation by ID
- `isVacationDay(_ date: Date) -> Bool` - check if date falls in any vacation

## Feature Computation Changes

`FeatureComputer.computeFeatures` accepts vacation periods:

```swift
static func computeFeatures(
    from samples: [RHRSample],
    excluding vacations: [VacationPeriod]
) -> ThyroidFeatures?
```

- Filter out samples where date falls within any vacation period
- Aggregate remaining days normally
- Feature windows use only non-vacation days
- Return nil if <10 days remain after filtering

## UI Components

### Vacation Section (ContentView)

Location: Below RHR trend chart, expandable section.

**Collapsed:**
- Airplane icon, "Vacation Periods (N)", chevron

**Expanded:**
- List of vacation periods with date range and delete button
- "Add Vacation Period" button

### Add Vacation Sheet

- Two date pickers: Start Date, End Date
- Validation: end >= start
- Cancel and Save buttons

### Delete Behavior

- Tap trash icon removes immediately (no confirmation)

## Chart Visualization

### RHR Trend Chart
- Vacation days: gray dot with reduced opacity
- Dashed line connecting adjacent non-vacation days

### 7-Day Risk Trend
- Vacation days: gray striped bar at minimal height
- Shows "—" instead of percentage

### Context Text
- Updated to show: "Based on 40 days of data (N vacation days excluded)"

## Files to Modify

1. **Models.swift** - Add `VacationPeriod` struct
2. **VacationManager.swift** (new) - UserDefaults persistence
3. **FeatureComputer.swift** - Add vacation filtering
4. **HealthKitManager.swift** - Pass vacations to feature computation
5. **ContentView.swift** - Add vacation section UI and add sheet
