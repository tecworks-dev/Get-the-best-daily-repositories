package db

import (
	"fmt"
	"strconv"
	"time"
)

// Parse retention period
func ParseRetentionPeriod(retentionPeriod string) (time.Duration, error) {
	// Split the value and unit (e.g., "1d" -> "1", "d")
	value, unit := retentionPeriod[:len(retentionPeriod)-1], retentionPeriod[len(retentionPeriod)-1:]

	// Parse the numeric part
	num, err := strconv.Atoi(value)
	if err != nil {
		return 0, fmt.Errorf("invalid number in retention period: %s", retentionPeriod)
	}

	// Calculate the appropriate duration based on the unit
	switch unit {
	case "d": // 1 day ~ 7 days
		if num < 1 || num > 7 {
			return 0, fmt.Errorf("days out of range: %d (valid range: 1d to 7d)", num)
		}
		return time.Duration(-num) * 24 * time.Hour, nil
	case "w": // 1 week ~ 4 weeks
		if num < 1 || num > 4 {
			return 0, fmt.Errorf("weeks out of range: %d (valid range: 1w to 4w)", num)
		}
		return time.Duration(-num) * 7 * 24 * time.Hour, nil
	case "h": // 1 hour ~ 24 hours
		if num < 1 || num > 24 {
			return 0, fmt.Errorf("hours out of range: %d (valid range: 1h to 24h)", num)
		}
		return time.Duration(-num) * time.Hour, nil
	case "m": // 1 month ~ 12 months
		if num < 1 || num > 12 {
			return 0, fmt.Errorf("months out of range: %d (valid range: 1m to 12m)", num)
		}
		return time.Duration(-num) * 30 * 24 * time.Hour, nil // Assuming 1 month is 30 days
	default:
		return 0, fmt.Errorf("invalid unit in retention period: %s", retentionPeriod)
	}
}
