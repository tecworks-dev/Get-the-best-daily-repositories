import { useSettingsStore } from "@/store/settings";
import { useCalendarStore } from "@/store/calendar";
import { SettingsSection, SettingRow } from "./SettingsSection";
import { useEffect } from "react";

export function CalendarSettings() {
  const { calendar, updateCalendarSettings } = useSettingsStore();
  const { feeds, loadFromDatabase } = useCalendarStore();

  // Load feeds when component mounts
  useEffect(() => {
    loadFromDatabase();
  }, [loadFromDatabase]);

  const workingDays = [
    { value: 0, label: "Sunday" },
    { value: 1, label: "Monday" },
    { value: 2, label: "Tuesday" },
    { value: 3, label: "Wednesday" },
    { value: 4, label: "Thursday" },
    { value: 5, label: "Friday" },
    { value: 6, label: "Saturday" },
  ];

  return (
    <SettingsSection
      title="Calendar Settings"
      description="Configure your calendar display and event defaults."
    >
      <SettingRow
        label="Default Calendar"
        description="Choose which calendar new events are added to by default"
      >
        <select
          value={calendar.defaultCalendarId || ""}
          onChange={(e) =>
            updateCalendarSettings({ defaultCalendarId: e.target.value })
          }
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        >
          <option value="">Select a default calendar</option>
          {feeds
            .filter((feed) => feed.enabled)
            .map((feed) => (
              <option key={feed.id} value={feed.id}>
                {feed.name}
              </option>
            ))}
        </select>
      </SettingRow>

      <SettingRow
        label="Working Hours"
        description="Set your working hours for better calendar visualization"
      >
        <div className="space-y-4">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={calendar.workingHours.enabled}
              onChange={(e) =>
                updateCalendarSettings({
                  workingHours: {
                    ...calendar.workingHours,
                    enabled: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm">Show working hours</span>
          </label>

          <div className="flex space-x-4">
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700">
                Start Time
              </label>
              <input
                type="time"
                value={calendar.workingHours.start}
                onChange={(e) =>
                  updateCalendarSettings({
                    workingHours: {
                      ...calendar.workingHours,
                      start: e.target.value,
                    },
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>
            <div className="flex-1">
              <label className="block text-sm font-medium text-gray-700">
                End Time
              </label>
              <input
                type="time"
                value={calendar.workingHours.end}
                onChange={(e) =>
                  updateCalendarSettings({
                    workingHours: {
                      ...calendar.workingHours,
                      end: e.target.value,
                    },
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Working Days
            </label>
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {workingDays.map((day) => (
                <label key={day.value} className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={calendar.workingHours.days.includes(day.value)}
                    onChange={(e) => {
                      const days = e.target.checked
                        ? [...calendar.workingHours.days, day.value]
                        : calendar.workingHours.days.filter(
                            (d) => d !== day.value
                          );
                      updateCalendarSettings({
                        workingHours: {
                          ...calendar.workingHours,
                          days,
                        },
                      });
                    }}
                    className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm">{day.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </SettingRow>
    </SettingsSection>
  );
}
