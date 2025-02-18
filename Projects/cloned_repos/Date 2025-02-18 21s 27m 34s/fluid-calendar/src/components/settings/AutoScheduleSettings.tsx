import { useSettingsStore } from "@/store/settings";
import { useCalendarStore } from "@/store/calendar";
import { SettingsSection, SettingRow } from "./SettingsSection";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import {
  parseWorkDays,
  parseSelectedCalendars,
  stringifyWorkDays,
  stringifySelectedCalendars,
  formatTime,
} from "@/lib/autoSchedule";
import { useEffect } from "react";

export function AutoScheduleSettings() {
  const { autoSchedule, updateAutoScheduleSettings } = useSettingsStore();
  const { feeds, loadFromDatabase } = useCalendarStore();

  // Load calendar feeds when component mounts
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

  const timeOptions = Array.from({ length: 24 }, (_, i) => ({
    value: i,
    label: formatTime(i),
  }));

  const selectedCalendars = parseSelectedCalendars(
    autoSchedule.selectedCalendars
  );
  const workDays = parseWorkDays(autoSchedule.workDays);

  return (
    <SettingsSection
      title="Auto-Schedule Settings"
      description="Configure how tasks are automatically scheduled in your calendar."
    >
      <SettingRow
        label="Calendars to Consider"
        description="Select which calendars to check for conflicts when auto-scheduling"
      >
        <div className="space-y-2">
          {feeds.map((feed) => (
            <div key={feed.id} className="flex items-center space-x-2">
              <Switch
                checked={selectedCalendars.includes(feed.id)}
                onCheckedChange={(checked) => {
                  const calendars = checked
                    ? [...selectedCalendars, feed.id]
                    : selectedCalendars.filter((id) => id !== feed.id);
                  updateAutoScheduleSettings({
                    selectedCalendars: stringifySelectedCalendars(calendars),
                  });
                }}
              />
              <Label className="flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: feed.color || "#E5E7EB" }}
                />
                {feed.name}
              </Label>
            </div>
          ))}
          {feeds.length === 0 && (
            <div className="text-sm text-gray-500">
              No calendars found. Please add calendars in the Calendar Settings.
            </div>
          )}
        </div>
      </SettingRow>

      <SettingRow
        label="Working Hours"
        description="Set your preferred working hours for task scheduling"
      >
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label>Start Time</Label>
              <select
                value={autoSchedule.workHourStart}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    workHourStart: parseInt(e.target.value),
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <Label>End Time</Label>
              <select
                value={autoSchedule.workHourEnd}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    workHourEnd: parseInt(e.target.value),
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div>
            <Label>Working Days</Label>
            <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
              {workingDays.map((day) => (
                <label key={day.value} className="flex items-center space-x-2">
                  <Switch
                    checked={workDays.includes(day.value)}
                    onCheckedChange={(checked) => {
                      const days = checked
                        ? [...workDays, day.value]
                        : workDays.filter((d) => d !== day.value);
                      updateAutoScheduleSettings({
                        workDays: stringifyWorkDays(days),
                      });
                    }}
                  />
                  <span className="text-sm">{day.label}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
      </SettingRow>

      <SettingRow
        label="Energy Level Time Preferences"
        description="Map your energy levels to specific time ranges"
      >
        <div className="space-y-6">
          <div className="space-y-2">
            <Label>High Energy Hours</Label>
            <div className="grid grid-cols-2 gap-4">
              <select
                value={autoSchedule.highEnergyStart ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    highEnergyStart: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
              <select
                value={autoSchedule.highEnergyEnd ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    highEnergyEnd: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Medium Energy Hours</Label>
            <div className="grid grid-cols-2 gap-4">
              <select
                value={autoSchedule.mediumEnergyStart ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    mediumEnergyStart: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
              <select
                value={autoSchedule.mediumEnergyEnd ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    mediumEnergyEnd: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="space-y-2">
            <Label>Low Energy Hours</Label>
            <div className="grid grid-cols-2 gap-4">
              <select
                value={autoSchedule.lowEnergyStart ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    lowEnergyStart: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
              <select
                value={autoSchedule.lowEnergyEnd ?? ""}
                onChange={(e) =>
                  updateAutoScheduleSettings({
                    lowEnergyEnd: e.target.value
                      ? parseInt(e.target.value)
                      : null,
                  })
                }
                className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="">Not Set</option>
                {timeOptions.map((time) => (
                  <option key={time.value} value={time.value}>
                    {time.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </SettingRow>

      <SettingRow
        label="Buffer Time"
        description="Minutes to leave between scheduled tasks"
      >
        <div className="space-y-4">
          <Slider
            value={[autoSchedule.bufferMinutes]}
            onValueChange={([value]) =>
              updateAutoScheduleSettings({ bufferMinutes: value })
            }
            min={0}
            max={60}
            step={5}
          />
          <div className="text-sm text-muted-foreground">
            Current buffer: {autoSchedule.bufferMinutes} minutes
          </div>
        </div>
      </SettingRow>

      <SettingRow
        label="Project Grouping"
        description="Try to schedule tasks from the same project together"
      >
        <Switch
          checked={autoSchedule.groupByProject}
          onCheckedChange={(checked) =>
            updateAutoScheduleSettings({ groupByProject: checked })
          }
        />
      </SettingRow>
    </SettingsSection>
  );
}
