import { useSettingsStore } from "@/store/settings";
import { SettingsSection, SettingRow } from "./SettingsSection";
import { useSession } from "next-auth/react";
import Image from "next/image";
import { TimeFormat, WeekStartDay } from "@/types/settings";

export function UserSettings() {
  const { data: session } = useSession();
  const { user, updateUserSettings } = useSettingsStore();

  const timeFormats: { value: TimeFormat; label: string }[] = [
    { value: "12h", label: "12-hour" },
    { value: "24h", label: "24-hour" },
  ];

  const weekStarts: { value: WeekStartDay; label: string }[] = [
    { value: "sunday", label: "Sunday" },
    { value: "monday", label: "Monday" },
  ];

  // Static list of common timezones
  const timeZones = [
    "UTC",
    "America/New_York",
    "America/Chicago",
    "America/Denver",
    "America/Los_Angeles",
    "America/Toronto",
    "Europe/London",
    "Europe/Paris",
    "Europe/Berlin",
    "Asia/Tokyo",
    "Asia/Shanghai",
    "Asia/Singapore",
    "Australia/Sydney",
    "Pacific/Auckland",
  ];

  return (
    <SettingsSection
      title="User Settings"
      description="Manage your personal preferences for the calendar application."
    >
      {session?.user && (
        <SettingRow label="Profile" description="Your account information">
          <div className="flex items-center space-x-3">
            {session.user.image && (
              <Image
                src={session.user.image}
                alt={session.user.name || ""}
                width={40}
                height={40}
                className="rounded-full"
              />
            )}
            <div>
              <div className="font-medium">{session.user.name}</div>
              <div className="text-sm text-gray-500">{session.user.email}</div>
            </div>
          </div>
        </SettingRow>
      )}

      <SettingRow
        label="Time Format"
        description="Choose how times are displayed"
      >
        <select
          value={user.timeFormat}
          onChange={(e) =>
            updateUserSettings({ timeFormat: e.target.value as TimeFormat })
          }
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        >
          {timeFormats.map((format) => (
            <option key={format.value} value={format.value}>
              {format.label}
            </option>
          ))}
        </select>
      </SettingRow>

      <SettingRow
        label="Week Starts On"
        description="Choose which day your week starts on"
      >
        <select
          value={user.weekStartDay}
          onChange={(e) =>
            updateUserSettings({ weekStartDay: e.target.value as WeekStartDay })
          }
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        >
          {weekStarts.map((day) => (
            <option key={day.value} value={day.value}>
              {day.label}
            </option>
          ))}
        </select>
      </SettingRow>

      <SettingRow
        label="Time Zone"
        description="Your current time zone setting"
      >
        <select
          value={user.timeZone}
          onChange={(e) => updateUserSettings({ timeZone: e.target.value })}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
        >
          {timeZones.map((zone) => (
            <option key={zone} value={zone}>
              {zone.replace("_", " ")}
            </option>
          ))}
        </select>
      </SettingRow>
    </SettingsSection>
  );
}
