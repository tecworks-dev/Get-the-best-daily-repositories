import { useSettingsStore } from "@/store/settings";
import { SettingsSection, SettingRow } from "./SettingsSection";

export function NotificationSettings() {
  const { notifications, updateNotificationSettings } = useSettingsStore();

  const handleReminderChange = (index: number, value: string) => {
    const newReminders = [...notifications.defaultReminderTiming];
    newReminders[index] = Number(value);
    updateNotificationSettings({ defaultReminderTiming: newReminders });
  };

  const addReminder = () => {
    updateNotificationSettings({
      defaultReminderTiming: [...notifications.defaultReminderTiming, 30],
    });
  };

  const removeReminder = (index: number) => {
    const newReminders = notifications.defaultReminderTiming.filter(
      (_, i) => i !== index
    );
    updateNotificationSettings({ defaultReminderTiming: newReminders });
  };

  return (
    <SettingsSection
      title="Notification Settings"
      description="Configure how and when you receive notifications about your events."
    >
      <SettingRow
        label="Email Notifications"
        description="Receive notifications via email"
      >
        <label className="flex items-center">
          <input
            type="checkbox"
            checked={notifications.emailNotifications}
            onChange={(e) =>
              updateNotificationSettings({
                emailNotifications: e.target.checked,
              })
            }
            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="ml-2 text-sm">Enable email notifications</span>
        </label>
      </SettingRow>

      <SettingRow
        label="Notification Types"
        description="Choose which types of notifications you want to receive"
      >
        <div className="space-y-2">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={notifications.notifyFor.eventInvites}
              onChange={(e) =>
                updateNotificationSettings({
                  notifyFor: {
                    ...notifications.notifyFor,
                    eventInvites: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm">Event invitations</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={notifications.notifyFor.eventUpdates}
              onChange={(e) =>
                updateNotificationSettings({
                  notifyFor: {
                    ...notifications.notifyFor,
                    eventUpdates: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm">Event updates</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={notifications.notifyFor.eventCancellations}
              onChange={(e) =>
                updateNotificationSettings({
                  notifyFor: {
                    ...notifications.notifyFor,
                    eventCancellations: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm">Event cancellations</span>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={notifications.notifyFor.eventReminders}
              onChange={(e) =>
                updateNotificationSettings({
                  notifyFor: {
                    ...notifications.notifyFor,
                    eventReminders: e.target.checked,
                  },
                })
              }
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            />
            <span className="ml-2 text-sm">Event reminders</span>
          </label>
        </div>
      </SettingRow>

      <SettingRow
        label="Default Reminders"
        description="Set when you want to be reminded about events"
      >
        <div className="space-y-3">
          {notifications.defaultReminderTiming.map((minutes, index) => (
            <div key={index} className="flex items-center space-x-2">
              <input
                type="number"
                min="0"
                max="10080"
                value={minutes}
                onChange={(e) => handleReminderChange(index, e.target.value)}
                className="block w-24 rounded-md border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              />
              <span className="text-sm text-gray-500">minutes before</span>
              <button
                type="button"
                onClick={() => removeReminder(index)}
                className="ml-2 text-red-600 hover:text-red-700"
              >
                Remove
              </button>
            </div>
          ))}
          <button
            type="button"
            onClick={addReminder}
            className="inline-flex items-center rounded-md border border-transparent bg-blue-600 px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
          >
            Add Reminder
          </button>
        </div>
      </SettingRow>
    </SettingsSection>
  );
}
