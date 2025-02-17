"use client";

import { useState } from "react";
import { UserSettings } from "@/components/settings/UserSettings";
import { CalendarSettings } from "@/components/settings/CalendarSettings";
import { NotificationSettings } from "@/components/settings/NotificationSettings";
import { IntegrationSettings } from "@/components/settings/IntegrationSettings";
import { DataSettings } from "@/components/settings/DataSettings";
import { Separator } from "@/components/ui/separator";
import { AccountManager } from "@/components/settings/AccountManager";
import { AutoScheduleSettings } from "@/components/settings/AutoScheduleSettings";
import { cn } from "@/lib/utils";

type SettingsTab =
  | "accounts"
  | "user"
  | "calendar"
  | "auto-schedule"
  | "notifications"
  | "integrations"
  | "data";

export default function SettingsPage() {
  const [activeTab, setActiveTab] = useState<SettingsTab>("accounts");

  const tabs = [
    { id: "accounts", label: "Accounts" },
    { id: "user", label: "User" },
    { id: "calendar", label: "Calendar" },
    { id: "auto-schedule", label: "Auto-Schedule" },
    { id: "notifications", label: "Notifications" },
    { id: "integrations", label: "Integrations" },
    { id: "data", label: "Data" },
  ] as const;

  const renderContent = () => {
    switch (activeTab) {
      case "accounts":
        return <AccountManager />;
      case "user":
        return <UserSettings />;
      case "calendar":
        return <CalendarSettings />;
      case "auto-schedule":
        return <AutoScheduleSettings />;
      case "notifications":
        return <NotificationSettings />;
      case "integrations":
        return <IntegrationSettings />;
      case "data":
        return <DataSettings />;
      default:
        return null;
    }
  };

  return (
    <div className="container py-6">
      <div className="flex flex-col lg:flex-row lg:space-x-12 lg:space-y-0">
        <aside className="lg:w-1/5">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as SettingsTab)}
                className={cn(
                  "flex w-full items-center rounded-lg px-3 py-2 text-sm font-medium",
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </aside>
        <div className="flex-1">
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold tracking-tight">Settings</h2>
              <p className="text-muted-foreground">
                Manage your account settings and preferences.
              </p>
            </div>
            <Separator />
            <div className="space-y-8">{renderContent()}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
