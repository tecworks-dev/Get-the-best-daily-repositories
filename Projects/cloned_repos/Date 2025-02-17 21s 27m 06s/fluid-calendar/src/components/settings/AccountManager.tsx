import { useState, useCallback } from "react";
import { useSettingsStore } from "@/store/settings";
import { useEffect } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { AvailableCalendars } from "./AvailableCalendars";

export function AccountManager() {
  const { accounts, refreshAccounts, removeAccount } = useSettingsStore();
  const [showAvailableFor, setShowAvailableFor] = useState<string | null>(null);

  useEffect(() => {
    refreshAccounts();
  }, [refreshAccounts]);

  const handleConnect = (provider: "GOOGLE" | "OUTLOOK") => {
    if (provider === "GOOGLE") {
      window.location.href = `/api/calendar/google/auth`;
    }
    // TODO: Add Outlook support
  };

  const handleRemove = async (accountId: string) => {
    try {
      await removeAccount(accountId);
    } catch (error) {
      console.error("Failed to remove account:", error);
    }
  };

  const toggleAvailableCalendars = useCallback((accountId: string) => {
    setShowAvailableFor((current) =>
      current === accountId ? null : accountId
    );
  }, []);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Connected Accounts</CardTitle>
          <CardDescription>
            Manage your connected calendar accounts
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button onClick={() => handleConnect("GOOGLE")}>
              Connect Google Calendar
            </Button>
            <Button
              onClick={() => handleConnect("OUTLOOK")}
              disabled
              title="Coming soon"
            >
              Connect Outlook Calendar
            </Button>
          </div>

          <div className="space-y-2">
            {accounts.map((account) => (
              <div key={account.id} className="space-y-2">
                <div className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-2">
                    <Badge
                      variant={
                        account.provider === "GOOGLE" ? "default" : "secondary"
                      }
                    >
                      {account.provider}
                    </Badge>
                    <span>{account.email}</span>
                    <Badge variant="outline">
                      {account.calendars.length} calendars
                    </Badge>
                  </div>
                  <div className="flex gap-2">
                    {account.provider === "GOOGLE" && (
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => toggleAvailableCalendars(account.id)}
                      >
                        {showAvailableFor === account.id ? "Hide" : "Show"}{" "}
                        Available Calendars
                      </Button>
                    )}
                    <Button
                      variant="destructive"
                      size="sm"
                      onClick={() => handleRemove(account.id)}
                    >
                      Remove
                    </Button>
                  </div>
                </div>
                {showAvailableFor === account.id && (
                  <AvailableCalendars accountId={account.id} />
                )}
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
