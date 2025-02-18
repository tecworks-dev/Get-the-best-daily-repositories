import { ReactNode } from "react";

interface SettingsSectionProps {
  title: string;
  description: string;
  children: ReactNode;
}

export function SettingsSection({
  title,
  description,
  children,
}: SettingsSectionProps) {
  return (
    <div className="py-6">
      <div className="mb-5">
        <h3 className="text-lg font-medium leading-6 text-gray-900">{title}</h3>
        {description && (
          <p className="mt-1 text-sm text-gray-500">{description}</p>
        )}
      </div>
      <div className="mt-6 space-y-6">{children}</div>
    </div>
  );
}

interface SettingRowProps {
  label: string;
  description: ReactNode;
  children: ReactNode;
}

export function SettingRow({ label, description, children }: SettingRowProps) {
  return (
    <div className="flex flex-col sm:flex-row sm:items-start">
      <div className="w-full sm:w-1/3">
        <label className="text-sm font-medium text-gray-900">{label}</label>
        {description && (
          <div className="mt-1 text-sm text-gray-500">{description}</div>
        )}
      </div>
      <div className="mt-2 sm:mt-0 sm:ml-4 sm:w-2/3">{children}</div>
    </div>
  );
}
