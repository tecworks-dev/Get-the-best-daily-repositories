import { ReactNode } from 'react';

interface MetricCardProps {
  number: string;
  label: string;
  icon: ReactNode;
}

export const MetricCard = ({ number, label, icon }: MetricCardProps) => (
  <div className="bg-slate-50 rounded-lg p-4">
    {/* Center icon */}
    <div className="flex justify-center mb-2">
      {icon}
    </div>
    {/* Center number and label */}
    <div className="text-center">
      <div className="text-2xl font-bold text-slate-900">{number}</div>
      <div className="text-sm text-slate-600">{label}</div>
    </div>
  </div>
); 