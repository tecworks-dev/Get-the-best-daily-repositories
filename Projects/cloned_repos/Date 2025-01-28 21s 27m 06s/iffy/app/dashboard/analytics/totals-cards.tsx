import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface TotalsCardsProps {
  totals: {
    moderations: number;
    flagged: number;
  };
}

export function TotalsCards({ totals }: TotalsCardsProps) {
  return (
    <>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium dark:text-stone-100">Total moderations</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-black">{totals.moderations.toLocaleString()}</div>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
          <CardTitle className="text-sm font-medium dark:text-stone-100">Total flagged</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold text-red-600">{totals.flagged.toLocaleString()}</div>
        </CardContent>
      </Card>
    </>
  );
}
