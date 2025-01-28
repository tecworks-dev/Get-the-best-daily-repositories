"use client";

import type { Row, Table as TableImpl } from "@tanstack/react-table";
import { flexRender } from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface DataTableProps<TData> {
  table: TableImpl<TData>;
}

export function DataTableLoading<TData>({ table }: DataTableProps<TData>) {
  return (
    <Table className="h-full">
      <TableHeader>
        {table.getHeaderGroups().map((headerGroup) => (
          <TableRow key={headerGroup.id}>
            {headerGroup.headers.map((header) => {
              return (
                <TableHead key={header.id} colSpan={header.colSpan}>
                  {header.isPlaceholder ? null : flexRender(header.column.columnDef.header, header.getContext())}
                </TableHead>
              );
            })}
          </TableRow>
        ))}
      </TableHeader>
      <TableBody>
        <TableRow>
          <TableCell colSpan={table.getAllColumns().length} className="h-24 text-center">
            Loading...
          </TableCell>
        </TableRow>
      </TableBody>
    </Table>
  );
}
