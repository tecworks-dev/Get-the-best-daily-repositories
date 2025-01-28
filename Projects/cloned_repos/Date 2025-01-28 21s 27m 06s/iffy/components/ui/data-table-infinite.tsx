"use client";

import type { Row, Table as TableImpl } from "@tanstack/react-table";
import { flexRender } from "@tanstack/react-table";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface DataTableProps<TData> {
  table: TableImpl<TData>;
  onRowClick?: (row: Row<TData>) => void;
}

export function DataTableInfinite<TData>({ table, onRowClick }: DataTableProps<TData>) {
  return (
    <Table>
      <TableHeader className="sticky top-0 z-10 bg-white dark:bg-zinc-900 [&_tr:after]:absolute [&_tr:after]:bottom-0 [&_tr:after]:left-0 [&_tr:after]:w-full [&_tr:after]:border-b [&_tr:after]:border-stone-300 [&_tr:after]:content-[''] dark:[&_tr:after]:border-zinc-800">
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
        {table.getRowModel().rows?.length ? (
          table.getRowModel().rows.map((row) => (
            <TableRow
              key={row.id}
              data-state={row.getIsSelected() && "selected"}
              onClick={() => onRowClick?.(row)}
              className={cn(onRowClick && "cursor-pointer")}
            >
              {row.getVisibleCells().map((cell) => (
                <TableCell key={cell.id}>{flexRender(cell.column.columnDef.cell, cell.getContext())}</TableCell>
              ))}
            </TableRow>
          ))
        ) : (
          <TableRow>
            <TableCell colSpan={table.getAllColumns().length} className="h-24 text-center">
              No results.
            </TableCell>
          </TableRow>
        )}
      </TableBody>
    </Table>
  );
}
