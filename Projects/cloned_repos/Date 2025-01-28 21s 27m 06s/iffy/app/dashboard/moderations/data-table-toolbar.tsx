"use client";

import { Link, MoreHorizontal, XIcon } from "lucide-react";
import { Table } from "@tanstack/react-table";

import { Button } from "@/components/ui/button";
import { DataTableViewOptions } from "@/components/ui/data-table-view-options";
import { DataTableFacetedFilter } from "@/components/ui/data-table-faceted-filter";

import type { Record } from "../records/types";
import { DebouncedInput } from "@/components/debounced-input";
import { BulkActionMenu } from "../records/action-menu";

import * as schema from "@/db/schema";

const STATUS_OPTIONS = [
  { label: "Compliant", value: "Compliant" },
  { label: "Flagged", value: "Flagged" },
];

interface DataTableToolbarProps<TData> {
  table: Table<TData>;
  data: TData[];
}

function DataTableToolbarActions<TData extends Record>({ table, data }: DataTableToolbarProps<TData>) {
  const selected = table.getFilteredSelectedRowModel().rows.map((row) => row.original);
  return (
    <div className="flex items-center text-sm">
      <span className="text-gray-500">{selected.length} selected</span>
      <Button variant="link" className="h-8" onClick={() => table.toggleAllRowsSelected(false)}>
        Deselect
      </Button>
      <BulkActionMenu records={selected} />
    </div>
  );
}

export function DataTableToolbar<TData extends Record>({ table, data }: DataTableToolbarProps<TData>) {
  const isFiltered = !!table.getState().globalFilter || table.getState().columnFilters.length > 0;
  const hasSelected = table.getFilteredSelectedRowModel().rows.length > 0;

  const ENTITY_OPTIONS = data
    .reduce(
      (entities, record) => (entities.includes(record.entity) ? entities : [...entities, record.entity]),
      [] as string[],
    )
    .map((entity) => ({
      label: entity,
      value: entity,
    }));

  return (
    <div className="flex items-center justify-between px-4">
      <div className="flex flex-1 items-center space-x-2">
        <DebouncedInput
          placeholder="Search moderations..."
          value={(table.getState().globalFilter as string) ?? ""}
          onChange={(value) => table.setGlobalFilter(value)}
          className="h-8 w-[150px] lg:w-[250px]"
        />
        {table.getColumn("status") && (
          <DataTableFacetedFilter column={table.getColumn("status")} title="Status" options={STATUS_OPTIONS} />
        )}
        {table.getColumn("entity") && (
          <DataTableFacetedFilter column={table.getColumn("entity")} title="Entity" options={ENTITY_OPTIONS} />
        )}
        {isFiltered && (
          <Button
            variant="ghost"
            onClick={() => {
              table.resetGlobalFilter();
              table.resetColumnFilters();
            }}
            className="h-8 px-2 lg:px-3"
          >
            Clear
            <XIcon className="ml-2 h-4 w-4" />
          </Button>
        )}
      </div>
      {hasSelected ? <DataTableToolbarActions table={table} data={data} /> : <DataTableViewOptions table={table} />}
    </div>
  );
}
