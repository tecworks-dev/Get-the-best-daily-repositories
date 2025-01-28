"use client";

import { useState } from "react";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { MoreHorizontal, Plus } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { RuleWithStrategies } from "@/services/rules";
import type * as schema from "@/db/schema";
import { RuleDialog } from "./rule-dialog";
import { RuleFormValues } from "./schema";
import { useConfirm } from "@/components/ui/confirm";
import { createRule, deleteRule, updateRule } from "./actions";

interface RuleRowProps {
  rule: RuleWithStrategies;
  onEdit: (rule: RuleWithStrategies) => void;
  onDelete: (rule: RuleWithStrategies) => void;
}

function RuleRow({ rule, onEdit, onDelete }: RuleRowProps) {
  const name = rule.preset?.name || rule.name;
  const description = rule.preset?.description || rule.description;

  return (
    <TableRow>
      <TableCell className="w-1/4">{name}</TableCell>
      <TableCell className="w-3/4">{description}</TableCell>
      <TableCell className="w-[50px]">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 w-8 p-0">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => onEdit(rule)}>Edit</DropdownMenuItem>
            <DropdownMenuItem className="text-red-600" onClick={() => onDelete(rule)}>
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
}

export function Rules({
  rulesetId,
  rules,
  presets,
}: {
  rulesetId: string;
  rules: RuleWithStrategies[];
  presets: (typeof schema.presets.$inferSelect)[];
}) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedRule, setSelectedRule] = useState<RuleWithStrategies | undefined>();
  const confirm = useConfirm();

  const createRuleWithRulesetId = createRule.bind(null, rulesetId);

  const handleEdit = (rule: RuleWithStrategies) => {
    setSelectedRule(rule);
    setDialogOpen(true);
  };

  const handleDelete = async (rule: RuleWithStrategies) => {
    if (
      await confirm({
        title: "Delete rule",
        description: "Are you sure you want to delete this rule? This action cannot be undone.",
      })
    ) {
      await deleteRule(rule.id);
    }
  };

  const handleSubmit = async (data: RuleFormValues) => {
    if (selectedRule) {
      await updateRule({ id: selectedRule.id, ...data });
    } else {
      await createRuleWithRulesetId(data);
    }
    setDialogOpen(false);
    setSelectedRule(undefined);
  };

  const remainingPresets = presets.filter((preset) => !rules.some((rule) => rule.presetId === preset.id));

  return (
    <>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold">Rules</h2>
          <Button onClick={() => setDialogOpen(true)}>
            <Plus className="mr-2 h-4 w-4" /> New rule
          </Button>
        </div>

        {rules.length === 0 ? (
          <p className="text-center text-gray-500">No rules set.</p>
        ) : (
          <div className="rounded-md border">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-1/4">Name</TableHead>
                  <TableHead className="w-3/4">Description</TableHead>
                  <TableHead className="w-[50px]" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {rules.map((rule) => (
                  <RuleRow key={rule.id} rule={rule} onEdit={handleEdit} onDelete={handleDelete} />
                ))}
              </TableBody>
            </Table>
          </div>
        )}
      </div>

      <RuleDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        onSubmit={handleSubmit}
        initialData={selectedRule}
        presets={remainingPresets}
      />
    </>
  );
}
