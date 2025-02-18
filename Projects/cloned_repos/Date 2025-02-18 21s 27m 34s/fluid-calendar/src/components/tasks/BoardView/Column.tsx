"use client";

import { Task, TaskStatus } from "@/types/task";
import { useDroppable } from "@dnd-kit/core";
import { BoardTask } from "./BoardTask";

interface ColumnProps {
  status: TaskStatus;
  tasks: Task[];
  onEdit: (task: Task) => void;
  onDelete: (taskId: string) => void;
}

const statusColors = {
  [TaskStatus.TODO]: "bg-yellow-50 border-yellow-200",
  [TaskStatus.IN_PROGRESS]: "bg-blue-50 border-blue-200",
  [TaskStatus.COMPLETED]: "bg-green-50 border-green-200",
};

const statusHeaderColors = {
  [TaskStatus.TODO]: "bg-yellow-100 text-yellow-800",
  [TaskStatus.IN_PROGRESS]: "bg-blue-100 text-blue-800",
  [TaskStatus.COMPLETED]: "bg-green-100 text-green-800",
};

// Helper function to format enum values for display
const formatEnumValue = (value: string) => {
  return value
    .toLowerCase()
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

export function Column({
  status,
  tasks,
  onEdit,
  onDelete,
}: ColumnProps) {
  const { setNodeRef, isOver } = useDroppable({
    id: status,
  });

  return (
    <div
      ref={setNodeRef}
      className={`flex-shrink-0 w-80 flex flex-col bg-white rounded-lg border ${
        statusColors[status]
      } ${isOver ? "ring-2 ring-blue-400" : ""}`}
    >
      <div className="p-2 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span
              className={`px-2.5 py-0.5 rounded-full text-sm font-medium ${statusHeaderColors[status]}`}
            >
              {formatEnumValue(status)}
            </span>
            <span className="text-sm text-gray-500">{tasks.length}</span>
          </div>
        </div>
      </div>
      <div className="flex-1 min-h-0 p-2 overflow-y-auto">
        <div className="space-y-2">
          {tasks.map((task) => (
            <BoardTask
              key={task.id}
              task={task}
              onEdit={onEdit}
              onDelete={onDelete}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
