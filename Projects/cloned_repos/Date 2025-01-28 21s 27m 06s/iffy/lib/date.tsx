export function formatDay(date: Date) {
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
  });
}

export function formatDayFull(date: Date) {
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatDate(date: Date) {
  const now = new Date();
  const isOld = now.getFullYear() - date.getFullYear() > 0;
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: isOld ? "numeric" : undefined,
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}

export function formatDateFull(date: Date) {
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
  });
}
