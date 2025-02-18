import { CalendarEvent } from "@prisma/client";
import { TimeSlot, Conflict } from "@/types/scheduling";

export interface CalendarService {
  findConflicts(
    slot: TimeSlot,
    selectedCalendarIds: string[]
  ): Promise<Conflict[]>;
  
  getEvents(
    start: Date,
    end: Date,
    selectedCalendarIds: string[]
  ): Promise<CalendarEvent[]>;
} 