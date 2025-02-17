import { useEffect, useState, useRef, useCallback } from "react";
import FullCalendar from "@fullcalendar/react";
import timeGridPlugin from "@fullcalendar/timegrid";
import interactionPlugin from "@fullcalendar/interaction";
import type {
  EventClickArg,
  EventContentArg,
  DatesSetArg,
} from "@fullcalendar/core";
import type { DateSelectArg } from "@fullcalendar/core";
import { useCalendarStore } from "@/store/calendar";
import { useSettingsStore } from "@/store/settings";
import { useTaskStore } from "@/store/task";
import { EventModal } from "./EventModal";
import { TaskModal } from "@/components/tasks/TaskModal";
import { CalendarEvent, ExtendedEventProps } from "@/types/calendar";
import { Task } from "@/types/task";
import { IoRepeat, IoCheckmarkCircle, IoTimeOutline } from "react-icons/io5";
import { cn } from "@/lib/utils";

interface WeekViewProps {
  currentDate: Date;
  onDateClick?: (date: Date) => void;
}

export function WeekView({ currentDate, onDateClick }: WeekViewProps) {
  const { feeds, getAllCalendarItems, isLoading } = useCalendarStore();
  const { user: userSettings, calendar: calendarSettings } = useSettingsStore();
  const { updateTask } = useTaskStore();
  const [selectedEvent, setSelectedEvent] = useState<Partial<CalendarEvent>>();
  const [selectedTask, setSelectedTask] = useState<Task>();
  const [selectedDate, setSelectedDate] = useState<Date>();
  const [selectedEndDate, setSelectedEndDate] = useState<Date>();
  const [isEventModalOpen, setIsEventModalOpen] = useState(false);
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false);
  const [events, setEvents] = useState<
    Array<{
      id: string;
      title: string;
      start: Date;
      end: Date;
      location?: string;
      backgroundColor: string;
      borderColor: string;
      allDay: boolean;
      classNames: string[];
      extendedProps?: ExtendedEventProps;
    }>
  >([]);
  const calendarRef = useRef<FullCalendar>(null);
  const tasks = useTaskStore((state) => state.tasks);

  // Update events when the calendar view changes
  const handleDatesSet = useCallback(
    async (arg: DatesSetArg) => {
      console.log("Calendar dates set:", {
        start: arg.start.toISOString(),
        end: arg.end.toISOString(),
        timeZone: arg.timeZone,
      });

      // Get all calendar items with current task data
      const items = getAllCalendarItems(arg.start, arg.end);
      console.log("Retrieved calendar items:", {
        total: items.length,
        tasks: items.filter((item) => item.feedId === "tasks").length,
        events: items.filter((item) => item.feedId !== "tasks").length,
      });

      const formattedItems = items
        .filter((item) => {
          if (item.feedId === "tasks") return true;
          const feed = feeds.find((f) => f.id === item.feedId);
          return feed?.enabled;
        })
        .map((item) => ({
          id: item.id,
          title: item.title,
          start: new Date(item.start),
          end: new Date(item.end),
          location: item.location,
          backgroundColor:
            item.feedId === "tasks"
              ? item.color || "#4f46e5"
              : feeds.find((f) => f.id === item.feedId)?.color || "#3b82f6",
          borderColor:
            item.feedId === "tasks"
              ? item.color || "#4f46e5"
              : feeds.find((f) => f.id === item.feedId)?.color || "#3b82f6",
          allDay: item.allDay,
          classNames: [
            item.extendedProps?.isTask ? "calendar-task" : "calendar-event",
          ],
          // Store the original event data
          extendedProps: {
            ...item,
            // Bring important flags to top level of extendedProps for easy access
            isTask: item.extendedProps?.isTask,
            isRecurring: item.isRecurring,
            status: item.extendedProps?.status,
          },
        }));

      console.log("Setting formatted calendar items:", {
        total: formattedItems.length,
        tasks: formattedItems.filter((item) => item.extendedProps?.isTask)
          .length,
        events: formattedItems.filter((item) => !item.extendedProps?.isTask)
          .length,
      });
      setEvents(formattedItems);
    },
    [feeds, getAllCalendarItems]
  );

  // Initial data load
  useEffect(() => {
    console.log("Loading initial data...");
    Promise.all([
      useCalendarStore.getState().loadFromDatabase(),
      useTaskStore.getState().fetchTasks(),
    ]);
  }, []);

  // Update items when loading state changes, feeds change, or tasks change
  useEffect(() => {
    if (!isLoading && calendarRef.current) {
      console.log("Updating calendar items due to dependency change");
      const calendar = calendarRef.current.getApi();
      handleDatesSet({
        start: calendar.view.activeStart,
        end: calendar.view.activeEnd,
        startStr: calendar.view.activeStart.toISOString(),
        endStr: calendar.view.activeEnd.toISOString(),
        timeZone: userSettings.timeZone,
        view: calendar.view,
      });
    }
  }, [isLoading, feeds, userSettings.timeZone, handleDatesSet, tasks]);

  // Update calendar date when currentDate changes
  useEffect(() => {
    if (calendarRef.current) {
      setTimeout(() => {
        if (calendarRef.current) {
          const calendar = calendarRef.current.getApi();
          calendar.gotoDate(currentDate);
        }
      }, 0);
    }
  }, [currentDate]);

  const handleEventClick = (info: EventClickArg) => {
    const item = info.event.extendedProps;
    const itemId = info.event.id;
    const isTask = item.isTask;
    console.log("Event clicked:", info.event);
    console.log("info.event.id:", info.event.id);
    if (isTask) {
      // Handle task click
      console.log("Task ID:", itemId);

      const task = useTaskStore.getState().tasks.find((t) => t.id === itemId);
      if (task) {
        setSelectedTask(task);
        setIsTaskModalOpen(true);
      }
    } else {
      // Handle regular event click
      const event = useCalendarStore.getState().events.find(
        (e) => e.id === itemId
      );
      setSelectedEvent(event as CalendarEvent);
      setIsEventModalOpen(true);
    }
  };

  const handleDateSelect = (selectInfo: DateSelectArg) => {
    const start = selectInfo.start;
    const end = selectInfo.allDay ? start : selectInfo.end;

    setSelectedDate(start);
    setSelectedEndDate(end);
    setSelectedEvent({
      allDay: selectInfo.allDay,
    });
    setIsEventModalOpen(true);
  };

  const handleEventModalClose = () => {
    setIsEventModalOpen(false);
    setSelectedEvent(undefined);
    setSelectedDate(undefined);
    setSelectedEndDate(undefined);
  };

  const handleTaskModalClose = () => {
    setIsTaskModalOpen(false);
    setSelectedTask(undefined);
  };

  const renderEventContent = (eventInfo: EventContentArg) => {
    const isTask = eventInfo.event.extendedProps.isTask;
    const isRecurring = eventInfo.event.extendedProps.isRecurring;

    return (
      <div
        data-testid={isTask ? "calendar-task" : "calendar-event"}
        className={cn(
          "flex items-center gap-1 p-1 text-xs overflow-hidden h-full",
          isTask && "border-l-4",
          isTask && {
            "border-green-500":
              eventInfo.event.extendedProps.status === "completed",
            "border-yellow-500":
              eventInfo.event.extendedProps.status === "in_progress",
            "border-gray-500": eventInfo.event.extendedProps.status === "todo",
          }
        )}
      >
        {isTask ? (
          <IoCheckmarkCircle className="flex-shrink-0 h-3 w-3 text-current opacity-75" />
        ) : isRecurring ? (
          <IoRepeat className="flex-shrink-0 h-3 w-3 text-current opacity-75" />
        ) : (
          <IoTimeOutline className="flex-shrink-0 h-3 w-3 text-current opacity-75" />
        )}
        <div className="flex-1">
          <div className="font-medium truncate">{eventInfo.event.title}</div>
          {eventInfo.event.extendedProps.location && (
            <div className="truncate opacity-80 text-[10px]">
              {eventInfo.event.extendedProps.location}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="h-full [&_.fc-timegrid-slot]:!h-[25px]">
      <button
        onClick={() => {
          setSelectedDate(new Date());
          setSelectedEndDate(new Date(Date.now() + 3600000));
          setIsEventModalOpen(true);
        }}
        data-testid="create-event-button"
        className="fixed bottom-4 right-4 z-10 rounded-full bg-blue-600 p-3 text-white shadow-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-6 w-6"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 4v16m8-8H4"
          />
        </svg>
      </button>

      <FullCalendar
        ref={calendarRef}
        plugins={[timeGridPlugin, interactionPlugin]}
        initialView="timeGridWeek"
        headerToolbar={false}
        initialDate={currentDate}
        events={events}
        nowIndicator={true}
        allDaySlot={true}
        slotMinTime="00:00:00"
        slotMaxTime="24:00:00"
        scrollTime={calendarSettings.workingHours.start}
        expandRows={true}
        stickyHeaderDates={true}
        timeZone="local"
        displayEventEnd={true}
        eventTimeFormat={{
          hour: userSettings.timeFormat === "12h" ? "numeric" : "2-digit",
          minute: "2-digit",
          meridiem: userSettings.timeFormat === "12h" ? "short" : false,
          hour12: userSettings.timeFormat === "12h",
        }}
        slotLabelFormat={{
          hour: userSettings.timeFormat === "12h" ? "numeric" : "2-digit",
          minute: "2-digit",
          meridiem: userSettings.timeFormat === "12h" ? "short" : false,
          hour12: userSettings.timeFormat === "12h",
        }}
        firstDay={userSettings.weekStartDay === "monday" ? 1 : 0}
        businessHours={{
          daysOfWeek: calendarSettings.workingHours.enabled
            ? calendarSettings.workingHours.days
            : [0, 1, 2, 3, 4, 5, 6],
          startTime: calendarSettings.workingHours.start,
          endTime: calendarSettings.workingHours.end,
        }}
        dayHeaderFormat={{
          weekday: "short",
          month: "numeric",
          day: "numeric",
          omitCommas: true,
        }}
        height="100%"
        dateClick={(arg) => onDateClick?.(arg.date)}
        eventClick={handleEventClick}
        select={handleDateSelect}
        selectable={true}
        selectMirror={true}
        datesSet={handleDatesSet}
        eventContent={renderEventContent}
      />

      <EventModal
        isOpen={isEventModalOpen}
        onClose={handleEventModalClose}
        event={selectedEvent}
        defaultDate={selectedDate}
        defaultEndDate={selectedEndDate}
      />

      {selectedTask && (
        <TaskModal
          isOpen={isTaskModalOpen}
          onClose={handleTaskModalClose}
          task={selectedTask}
          tags={useTaskStore.getState().tags}
          onSave={async (updates) => {
            await updateTask(selectedTask.id, updates);
            handleTaskModalClose();
          }}
          onCreateTag={async (name: string, color?: string) => {
            return useTaskStore.getState().createTag({ name, color });
          }}
        />
      )}
    </div>
  );
}
