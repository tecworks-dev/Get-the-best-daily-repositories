'use client';

import {
  createContext,
  useContext,
  useReducer,
  ReactNode,
  useCallback,
} from 'react';

interface ActivityItem {
  type:
    | 'search'
    | 'extract'
    | 'analyze'
    | 'reasoning'
    | 'synthesis'
    | 'thought';
  status: 'pending' | 'complete' | 'error';
  message: string;
  timestamp: string;
  depth?: number;
}

interface SourceItem {
  url: string;
  title: string;
  relevance: number;
}

interface DeepResearchState {
  isActive: boolean;
  activity: ActivityItem[];
  sources: SourceItem[];
  currentDepth: number;
  maxDepth: number;
  completedSteps: number;
  totalExpectedSteps: number;
}

type DeepResearchAction =
  | { type: 'TOGGLE_ACTIVE' }
  | { type: 'SET_ACTIVE'; payload: boolean }
  | {
      type: 'ADD_ACTIVITY';
      payload: ActivityItem & { completedSteps?: number; totalSteps?: number };
    }
  | { type: 'ADD_SOURCE'; payload: SourceItem }
  | { type: 'SET_DEPTH'; payload: { current: number; max: number } }
  | { type: 'INIT_PROGRESS'; payload: { maxDepth: number; totalSteps: number } }
  | { type: 'UPDATE_PROGRESS'; payload: { completed: number; total: number } }
  | { type: 'CLEAR_STATE' };

interface DeepResearchContextType {
  state: DeepResearchState;
  toggleActive: () => void;
  setActive: (active: boolean) => void;
  addActivity: (
    activity: ActivityItem & { completedSteps?: number; totalSteps?: number },
  ) => void;
  addSource: (source: SourceItem) => void;
  setDepth: (current: number, max: number) => void;
  initProgress: (maxDepth: number, totalSteps: number) => void;
  updateProgress: (completed: number, total: number) => void;
  clearState: () => void;
}

const initialState: DeepResearchState = {
  isActive: true,
  activity: [],
  sources: [],
  currentDepth: 0,
  maxDepth: 7,
  completedSteps: 0,
  totalExpectedSteps: 0,
};

function deepResearchReducer(
  state: DeepResearchState,
  action: DeepResearchAction,
): DeepResearchState {
  switch (action.type) {
    case 'TOGGLE_ACTIVE':
      return {
        ...state,
        isActive: !state.isActive,
        ...(state.isActive && {
          activity: [],
          sources: [],
          currentDepth: 0,
          completedSteps: 0,
          totalExpectedSteps: 0,
        }),
      };
    case 'SET_ACTIVE':
      return {
        ...state,
        isActive: action.payload,
        ...(action.payload === false && {
          activity: [],
          sources: [],
          currentDepth: 0,
          completedSteps: 0,
          totalExpectedSteps: 0,
        }),
      };
    case 'ADD_ACTIVITY':
      return {
        ...state,
        activity: [...state.activity, action.payload],
        completedSteps:
          action.payload.completedSteps ??
          (action.payload.status === 'complete'
            ? state.completedSteps + 1
            : state.completedSteps),
        totalExpectedSteps:
          action.payload.totalSteps ?? state.totalExpectedSteps,
      };
    case 'ADD_SOURCE':
      return {
        ...state,
        sources: [...state.sources, action.payload],
      };
    case 'SET_DEPTH':
      return {
        ...state,
        currentDepth: action.payload.current,
        maxDepth: action.payload.max,
      };
    case 'INIT_PROGRESS':
      return {
        ...state,
        maxDepth: action.payload.maxDepth,
        totalExpectedSteps: action.payload.totalSteps,
        completedSteps: 0,
        currentDepth: 0,
      };
    case 'UPDATE_PROGRESS':
      return {
        ...state,
        completedSteps: action.payload.completed,
        totalExpectedSteps: action.payload.total,
      };
    case 'CLEAR_STATE':
      return {
        ...initialState,
        activity: [],
        sources: [],
        currentDepth: 0,
        completedSteps: 0,
        totalExpectedSteps: 0,
      };
    default:
      return state;
  }
}

const DeepResearchContext = createContext<DeepResearchContextType | undefined>(
  undefined,
);

export function DeepResearchProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(deepResearchReducer, initialState);

  const toggleActive = useCallback(() => {
    dispatch({ type: 'TOGGLE_ACTIVE' });
  }, []);

  const setActive = useCallback((active: boolean) => {
    dispatch({ type: 'SET_ACTIVE', payload: active });
  }, []);

  const addActivity = useCallback(
    (
      activity: ActivityItem & { completedSteps?: number; totalSteps?: number },
    ) => {
      dispatch({ type: 'ADD_ACTIVITY', payload: activity });
    },
    [],
  );

  const addSource = useCallback((source: SourceItem) => {
    dispatch({ type: 'ADD_SOURCE', payload: source });
  }, []);

  const setDepth = useCallback((current: number, max: number) => {
    dispatch({ type: 'SET_DEPTH', payload: { current, max } });
  }, []);

  const initProgress = useCallback((maxDepth: number, totalSteps: number) => {
    dispatch({ type: 'INIT_PROGRESS', payload: { maxDepth, totalSteps } });
  }, []);

  const updateProgress = useCallback((completed: number, total: number) => {
    dispatch({ type: 'UPDATE_PROGRESS', payload: { completed, total } });
  }, []);

  const clearState = useCallback(() => {
    dispatch({ type: 'CLEAR_STATE' });
  }, []);

  return (
    <DeepResearchContext.Provider
      value={{
        state,
        toggleActive,
        setActive,
        addActivity,
        addSource,
        setDepth,
        initProgress,
        updateProgress,
        clearState,
      }}
    >
      {children}
    </DeepResearchContext.Provider>
  );
}

export function useDeepResearch() {
  const context = useContext(DeepResearchContext);
  if (context === undefined) {
    throw new Error(
      'useDeepResearch must be used within a DeepResearchProvider',
    );
  }
  return context;
}
