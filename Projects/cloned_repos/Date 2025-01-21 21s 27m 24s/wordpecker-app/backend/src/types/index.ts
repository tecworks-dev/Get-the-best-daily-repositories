import { Request } from 'express';

declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
      };
    }
  }
}

export interface WordList {
  id: string;
  user_id: string;
  name: string;
  description?: string;
  context?: string;
  created_at: string;
  updated_at: string;
}

export interface Word {
  id: string;
  list_id: string;
  value: string;
  meaning: string;
  created_at: string;
  updated_at: string;
}

export interface User {
  id: string;
}

export interface Exercise {
  word_id: string;
  type: 'multiple_choice' | 'fill_blank';
  question: string;
  options?: string[];
  correct_answer: string;
}

export interface Question {
  word_id: string;
  type: 'quiz';
  question: string;
  options: string[];
  correct_answer: string;
}

export interface ExerciseResponse {
  exercises: Exercise[];
  completed?: boolean;
}

export interface QuizResponse {
  questions: Question[];
  total_questions?: number;
  completed?: boolean;
} 