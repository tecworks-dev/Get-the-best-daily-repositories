import { Request, Response } from 'express';
import { supabase } from '../config/supabase';
import { Question, QuizResponse } from '../types';
import { openaiService } from '../services/openaiService';
import { shuffleArray } from '../utils/arrayUtils';

export const quizController = {
  async startQuiz(req: Request, res: Response) {
    try {
      const { listId } = req.params;
      const userId = req.user!.id;

      // First verify the list belongs to the user and get its words
      const { data: list, error: listError } = await supabase
        .from('word_lists')
        .select(`
          id,
          name,
          context,
          words (
            id,
            value,
            meaning
          )
        `)
        .eq('id', listId)
        .eq('user_id', userId)
        .single();

      if (listError || !list) {
        return res.status(404).json({ message: 'List not found' });
      }

      if (!list.words?.length) {
        return res.status(400).json({ message: 'List has no words' });
      }

      // Shuffle the words array to randomize the quiz
      const shuffledWords = list.words.sort(() => Math.random() - 0.5);

      // Generate quiz questions using OpenAI
      const questions = await openaiService.createQuizQuestions(
        shuffledWords.slice(0, 5),
        list.context
      );

      // Return list details along with questions
      res.json({ 
        questions,
        total_questions: shuffledWords.length,
        list: {
          id: list.id,
          name: list.name,
          context: list.context
        }
      });
    } catch (error) {
      console.error('Start quiz error:', error);
      res.status(500).json({ message: 'Error starting quiz' });
    }
  },

  async getMoreQuestions(req: Request, res: Response) {
    try {
      const { listId } = req.params;
      const userId = req.user!.id;

      // First verify the list belongs to the user
      const { data: list, error: listError } = await supabase
        .from('word_lists')
        .select(`
          id,
          context,
          words (
            id,
            value,
            meaning
          )
        `)
        .eq('id', listId)
        .eq('user_id', userId)
        .single();

      if (listError || !list) {
        return res.status(404).json({ message: 'List not found' });
      }

      if (!list.words?.length) {
        return res.status(400).json({ message: 'List has no words' });
      }

      // Shuffle all words and take 5 random ones
      const shuffledWords = shuffleArray(list.words);
      const selectedWords = shuffledWords.slice(0, Math.min(5, shuffledWords.length));

      // Generate quiz questions using OpenAI
      const questions = await openaiService.createQuizQuestions(
        selectedWords,
        list.context
      );

      res.json({ questions });
    } catch (error) {
      console.error('Get more questions error:', error);
      res.status(500).json({ message: 'Error getting more questions' });
    }
  }
}; 