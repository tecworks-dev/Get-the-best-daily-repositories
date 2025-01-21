import { Request, Response } from 'express';
import { supabase } from '../config/supabase';
import { Exercise, ExerciseResponse, WordList } from '../types';
import { openaiService } from '../services/openaiService';
import { shuffleArray } from '../utils/arrayUtils';

export const learnController = {
  async startLearning(req: Request, res: Response) {
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

      // Shuffle words and take 5 random ones
      const shuffledWords = shuffleArray(list.words);
      const selectedWords = shuffledWords.slice(0, Math.min(5, shuffledWords.length));

      // Generate exercises using OpenAI
      const exercises = await openaiService.createExercises(
        selectedWords,
        list.context
      );

      // Return both list and exercises
      res.json({
        exercises,
        list: {
          id: list.id,
          name: list.name,
          context: list.context
        }
      });
    } catch (error) {
      console.error('Start learning error:', error);
      res.status(500).json({ message: 'Error starting learning session' });
    }
  },

  async getMoreExercises(req: Request, res: Response) {
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

      // Generate exercises using OpenAI
      const exercises = await openaiService.createExercises(
        selectedWords,
        list.context
      );

      res.json({ exercises });
    } catch (error) {
      console.error('Get more exercises error:', error);
      res.status(500).json({ message: 'Error getting more exercises' });
    }
  }
}; 