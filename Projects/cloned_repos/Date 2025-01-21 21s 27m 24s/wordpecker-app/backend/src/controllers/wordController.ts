import { Request, Response } from 'express';
import { supabase } from '../config/supabase';
import { openaiService } from '../services/openaiService';

export const wordController = {
  async addWord(req: Request, res: Response) {
    try {
      const { listId } = req.params;
      const { word: value } = req.body;
      const userId = req.user!.id;

      // First verify the list belongs to the user and get context
      const { data: list, error: listError } = await supabase
        .from('word_lists')
        .select('id, context')
        .eq('id', listId)
        .eq('user_id', userId)
        .single();

      if (listError || !list) {
        console.error('List verification error:', listError);
        return res.status(404).json({ message: 'List not found' });
      }

      // Generate word meaning using OpenAI
      const meaning = await openaiService.generateWordMeaning(value, list.context);

      // Add the word with generated meaning
      const { data, error } = await supabase
        .from('words')
        .insert([{
          list_id: listId,
          value,
          meaning
        }])
        .select()
        .single();

      if (error) {
        console.error('Add word error:', error);
        return res.status(500).json({ message: 'Error adding word' });
      }

      res.status(201).json(data);
    } catch (error) {
      console.error('Add word error:', error);
      res.status(500).json({ message: 'Error adding word' });
    }
  },

  async getWords(req: Request, res: Response) {
    try {
      const { listId } = req.params;
      const userId = req.user!.id;

      // First verify the list belongs to the user
      const { data: list, error: listError } = await supabase
        .from('word_lists')
        .select('id')
        .eq('id', listId)
        .eq('user_id', userId)
        .single();

      if (listError || !list) {
        console.error('List verification error:', listError);
        return res.status(404).json({ message: 'List not found' });
      }

      const { data, error } = await supabase
        .from('words')
        .select('*')
        .eq('list_id', listId)
        .order('created_at', { ascending: true });

      if (error) {
        console.error('Get words error:', error);
        return res.status(500).json({ message: 'Error fetching words' });
      }

      res.json(data);
    } catch (error) {
      console.error('Get words error:', error);
      res.status(500).json({ message: 'Error fetching words' });
    }
  },

  async deleteWord(req: Request, res: Response) {
    try {
      const { listId, wordId } = req.params;
      const userId = req.user!.id;

      // First verify the list belongs to the user
      const { data: list, error: listError } = await supabase
        .from('word_lists')
        .select('id')
        .eq('id', listId)
        .eq('user_id', userId)
        .single();

      if (listError || !list) {
        console.error('List verification error:', listError);
        return res.status(404).json({ message: 'List not found' });
      }

      const { error } = await supabase
        .from('words')
        .delete()
        .eq('id', wordId)
        .eq('list_id', listId);

      if (error) {
        console.error('Delete word error:', error);
        return res.status(500).json({ message: 'Error deleting word' });
      }

      res.status(204).send();
    } catch (error) {
      console.error('Delete word error:', error);
      res.status(500).json({ message: 'Error deleting word' });
    }
  }
}; 