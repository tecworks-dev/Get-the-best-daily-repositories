import { Request, Response } from 'express';
import { supabase } from '../config/supabase';

export const listController = {
  async createList(req: Request, res: Response) {
    try {
      const { name, description, context } = req.body;
      const userId = req.user!.id;

      const { data, error } = await supabase
        .from('word_lists')
        .insert([{
          user_id: userId,
          name,
          description,
          context
        }])
        .select()
        .single();

      if (error) {
        console.error('Create list error:', error);
        return res.status(500).json({ message: 'Error creating list' });
      }

      res.status(201).json(data);
    } catch (error) {
      console.error('Create list error:', error);
      res.status(500).json({ message: 'Error creating list' });
    }
  },

  async getLists(req: Request, res: Response) {
    try {
      const userId = req.user!.id;

      const { data, error } = await supabase
        .from('word_lists')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Get lists error:', error);
        return res.status(500).json({ message: 'Error fetching lists' });
      }

      res.json(data);
    } catch (error) {
      console.error('Get lists error:', error);
      res.status(500).json({ message: 'Error fetching lists' });
    }
  },

  async getList(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const userId = req.user!.id;

      const { data, error } = await supabase
        .from('word_lists')
        .select('*')
        .eq('id', id)
        .eq('user_id', userId)
        .single();

      if (error) {
        console.error('Get list error:', error);
        if (error.code === 'PGRST116') {
          return res.status(404).json({ message: 'List not found' });
        }
        return res.status(500).json({ message: 'Error fetching list' });
      }

      if (!data) {
        return res.status(404).json({ message: 'List not found' });
      }

      res.json(data);
    } catch (error) {
      console.error('Get list error:', error);
      res.status(500).json({ message: 'Error fetching list' });
    }
  },

  async updateList(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const { name, description, context } = req.body;
      const userId = req.user!.id;

      const { data, error } = await supabase
        .from('word_lists')
        .update({ name, description, context })
        .eq('id', id)
        .eq('user_id', userId)
        .select()
        .single();

      if (error) {
        console.error('Update list error:', error);
        if (error.code === 'PGRST116') {
          return res.status(404).json({ message: 'List not found' });
        }
        return res.status(500).json({ message: 'Error updating list' });
      }

      if (!data) {
        return res.status(404).json({ message: 'List not found' });
      }

      res.json(data);
    } catch (error) {
      console.error('Update list error:', error);
      res.status(500).json({ message: 'Error updating list' });
    }
  },

  async deleteList(req: Request, res: Response) {
    try {
      const { id } = req.params;
      const userId = req.user!.id;

      // First delete all words in the list
      await supabase
        .from('words')
        .delete()
        .eq('list_id', id);

      const { error } = await supabase
        .from('word_lists')
        .delete()
        .eq('id', id)
        .eq('user_id', userId);

      if (error) {
        console.error('Delete list error:', error);
        if (error.code === 'PGRST116') {
          return res.status(404).json({ message: 'List not found' });
        }
        return res.status(500).json({ message: 'Error deleting list' });
      }

      res.status(204).send();
    } catch (error) {
      console.error('Delete list error:', error);
      res.status(500).json({ message: 'Error deleting list' });
    }
  }
}; 