import request from 'supertest';
import app from '../app';
import { supabase } from '../config/supabase';

describe('Word Routes', () => {
  const testUser = {
    id: process.env.TEST_USER_UUID,
    token: process.env.TEST_USER_TOKEN
  };

  let testListId: string;
  let testWordId: string;

  beforeAll(async () => {
    // Create a test list
    const { data: listData, error: listError } = await supabase
      .from('word_lists')
      .insert([{
        user_id: testUser.id,
        name: 'Test List',
        description: 'Test Description'
      }])
      .select()
      .single();

    if (listError) {
      console.error('Error creating test list:', listError);
      throw listError;
    }

    testListId = listData.id;

    // Wait for the list to be created
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Create initial test word
    const { data: wordData, error: wordError } = await supabase
      .from('words')
      .insert([{
        list_id: testListId,
        value: 'initial test word',
        meaning: 'test meaning'
      }])
      .select()
      .single();

    if (wordError) {
      console.error('Error creating test word:', wordError);
      throw wordError;
    }

    testWordId = wordData.id;
  });

  describe('POST /api/lists/:listId/words', () => {
    it('should add a word to the list', async () => {
      const response = await request(app)
        .post(`/api/lists/${testListId}/words`)
        .set('Authorization', `Bearer ${testUser.token}`)
        .send({
          word: 'new test word'
        });

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('id');
      expect(response.body.value).toBe('new test word');
    });
  });

  describe('GET /api/lists/:listId/words', () => {
    it('should return list words', async () => {
      const response = await request(app)
        .get(`/api/lists/${testListId}/words`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
      expect(response.body.length).toBeGreaterThan(0);
    });
  });

  describe('DELETE /api/lists/:listId/words/:wordId', () => {
    it('should delete a word', async () => {
      const response = await request(app)
        .delete(`/api/lists/${testListId}/words/${testWordId}`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(204);
    });
  });

  afterAll(async () => {
    if (testListId) {
      // Delete all words first
      await supabase
        .from('words')
        .delete()
        .eq('list_id', testListId);

      // Then delete the list
      await supabase
        .from('word_lists')
        .delete()
        .eq('id', testListId);
    }
  });
}); 