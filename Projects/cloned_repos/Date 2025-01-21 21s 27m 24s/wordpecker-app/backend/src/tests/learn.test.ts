import request from 'supertest';
import app from '../app';
import { supabase } from '../config/supabase';

describe('Learning Routes', () => {
  const testUser = {
    id: process.env.TEST_USER_UUID,
    token: process.env.TEST_USER_TOKEN
  };

  let testListId: string;

  beforeAll(async () => {
    // Create a test list with words
    const { data: listData, error: listError } = await supabase
      .from('word_lists')
      .insert([{
        user_id: testUser.id,
        name: 'Test Learning List',
        description: 'Test Description'
      }])
      .select()
      .single();

    if (listError) {
      throw listError;
    }

    testListId = listData.id;

    // Add test words
    const testWords = [
      { list_id: testListId, value: 'test word 1', meaning: 'meaning 1' },
      { list_id: testListId, value: 'test word 2', meaning: 'meaning 2' },
      { list_id: testListId, value: 'test word 3', meaning: 'meaning 3' }
    ];

    const { error: wordsError } = await supabase
      .from('words')
      .insert(testWords);

    if (wordsError) {
      throw wordsError;
    }

    // Wait for data to be available
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  describe('POST /api/learn/:listId/start', () => {
    it('should start a learning session', async () => {
      const response = await request(app)
        .post(`/api/learn/${testListId}/start`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('exercises');
      expect(Array.isArray(response.body.exercises)).toBe(true);
      expect(response.body.exercises.length).toBeGreaterThan(0);

      const exercise = response.body.exercises[0];
      expect(exercise).toHaveProperty('word_id');
      expect(exercise).toHaveProperty('type');
      expect(exercise).toHaveProperty('question');
      expect(exercise).toHaveProperty('options');
      expect(exercise).toHaveProperty('correct_answer');
      expect(exercise.type).toBe('multiple_choice');
    });

    it('should return 404 for non-existent list', async () => {
      const response = await request(app)
        .post('/api/learn/00000000-0000-0000-0000-000000000000/start')
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(404);
      expect(response.body).toHaveProperty('message');
      expect(response.body.message).toBe('List not found');
    });

    it('should return 400 for list with no words', async () => {
      // Create an empty list
      const { data: emptyList } = await supabase
        .from('word_lists')
        .insert([{
          user_id: testUser.id,
          name: 'Empty List',
          description: 'No words'
        }])
        .select()
        .single();

      const response = await request(app)
        .post(`/api/learn/${emptyList.id}/start`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(400);
      expect(response.body).toHaveProperty('message');
      expect(response.body.message).toBe('List has no words');

      // Clean up empty list
      await supabase
        .from('word_lists')
        .delete()
        .eq('id', emptyList.id);
    });
  });

  describe('GET /api/learn/:listId/exercises', () => {
    it('should get more exercises', async () => {
      const response = await request(app)
        .get(`/api/learn/${testListId}/exercises`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('exercises');
      expect(response.body).toHaveProperty('completed');
      expect(Array.isArray(response.body.exercises)).toBe(true);

      if (response.body.exercises.length > 0) {
        const exercise = response.body.exercises[0];
        expect(exercise).toHaveProperty('word_id');
        expect(exercise).toHaveProperty('type');
        expect(exercise).toHaveProperty('question');
        expect(exercise).toHaveProperty('options');
        expect(exercise).toHaveProperty('correct_answer');
        expect(exercise.type).toBe('multiple_choice');
      }
    });

    it('should handle lastWordId parameter', async () => {
      // Get first batch of exercises
      const firstResponse = await request(app)
        .get(`/api/learn/${testListId}/exercises`)
        .set('Authorization', `Bearer ${testUser.token}`);

      const lastWordId = firstResponse.body.exercises[0].word_id;

      // Get next batch using lastWordId
      const response = await request(app)
        .get(`/api/learn/${testListId}/exercises`)
        .query({ lastWordId })
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('exercises');
      expect(response.body).toHaveProperty('completed');
    });
  });

  afterAll(async () => {
    if (testListId) {
      // Clean up test data
      await supabase
        .from('words')
        .delete()
        .eq('list_id', testListId);

      await supabase
        .from('word_lists')
        .delete()
        .eq('id', testListId);
    }
  });
}); 