import request from 'supertest';
import app from '../app';
import { supabase } from '../config/supabase';

describe('Quiz Routes', () => {
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
        name: 'Test Quiz List',
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
      { list_id: testListId, value: 'quiz word 1', meaning: 'meaning 1' },
      { list_id: testListId, value: 'quiz word 2', meaning: 'meaning 2' },
      { list_id: testListId, value: 'quiz word 3', meaning: 'meaning 3' },
      { list_id: testListId, value: 'quiz word 4', meaning: 'meaning 4' },
      { list_id: testListId, value: 'quiz word 5', meaning: 'meaning 5' }
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

  describe('POST /api/quiz/:listId/start', () => {
    it('should start a quiz session', async () => {
      const response = await request(app)
        .post(`/api/quiz/${testListId}/start`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('questions');
      expect(response.body).toHaveProperty('total_questions');
      expect(Array.isArray(response.body.questions)).toBe(true);
      expect(response.body.questions.length).toBeGreaterThan(0);

      const question = response.body.questions[0];
      expect(question).toHaveProperty('word_id');
      expect(question).toHaveProperty('type');
      expect(question).toHaveProperty('question');
      expect(question).toHaveProperty('options');
      expect(question).toHaveProperty('correct_answer');
      expect(question.type).toBe('quiz');
      expect(Array.isArray(question.options)).toBe(true);
    });

    it('should return 404 for non-existent list', async () => {
      const response = await request(app)
        .post('/api/quiz/00000000-0000-0000-0000-000000000000/start')
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
          name: 'Empty Quiz List',
          description: 'No words'
        }])
        .select()
        .single();

      const response = await request(app)
        .post(`/api/quiz/${emptyList.id}/start`)
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

  describe('GET /api/quiz/:listId/questions', () => {
    it('should get more questions', async () => {
      const response = await request(app)
        .get(`/api/quiz/${testListId}/questions`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('questions');
      expect(response.body).toHaveProperty('completed');
      expect(Array.isArray(response.body.questions)).toBe(true);

      if (response.body.questions.length > 0) {
        const question = response.body.questions[0];
        expect(question).toHaveProperty('word_id');
        expect(question).toHaveProperty('type');
        expect(question).toHaveProperty('question');
        expect(question).toHaveProperty('options');
        expect(question).toHaveProperty('correct_answer');
        expect(question.type).toBe('quiz');
        expect(Array.isArray(question.options)).toBe(true);
      }
    });

    it('should handle lastWordId parameter', async () => {
      // Get first batch of questions
      const firstResponse = await request(app)
        .get(`/api/quiz/${testListId}/questions`)
        .set('Authorization', `Bearer ${testUser.token}`);

      const lastWordId = firstResponse.body.questions[0].word_id;

      // Get next batch using lastWordId
      const response = await request(app)
        .get(`/api/quiz/${testListId}/questions`)
        .query({ lastWordId })
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('questions');
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