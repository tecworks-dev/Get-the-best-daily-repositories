import request from 'supertest';
import app from '../app';
import { supabase } from '../config/supabase';

describe('List Routes', () => {
  const testUser = {
    id: process.env.TEST_USER_UUID,
    token: process.env.TEST_USER_TOKEN
  };

  let testListId: string;

  beforeAll(async () => {
    // Clean up any existing test data
    await supabase
      .from('words')
      .delete()
      .neq('id', '00000000-0000-0000-0000-000000000000');

    await supabase
      .from('word_lists')
      .delete()
      .neq('id', '00000000-0000-0000-0000-000000000000');

    // Wait for deletions to complete
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  describe('POST /api/lists', () => {
    it('should create a new list', async () => {
      const response = await request(app)
        .post('/api/lists')
        .set('Authorization', `Bearer ${testUser.token}`)
        .send({
          name: 'Test List',
          description: 'Test Description',
          context: 'Test Context'
        });

      expect(response.status).toBe(201);
      expect(response.body).toHaveProperty('id');
      expect(response.body.name).toBe('Test List');
      testListId = response.body.id;
    });

    it('should return 400 if name is missing', async () => {
      const response = await request(app)
        .post('/api/lists')
        .set('Authorization', `Bearer ${testUser.token}`)
        .send({
          description: 'Test Description'
        });

      expect(response.status).toBe(400);
    });
  });

  describe('GET /api/lists/:id', () => {
    beforeEach(async () => {
      // Create a test list for these tests
      const { data, error } = await supabase
        .from('word_lists')
        .insert([{
          user_id: testUser.id,
          name: 'Test List',
          description: 'Test Description'
        }])
        .select()
        .single();

      if (error || !data) {
        throw new Error('Failed to create test list');
      }

      testListId = data.id;
      // Wait for the list to be created
      await new Promise(resolve => setTimeout(resolve, 1000));
    });

    afterEach(async () => {
      // Clean up test list
      if (testListId) {
        await supabase
          .from('words')
          .delete()
          .eq('list_id', testListId);

        await supabase
          .from('word_lists')
          .delete()
          .eq('id', testListId);

        // Wait for deletions to complete
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    });

    it('should return a specific list', async () => {
      const response = await request(app)
        .get(`/api/lists/${testListId}`)
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(response.body).toHaveProperty('id');
      expect(response.body.id).toBe(testListId);
      expect(response.body.name).toBe('Test List');
    });

    it('should return 404 for non-existent list', async () => {
      const response = await request(app)
        .get('/api/lists/00000000-0000-0000-0000-000000000000')
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(404);
      expect(response.body).toHaveProperty('message');
      expect(response.body.message).toBe('List not found');
    });
  });

  describe('GET /api/lists', () => {
    it('should return user lists', async () => {
      const response = await request(app)
        .get('/api/lists')
        .set('Authorization', `Bearer ${testUser.token}`);

      expect(response.status).toBe(200);
      expect(Array.isArray(response.body)).toBe(true);
    });
  });

  afterAll(async () => {
    // Final cleanup
    await supabase
      .from('words')
      .delete()
      .neq('id', '00000000-0000-0000-0000-000000000000');

    await supabase
      .from('word_lists')
      .delete()
      .neq('id', '00000000-0000-0000-0000-000000000000');
  });
}); 