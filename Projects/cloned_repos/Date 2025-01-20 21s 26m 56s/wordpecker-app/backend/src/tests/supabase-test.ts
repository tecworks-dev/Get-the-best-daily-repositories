import { supabase } from '../config/supabase';
import { environment } from '../config/environment';

async function testSupabaseConnection() {
  try {
    const userId = process.env.TEST_USER_UUID;
    if (!userId) {
      throw new Error('TEST_USER_UUID not found in environment variables');
    }

    // First, let's create a test list
    const { data: list, error: listError } = await supabase
      .from('word_lists')
      .insert([
        {
          user_id: userId,
          name: 'Test List',
          description: 'A test list',
          context: 'Testing'
        }
      ])
      .select()
      .single();

    if (listError) {
      throw listError;
    }

    console.log('Created list:', list);

    // Now, let's add a word to the list
    const { data: word, error: wordError } = await supabase
      .from('words')
      .insert([
        {
          list_id: list.id,
          value: 'test',
          meaning: 'a procedure intended to establish the quality, performance, or reliability of something'
        }
      ])
      .select()
      .single();

    if (wordError) {
      throw wordError;
    }

    console.log('Added word:', word);

    // Clean up - delete the test list (this will cascade delete the word)
    const { error: deleteError } = await supabase
      .from('word_lists')
      .delete()
      .eq('id', list.id)
      .eq('user_id', userId);

    if (deleteError) {
      throw deleteError;
    }

    console.log('Successfully cleaned up test data');

  } catch (error) {
    console.error('Test failed:', error);
  }
}

// Run the test
testSupabaseConnection()
  .then(() => console.log('Test completed'))
  .catch(console.error); 