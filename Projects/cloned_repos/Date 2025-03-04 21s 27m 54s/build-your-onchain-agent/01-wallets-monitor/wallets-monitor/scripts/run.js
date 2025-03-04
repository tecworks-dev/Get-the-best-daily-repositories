import { setupSwapWebhook } from './heliusSetup.js';

// Run setup
(async () => {
  try {
    await setupSwapWebhook();
    console.log('Webhook setup completed');
  } catch (err) {
    console.error('Setup failed:', err);
  }
})(); 
