
/**
 * API configuration for the ML model
 */
export const API_CONFIG = {
  // Use environment-specific API URL
  BASE_URL: import.meta.env.PROD 
    ? 'https://your-vercel-api-url.vercel.app' // Replace with your actual Vercel API URL
    : 'http://localhost:8000',
  ENDPOINTS: {
    PREDICT: '/predict',
    HEALTH: '/health'
  }
};
