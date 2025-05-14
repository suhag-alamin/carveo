/**
 * API configuration for the ML model
 */
export const API_CONFIG = {
  // Use environment-specific API URL
  BASE_URL: import.meta.env.PROD
    ? "https://carveo-ml-api.onrender.com"
    : "https://carveo-ml-api.onrender.com",
  ENDPOINTS: {
    PREDICT: "/predict",
    HEALTH: "/health",
  },
};
