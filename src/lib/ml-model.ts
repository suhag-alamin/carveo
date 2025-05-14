
import { CarPredictionFormValues } from "@/components/car/CarPredictionForm";
import { API_CONFIG } from "./api-config";

/**
 * Car price prediction function that connects to a Python ML model API
 * Falls back to the simulation when API is unavailable
 */
export async function predictCarPrice(formValues: CarPredictionFormValues): Promise<{
  predictedPrice: number;
  confidenceScore: number;
}> {
  try {
    // Call the Python ML model API with the configured endpoint
    const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.PREDICT}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        make: formValues.make,
        model: formValues.model,
        year: formValues.year,
        mileage: formValues.mileage,
        fuelType: formValues.fuelType,
        transmission: formValues.transmission,
      }),
    });
    
    if (!response.ok) {
      throw new Error('API request failed');
    }
    
    const result = await response.json();
    return {
      predictedPrice: result.predictedPrice,
      confidenceScore: result.confidenceScore,
    };
  } catch (error) {
    console.warn('Failed to reach ML API, falling back to simulation', error);
    // Fall back to the simulation
    return simulatePrediction(formValues);
  }
}

/**
 * Simplified simulation as fallback when API is not available
 */
function simulatePrediction(formValues: CarPredictionFormValues): Promise<{
  predictedPrice: number;
  confidenceScore: number;
}> {
  // Simulate API call delay
  return new Promise(resolve => setTimeout(() => {
    // Base price starting point
    let basePrice = 20000;
    
    // Make adjustment (premium brands cost more)
    const premiumBrands = ["BMW", "Mercedes-Benz", "Audi", "Lexus", "Tesla"];
    const midBrands = ["Toyota", "Honda", "Mazda", "Subaru", "Volkswagen"];
    
    if (premiumBrands.includes(formValues.make)) {
      basePrice *= 1.5; // 50% premium
    } else if (midBrands.includes(formValues.make)) {
      basePrice *= 1.2; // 20% premium
    }
    
    // Adjust for year (newer cars cost more)
    const currentYear = new Date().getFullYear();
    const yearFactor = 1 - ((currentYear - parseInt(formValues.year)) * 0.05);
    basePrice *= Math.max(yearFactor, 0.3); // Car won't be worth less than 30% of base due to age
    
    // Adjust for mileage (higher mileage means lower price)
    const mileageFactor = 1 - (formValues.mileage / 300000);
    basePrice *= Math.max(mileageFactor, 0.4); // Car won't be worth less than 40% of base due to mileage
    
    // Adjust for fuel type
    if (formValues.fuelType === "Electric") {
      basePrice *= 1.25; // Electric cars have premium
    } else if (formValues.fuelType === "Hybrid" || formValues.fuelType === "Plug-in Hybrid") {
      basePrice *= 1.15; // Hybrids have smaller premium
    }
    
    // Adjust for transmission type
    if (formValues.transmission === "Automatic") {
      basePrice *= 1.05; // Small premium for automatic
    }
    
    // Add some random variation to simulate real-world price fluctuations
    const randomVariation = 0.9 + (Math.random() * 0.2); // ±10% random variation
    basePrice *= randomVariation;
    
    // Round to nearest hundred
    const predictedPrice = Math.round(basePrice / 100) * 100;
    
    // Generate a confidence score (70-95%)
    const confidenceScore = 70 + Math.random() * 25;
    
    resolve({
      predictedPrice,
      confidenceScore
    });
  }, 1000));
}
