
import { CarPredictionFormValues } from "@/components/car/CarPredictionForm";
import { API_CONFIG } from "./api-config";

export interface FeatureImportance {
  name: string;
  importance: number;
}

/**
 * Car price prediction function that connects to a Python ML model API
 * Falls back to the simulation when API is unavailable
 */
export async function predictCarPrice(formValues: CarPredictionFormValues): Promise<{
  predictedPrice: number;
  confidenceScore: number;
  featureImportance: FeatureImportance[];
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
      featureImportance: result.featureImportance || [],
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
  featureImportance: FeatureImportance[];
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
    
    // Generate simulated feature importance
    const featureImportance: FeatureImportance[] = [];
    
    // Year importance (higher for newer cars)
    const yearImportance = 25 + (parseInt(formValues.year) - 2000) / 5;
    
    // Mileage importance (higher for high-mileage cars)
    const mileageImportance = 20 + (formValues.mileage > 100000 ? 10 : 0);
    
    // Make importance (higher for premium brands)
    const makeImportance = premiumBrands.includes(formValues.make) ? 25 : 15;
    
    // Fuel type importance (higher for electric/hybrid)
    const fuelImportance = 
      formValues.fuelType === "Electric" ? 20 : 
      (formValues.fuelType === "Hybrid" || formValues.fuelType === "Plug-in Hybrid") ? 15 : 10;
    
    // Transmission importance
    const transmissionImportance = 10;
    
    // Normalize to ensure they add up to 100%
    const total = yearImportance + mileageImportance + makeImportance + fuelImportance + transmissionImportance;
    
    featureImportance.push({ 
      name: "Year", 
      importance: Math.round((yearImportance / total) * 100)
    });
    
    featureImportance.push({ 
      name: "Mileage", 
      importance: Math.round((mileageImportance / total) * 100)
    });
    
    featureImportance.push({ 
      name: "Make", 
      importance: Math.round((makeImportance / total) * 100)
    });
    
    featureImportance.push({ 
      name: "Fuel Type", 
      importance: Math.round((fuelImportance / total) * 100)
    });
    
    featureImportance.push({ 
      name: "Transmission", 
      importance: Math.round((transmissionImportance / total) * 100)
    });
    
    // Sort by importance (descending)
    featureImportance.sort((a, b) => b.importance - a.importance);
    
    resolve({
      predictedPrice,
      confidenceScore,
      featureImportance
    });
  }, 1000));
}
