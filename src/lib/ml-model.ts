
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
    console.log('Calling ML API at:', `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.PREDICT}`);
    
    // Set a timeout to prevent long waiting times for the API
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
    
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
      signal: controller.signal,
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error('API request failed');
    }
    
    const result = await response.json();
    console.log('API Response:', result);
    
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
  console.log('Using simulation for prediction with values:', formValues);
  
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
    
    // Generate truly dynamic feature importance based on the input data
    let featureImportance: FeatureImportance[] = [];
    
    // Calculate year importance (newer cars have higher year importance)
    const yearValue = parseInt(formValues.year);
    const yearImportance = Math.round(15 + ((yearValue - 2000) / 23) * 15);
    
    // Calculate mileage importance (higher for high-mileage cars)
    const mileageImportance = Math.round(15 + (formValues.mileage / 200000) * 15);
    
    // Calculate make importance (higher for premium brands)
    const makeImportance = premiumBrands.includes(formValues.make) ? 30 : 
                           midBrands.includes(formValues.make) ? 22 : 15;
    
    // Calculate fuel type importance
    let fuelImportance = 10;
    if (formValues.fuelType === "Electric") {
      fuelImportance = 25;
    } else if (formValues.fuelType === "Hybrid" || formValues.fuelType === "Plug-in Hybrid") {
      fuelImportance = 18;
    }
    
    // Transmission importance varies less
    const transmissionImportance = formValues.transmission === "Automatic" ? 12 : 8;
    
    // Normalize to ensure they add up to 100%
    const totalImportance = yearImportance + mileageImportance + makeImportance + 
                            fuelImportance + transmissionImportance;
    
    featureImportance = [
      {
        name: "Year",
        importance: Math.round((yearImportance / totalImportance) * 100)
      },
      {
        name: "Mileage",
        importance: Math.round((mileageImportance / totalImportance) * 100)
      },
      {
        name: "Make",
        importance: Math.round((makeImportance / totalImportance) * 100)
      },
      {
        name: "Fuel Type",
        importance: Math.round((fuelImportance / totalImportance) * 100)
      },
      {
        name: "Transmission",
        importance: Math.round((transmissionImportance / totalImportance) * 100)
      }
    ];
    
    // Sort by importance (descending)
    featureImportance.sort((a, b) => b.importance - a.importance);
    
    // Ensure percentages sum to 100% (adjust the last element if needed)
    const sum = featureImportance.reduce((total, item) => total + item.importance, 0);
    if (sum !== 100) {
      const lastItem = featureImportance[featureImportance.length - 1];
      lastItem.importance += (100 - sum);
    }
    
    console.log('Simulated feature importance:', featureImportance);
    
    resolve({
      predictedPrice,
      confidenceScore,
      featureImportance
    });
  }, 1000));
}
