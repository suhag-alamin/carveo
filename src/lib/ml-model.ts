
import { CarPredictionFormValues } from "@/components/car/CarPredictionForm";

/**
 * Simplified simulation of a Linear Regression ML model for car price prediction
 * In a real application, this would call a backend API that uses a trained ML model
 */
export async function predictCarPrice(formValues: CarPredictionFormValues): Promise<{
  predictedPrice: number;
  confidenceScore: number;
}> {
  // Simulate API call delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
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
  
  return {
    predictedPrice,
    confidenceScore
  };
}
