
import { useState } from "react";
import { CarPredictionForm, CarPredictionFormValues } from "@/components/car/CarPredictionForm";
import { PredictionResult } from "@/components/car/PredictionResult";
import { predictCarPrice } from "@/lib/ml-model";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Car, Sparkles } from "lucide-react";

const Index = () => {
  const [isLoading, setIsLoading] = useState(false);
  const [prediction, setPrediction] = useState<{
    price: number;
    confidence: number;
    formValues: CarPredictionFormValues;
  } | null>(null);

  const handleSubmit = async (formValues: CarPredictionFormValues) => {
    setIsLoading(true);
    try {
      // In a real app, this would call a backend API with the trained ML model
      const result = await predictCarPrice(formValues);
      
      setPrediction({
        price: result.predictedPrice,
        confidence: result.confidenceScore,
        formValues
      });
    } catch (error) {
      console.error("Prediction error:", error);
      // In a real app, handle errors properly
    } finally {
      setIsLoading(false);
    }
  };

  const handleNewPrediction = () => {
    setPrediction(null);
  };

  return (
    <div className="min-h-screen bg-tesla-gradient">
      <div className="container mx-auto py-12">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center mb-4">
            <div className="bg-primary rounded-full p-3">
              <Car className="h-8 w-8 text-white" />
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            AI Car Price Prediction
          </h1>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Get an accurate estimate of your car's value using our advanced machine learning algorithm
          </p>
        </div>

        {/* Main Content */}
        <div className="max-w-4xl mx-auto">
          <Card className="form-container">
            {!prediction ? (
              <>
                <div className="mb-6 flex items-center gap-2">
                  <Sparkles className="h-5 w-5 text-primary" />
                  <h2 className="text-2xl font-medium">Enter Your Car Details</h2>
                </div>
                <Separator className="mb-8" />
                <CarPredictionForm onSubmit={handleSubmit} isLoading={isLoading} />
              </>
            ) : (
              <PredictionResult 
                prediction={prediction.price}
                confidenceScore={prediction.confidence}
                formValues={prediction.formValues}
                onNewPrediction={handleNewPrediction}
              />
            )}
          </Card>
          
          {/* Features Section */}
          <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center p-6">
              <div className="bg-primary/10 rounded-full p-3 w-14 h-14 flex items-center justify-center mx-auto mb-4">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-medium mb-2">Advanced AI</h3>
              <p className="text-muted-foreground">
                Using machine learning models trained on thousands of real car sales
              </p>
            </div>
            
            <div className="text-center p-6">
              <div className="bg-primary/10 rounded-full p-3 w-14 h-14 flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6 text-primary">
                  <path d="m12 14 4-4" />
                  <path d="M3.34 19a10 10 0 1 1 17.32 0" />
                </svg>
              </div>
              <h3 className="text-xl font-medium mb-2">Precise Valuation</h3>
              <p className="text-muted-foreground">
                Detailed analysis accounting for make, model, year, mileage, and more
              </p>
            </div>
            
            <div className="text-center p-6">
              <div className="bg-primary/10 rounded-full p-3 w-14 h-14 flex items-center justify-center mx-auto mb-4">
                <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-6 w-6 text-primary">
                  <path d="M12 2v20" />
                  <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
                </svg>
              </div>
              <h3 className="text-xl font-medium mb-2">Market Insights</h3>
              <p className="text-muted-foreground">
                Real-time data reflecting current market trends and valuations
              </p>
            </div>
          </div>
        </div>
        
        {/* Footer */}
        <div className="mt-20 text-center text-sm text-muted-foreground">
          <p>© 2025 AI Car Price Prediction. All rights reserved.</p>
          <p className="mt-1">Powered by Linear Regression ML Model</p>
        </div>
      </div>
    </div>
  );
};

export default Index;
