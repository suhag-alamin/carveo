import { useEffect, useState } from "react";
import { Check, TrendingUp, Info } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { CarPredictionFormValues } from "./CarPredictionForm";

interface FeatureImportance {
  name: string;
  importance: number;
}

interface PredictionResultProps {
  prediction: number;
  confidenceScore: number; 
  formValues: CarPredictionFormValues;
  onNewPrediction: () => void;
  featureImportance?: FeatureImportance[];
}

export function PredictionResult({ 
  prediction, 
  confidenceScore, 
  formValues, 
  onNewPrediction,
  featureImportance 
}: PredictionResultProps) {
  const [animatedPrice, setAnimatedPrice] = useState(0);
  const [animatedProgress, setAnimatedProgress] = useState(0);

  useEffect(() => {
    const duration = 1500;
    const priceSteps = 40;
    const progressSteps = 50;
    
    // Animate price
    let priceInterval = prediction / priceSteps;
    let priceCounter = 0;
    
    const priceTimer = setInterval(() => {
      priceCounter++;
      setAnimatedPrice(Math.floor(priceInterval * priceCounter));
      
      if (priceCounter >= priceSteps) {
        clearInterval(priceTimer);
        setAnimatedPrice(prediction);
      }
    }, duration / priceSteps);
    
    // Animate progress
    let progressInterval = confidenceScore / progressSteps;
    let progressCounter = 0;
    
    const progressTimer = setInterval(() => {
      progressCounter++;
      setAnimatedProgress(progressInterval * progressCounter);
      
      if (progressCounter >= progressSteps) {
        clearInterval(progressTimer);
        setAnimatedProgress(confidenceScore);
      }
    }, duration / progressSteps);
    
    return () => {
      clearInterval(priceTimer);
      clearInterval(progressTimer);
    };
  }, [prediction, confidenceScore]);

  // Format price with commas
  const formattedPrice = animatedPrice.toLocaleString('en-US', { 
    style: 'currency', 
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0 
  });

  return (
    <div className="space-y-6">
      <Card className="border-0 bg-gradient-to-br from-blue-500 to-indigo-600 text-white">
        <CardHeader>
          <CardTitle className="text-2xl font-normal">Estimated Value</CardTitle>
          <CardDescription className="text-blue-100">
            Based on current market analysis
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-5xl font-bold">{formattedPrice}</div>
          <div className="mt-4 space-y-2">
            <div className="flex justify-between text-sm">
              <span>Confidence Score</span>
              <span>{Math.round(animatedProgress)}%</span>
            </div>
            <Progress value={animatedProgress} className="h-2 bg-blue-400/30" />
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Vehicle Details
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <h4 className="text-sm font-medium text-muted-foreground">Make & Model</h4>
              <p className="font-medium">{formValues.make} {formValues.model}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-muted-foreground">Year</h4>
              <p className="font-medium">{formValues.year}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-muted-foreground">Mileage</h4>
              <p className="font-medium">{formValues.mileage.toLocaleString()} miles</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-muted-foreground">Fuel Type</h4>
              <p className="font-medium">{formValues.fuelType}</p>
            </div>
            <div>
              <h4 className="text-sm font-medium text-muted-foreground">Transmission</h4>
              <p className="font-medium">{formValues.transmission}</p>
            </div>
          </div>
        </CardContent>
      </Card>
      
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            What's Influencing This Price
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {featureImportance && featureImportance.map((feature) => (
              <div key={feature.name} className="space-y-1">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Check className="h-4 w-4 text-primary" />
                    <span>{feature.name}</span>
                  </div>
                  <Badge variant="outline">{feature.importance}%</Badge>
                </div>
                <Progress value={feature.importance} className="h-1" />
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
      
      <Button onClick={onNewPrediction} variant="outline" className="w-full">
        Get Another Prediction
      </Button>
    </div>
  );
}
