
import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import * as z from "zod";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Car, DollarSign, Fuel } from "lucide-react";
import { carMakes, fuelTypes, transmissionTypes } from "@/lib/car-data";
import { cn } from "@/lib/utils";

const formSchema = z.object({
  make: z.string().min(1, "Please select a car make"),
  model: z.string().min(1, "Please enter a model"),
  year: z.string()
    .refine((val) => !isNaN(Number(val)), "Year must be a number")
    .refine((val) => {
      const year = Number(val);
      return year >= 1980 && year <= new Date().getFullYear();
    }, `Year must be between 1980 and ${new Date().getFullYear()}`),
  mileage: z.number().min(0, "Mileage cannot be negative").max(500000, "Mileage seems too high"),
  fuelType: z.string().min(1, "Please select a fuel type"),
  transmission: z.string().min(1, "Please select a transmission type"),
});

export type CarPredictionFormValues = z.infer<typeof formSchema>;

interface CarPredictionFormProps {
  onSubmit: (values: CarPredictionFormValues) => void;
  isLoading?: boolean;
}

export function CarPredictionForm({ onSubmit, isLoading = false }: CarPredictionFormProps) {
  const [mileage, setMileage] = useState<number>(50000);

  const form = useForm<CarPredictionFormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      make: "",
      model: "",
      year: new Date().getFullYear().toString(),
      mileage: 50000,
      fuelType: "",
      transmission: "",
    },
  });

  const handleMileageChange = (value: number[]) => {
    setMileage(value[0]);
    form.setValue("mileage", value[0]);
  };

  const handleFormSubmit = (values: CarPredictionFormValues) => {
    onSubmit(values);
  };

  return (
    <Form {...form}>
      <form onSubmit={form.handleSubmit(handleFormSubmit)} className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            control={form.control}
            name="make"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="flex items-center gap-2">
                  <Car className="h-4 w-4" />
                  Car Make
                </FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  disabled={isLoading}
                >
                  <FormControl>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select make" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {carMakes.map((make) => (
                      <SelectItem key={make} value={make}>
                        {make}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="model"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Car Model</FormLabel>
                <FormControl>
                  <Input 
                    placeholder="e.g. Accord, Model 3, Camry" 
                    {...field}
                    disabled={isLoading}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            control={form.control}
            name="year"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Year</FormLabel>
                <FormControl>
                  <Input 
                    type="number" 
                    placeholder="e.g. 2018" 
                    min="1980"
                    max={new Date().getFullYear()}
                    {...field}
                    disabled={isLoading}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="mileage"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="flex justify-between">
                  <span>Mileage</span>
                  <span className="font-normal text-muted-foreground">
                    {mileage.toLocaleString()} miles
                  </span>
                </FormLabel>
                <FormControl>
                  <Slider
                    disabled={isLoading}
                    defaultValue={[field.value]}
                    min={0}
                    max={300000}
                    step={1000}
                    onValueChange={handleMileageChange}
                    className={cn("py-4", isLoading && "opacity-50")}
                  />
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <FormField
            control={form.control}
            name="fuelType"
            render={({ field }) => (
              <FormItem>
                <FormLabel className="flex items-center gap-2">
                  <Fuel className="h-4 w-4" />
                  Fuel Type
                </FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  disabled={isLoading}
                >
                  <FormControl>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select fuel type" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {fuelTypes.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />

          <FormField
            control={form.control}
            name="transmission"
            render={({ field }) => (
              <FormItem>
                <FormLabel>Transmission</FormLabel>
                <Select
                  onValueChange={field.onChange}
                  defaultValue={field.value}
                  disabled={isLoading}
                >
                  <FormControl>
                    <SelectTrigger className="w-full">
                      <SelectValue placeholder="Select transmission" />
                    </SelectTrigger>
                  </FormControl>
                  <SelectContent>
                    {transmissionTypes.map((type) => (
                      <SelectItem key={type} value={type}>
                        {type}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <FormMessage />
              </FormItem>
            )}
          />
        </div>

        <Button 
          type="submit" 
          className="w-full py-6 text-lg"
          disabled={isLoading}
        >
          <DollarSign className="mr-2 h-5 w-5" />
          Get Price Prediction
        </Button>
      </form>
    </Form>
  );
}
