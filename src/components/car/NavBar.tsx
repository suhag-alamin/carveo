import { Car } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import logo from "../../../public/logo.png";

export function NavBar() {
  return (
    <header className="border-b bg-white/50 backdrop-blur-md sticky top-0 z-50">
      <div className="container mx-auto flex h-16 items-center justify-between px-4">
        <Link to="/" className="flex items-center gap-2">
          {/* <div className="bg-primary rounded-full p-1">
            <Car className="h-5 w-5 text-white" />
          </div> */}
          <img src={logo} alt="Carveo Logo" className="h-8 w-8" />
          <span className="font-medium text-lg">Carveo</span>
        </Link>

        <nav className="hidden md:flex items-center gap-6">
          <Link
            to="/"
            className="text-sm font-medium hover:text-primary transition-colors"
          >
            Home
          </Link>
          <Link
            to="/"
            className="text-sm font-medium hover:text-primary transition-colors"
          >
            How It Works
          </Link>
          <Link
            to="/"
            className="text-sm font-medium hover:text-primary transition-colors"
          >
            About
          </Link>
        </nav>

        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm">
            Sign In
          </Button>
          <Button size="sm">Get Started</Button>
        </div>
      </div>
    </header>
  );
}
