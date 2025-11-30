import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatPercentage(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function getConfidenceColor(probability: number): string {
  if (probability >= 0.7) return 'text-red-500'
  if (probability >= 0.5) return 'text-orange-500'
  if (probability >= 0.3) return 'text-yellow-500'
  return 'text-green-500'
}

export function getConfidenceBgColor(probability: number): string {
  if (probability >= 0.7) return 'bg-red-500'
  if (probability >= 0.5) return 'bg-orange-500'
  if (probability >= 0.3) return 'bg-yellow-500'
  return 'bg-blue-500'
}
