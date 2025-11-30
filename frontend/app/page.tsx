'use client'

import { useState } from 'react'
import HeroSection from './components/HeroSection'
import UploadSection from './components/UploadSection'
import ResultsPanel from './components/ResultsPanel'
import HeatmapViewer from './components/HeatmapViewer'
import CoOccurrenceMatrix from './components/CoOccurrenceMatrix'
import { AnalysisResult } from './lib/api'
import { ArrowUp } from 'lucide-react'

export default function Home() {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null)

  const handleAnalysisComplete = (result: AnalysisResult) => {
    setAnalysisResult(result)
    // Smooth scroll to results
    setTimeout(() => {
      const resultsSection = document.getElementById('results')
      resultsSection?.scrollIntoView({ behavior: 'smooth' })
    }, 100)
  }

  const scrollToTop = () => {
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const resetAnalysis = () => {
    setAnalysisResult(null)
    scrollToTop()
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50">
      <HeroSection />

      <UploadSection onAnalysisComplete={handleAnalysisComplete} />

      {analysisResult && (
        <div id="results" className="space-y-12 pb-20">
          <ResultsPanel result={analysisResult} />

          <HeatmapViewer
            originalImage={analysisResult.original_image}
            heatmapOverlay={analysisResult.heatmap_overlay}
            topDisease={analysisResult.top_disease}
          />

          <CoOccurrenceMatrix
            coOccurrence={analysisResult.co_occurrence}
            diseaseClasses={analysisResult.disease_classes}
          />

          {/* Action buttons */}
          <div className="max-w-7xl mx-auto px-4 flex gap-4 justify-center">
            <button
              onClick={resetAnalysis}
              className="px-8 py-4 bg-gradient-medical text-white font-semibold rounded-xl hover:shadow-xl transition-all transform hover:scale-105"
            >
              Analyze Another Image
            </button>
            <button
              onClick={scrollToTop}
              className="px-8 py-4 bg-white text-gray-700 font-semibold rounded-xl border-2 border-gray-300 hover:shadow-xl transition-all transform hover:scale-105 flex items-center gap-2"
            >
              <ArrowUp className="w-5 h-5" />
              Back to Top
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="py-8 text-center text-gray-600 bg-white/50 backdrop-blur-sm mt-20">
        <p className="text-sm">
          CheXCA - Chest X-ray AI Diagnosis System
        </p>
        <p className="text-xs mt-2 text-gray-500">
          For research and educational purposes only. Not for clinical use.
        </p>
      </footer>
    </main>
  )
}
