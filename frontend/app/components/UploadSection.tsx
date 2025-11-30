'use client'

import { useState, useCallback } from 'react'
import { Upload, X, Loader2, Image as ImageIcon } from 'lucide-react'
import { analyzeImage, AnalysisResult } from '../lib/api'

interface UploadSectionProps {
  onAnalysisComplete: (result: AnalysisResult) => void
}

export default function UploadSection({ onAnalysisComplete }: UploadSectionProps) {
  const [isDragging, setIsDragging] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)

    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      handleFileSelect(file)
    } else {
      setError('Please upload a valid image file')
    }
  }, [])

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      handleFileSelect(file)
    }
  }

  const handleFileSelect = (file: File) => {
    setSelectedFile(file)
    setError(null)

    // Create preview
    const reader = new FileReader()
    reader.onloadend = () => {
      setPreview(reader.result as string)
    }
    reader.readAsDataURL(file)
  }

  const handleAnalyze = async () => {
    if (!selectedFile) return

    setIsAnalyzing(true)
    setError(null)

    try {
      const result = await analyzeImage(selectedFile)
      onAnalysisComplete(result)
    } catch (err) {
      setError('Analysis failed. Please make sure the backend server is running.')
      console.error('Analysis error:', err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleClear = () => {
    setSelectedFile(null)
    setPreview(null)
    setError(null)
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-12">
      <div className="bg-white rounded-2xl shadow-2xl p-8 border border-gray-100">
        <h2 className="text-2xl font-bold text-gray-800 mb-6 text-center">
          Upload Chest X-ray Image
        </h2>

        {!preview ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              border-3 border-dashed rounded-xl p-12 text-center cursor-pointer
              transition-all duration-300 ease-in-out
              ${isDragging
                ? 'border-primary-500 bg-primary-50 scale-105'
                : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
              }
            `}
          >
            <input
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
              id="file-upload"
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              <div className="flex flex-col items-center gap-4">
                <div className="bg-gradient-medical p-6 rounded-full">
                  <Upload className="w-12 h-12 text-white" />
                </div>
                <div>
                  <p className="text-lg font-semibold text-gray-700 mb-2">
                    Drop your X-ray image here
                  </p>
                  <p className="text-sm text-gray-500">
                    or click to browse files
                  </p>
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  Supports: JPG, PNG, DICOM
                </p>
              </div>
            </label>
          </div>
        ) : (
          <div className="space-y-6">
            <div className="relative bg-gray-100 rounded-xl overflow-hidden">
              <img
                src={preview}
                alt="Preview"
                className="w-full h-96 object-contain"
              />
              <button
                onClick={handleClear}
                className="absolute top-4 right-4 bg-red-500 text-white p-2 rounded-full hover:bg-red-600 transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className={`
                w-full py-4 px-6 rounded-xl font-semibold text-white text-lg
                transition-all duration-300 transform hover:scale-105
                ${isAnalyzing
                  ? 'bg-gray-400 cursor-not-allowed'
                  : 'bg-gradient-medical hover:shadow-xl'
                }
              `}
            >
              {isAnalyzing ? (
                <span className="flex items-center justify-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin" />
                  Analyzing...
                </span>
              ) : (
                <span className="flex items-center justify-center gap-3">
                  <ImageIcon className="w-6 h-6" />
                  Analyze X-ray
                </span>
              )}
            </button>
          </div>
        )}

        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}
      </div>
    </div>
  )
}
