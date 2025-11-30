'use client'

import Image from 'next/image'

export default function HeroSection() {
  return (
    <div className="relative overflow-hidden bg-gradient-medical animate-gradient py-20 px-4">
      <div className="absolute inset-0 bg-black opacity-5"></div>
      <div className="relative max-w-6xl mx-auto text-center">
        <div className="flex justify-center mb-6">
          <div className="bg-white/20 backdrop-blur-sm p-4 rounded-full">
            <Image
              src="/logo.jpg"
              alt="CheXCA Logo"
              width={64}
              height={64}
              className="w-16 h-16 rounded-full object-cover"
              priority
            />
          </div>
        </div>

        <h1 className="text-5xl md:text-6xl font-bold text-white mb-4 tracking-tight">
          CheXCA
        </h1>

        <p className="text-2xl md:text-3xl text-white/90 font-light mb-6">
          Chest X-ray AI Diagnosis
        </p>

        <p className="text-lg md:text-xl text-white/80 max-w-3xl mx-auto leading-relaxed">
          Intelligent diagnosis with <span className="font-semibold">explainable AI</span>.
          <br />
          Get visual insights, disease co-occurrence analysis, and comprehensive medical intelligence.
        </p>

        <div className="mt-10 flex justify-center gap-4 flex-wrap text-sm text-white/70">
          <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-full">
            14 Disease Classifications
          </div>
          <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-full">
            Grad-CAM Visualization
          </div>
          <div className="bg-white/10 backdrop-blur-sm px-4 py-2 rounded-full">
            Real-time Analysis
          </div>
        </div>
      </div>
    </div>
  )
}
