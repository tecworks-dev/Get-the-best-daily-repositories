import { Fragment, useState, useEffect, useMemo } from 'react';
import { Dialog, Transition } from '@headlessui/react';
import { supabase } from '@/utils/supabase';

interface OnboardingTourProps {
  isFirstTime: boolean;
  onComplete: () => void;
}

interface Step {
  title: string;
  description: React.ReactNode;
  targetClass: string;
}

export function OnboardingTour({ isFirstTime, onComplete }: OnboardingTourProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);

  const steps = useMemo<Step[]>(() => [
    {
      title: "Start with a Template",
      description: (
        <div className="flex flex-col gap-2">
          <p>Click  <button className="px-2 py-1 text-xs rounded bg-primary hover:bg-primary-darker text-white w-fit">Use Template</button> to import one of our pre-made recipes into your collection.</p>
          
        </div>
      ),
      targetClass: "recipe-templates"
    },
    {
      title: "Voice Start",
      description: (
        <div className="flex flex-col gap-2 items-center">
          <p>Click the microphone to START speaking.</p>
          <button className="w-12 h-12 mt-2 rounded-full flex items-center justify-center bg-primary hover:bg-primary-darker shadow-lg"><span className="text-xl text-white">🎤</span></button>
        </div>
      ),
      targetClass: "ai-assistant-button"
    },
    {
        title: "Voice Stop",
        description: (
            <div className="flex flex-col gap-2 items-center">
              <p>Click the red button to STOP voice interaction.</p>
              <button className="w-12 h-12 mt-2 rounded-full flex items-center justify-center bg-red-500 hover:bg-red-600 shadow-lg"><span className="text-xl text-white">⏹</span></button>
            </div>
          ),
          targetClass: "ai-assistant-button"
    },
    {
      title: "Add Your Own Recipe",
      description: (
        <div className="flex flex-col gap-2 items-center">
          <p>Click the button below to add your own recipe:</p>
          <button className="w-48 mt-2 sm:w-auto px-4 py-2 rounded-lg flex items-center justify-center gap-2 bg-primary hover:bg-primary-darker text-white">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
            <span>Add New Recipe</span>
          </button>
        </div>
      ),
      targetClass: "add-recipe-button"
    }
  ], []);

  useEffect(() => {
    const checkOnboardingStatus = async () => {
      const user = (await supabase.auth.getUser()).data.user;
      
      if (!user) {
        console.error('No user found');
        return;
      }

      // Check if user has completed onboarding
      const { data, error } = await supabase
        .from('user_preferences')
        .select('has_completed_onboarding')
        .eq('user_id', user.id)
        .single();

      if (error) {
        console.error('Failed to fetch onboarding status:', error);
        return;
      }

      // Only show onboarding if it's first time and hasn't been completed
      if (isFirstTime && (!data || !data.has_completed_onboarding)) {
        setIsOpen(true);
      }
    };

    checkOnboardingStatus();
  }, [isFirstTime]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
      highlightElement(steps[currentStep + 1].targetClass);
    } else {
      handleComplete();
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      highlightElement(steps[currentStep - 1].targetClass);
    }
  };

  const handleComplete = async () => {
    setIsOpen(false);
    
    const user = (await supabase.auth.getUser()).data.user;
    
    if (!user) {
      console.error('No user found');
      return;
    }

    // Upsert user preferences
    const { error } = await supabase
      .from('user_preferences')
      .upsert({
        user_id: user.id,
        has_completed_onboarding: true
      }, {
        onConflict: 'user_id'
      });

    if (error) {
      console.error('Failed to update onboarding status:', error);
    }
    
    onComplete();
  };

  // Keep track of highlighted element
  const highlightElement = (className: string) => {
    const element = document.querySelector(`.${className}`);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      element.classList.add('ring-4', 'ring-primary', 'ring-opacity-50', 'transition-all', 'duration-500');
      setTimeout(() => {
        element.classList.remove('ring-4', 'ring-primary', 'ring-opacity-50');
      }, 2000);
    }
  };

  useEffect(() => {
    if (isOpen) {
      highlightElement(steps[currentStep].targetClass);
    }
  }, [currentStep, isOpen, steps]);

  return (
    <Transition show={isOpen} as={Fragment}>
      <Dialog 
        onClose={() => {}}
        className="fixed inset-0 z-50 overflow-y-auto"
      >
        <div className="flex min-h-screen items-center justify-center">
          <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

          <div className="relative bg-white dark:bg-gray-800 rounded-lg p-4 sm:p-6 w-[400px] min-h-[250px] max-w-[calc(100%-2rem)]">
            <h2 className="text-lg font-medium text-gray-900 dark:text-white">
              {steps[currentStep].title}
            </h2>

            <div className="min-h-[120px] mt-4">
              <div className="text-gray-600 dark:text-gray-300">
                {steps[currentStep].description}
              </div>
            </div>

            <div className="mt-2 flex justify-between items-center">
              <button
                onClick={handlePrevious}
                className={`bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 
                  text-gray-700 dark:text-gray-300 px-4 py-2 rounded-full 
                  ${currentStep === 0 ? 'invisible' : ''}`}
              >
                Previous
              </button>
              <div className="flex items-center gap-2">
                <span className="text-sm text-gray-500">
                  {currentStep + 1} of {steps.length}
                </span>
                <button
                  onClick={handleNext}
                  className="bg-primary hover:bg-primary-dark text-white px-4 py-2 ml-2 rounded-full"
                >
                  {currentStep === steps.length - 1 ? 'Finish' : 'Next'}
                </button>
              </div>
            </div>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}