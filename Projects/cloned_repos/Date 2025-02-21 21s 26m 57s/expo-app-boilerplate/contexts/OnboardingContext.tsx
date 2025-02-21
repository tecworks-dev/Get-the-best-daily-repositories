import { createContext, useContext, useEffect, useState } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter, useSegments } from 'expo-router';

type OnboardingContextType = {
  isOnboarded: boolean;
  setIsOnboarded: (value: boolean) => void;
};

const OnboardingContext = createContext<OnboardingContextType | undefined>(undefined);

const STORAGE_KEY = 'hasCompletedOnboarding';

export function OnboardingProvider({ children }: { children: React.ReactNode }) {
  const [isOnboarded, setIsOnboarded] = useState<boolean | null>(null);
  const segments = useSegments();
  const router = useRouter();

  useEffect(() => {
    checkOnboardingStatus();
  }, []);

  useEffect(() => {
    if (isOnboarded === null) return;

    const inAuthGroup = segments[0] === 'onboarding';

    if (!isOnboarded && !inAuthGroup) {
      router.replace('/onboarding');
    } else if (isOnboarded && inAuthGroup) {
      router.replace('/');
    }
  }, [isOnboarded, segments]);

  const checkOnboardingStatus = async () => {
    try {
      const value = await AsyncStorage.getItem(STORAGE_KEY);
      setIsOnboarded(value === 'true');
    } catch (error) {
      console.error('Error checking onboarding status:', error);
      setIsOnboarded(false);
    }
  };

  const handleSetIsOnboarded = async (value: boolean) => {
    try {
      await AsyncStorage.setItem(STORAGE_KEY, String(value));
      setIsOnboarded(value);
    } catch (error) {
      console.error('Error setting onboarding status:', error);
    }
  };

  return (
    <OnboardingContext.Provider
      value={{
        isOnboarded: isOnboarded ?? false,
        setIsOnboarded: handleSetIsOnboarded,
      }}>
      {children}
    </OnboardingContext.Provider>
  );
}

export function useOnboarding() {
  const context = useContext(OnboardingContext);
  if (context === undefined) {
    throw new Error('useOnboarding must be used within an OnboardingProvider');
  }
  return context;
} 