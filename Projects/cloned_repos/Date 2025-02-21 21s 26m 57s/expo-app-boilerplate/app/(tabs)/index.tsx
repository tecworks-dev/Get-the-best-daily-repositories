import { Image, StyleSheet, Platform, View } from 'react-native';
import { TouchableOpacity } from 'react-native-gesture-handler';

import { HelloWave } from '@/components/HelloWave';
import ParallaxScrollView from '@/components/ParallaxScrollView';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { useSuperwall } from '@/hooks/useSuperwall';
import { useOnboarding } from '@/contexts/OnboardingContext';
import { useRouter } from 'expo-router';
import { SUPERWALL_TRIGGERS } from '@/config/superwall';

export default function HomeScreen() {
  const { setIsOnboarded } = useOnboarding();
  const { showPaywall } = useSuperwall();
  const router = useRouter();

  const handleRestartOnboarding = async () => {
    await setIsOnboarded(false);
    router.push('/onboarding');
  };

  const handleShowPaywall = () => {
    showPaywall(SUPERWALL_TRIGGERS.ONBOARDING);
  };

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
      headerImage={
        <Image
          source={require('@/assets/images/partial-react-logo.png')}
          style={styles.reactLogo}
        />
      }>
      <ThemedView style={styles.titleContainer}>
        <ThemedText type="title">Welcome!</ThemedText>
        <HelloWave />
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">Step 1: Try it</ThemedText>
        <ThemedText>
          Edit <ThemedText type="defaultSemiBold">app/(tabs)/index.tsx</ThemedText> to see changes.
          Press{' '}
          <ThemedText type="defaultSemiBold">
            {Platform.select({
              ios: 'cmd + d',
              android: 'cmd + m',
              web: 'F12'
            })}
          </ThemedText>{' '}
          to open developer tools.
        </ThemedText>
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">Step 2: Explore</ThemedText>
        <ThemedText>
          Tap the Explore tab to learn more about what's included in this starter app.
        </ThemedText>
      </ThemedView>
      <ThemedView style={styles.stepContainer}>
        <ThemedText type="subtitle">Step 3: Get a fresh start</ThemedText>
        <ThemedText>
          When you're ready, run{' '}
          <ThemedText type="defaultSemiBold">npm run reset-project</ThemedText> to get a fresh{' '}
          <ThemedText type="defaultSemiBold">app</ThemedText> directory. This will move the current{' '}
          <ThemedText type="defaultSemiBold">app</ThemedText> to{' '}
          <ThemedText type="defaultSemiBold">app-example</ThemedText>.
        </ThemedText>
      </ThemedView>
      
      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.button} onPress={handleShowPaywall}>
          <ThemedText type="defaultSemiBold" style={styles.buttonText}>
            Show Paywall
          </ThemedText>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, styles.secondaryButton]} onPress={handleRestartOnboarding}>
          <ThemedText type="defaultSemiBold" style={styles.secondaryButtonText}>
            Restart Onboarding
          </ThemedText>
        </TouchableOpacity>
      </View>
    </ParallaxScrollView>
  );
}

const styles = StyleSheet.create({
  titleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  stepContainer: {
    gap: 8,
    marginBottom: 8,
  },
  reactLogo: {
    height: 178,
    width: 290,
    bottom: 0,
    left: 0,
    position: 'absolute',
  },
  buttonContainer: {
    marginTop: 24,
    gap: 12,
  },
  button: {
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    backgroundColor: '#0A7EA4',
  },
  buttonText: {
    color: 'white',
  },
  secondaryButton: {
    backgroundColor: '#0A7EA420',
  },
  secondaryButtonText: {
    color: '#0A7EA4',
  },
});
