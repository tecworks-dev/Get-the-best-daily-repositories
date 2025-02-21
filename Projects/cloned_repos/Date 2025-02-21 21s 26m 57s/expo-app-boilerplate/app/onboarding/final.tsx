import { StyleSheet, View, ScrollView } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { TouchableOpacity } from 'react-native-gesture-handler';
import { useSuperwall } from '@/hooks/useSuperwall';
import { useOnboarding } from '@/contexts/OnboardingContext';
import { SUPERWALL_TRIGGERS } from '@/config/superwall';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import type { MaterialCommunityIcons as IconType } from '@expo/vector-icons';

export default function FinalScreen() {
  const { showPaywall } = useSuperwall();
  const { setIsOnboarded } = useOnboarding();

  const handleGetStarted = async () => {
    try {
      await showPaywall(SUPERWALL_TRIGGERS.ONBOARDING);
      setIsOnboarded(true);
    } catch (error) {
      console.error('Failed to show paywall:', error);
    }
  };

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView style={styles.scroll} contentContainerStyle={styles.scrollContent}>
          <View style={styles.header}>
            <MaterialCommunityIcons name="rocket-launch" size={48} color="#0A7EA4" />
            <ThemedText type="title" style={styles.title}>
              Start Building Today
            </ThemedText>
            <ThemedText style={styles.description}>
              You're all set to create your next great app. Get started now and save weeks of development time!
            </ThemedText>
          </View>

          <View style={styles.benefits}>
            <Benefit icon="lightning-bolt" text="Launch faster" />
            <Benefit icon="palette" text="Professional design" />
            <Benefit icon="cash-multiple" text="Ready for monetization" />
          </View>
        </ScrollView>

        <TouchableOpacity style={styles.button} onPress={handleGetStarted}>
          <ThemedText type="defaultSemiBold" style={styles.buttonText}>
            Get Started Now
          </ThemedText>
        </TouchableOpacity>
      </SafeAreaView>
    </ThemedView>
  );
}

function Benefit({ icon, text }: { icon: keyof typeof IconType.glyphMap; text: string }) {
  return (
    <View style={styles.benefitContainer}>
      <View style={styles.iconContainer}>
        <MaterialCommunityIcons name={icon} size={24} color="#0A7EA4" />
      </View>
      <ThemedText style={styles.benefitText}>{text}</ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    paddingHorizontal: 24,
    paddingTop: 24,
    gap: 32,
  },
  header: {
    alignItems: 'center',
    gap: 16,
  },
  title: {
    fontSize: 32,
    textAlign: 'center',
  },
  description: {
    textAlign: 'center',
    fontSize: 18,
    lineHeight: 28,
    opacity: 0.7,
  },
  benefits: {
    gap: 16,
    paddingBottom: 24,
  },
  benefitContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
    backgroundColor: '#0A7EA410',
    padding: 16,
    borderRadius: 12,
  },
  iconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#0A7EA420',
    alignItems: 'center',
    justifyContent: 'center',
  },
  benefitText: {
    fontSize: 17,
  },
  button: {
    backgroundColor: '#0A7EA4',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginHorizontal: 24,
    marginBottom: 16,
  },
  buttonText: {
    color: 'white',
    fontSize: 18,
  },
}); 