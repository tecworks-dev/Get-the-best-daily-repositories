import { StyleSheet, View, ScrollView } from 'react-native';
import { useRouter } from 'expo-router';
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { TouchableOpacity } from 'react-native-gesture-handler';
import { SafeAreaView } from 'react-native-safe-area-context';
import { MaterialCommunityIcons } from '@expo/vector-icons';

export default function ProblemScreen() {
  const router = useRouter();

  const handleNext = () => {
    router.push('/onboarding/solution');
  };

  return (
    <ThemedView style={styles.container}>
      <SafeAreaView style={styles.safeArea}>
        <ScrollView 
          style={styles.scroll} 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
        >
          <View style={styles.header}>
            <ThemedText type="title" style={styles.title}>
              The Problem
            </ThemedText>
            <ThemedText style={styles.description}>
              Describe the main challenge or pain point your users face. Make it relatable and specific.
            </ThemedText>
          </View>

          <View style={styles.content}>
            <View style={styles.example}>
              <MaterialCommunityIcons name="alert-circle" size={32} color="#0A7EA4" />
              <ThemedText style={styles.exampleText}>
                "I struggle with X every day, and it costs me Y hours per week..."
              </ThemedText>
            </View>

            <View style={styles.points}>
              <View style={styles.point}>
                <MaterialCommunityIcons name="close" size={24} color="#E11D48" />
                <ThemedText style={styles.pointText}>
                  Current solutions are expensive and complex
                </ThemedText>
              </View>
              <View style={styles.point}>
                <MaterialCommunityIcons name="close" size={24} color="#E11D48" />
                <ThemedText style={styles.pointText}>
                  Users waste time on manual workarounds
                </ThemedText>
              </View>
              <View style={styles.point}>
                <MaterialCommunityIcons name="close" size={24} color="#E11D48" />
                <ThemedText style={styles.pointText}>
                  Existing tools lack key features
                </ThemedText>
              </View>
            </View>
          </View>
        </ScrollView>

        <View style={styles.buttonContainer}>
          <TouchableOpacity style={styles.button} onPress={handleNext}>
            <ThemedText type="defaultSemiBold" style={styles.buttonText}>
              See the Solution
            </ThemedText>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    </ThemedView>
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
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 24,
  },
  header: {
    gap: 8,
    marginBottom: 32,
  },
  title: {
    fontSize: 36,
  },
  description: {
    fontSize: 16,
    opacity: 0.7,
    lineHeight: 22,
  },
  content: {
    gap: 24,
  },
  example: {
    backgroundColor: '#0A7EA410',
    padding: 16,
    borderRadius: 16,
    flexDirection: 'row',
    gap: 12,
    alignItems: 'center',
  },
  exampleText: {
    flex: 1,
    fontSize: 16,
    fontStyle: 'italic',
    lineHeight: 22,
  },
  points: {
    gap: 12,
  },
  point: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: '#E11D4810',
    padding: 14,
    borderRadius: 12,
  },
  pointText: {
    flex: 1,
    fontSize: 15,
    lineHeight: 20,
  },
  buttonContainer: {
    paddingHorizontal: 24,
    paddingVertical: 12,
  },
  button: {
    backgroundColor: '#0A7EA4',
    padding: 16,
    borderRadius: 16,
    alignItems: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 17,
  },
}); 